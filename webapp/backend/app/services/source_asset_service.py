from __future__ import annotations

import asyncio
import mimetypes
import time
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.core.config import settings
from app.schemas.source import SourceAsset, SourceAssetKind
from app.storage.base import ObjectInfo, Storage

# Tokens that, when they lead a filename, are read as the asset's gender group.
_GENDER_TOKENS = {"female", "male"}


@dataclass(frozen=True)
class _CatalogSpec:
    """Where a catalog lives in storage and which files count as members of it."""

    prefix: str
    extensions: frozenset[str]


@dataclass
class _CacheEntry:
    """A cached object listing for one catalog, with the time it was fetched."""

    fetched_at: float
    objects: list[ObjectInfo]


class SourceAssetService:
    """Read-only catalog of the curated faces & voices users pick from.

    The assets are shared (not per-user) objects under fixed R2 prefixes, so this
    service owns no database state: it lists the prefix, derives display metadata from
    each key, and mints a short-lived presigned URL for the preview. The listing is
    cached for a short TTL because the catalogs change rarely; the presigned URLs are
    always minted fresh so a cached entry never hands out an expired link.
    """

    def __init__(self, *, storage: Storage) -> None:
        self.storage = storage
        self._cache: dict[SourceAssetKind, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def list_assets(self, kind: SourceAssetKind) -> list[SourceAsset]:
        """Return every asset in ``kind``'s catalog, each with a fresh preview URL."""
        spec = self._spec(kind)
        objects = await self._listing(kind, spec)

        # Presign concurrently — each URL is an independent storage round-trip.
        urls = await asyncio.gather(
            *(self.storage.generate_presigned_get_url(obj.key) for obj in objects)
        )
        return [self._to_asset(kind, obj, url) for obj, url in zip(objects, urls)]

    # ------------------------------------------------------------------ #
    # Listing (TTL-cached)                                                #
    # ------------------------------------------------------------------ #
    async def _listing(
        self, kind: SourceAssetKind, spec: _CatalogSpec
    ) -> list[ObjectInfo]:
        cached = self._cache.get(kind)
        if cached is not None and not self._expired(cached):
            return cached.objects

        async with self._lock:
            # Re-check under the lock so a stampede of requests lists R2 only once.
            cached = self._cache.get(kind)
            if cached is not None and not self._expired(cached):
                return cached.objects

            objects = await self.storage.list_objects(spec.prefix)
            objects = [
                obj
                for obj in objects
                if PurePosixPath(obj.key).suffix.lower() in spec.extensions
            ]
            objects.sort(key=lambda obj: obj.key)
            self._cache[kind] = _CacheEntry(fetched_at=time.monotonic(), objects=objects)
            return objects

    @staticmethod
    def _expired(entry: _CacheEntry) -> bool:
        ttl = settings.source_assets_cache_ttl_seconds
        return ttl <= 0 or (time.monotonic() - entry.fetched_at) >= ttl

    # ------------------------------------------------------------------ #
    # Mapping                                                             #
    # ------------------------------------------------------------------ #
    def _to_asset(
        self, kind: SourceAssetKind, obj: ObjectInfo, url: str
    ) -> SourceAsset:
        name, gender = self._humanize(obj.key)
        return SourceAsset(
            kind=kind,
            key=obj.key,
            name=name,
            gender=gender,
            url=url,
            content_type=mimetypes.guess_type(obj.key)[0],
            size_bytes=obj.size_bytes,
            expires_in=settings.r2_presign_expiry_seconds,
        )

    @staticmethod
    def _humanize(key: str) -> tuple[str, str | None]:
        """Derive a display name and (optional) gender from an object key.

        ``source_faces/female_1.jpeg`` -> ``("Female 1", "female")``; a stem without a
        leading gender token keeps every part of its name and reports no gender.
        """
        stem = PurePosixPath(key).stem
        tokens = [tok for tok in stem.replace("-", "_").split("_") if tok]
        gender = tokens[0].lower() if tokens and tokens[0].lower() in _GENDER_TOKENS else None
        name = " ".join(tok.capitalize() for tok in tokens) or stem
        return name, gender

    @staticmethod
    def _spec(kind: SourceAssetKind) -> _CatalogSpec:
        if kind is SourceAssetKind.FACE:
            return _CatalogSpec(
                prefix=settings.source_faces_prefix,
                extensions=frozenset(settings.resolved_source_face_extensions),
            )
        return _CatalogSpec(
            prefix=settings.source_voices_prefix,
            extensions=frozenset(settings.resolved_source_voice_extensions),
        )
