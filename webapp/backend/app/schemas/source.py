from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class SourceAssetKind(str, Enum):
    """Which catalog a :class:`SourceAsset` belongs to."""

    FACE = "face"
    VOICE = "voice"


class SourceAsset(BaseModel):
    """One curated, selectable asset (a face image or a voice clip).

    ``key`` is the asset's stable identifier: it is the R2 object key, and it is what
    a future edit request will reference to pick this face/voice. ``url`` is a
    short-lived presigned link the browser uses only to preview the asset (show the
    thumbnail / play the clip) and must not be persisted.
    """

    kind: SourceAssetKind
    key: str
    # Human-friendly label derived from the filename, e.g. "Female 1".
    name: str
    # Best-effort group parsed from the filename ("female" / "male"); ``None`` when the
    # filename does not encode one.
    gender: str | None
    url: str
    content_type: str | None
    size_bytes: int
    # Lifetime of ``url`` in seconds; refetch the catalog once it elapses.
    expires_in: int
