from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import quote

from app.storage.base import ObjectInfo, ObjectStat, Storage, StorageError

if TYPE_CHECKING:
    from app.core.config import Settings


def _content_disposition(filename: str) -> str:
    """Build an RFC 6266 ``Content-Disposition`` value that forces a download.

    Emits an ASCII ``filename`` fallback plus a UTF-8 ``filename*`` so names with
    non-ASCII characters (e.g. Vietnamese) survive on modern browsers.
    """
    ascii_name = (
        filename.encode("ascii", "ignore").decode("ascii").replace('"', "").strip()
        or "download"
    )
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


def _is_not_found(exc: Exception) -> bool:
    """True if a botocore error means "object does not exist" (404 / NoSuchKey)."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    http_status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or http_status == 404


class R2Storage(Storage):
    """Cloudflare R2 storage via the S3-compatible API (boto3).

    The boto3 client is created lazily on first use, so the app can boot without R2
    credentials configured; operations then fail with a clear :class:`StorageError`.
    Blocking boto3 calls are offloaded to a worker thread.
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
        bucket: str | None,
        region: str = "auto",
        presign_expiry_seconds: int = 3600,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._bucket = bucket
        self._region = region
        self._presign_expiry_seconds = presign_expiry_seconds
        self._client: Any | None = None

    @classmethod
    def from_settings(cls, settings: "Settings") -> "R2Storage":
        return cls(
            endpoint_url=settings.r2_endpoint_url,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket=settings.r2_bucket,
            region=settings.r2_region,
            presign_expiry_seconds=settings.r2_presign_expiry_seconds,
        )

    @property
    def is_configured(self) -> bool:
        return all(
            (
                self._endpoint_url,
                self._access_key_id,
                self._secret_access_key,
                self._bucket,
            )
        )

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.is_configured:
            raise StorageError(
                "Cloudflare R2 is not configured. Set R2_ENDPOINT_URL, "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET."
            )
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - depends on install
            raise StorageError(
                "boto3 is required for R2 storage but is not installed."
            ) from exc

        self._client = boto3.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key_id,
            aws_secret_access_key=self._secret_access_key,
            region_name=self._region,
            config=Config(signature_version="s3v4"),
        )
        return self._client

    async def upload_fileobj(
        self,
        key: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
    ) -> None:
        client = self._client_or_raise()
        extra_args = {"ContentType": content_type} if content_type else None
        try:
            await asyncio.to_thread(
                client.upload_fileobj, fileobj, self._bucket, key, ExtraArgs=extra_args
            )
        except StorageError:
            raise
        except Exception as exc:  # botocore.ClientError and friends
            raise StorageError(f"Failed to upload object '{key}' to R2: {exc}") from exc

    async def download_to_path(self, key: str, dest_path: str) -> None:
        client = self._client_or_raise()
        try:
            await asyncio.to_thread(
                client.download_file, self._bucket, key, dest_path
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to download object '{key}' from R2: {exc}"
            ) from exc

    async def generate_presigned_get_url(
        self,
        key: str,
        expires_in: int | None = None,
        download_filename: str | None = None,
    ) -> str:
        client = self._client_or_raise()
        expires = int(expires_in) if expires_in else self._presign_expiry_seconds
        params: dict[str, Any] = {"Bucket": self._bucket, "Key": key}
        if download_filename:
            params["ResponseContentDisposition"] = _content_disposition(download_filename)
        try:
            return await asyncio.to_thread(
                client.generate_presigned_url,
                "get_object",
                Params=params,
                ExpiresIn=expires,
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to presign download URL for '{key}': {exc}"
            ) from exc

    async def generate_presigned_put_url(
        self,
        key: str,
        expires_in: int | None = None,
    ) -> str:
        client = self._client_or_raise()
        expires = int(expires_in) if expires_in else self._presign_expiry_seconds
        try:
            # Sign only the bucket + key: the browser may send a Content-Type header
            # (which R2 stores on the object) without it being part of the signature,
            # so a header mismatch can never break the upload.
            return await asyncio.to_thread(
                client.generate_presigned_url,
                "put_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires,
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(
                f"Failed to presign upload URL for '{key}': {exc}"
            ) from exc

    async def list_objects(self, prefix: str) -> list[ObjectInfo]:
        client = self._client_or_raise()
        try:
            pages = await asyncio.to_thread(self._list_all_pages, client, prefix)
        except StorageError:
            raise
        except Exception as exc:  # botocore.ClientError and friends
            raise StorageError(
                f"Failed to list objects under '{prefix}' in R2: {exc}"
            ) from exc

        objects: list[ObjectInfo] = []
        for entry in pages:
            key = str(entry.get("Key", ""))
            # Skip the zero-byte "directory marker" some clients create for a prefix.
            if not key or key == prefix or key.endswith("/"):
                continue
            objects.append(ObjectInfo(key=key, size_bytes=int(entry.get("Size", 0))))
        return objects

    def _list_all_pages(self, client: Any, prefix: str) -> list[dict[str, Any]]:
        """Walk every page of ``list_objects_v2`` (blocking; runs in a worker thread)."""
        paginator = client.get_paginator("list_objects_v2")
        contents: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            contents.extend(page.get("Contents", []))
        return contents

    async def head(self, key: str) -> ObjectStat | None:
        client = self._client_or_raise()
        try:
            response = await asyncio.to_thread(
                client.head_object, Bucket=self._bucket, Key=key
            )
        except StorageError:
            raise
        except Exception as exc:  # botocore.ClientError and friends
            if _is_not_found(exc):
                return None
            raise StorageError(f"Failed to head object '{key}' in R2: {exc}") from exc
        return ObjectStat(
            size_bytes=int(response.get("ContentLength", 0)),
            content_type=response.get("ContentType"),
        )

    async def delete(self, key: str) -> None:
        client = self._client_or_raise()
        try:
            await asyncio.to_thread(
                client.delete_object, Bucket=self._bucket, Key=key
            )
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to delete object '{key}' from R2: {exc}") from exc
