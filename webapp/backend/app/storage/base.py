from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO


class StorageError(RuntimeError):
    """Raised when an object-storage operation cannot be completed.

    Covers both misconfiguration (no credentials) and runtime failures (network,
    permissions). The API layer maps it to a 502 response.
    """


@dataclass(frozen=True)
class ObjectStat:
    """Metadata of a stored object, as returned by :meth:`Storage.head`."""

    size_bytes: int
    content_type: str | None = None


class Storage(ABC):
    """Abstract object-storage backend (S3 / Cloudflare R2 / ...).

    Keys are opaque object keys. Methods are async so a blocking SDK can be offloaded
    to a worker thread inside the implementation, keeping the event loop free.
    """

    @abstractmethod
    async def upload_fileobj(
        self,
        key: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
    ) -> None:
        """Stream a binary file object to ``key`` (overwriting any existing object)."""

    @abstractmethod
    async def download_to_path(self, key: str, dest_path: str) -> None:
        """Download the object at ``key`` to the local file ``dest_path``."""

    @abstractmethod
    async def generate_presigned_get_url(
        self,
        key: str,
        expires_in: int | None = None,
        download_filename: str | None = None,
    ) -> str:
        """Return a time-limited URL to download the object at ``key``.

        When ``download_filename`` is given, the URL is signed so the storage host
        responds with ``Content-Disposition: attachment``, making the browser save
        the file under that name rather than display it inline — which is what lets a
        cross-origin link download instead of opening in a new tab.
        """

    @abstractmethod
    async def generate_presigned_put_url(
        self,
        key: str,
        expires_in: int | None = None,
    ) -> str:
        """Return a time-limited URL the client can ``PUT`` an object to directly.

        Lets the browser upload straight to object storage, so large files never pass
        through the application servers. Only the bucket + key are signed, so the
        client may send a ``Content-Type`` header without breaking the signature.
        """

    @abstractmethod
    async def head(self, key: str) -> ObjectStat | None:
        """Return the object's metadata, or ``None`` if it does not exist."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete the object at ``key`` (no error if it does not exist)."""
