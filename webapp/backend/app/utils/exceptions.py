from fastapi import Request, status
from fastapi.responses import JSONResponse

from app.storage.base import StorageError


class AppException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


async def storage_error_handler(_: Request, exc: StorageError) -> JSONResponse:
    # Storage misconfiguration or an object-store failure is an upstream problem.
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )
