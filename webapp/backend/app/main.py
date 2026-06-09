from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import AsyncSessionLocal, engine
# Import models so SQLAlchemy registers all tables before create_all runs.
from app import models as _models  # noqa: F401
from app.processing.pipeline import AnonymizationPipeline
from app.processing.processor import LocalVideoProcessor
from app.services.source_asset_service import SourceAssetService
from app.services.video_service import VideoService
from app.storage.base import StorageError
from app.storage.r2 import R2Storage
from app.utils.exceptions import (
    AppException,
    app_exception_handler,
    storage_error_handler,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Compose the dependencies here so the request layer only sees the VideoService
    # interface. The anonymization engine itself loads lazily on the first edit.
    # One pipeline is shared by the offline edit worker and the live socket, so the
    # ONNX models are loaded into memory only once.
    storage = R2Storage.from_settings(settings)
    pipeline = AnonymizationPipeline.from_settings(settings)
    processor = LocalVideoProcessor(
        storage=storage,
        pipeline=pipeline,
        session_factory=AsyncSessionLocal,
    )
    app.state.video_service = VideoService(storage=storage, processor=processor)

    # Curated face/voice catalogs share the one storage client; the service is
    # stateless beyond a short-lived listing cache, so it is built here once.
    app.state.source_asset_service = SourceAssetService(storage=storage)

    # Live camera (real-time): share the pipeline and bound concurrent inferences so
    # a burst of viewers cannot starve the event loop or the offline edit worker.
    app.state.live_pipeline = pipeline
    app.state.live_limiter = anyio.CapacityLimiter(settings.live_max_concurrent_frames)
    app.state.live_jpeg_quality = settings.live_jpeg_quality
    try:
        yield
    finally:
        app.state.video_service = None
        app.state.source_asset_service = None
        app.state.live_pipeline = None
        app.state.live_limiter = None


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(StorageError, storage_error_handler)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_application()
