from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
# Import models so SQLAlchemy registers tables before create_all runs.
from app.models import user as _user_models  # noqa: F401
from app.services.face_swap_service import FaceSwapService
from app.services.video_service import VideoPipelineService
from app.utils.exceptions import AppException, app_exception_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    video_service = VideoPipelineService.create_from_settings(settings)
    app.state.video_service = video_service
    app.state.face_swap_service = FaceSwapService(
        settings=settings,
        video_service=video_service,
    )
    try:
        yield
    finally:
        app.state.video_service = None
        app.state.face_swap_service = None


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

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_application()
