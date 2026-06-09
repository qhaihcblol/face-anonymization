from fastapi import APIRouter

from app.api.endpoints import auth, live, sources, user, video


api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(video.router, prefix="/videos", tags=["videos"])
api_router.include_router(sources.router, prefix="/sources", tags=["sources"])
api_router.include_router(live.router, prefix="/live", tags=["live"])
