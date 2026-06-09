"""Read-only catalogs of the curated faces & voices a user picks from.

Each asset carries a stable ``key`` (the storage object key, to reference it in a
future anonymization edit) plus a short-lived presigned ``url`` for previewing it.
"""
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_source_asset_service
from app.models.user import User
from app.schemas.source import SourceAsset, SourceAssetKind
from app.services.source_asset_service import SourceAssetService

router = APIRouter()


@router.get("/faces", response_model=list[SourceAsset])
async def list_source_faces(
    current_user: User = Depends(get_current_user),
    service: SourceAssetService = Depends(get_source_asset_service),
) -> list[SourceAsset]:
    """List the curated source faces available for the face-swap method."""
    return await service.list_assets(SourceAssetKind.FACE)


@router.get("/voices", response_model=list[SourceAsset])
async def list_source_voices(
    current_user: User = Depends(get_current_user),
    service: SourceAssetService = Depends(get_source_asset_service),
) -> list[SourceAsset]:
    """List the curated source voices available for the voice-convert method."""
    return await service.list_assets(SourceAssetKind.VOICE)
