from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualMethod(str, Enum):
    """How faces are anonymized. Mirrors ai_core but kept independent of it."""

    NONE = "none"
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"
    BLACKOUT = "blackout"
    SWAP = "swap"


class VoiceMethod(str, Enum):
    """How the voice is anonymized (only when ``anonymize_voice`` is set)."""

    NONE = "none"
    MCADAMS = "mcadams"
    PITCH = "pitch"
    FORMANT = "formant"
    PITCH_FORMANT = "pitch_formant"
    CONVERT = "convert"


class VideoEditCreate(BaseModel):
    """Parameters for a new anonymization edit; persisted as ``VideoEdit.params``."""

    visual_method: VisualMethod = VisualMethod.BLUR
    keep_audio: bool = True
    anonymize_voice: bool = False
    voice_method: VoiceMethod = VoiceMethod.MCADAMS
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, gt=0)

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoEditCreate":
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec <= self.start_sec
        ):
            raise ValueError("end_sec must be greater than start_sec.")
        return self


class VideoPublic(BaseModel):
    id: int
    original_filename: str
    content_type: str | None
    size_bytes: int | None
    duration_sec: float | None
    width: int | None
    height: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoEditPublic(BaseModel):
    id: int
    video_id: int
    status: str
    params: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    model_config = ConfigDict(from_attributes=True)


class PresignedUrlResponse(BaseModel):
    url: str
    expires_in: int
