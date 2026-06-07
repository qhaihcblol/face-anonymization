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
    """Parameters for a new anonymization edit; persisted as ``VideoEdit.params``.

    Mirrors the knobs the ``ai_core`` pipeline accepts. Defaults match ``ai_core`` so
    an empty request reproduces the engine's out-of-the-box behaviour. Each field is
    validated here at the edge so the worker never has to defend against bad input.
    """

    # --- Visual (face) ---
    visual_method: VisualMethod = VisualMethod.BLUR
    # Gaussian kernel size for BLUR; coerced to an odd number >= 3 downstream.
    blur_strength: int = Field(default=31, ge=3, le=199)
    # PIXELATE block coarseness — lower means chunkier blocks (more obscured).
    pixelation_level: int = Field(default=16, ge=4, le=256)
    # Solid fill for MASK, as a ``#RRGGBB`` hex colour.
    mask_color: str = Field(default="#A0A0A0", pattern=r"^#?[0-9a-fA-F]{6}$")
    # Overlay tracker boxes on the output (ignored for the SWAP method).
    draw_boxes: bool = False

    # --- Audio (voice) ---
    keep_audio: bool = True
    anonymize_voice: bool = False
    voice_method: VoiceMethod = VoiceMethod.MCADAMS
    # McAdams warp strength; values further from 1.0 are stronger.
    mcadams_alpha: float = Field(default=0.8, gt=0, le=2.0)
    # Pitch shift in semitones (negative lowers the pitch).
    pitch_steps: float = Field(default=-4.0, ge=-12.0, le=12.0)
    # Formant scale; > 1 raises formants, < 1 lowers them.
    formant_shift: float = Field(default=1.2, gt=0, le=3.0)

    # --- Processing range ---
    # Downsample to this FPS (never upsamples); ``None`` keeps the source rate.
    target_fps: int | None = Field(default=None, gt=0, le=240)
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
