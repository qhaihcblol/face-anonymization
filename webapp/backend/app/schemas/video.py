from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FilterMethod = Literal["none", "blur", "pixelate", "mask", "blackout"]
VoiceMethod = Literal[
    "none", "mcadams", "pitch", "formant", "pitch_formant", "convert"
]


class VoiceOptions(BaseModel):
    """Voice anonymization controls shared by the anonymize and face-swap paths.

    ``voice_method="none"`` keeps the original audio; any other method runs the
    :class:`ai_core...VoiceAnonymizer`. ``pitch_steps``/``formant_shift`` apply to
    the pitch/formant DSP methods, ``mcadams_alpha`` to the McAdams transform.
    """

    voice_method: VoiceMethod = "none"
    pitch_steps: float = -4.0
    formant_shift: float = Field(default=1.2, gt=0)
    mcadams_alpha: float = Field(default=0.8, gt=0)


class VideoMetadataPublic(BaseModel):
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    size_bytes: int
    metadata: VideoMetadataPublic
    original_video_url: str
    anonymized_video_url: str


class VideoAnonymizeRequest(VoiceOptions):
    method: FilterMethod = "blur"
    detect_interval: int = Field(default=1, ge=1)
    target_fps: int | None = Field(default=None, gt=0)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    blur_new: bool = False
    draw_tracks: bool = False
    codec: str = "H264"
    progress_every: int = Field(default=60, ge=0)
    # Method-specific appearance controls. ``mask_color`` is RGB (web convention);
    # the service converts it to BGR for OpenCV.
    blur_strength: int = Field(default=31, ge=3)
    pixelation_level: int = Field(default=16, ge=4)
    mask_color: tuple[int, int, int] = (160, 160, 160)

    @field_validator("mask_color")
    @classmethod
    def validate_mask_color(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("mask_color channels must be within [0, 255]")
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "VideoAnonymizeRequest":
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec <= self.start_sec
        ):
            raise ValueError("end_sec must be greater than start_sec")

        if len(self.codec) != 4:
            raise ValueError("codec must be a 4-character string")

        return self


class VideoAnonymizeResponse(BaseModel):
    video_id: str
    method: FilterMethod
    voice_method: VoiceMethod
    target_fps: int | None
    start_sec: float | None
    end_sec: float | None
    output_video_url: str
    output_metadata: VideoMetadataPublic
    elapsed_sec: float
    throughput_fps: float


class VideoFaceSwapRequest(VoiceOptions):
    """Parameters for the BlendSwap (model-based) face-swap pipeline.

    Kept separate from :class:`VideoAnonymizeRequest` because the swap path takes
    its own controls (temporal stabilization + one-euro smoothing) and never uses
    the bbox-based options (``method``, ``detect_interval``, ``draw_tracks``). Voice
    controls are inherited from :class:`VoiceOptions`.
    """

    target_fps: int | None = Field(default=None, gt=0)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    codec: str = "H264"
    progress_every: int = Field(default=60, ge=0)
    stabilize: bool = True
    smooth_min_cutoff: float = Field(default=0.5, gt=0)
    smooth_beta: float = Field(default=0.05, ge=0)
    output_smooth: float = Field(default=0.4, ge=0, le=1)
    mask_smooth: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_request(self) -> "VideoFaceSwapRequest":
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec <= self.start_sec
        ):
            raise ValueError("end_sec must be greater than start_sec")

        if len(self.codec) != 4:
            raise ValueError("codec must be a 4-character string")

        return self


class VideoFaceSwapResponse(BaseModel):
    video_id: str
    method: Literal["swap"] = "swap"
    voice_method: VoiceMethod
    target_fps: int | None
    start_sec: float | None
    end_sec: float | None
    stabilize: bool
    output_video_url: str
    output_metadata: VideoMetadataPublic
    elapsed_sec: float
    throughput_fps: float
