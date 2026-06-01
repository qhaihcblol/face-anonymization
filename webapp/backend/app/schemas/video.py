from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

FilterMethod = Literal["none", "blur", "pixelate", "mask", "blackout"]


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


class VideoAnonymizeRequest(BaseModel):
    method: FilterMethod = "blur"
    detect_interval: int = Field(default=1, ge=1)
    target_fps: int | None = Field(default=None, gt=0)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    blur_new: bool = False
    draw_tracks: bool = False
    codec: str = "H264"
    progress_every: int = Field(default=60, ge=0)

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
    target_fps: int | None
    start_sec: float | None
    end_sec: float | None
    output_video_url: str
    output_metadata: VideoMetadataPublic
    elapsed_sec: float
    throughput_fps: float


class VideoFaceSwapRequest(BaseModel):
    """Parameters for the BlendSwap (model-based) face-swap pipeline.

    Kept separate from :class:`VideoAnonymizeRequest` because the swap path takes
    its own controls (temporal stabilization + one-euro smoothing) and never uses
    the bbox-based options (``method``, ``detect_interval``, ``draw_tracks``).
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
    target_fps: int | None
    start_sec: float | None
    end_sec: float | None
    stabilize: bool
    output_video_url: str
    output_metadata: VideoMetadataPublic
    elapsed_sec: float
    throughput_fps: float
