from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LiveVisualMethod(str, Enum):
    """Face filters available on the live path — the visual subset of ``VisualMethod``.

    Face swap is intentionally excluded: it needs a source identity and per-frame
    landmarks, which is offline-only.
    """

    NONE = "none"
    BLUR = "blur"
    PIXELATE = "pixelate"
    MASK = "mask"
    BLACKOUT = "blackout"


class MaskShape(str, Enum):
    """How the face region is masked on the live path (mirrors ai_core's ``MaskShape``).

    ``parser`` runs a precise BiSeNet segmentation that hugs the face (lower FPS);
    ``ellipse`` uses a coarse, model-free ellipse (much higher FPS). It is the single
    biggest real-time lever, so it is exposed to the user.
    """

    PARSER = "parser"
    ELLIPSE = "ellipse"


class LiveConfigMessage(BaseModel):
    """One real-time filter-config message sent by the browser over the live socket.

    Mirrors the visual knobs of :class:`~app.schemas.video.VideoEditCreate` and is
    validated at the edge so the live worker only ever sees clean values. Sent on
    connect and whenever the user changes a control.
    """

    visual_method: LiveVisualMethod = LiveVisualMethod.NONE
    # Gaussian kernel size for BLUR; coerced to an odd number >= 3 downstream.
    blur_strength: int = Field(default=31, ge=3, le=199)
    # PIXELATE block coarseness — lower means chunkier blocks (more obscured).
    pixelation_level: int = Field(default=16, ge=4, le=256)
    # Solid fill for MASK, as a ``#RRGGBB`` hex colour.
    mask_color: str = Field(default="#A0A0A0", pattern=r"^#?[0-9a-fA-F]{6}$")
    # How the face region is masked: precise BiSeNet parse vs cheap ellipse. The
    # ellipse skips a per-face model inference, so it is the main real-time FPS lever.
    mask_shape: MaskShape = MaskShape.PARSER
    # Overlay tracker boxes on the streamed-back frame.
    draw_boxes: bool = False
    # Detector cadence: run it every N frames (higher = lower latency).
    detect_interval: int = Field(default=2, ge=1, le=10)

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")
