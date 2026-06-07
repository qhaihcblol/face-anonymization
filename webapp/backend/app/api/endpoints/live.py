"""Real-time live-camera anonymization over a WebSocket.

Protocol (one connection == one camera session):

* Client → server
    - **Text (JSON)**: a :class:`~app.schemas.live.LiveConfigMessage`. Sent on connect
      and whenever the user changes a control. Updates the filter without resetting
      tracker state.
    - **Binary**: one JPEG-encoded frame to anonymize.
* Server → client
    - ``{"type": "ready"}``               — session built, start sending frames.
    - ``{"type": "config_ack"}``          — a config message was applied.
    - ``{"type": "error", "detail": …}``  — bad config / engine unavailable.
    - **Binary** processed JPEG frame, immediately followed by
      ``{"type": "frame", …meta}`` (dimensions, timing, detected faces). On a frame
      that cannot be decoded/encoded, ``{"type": "frame_error", …}`` is sent instead.

Each binary frame yields exactly one terminal response (``frame`` or ``frame_error``),
so the client can stream with backpressure — send the next frame only once the
previous one's response arrives.

Auth: the browser cannot set an ``Authorization`` header on a WebSocket, so the JWT
is passed as a ``?token=`` query parameter and validated before the handshake is
accepted.
"""
from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import anyio
import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from jose import JWTError
from pydantic import ValidationError

from app.core.security import decode_access_token
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.live import LiveConfigMessage

logger = logging.getLogger(__name__)

router = APIRouter()


async def _authenticate(token: str | None) -> User | None:
    """Resolve the user behind a ``?token=`` JWT, or ``None`` if it is invalid."""
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        subject = payload.get("sub")
        if subject is None:
            return None
        user_id = int(subject)
    except (JWTError, ValueError, TypeError):
        return None

    async with AsyncSessionLocal() as db:
        return await UserRepository.get_by_id(db, user_id)


def _render_frame(
    session: Any, data: bytes, quality: int
) -> tuple[bytes, dict[str, Any]] | None:
    """Decode → anonymize → re-encode one JPEG frame. CPU-bound: run in a thread.

    Returns ``(jpeg_bytes, meta)`` or ``None`` if the frame could not be decoded or
    re-encoded.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    result = session.process_frame(frame)

    ok, encoded = cv2.imencode(
        ".jpg", result.frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        return None

    height, width = result.frame.shape[:2]
    meta: dict[str, Any] = {
        "type": "frame",
        "width": int(width),
        "height": int(height),
        "detected": bool(result.detected),
        "detect_ms": round(result.detect_ms, 1),
        "process_ms": round(result.process_ms, 1),
        "faces": [
            {
                "id": int(track["track_id"]),
                "bbox": [round(float(value), 1) for value in track["bbox"]],
                "score": round(float(track["score"]), 3),
            }
            for track in result.tracks
        ],
    }
    return encoded.tobytes(), meta


async def _apply_config(
    websocket: WebSocket, pipeline: Any, session: Any, text: str
) -> None:
    """Validate a config message and apply it to ``session`` (keeps tracker state)."""
    try:
        message = LiveConfigMessage.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        await websocket.send_json({"type": "error", "detail": "Invalid filter config."})
        return

    # mode="json" serializes the LiveVisualMethod enum to its value ("blur"); the
    # default python mode keeps the member, and build_live_config's str() coercion
    # would then turn it into "livevisualmethod.blur" and reject it.
    session.configure(pipeline.build_live_config(message.model_dump(mode="json")))
    await websocket.send_json({"type": "config_ack"})


@router.websocket("/ws")
async def live_ws(
    websocket: WebSocket, token: str | None = Query(default=None)
) -> None:
    user = await _authenticate(token)
    if user is None:
        # Reject before the handshake completes -> the browser sees a failed connect.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    pipeline = getattr(websocket.app.state, "live_pipeline", None)
    limiter = getattr(websocket.app.state, "live_limiter", None)
    if pipeline is None or limiter is None:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    quality = int(getattr(websocket.app.state, "live_jpeg_quality", 80))

    await websocket.accept()

    # Building the session may trigger the lazy ONNX model load on first ever use.
    try:
        session = await anyio.to_thread.run_sync(pipeline.create_live_session)
    except Exception:  # noqa: BLE001 - surface as a clean close, not a 500
        logger.exception("Failed to build live session for user id=%s", user.id)
        await websocket.send_json({"type": "error", "detail": "Engine unavailable."})
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    await websocket.send_json({"type": "ready"})

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                await _apply_config(websocket, pipeline, session, text)
                continue

            data = message.get("bytes")
            if not data:
                continue

            rendered = await anyio.to_thread.run_sync(
                _render_frame, session, data, quality, limiter=limiter
            )
            if rendered is None:
                await websocket.send_json(
                    {"type": "frame_error", "detail": "Could not process frame."}
                )
                continue

            frame_bytes, meta = rendered
            await websocket.send_bytes(frame_bytes)
            await websocket.send_json(meta)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - log and tear the socket down cleanly
        logger.exception("Live session error for user id=%s", user.id)
    finally:
        with contextlib.suppress(RuntimeError):
            await websocket.close()
