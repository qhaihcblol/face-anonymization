from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    project_name: str = "FaceGuard Backend API"
    api_prefix: str = "/api"
    database_url: str
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    sql_echo: bool = False

    # --- Object storage (Cloudflare R2, via the S3-compatible API) ---
    r2_endpoint_url: str | None = None
    r2_access_key_id: str | None = None
    r2_secret_access_key: str | None = None
    r2_bucket: str | None = None
    r2_region: str = "auto"
    r2_presign_expiry_seconds: int = 3600

    # --- Video upload constraints ---
    video_max_upload_mb: int = 2048
    video_allowed_extensions: list[str] = [".mp4", ".mov", ".webm", ".mkv", ".avi"]

    # --- Anonymization pipeline (ai_core) ---
    # Optional override for the RetinaFace detector weights; defaults to the model
    # bundled inside the ai_core package when unset.
    retinaface_onnx_path: str | None = None

    # kNN-VC voice conversion (the "convert" voice method). Paths default to the
    # assets bundled in ai_core/voice_anonymization/ when unset; if the models are
    # missing the pipeline falls back to DSP voice methods (convert disabled).
    knnvc_reference_voice_path: str | None = None
    knnvc_encoder_onnx_path: str | None = None
    knnvc_vocoder_onnx_path: str | None = None
    knnvc_topk: int = 4

    # --- Live camera (real-time WebSocket) ---
    # JPEG quality (1-100) of the processed frames streamed back to the browser.
    live_jpeg_quality: int = Field(default=80, ge=1, le=100)
    # Run the detector every N frames; in-between frames reuse tracker predictions.
    live_detect_interval: int = Field(default=2, ge=1, le=10)
    # Cap concurrent (CPU-bound) frame inferences across all live sessions so a burst
    # of viewers cannot starve the event loop / the offline edit worker.
    live_max_concurrent_frames: int = Field(default=2, ge=1, le=32)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            raw_value = value.strip()
            if not raw_value:
                return []
            if raw_value.startswith("["):
                return value
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        return value

    @field_validator("video_allowed_extensions", mode="before")
    @classmethod
    def parse_video_extensions(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            raw_value = value.strip()
            if not raw_value:
                return []
            if raw_value.startswith("["):
                return value
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        return value

    @field_validator("video_max_upload_mb")
    @classmethod
    def validate_video_max_upload_mb(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("video_max_upload_mb must be > 0")
        return value

    @property
    def async_database_url(self) -> str:
        if self.database_url.startswith("postgresql+asyncpg://"):
            async_url = self.database_url
        elif self.database_url.startswith("postgresql://"):
            async_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif self.database_url.startswith("postgres://"):
            async_url = self.database_url.replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
        else:
            raise ValueError(
                "DATABASE_URL must start with postgresql://, postgres://, or postgresql+asyncpg://"
            )

        parsed = urlsplit(async_url)
        filtered_query: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered_key = key.lower()
            if lowered_key in {"sslmode", "channel_binding"}:
                continue
            filtered_query.append((key, value))

        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(filtered_query),
                parsed.fragment,
            )
        )

    @property
    def async_database_connect_args(self) -> dict[str, bool]:
        connect_args: dict[str, bool] = {}
        parsed = urlsplit(self.database_url)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key.lower() != "sslmode":
                continue

            mode = value.strip().lower()
            if mode in {"disable", "allow"}:
                connect_args["ssl"] = False
            elif mode:
                connect_args["ssl"] = True

        return connect_args

    @property
    def r2_configured(self) -> bool:
        return all(
            (
                self.r2_endpoint_url,
                self.r2_access_key_id,
                self.r2_secret_access_key,
                self.r2_bucket,
            )
        )

    @property
    def resolved_video_allowed_extensions(self) -> set[str]:
        return {
            f".{ext.lower().lstrip('.')}"
            for ext in self.video_allowed_extensions
            if ext and ext.strip()
        }

    @property
    def video_max_upload_bytes(self) -> int:
        return int(self.video_max_upload_mb) * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]


settings = get_settings()
