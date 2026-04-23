from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import field_validator
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
    video_upload_dir: str = "storage/uploads"
    video_output_dir: str = "storage/outputs"
    retinaface_onnx_path: str | None = None
    video_max_upload_mb: int = 2048
    video_allowed_extensions: list[str] = [".mp4", ".mov", ".webm", ".mkv", ".avi"]

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
    def backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[4]

    @property
    def ai_core_root(self) -> Path:
        return self.project_root / "ai_core"

    @property
    def resolved_retinaface_onnx_path(self) -> Path:
        if self.retinaface_onnx_path:
            configured = Path(self.retinaface_onnx_path)
            if not configured.is_absolute():
                configured = (self.backend_root / configured).resolve()
            return configured
        return (self.ai_core_root / "face_detection" / "onnx" / "retinaface_best.onnx").resolve()

    @property
    def resolved_video_upload_dir(self) -> Path:
        directory = Path(self.video_upload_dir)
        if not directory.is_absolute():
            directory = self.backend_root / directory
        return directory.resolve()

    @property
    def resolved_video_output_dir(self) -> Path:
        directory = Path(self.video_output_dir)
        if not directory.is_absolute():
            directory = self.backend_root / directory
        return directory.resolve()

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
