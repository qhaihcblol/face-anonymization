from functools import lru_cache
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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]


settings = get_settings()
