from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import jwt

from app.core.config import settings


MAX_BCRYPT_PASSWORD_BYTES = 72


def _encode_password(password: str) -> bytes:
    encoded_password = password.encode("utf-8")
    if len(encoded_password) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("Password cannot be longer than 72 bytes.")
    return encoded_password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            _encode_password(plain_password),
            hashed_password.encode("utf-8"),
        )
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(_encode_password(password), bcrypt.gensalt()).decode("utf-8")


def create_access_token(
    subject: str | int, expires_delta: timedelta | None = None
) -> str:
    expire_at = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload: dict[str, Any] = {
        "sub": str(subject),
        "exp": expire_at,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
