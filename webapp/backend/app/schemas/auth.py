from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.user import UserPublic


MAX_BCRYPT_PASSWORD_BYTES = 72


def _validate_password_bytes_limit(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_BCRYPT_PASSWORD_BYTES:
        raise ValueError("Password cannot be longer than 72 bytes.")
    return value


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120, alias="fullName")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128, alias="confirmPassword")

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    @field_validator("password", "confirm_password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        return _validate_password_bytes_limit(value)

    @model_validator(mode="after")
    def validate_password_match(self) -> "RegisterRequest":
        if self.password != self.confirm_password:
            raise ValueError("Password and confirm password do not match.")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, value: str) -> str:
        return _validate_password_bytes_limit(value)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
