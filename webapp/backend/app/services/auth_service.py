from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest
from app.schemas.user import UserPublic
from app.utils.exceptions import AppException


class AuthService:
    @staticmethod
    def _build_auth_response(user: User) -> AuthResponse:
        return AuthResponse(
            access_token=create_access_token(subject=user.id),
            user=UserPublic.model_validate(user),
        )

    @staticmethod
    async def register(db: AsyncSession, payload: RegisterRequest) -> AuthResponse:
        existing_user = await UserRepository.get_by_email(db, payload.email)
        if existing_user:
            raise AppException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered.",
            )

        hashed_password = get_password_hash(payload.password)
        user = await UserRepository.create(
            db=db,
            full_name=payload.full_name,
            email=payload.email,
            hashed_password=hashed_password,
        )

        return AuthService._build_auth_response(user)

    @staticmethod
    async def login(db: AsyncSession, payload: LoginRequest) -> AuthResponse:
        user = await UserRepository.get_by_email(db, payload.email)
        if user is None or not verify_password(payload.password, user.hashed_password):
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not user.is_active:
            raise AppException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive.",
            )

        return AuthService._build_auth_response(user)
