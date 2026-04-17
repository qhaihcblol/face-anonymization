from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User


class UserRepository:
    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.lower().strip()

    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> User | None:
        normalized_email = UserRepository._normalize_email(email)
        result = await db.execute(select(User).where(User.email == normalized_email))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> User | None:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        db: AsyncSession,
        full_name: str,
        email: str,
        hashed_password: str,
    ) -> User:
        user = User(
            full_name=full_name.strip(),
            email=UserRepository._normalize_email(email),
            hashed_password=hashed_password,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
