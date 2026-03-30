"""Auth service: JWT encode/decode and user upsert."""

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy import select

from app.config import get_settings
from app.models.database import get_session
from app.models.db_models import User


def create_jwt(user: User) -> str:
    settings = get_settings()
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "exp": datetime.now(tz=timezone.utc) + timedelta(days=settings.jwt_expire_days),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_jwt(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


async def upsert_user(google_id: str, email: str, name: str, avatar_url: str | None) -> User:
    """Create or update a user record from Google OAuth info."""
    async with get_session() as session:
        result = await session.execute(select(User).where(User.google_id == google_id))
        user = result.scalar_one_or_none()
        if user is None:
            # Fall back to email match — picks up pre-approved stub records
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()
        if user is None:
            user = User(
                id=uuid.uuid4(),
                google_id=google_id,
                email=email,
                name=name,
                avatar_url=avatar_url,
            )
            session.add(user)
        else:
            user.google_id = google_id  # stamp real google_id onto stub if needed
            user.email = email
            user.name = name
            user.avatar_url = avatar_url
            user.updated_at = datetime.now(tz=timezone.utc)
        await session.flush()
        await session.refresh(user)
        # Detach from session so we can use the object after the session closes
        user_data = User(
            id=user.id,
            google_id=user.google_id,
            email=user.email,
            name=user.name,
            avatar_url=user.avatar_url,
            is_approved=user.is_approved,
        )
    return user_data
