"""FastAPI dependencies for authentication."""

import logging

import jwt
from fastapi import Depends, Header, HTTPException

from app.auth.service import decode_jwt

logger = logging.getLogger(__name__)


def _extract_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):]


async def get_optional_user(authorization: str | None = Header(None)) -> dict | None:
    """Return decoded JWT payload if a valid token is present, else None."""
    token = _extract_token(authorization)
    if not token:
        return None
    try:
        return decode_jwt(token)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.PyJWTError:
        return None


async def require_user(user: dict | None = Depends(get_optional_user)) -> dict:
    """Like get_optional_user but raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
