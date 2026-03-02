"""FastAPI dependency injection utilities."""

from fastapi import Header, HTTPException

from app.config import get_settings


async def require_admin_key(x_admin_key: str = Header(default="")):
    """Require a valid X-Admin-Key header on protected routes.

    If ADMIN_API_KEY is not configured (empty string), the check is bypassed
    so local dev works without any setup.
    """
    settings = get_settings()
    if settings.admin_api_key and x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
