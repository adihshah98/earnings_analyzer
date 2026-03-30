"""Google OAuth 2.0 endpoints."""

import logging
import secrets
import urllib.parse
import uuid

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.auth.dependencies import require_user
from app.auth.service import create_jwt, upsert_user
from app.config import get_settings
from app.models.database import get_session
from app.models.db_models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google")
async def google_login():
    """Redirect the browser to Google's OAuth consent screen."""
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    state = secrets.token_urlsafe(32)
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": _callback_url(settings),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
    }
    url = f"{_GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url)


@router.get("/google/callback")
async def google_callback(
    code: str = Query(...),
    state: str | None = Query(None),
    error: str | None = Query(None),
):
    """Handle Google OAuth callback: exchange code, upsert user, return JWT."""
    settings = get_settings()
    if error:
        frontend_url = settings.frontend_url.rstrip("/")
        return RedirectResponse(f"{frontend_url}?auth_error={urllib.parse.quote(error)}")

    # Exchange authorization code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": _callback_url(settings),
                "grant_type": "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        logger.error("Google token exchange failed: %s", token_resp.text)
        raise HTTPException(status_code=400, detail="Failed to exchange OAuth code")

    token_data = token_resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token in Google response")

    # Fetch user info from Google
    async with httpx.AsyncClient() as client:
        userinfo_resp = await client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if userinfo_resp.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch Google user info")

    userinfo = userinfo_resp.json()
    google_id = userinfo.get("id")
    email = userinfo.get("email")
    name = userinfo.get("name") or email or "Unknown"
    avatar_url = userinfo.get("picture")

    if not google_id or not email:
        raise HTTPException(status_code=400, detail="Incomplete user info from Google")

    user = await upsert_user(google_id=google_id, email=email, name=name, avatar_url=avatar_url)

    if not user.is_approved:
        frontend_url = settings.frontend_url.rstrip("/")
        return RedirectResponse(f"{frontend_url}?auth_error=not_approved")

    jwt_token = create_jwt(user)
    frontend_url = settings.frontend_url.rstrip("/")
    return RedirectResponse(f"{frontend_url}?token={jwt_token}")


class MeResponse(BaseModel):
    sub: str
    email: str
    name: str
    avatar_url: str | None = None


@router.get("/me", response_model=MeResponse)
async def get_me(user: dict = Depends(require_user)):
    """Return the current authenticated user's info from the JWT."""
    return MeResponse(
        sub=user["sub"],
        email=user["email"],
        name=user["name"],
        avatar_url=user.get("avatar_url"),
    )


class SetApprovalRequest(BaseModel):
    email: str
    approved: bool = True


@router.post("/set-approval")
async def set_approval(
    body: SetApprovalRequest,
    x_admin_key: str | None = Header(None),
):
    """Approve or revoke a user's access by email. Requires X-Admin-Key header."""
    settings = get_settings()
    if settings.admin_api_key and x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    created = False
    async with get_session() as session:
        result = await session.execute(select(User).where(User.email == body.email))
        user = result.scalar_one_or_none()
        if user is None:
            if not body.approved:
                raise HTTPException(status_code=404, detail=f"No user found with email {body.email!r}")
            user = User(
                id=uuid.uuid4(),
                google_id=f"pre-approved:{body.email}",
                email=body.email,
                name=body.email,
                is_approved=True,
            )
            session.add(user)
            created = True
        else:
            user.is_approved = body.approved
        await session.flush()

    action = "approved" if body.approved else "revoked"
    detail = f"User pre-created and {action}." if created else f"Access {action}."
    return {"email": body.email, "is_approved": body.approved, "created": created, "detail": detail}


def _callback_url(settings) -> str:
    return f"{settings.backend_url.rstrip('/')}/auth/google/callback"
