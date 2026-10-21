from __future__ import annotations

import jwt

from app.access import Principal
from app.config import config


class AuthError(Exception):
    pass


def principal_from_token(token: str, secret: str | None = None, algorithm: str | None = None) -> Principal:
    secret = secret or config.jwt_secret
    algorithm = algorithm or config.jwt_algorithm
    if not secret:
        raise AuthError("no JWT secret configured")
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from exc

    user_id = str(claims.get("sub") or claims.get("user_id") or "anonymous")
    groups = claims.get("groups") or []
    if isinstance(groups, str):
        groups = [g.strip() for g in groups.split(",") if g.strip()]
    return Principal(user_id=user_id, groups=frozenset(groups))
