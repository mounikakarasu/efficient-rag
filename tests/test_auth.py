import time

import jwt
import pytest

from app.auth import AuthError, principal_from_token

SECRET = "test-secret"


def _token(claims, secret=SECRET, algorithm="HS256"):
    return jwt.encode(claims, secret, algorithm=algorithm)


def test_valid_token_yields_principal():
    token = _token({"sub": "alice", "groups": ["team-alpha", "team-beta"]})
    principal = principal_from_token(token, secret=SECRET, algorithm="HS256")
    assert principal.user_id == "alice"
    assert principal.groups == frozenset({"team-alpha", "team-beta"})


def test_groups_accepts_csv_string():
    token = _token({"sub": "bob", "groups": "team-a, team-b"})
    assert principal_from_token(token, secret=SECRET).groups == frozenset({"team-a", "team-b"})


def test_wrong_secret_rejected():
    token = _token({"sub": "x", "groups": []}, secret="other-secret")
    with pytest.raises(AuthError):
        principal_from_token(token, secret=SECRET)


def test_expired_token_rejected():
    token = _token({"sub": "x", "groups": [], "exp": int(time.time()) - 10})
    with pytest.raises(AuthError):
        principal_from_token(token, secret=SECRET)


def test_missing_secret_rejected():
    token = _token({"sub": "x"})
    with pytest.raises(AuthError):
        principal_from_token(token, secret="")
