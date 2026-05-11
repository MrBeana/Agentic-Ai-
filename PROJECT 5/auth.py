"""
auth.py — JWT Authentication module for a REST API.

Provides password hashing, verification, and JWT token management
for securing FastAPI / Flask endpoints.
"""
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "super-secret-key-change-in-production"
ALGORITHM  = "HS256"
TOKEN_EXPIRE_MINUTES = 30


def hash_password(password: str) -> str:
    """Hash a plaintext password using SHA-256 with a random salt.

    Args:
        password: The plaintext password to hash.

    Returns:
        A string in the format 'salt:hash'.

    Example:
        >>> h = hash_password("my_password")
        >>> verify_password("my_password", h)
        True
    """
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored hash.

    Args:
        password: The plaintext password to verify.
        stored_hash: The stored hash in 'salt:hash' format.

    Returns:
        True if the password matches, False otherwise.

    Raises:
        ValueError: If stored_hash is not in valid 'salt:hash' format.
    """
    parts = stored_hash.split(":")
    if len(parts) != 2:
        raise ValueError("Invalid hash format. Expected 'salt:hash'.")
    salt, hashed = parts
    return hashlib.sha256((password + salt).encode()).hexdigest() == hashed


def create_access_token(user_id: int, expires_minutes: int = TOKEN_EXPIRE_MINUTES) -> str:
    """Create a signed JWT access token.

    Args:
        user_id: The user's unique identifier (must be > 0).
        expires_minutes: Token validity duration in minutes. Defaults to 30.

    Returns:
        A signed JWT string.

    Raises:
        ValueError: If user_id is not a positive integer.
        jwt.PyJWTError: If token encoding fails.

    Example:
        >>> token = create_access_token(user_id=42, expires_minutes=60)
        >>> payload = decode_token(token)
        >>> payload["sub"]
        42
    """
    if user_id <= 0:
        raise ValueError("user_id must be a positive integer")
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token.

    Args:
        token: The JWT string to decode.

    Returns:
        The decoded payload as a dict (keys: sub, exp, iat),
        or None if the token is invalid or expired.

    Example:
        >>> token = create_access_token(1)
        >>> payload = decode_token(token)
        >>> payload["sub"]
        1
        >>> decode_token("bad.token.here")
        None
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def refresh_token(token: str, extend_minutes: int = TOKEN_EXPIRE_MINUTES) -> Optional[str]:
    """Refresh a valid (non-expired) JWT token.

    Args:
        token: The existing JWT string.
        extend_minutes: New expiry window in minutes from now.

    Returns:
        A new signed JWT string, or None if the original token is invalid.
    """
    payload = decode_token(token)
    if payload is None:
        return None
    return create_access_token(payload["sub"], extend_minutes)
