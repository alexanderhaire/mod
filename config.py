"""Centralized config loader.

Precedence: OS environment variables > .env file > .streamlit/secrets.toml.
Existing modules can migrate to get_config() at their own pace; until then
secrets.toml keeps working.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _streamlit_secrets() -> Any:
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return {}


def _secret(section: str, key: str, default: Any = None) -> Any:
    sec = _streamlit_secrets()
    try:
        return sec[section][key]
    except Exception:
        return default


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    _load_dotenv()

    auth_mode = os.getenv("GP_SQL_AUTH") or _secret("sql", "authentication", "windows")
    sql = {
        "driver": os.getenv("GP_SQL_DRIVER") or _secret("sql", "driver", "ODBC Driver 18 for SQL Server"),
        "server": os.getenv("GP_SQL_SERVER") or _secret("sql", "server", "DynamicsGP"),
        "database": os.getenv("GP_SQL_DATABASE") or _secret("sql", "database", "CDI"),
        "authentication": auth_mode,
        "username": os.getenv("GP_SQL_USERNAME") or _secret("sql", "username"),
        "password": os.getenv("GP_SQL_PASSWORD") or _secret("sql", "password"),
        "encrypt": os.getenv("GP_SQL_ENCRYPT") or _secret("sql", "encrypt", "yes"),
        "trust_server_certificate": (
            os.getenv("GP_SQL_TRUST_SERVER_CERTIFICATE")
            or _secret("sql", "trust_server_certificate", "yes")
        ),
    }

    shadow = os.getenv("SHADOW_MODE")
    shadow_mode = shadow.lower() == "true" if shadow is not None else True

    return {
        "sql": sql,
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY") or _secret("openai", "api_key"),
            "model": os.getenv("OPENAI_MODEL") or _secret("openai", "model", "gpt-4o-mini"),
        },
        "google": {
            "maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY") or _secret("google", "maps_api_key"),
        },
        "alpaca": {
            "endpoint": os.getenv("ALPACA_ENDPOINT") or _secret("alpaca", "endpoint"),
            "key": os.getenv("ALPACA_KEY") or _secret("alpaca", "key"),
            "secret": os.getenv("ALPACA_SECRET") or _secret("alpaca", "secret"),
        },
        "teams": {
            "webhook_url": os.getenv("TEAMS_WEBHOOK_URL") or _secret("teams", "webhook_url"),
            "enabled": (os.getenv("TEAMS_ENABLED", "true").lower() == "true"),
            "check_interval_minutes": int(
                os.getenv("TEAMS_CHECK_INTERVAL_MINUTES")
                or _secret("teams", "check_interval_minutes", 30)
            ),
        },
        "app": {"shadow_mode": shadow_mode},
    }


def build_odbc_connection_string() -> str:
    s = get_config()["sql"]
    parts = [
        f"DRIVER={{{s['driver']}}}",
        f"SERVER={s['server']}",
        f"DATABASE={s['database']}",
        f"Encrypt={s['encrypt']}",
        f"TrustServerCertificate={s['trust_server_certificate']}",
    ]
    if (s.get("authentication") or "").lower() == "windows":
        parts.append("Trusted_Connection=yes")
    else:
        parts.append(f"UID={s['username']}")
        parts.append(f"PWD={s['password']}")
    return ";".join(parts) + ";"
