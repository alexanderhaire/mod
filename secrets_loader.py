from collections.abc import Mapping
from functools import lru_cache

from constants import LOCAL_SECRETS_PATHS, OPENAI_BEST_MODEL, OPENAI_DEFAULT_MODEL

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        tomllib = None  # type: ignore


@lru_cache(maxsize=1)
def load_project_secrets() -> dict:
    if tomllib is None:
        return {}
    for candidate in LOCAL_SECRETS_PATHS:
        try:
            with candidate.open("rb") as handle:
                parsed = tomllib.load(handle)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (FileNotFoundError, OSError, ValueError):
            continue
    return {}


def load_local_secret_section(section_name: str) -> dict:
    secrets_blob = load_project_secrets()
    section = secrets_blob.get(section_name)
    return dict(section) if isinstance(section, Mapping) else {}


def load_sql_secrets() -> dict:
    return load_local_secret_section("sql")


def load_openai_settings() -> dict:
    openai_section = load_local_secret_section("openai")
    api_key = openai_section.get("api_key")
    if not api_key:
        api_key = load_project_secrets().get("openai_api_key")
    if not api_key:
        return {}

    def _coerce_model(value) -> str | None:
        if value is None:
            return None
        model_str = str(value).strip()
        if not model_str:
            return None
        if model_str.lower() == "best":
            return OPENAI_BEST_MODEL
        return model_str

    model = _coerce_model(openai_section.get("model")) or OPENAI_DEFAULT_MODEL
    sql_model = _coerce_model(openai_section.get("sql_model")) or model
    embedding_model = _coerce_model(openai_section.get("embedding_model")) or openai_section.get("embedding_model")
    return {
        "api_key": api_key,
        "model": model,
        "sql_model": sql_model,
        "embedding_model": embedding_model,
    }


def build_connection_string(force_flags: dict[str, str | bool] | None = None) -> tuple[str, str, str, str]:
    secrets_config = load_sql_secrets()
    driver = secrets_config.get("driver", "ODBC Driver 17 for SQL Server")
    server = secrets_config.get("server", "localhost")
    database = secrets_config.get("database", "CDI")
    auth_mode = secrets_config.get("authentication", "windows").lower()

    def _normalize_flag(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().lower()
        if value in {"yes", "true", "1"}:
            return "yes"
        if value in {"no", "false", "0"}:
            return "no"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return None

    encrypt = _normalize_flag(secrets_config.get("encrypt"))
    trust_cert = _normalize_flag(secrets_config.get("trust_server_certificate"))

    if force_flags:
        if "encrypt" in force_flags:
            override = _normalize_flag(force_flags.get("encrypt"))
            if override is not None:
                encrypt = override
        if "trust_server_certificate" in force_flags:
            override = _normalize_flag(force_flags.get("trust_server_certificate"))
            if override is not None:
                trust_cert = override

    extra_flags = ""
    if encrypt:
        extra_flags += f"Encrypt={encrypt};"
    if trust_cert:
        extra_flags += f"TrustServerCertificate={trust_cert};"

    if auth_mode == "sql":
        username = secrets_config.get("username")
        password = secrets_config.get("password")
        if not username or not password:
            raise RuntimeError("SQL auth requires username and password.")
        conn_str = (
            f"Driver={{{driver}}};Server={server};Database={database};"
            f"UID={username};PWD={password};{extra_flags}"
        )
        return conn_str, server, database, "SQL user"

    conn_str = (
        f"Driver={{{driver}}};Server={server};Database={database};"
        f"Trusted_Connection=yes;{extra_flags}"
    )
    return conn_str, server, database, "Windows auth"
