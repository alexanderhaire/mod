import re
from collections.abc import Iterable, Mapping

from constants import SQL_TABLE_TOKEN_PATTERN


def normalize_sql_params(raw_params) -> list:
    if raw_params is None: return []
    if isinstance(raw_params, (str, bytes, int, float, bool)): return [raw_params]
    if isinstance(raw_params, Iterable) and not isinstance(raw_params, (Mapping, str, bytes)):
        return [p for p in raw_params]
    return [raw_params]


def _sanitize_table_token(token: str) -> str:
    cleaned = token.strip().strip("[]")
    return cleaned.split(".")[-1].upper()


def extract_table_tokens(sql_text: str) -> set[str]:
    """Extract normalized table tokens from a SQL string."""
    return {_sanitize_table_token(m.group(1)) for m in SQL_TABLE_TOKEN_PATTERN.finditer(sql_text or "")}


def validate_custom_sql(sql: str, allowed_tables: Iterable[str]) -> tuple[bool, str | None]:
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL plan was empty or not a string."
    
    lowered = sql.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False, "Only SELECT statements (including CTEs starting with WITH) are allowed."
    
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "MERGE", "EXEC")
    if any(keyword in sql.upper() for keyword in forbidden):
        return False, "Disallowed DML/DDL keyword detected."

    cte_names = set()
    if lowered.startswith("with"):
        cte_names = {m.group(1).upper() for m in re.finditer(r"(\w+)\s+AS\s*\(", sql, re.IGNORECASE)}

    allowed = {name.upper() for name in allowed_tables}
    referenced = {_sanitize_table_token(m.group(1)) for m in SQL_TABLE_TOKEN_PATTERN.finditer(sql)}
    
    effective_referenced = referenced - cte_names
    if allowed and not effective_referenced.issubset(allowed):
        disallowed = effective_referenced - allowed
        return False, f"Disallowed tables referenced: {', '.join(disallowed)}"
        
    return True, None


def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return cleaned


def format_sql_preview(sql: str, params: Iterable) -> str:
    preview = " ".join(line.strip() for line in sql.strip().splitlines())
    for param in params:
        literal = f"'{param}'" if isinstance(param, str) else str(param)
        preview = preview.replace("?", literal, 1)
    return preview


def extract_invalid_column_names(err_text: str) -> list[str]:
    """Pull out column names from SQL Server 'Invalid column name' errors."""
    return re.findall(r"Invalid column name '([^']+)'", err_text or "", re.IGNORECASE)


def summarize_table_columns(table_names: set[str], schema: Mapping[str, list[dict]], max_cols: int = 10) -> str:
    if not table_names or not schema:
        return ""
    lines = []
    for name in sorted(table_names):
        columns = schema.get(name)
        if not columns:
            continue
        col_list = [col.get("name") for col in columns if isinstance(col, Mapping)]
        if not col_list:
            continue
        display = col_list[:max_cols]
        if len(col_list) > max_cols:
            display.append("...")
        lines.append(f"{name}: {', '.join(display)}")
    return "; ".join(lines)
