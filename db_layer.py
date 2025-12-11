import logging
from typing import Iterable, Mapping, Tuple

import pyodbc

from constants import CUSTOM_SQL_ALLOWED_TABLES, CUSTOM_SQL_MAX_ROWS
from sql_utils import (
    extract_invalid_column_names,
    extract_table_tokens,
    format_sql_preview,
    normalize_sql_params,
    summarize_table_columns,
    validate_custom_sql,
)

LOGGER = logging.getLogger(__name__)


def validate_sql_plan(
    sql_text: str,
    raw_params,
    allowed_tables: Iterable[str] | None = None,
) -> tuple[bool, str | None, list, str | None]:
    """
    Normalize parameters, enforce table allowlists, and provide a rendered SQL preview.
    Returns (is_valid, reason, normalized_params, sql_preview).
    """
    tables = allowed_tables or CUSTOM_SQL_ALLOWED_TABLES
    normalized_params = normalize_sql_params(raw_params)
    valid, reason = validate_custom_sql(sql_text, tables)
    preview = format_sql_preview(sql_text, normalized_params) if sql_text else None
    return valid, reason, normalized_params, preview


def execute_sql(
    cursor: pyodbc.Cursor,
    sql_text: str,
    params: Iterable,
    max_rows: int = CUSTOM_SQL_MAX_ROWS,
) -> tuple[list[dict], bool, list[str]]:
    """
    Run a parameterized SQL statement with a conservative row cap.
    Returns (rows, truncated_flag, column_names).
    """
    cursor.execute(sql_text, params)
    fetched = cursor.fetchmany(max_rows + 1)
    columns = [col[0] for col in cursor.description] if cursor.description else []
    rows = [dict(zip(columns, row)) for row in fetched[:max_rows]]
    truncated = len(fetched) > max_rows
    return rows, truncated, columns


def describe_sql_error(err_text: str, sql_text: str, schema: Mapping[str, list[dict]] | None = None) -> str:
    """
    Build a user-facing error message with column and table hints when available.
    """
    missing_cols = extract_invalid_column_names(err_text)
    tables_in_sql = extract_table_tokens(sql_text)
    column_hints = summarize_table_columns(tables_in_sql, schema) if schema else ""

    detail_parts = [f"Database rejected the SQL: {err_text}"]
    if missing_cols:
        detail_parts.append(f"Columns not found: {', '.join(missing_cols)}.")
    if column_hints:
        detail_parts.append(f"Try columns from these tables instead: {column_hints}.")

    return " ".join(detail_parts)
