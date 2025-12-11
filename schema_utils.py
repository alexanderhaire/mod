import pyodbc

from constants import (
    CUSTOM_SQL_ALLOWED_TABLES,
    LOGGER,
    SQL_SCHEMA_CACHE,
    SQL_SCHEMA_CACHE_KEY,
)


def load_allowed_sql_schema(cursor: pyodbc.Cursor) -> dict[str, list[dict]]:
    cached = SQL_SCHEMA_CACHE.get(SQL_SCHEMA_CACHE_KEY)
    if cached is not None:
        return cached
    if not CUSTOM_SQL_ALLOWED_TABLES:
        SQL_SCHEMA_CACHE[SQL_SCHEMA_CACHE_KEY] = {}
        return {}
    placeholders = ", ".join("?" for _ in CUSTOM_SQL_ALLOWED_TABLES)
    query = f"SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME IN ({placeholders}) ORDER BY TABLE_NAME, ORDINAL_POSITION"
    try:
        cursor.execute(query, CUSTOM_SQL_ALLOWED_TABLES)
        rows = cursor.fetchall()
    except pyodbc.Error as err:
        LOGGER.warning("Failed to load schema for custom SQL: %s", err)
        SQL_SCHEMA_CACHE[SQL_SCHEMA_CACHE_KEY] = {}
        return {}

    schema: dict[str, list[dict]] = {}
    for row in rows:
        table_name, column_name, data_type = row
        if not table_name or not column_name:
            continue
        schema.setdefault(table_name, []).append({"name": column_name, "type": data_type})
    SQL_SCHEMA_CACHE[SQL_SCHEMA_CACHE_KEY] = schema
    return schema


def summarize_schema_for_prompt(
    schema: dict[str, list[dict]], 
    max_columns: int = 12, 
    priority_columns: tuple[str, ...] = ()
) -> str:
    if not schema:
        return ""
    lines = []
    for table_name in sorted(schema):
        columns = schema[table_name]
        
        # Split columns into priority and others
        priority_cols = []
        other_cols = []
        
        for col in columns:
            if col["name"] in priority_columns:
                priority_cols.append(col)
            else:
                other_cols.append(col)
        
        # Combine: priority first, then others up to the limit
        # We always show all priority columns, even if it exceeds max_columns slightly
        display_cols = priority_cols + other_cols
        
        # If we have more than max_columns, truncate the *non-priority* ones
        if len(display_cols) > max_columns:
            # Keep all priority, fill the rest with others until max_columns is reached
            # But ensure we at least show the priority ones
            limit = max(len(priority_cols), max_columns)
            display_cols = display_cols[:limit]
            
        col_bits = [f"{col['name']} ({col['type']})" for col in display_cols]
        
        if len(columns) > len(display_cols):
            col_bits.append("...")
            
        lines.append(f"{table_name}: {', '.join(col_bits)}")
    return "\n".join(lines)
