"""
Aggregated exports for the reasoning core split across helper modules.
Keep imports centralized here so existing callers can continue using
`from reasoning_core import ...` without changes.
"""

from constants import (
    APP_ROOT,
    BILL_OF_MATERIALS_UI_REFERENCE,
    CUSTOM_SQL_ALLOWED_TABLES,
    CUSTOM_SQL_HINTS,
    CUSTOM_SQL_MAX_ROWS,
    FEW_SHOT_EXAMPLES,
    ITEM_RESOURCE_PLANNING_UI_REFERENCE,
    LOCAL_SECRETS_PATHS,
    LOGGER,
    OPENAI_BEST_MODEL,
    OPENAI_CHAT_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    SQL_SCHEMA_CACHE,
    SQL_SCHEMA_CACHE_KEY,
    SQL_TABLE_TOKEN_PATTERN,
)
from secrets_loader import (
    build_connection_string,
    load_local_secret_section,
    load_openai_settings,
    load_project_secrets,
    load_sql_secrets,
)
from schema_utils import load_allowed_sql_schema, summarize_schema_for_prompt
from context_utils import summarize_sql_context
from parsing_utils import (
    decimal_or_zero,
    extract_item_from_prompt,
    normalize_item_for_bom,
    parse_month_year_from_prompt,
    parse_percent_increase,
)
from sql_utils import (
    extract_invalid_column_names,
    extract_table_tokens,
    format_sql_preview,
    normalize_sql_params,
    summarize_table_columns,
    validate_custom_sql,
)
from bom_guidance import build_bom_guidance
from inventory_queries import fetch_on_hand_by_item, fetch_open_po_supply, fetch_mfg_bom_grouped_by_component
from openai_clients import (
    call_openai_general_response,
    call_openai_question_router,
    call_openai_sql_generator,
    call_openai_small_talk,
    enhance_prompt_for_complexity,
    is_erp_intent,
    is_likely_erp_question,
)
from web_handlers import fetch_web_context, handle_web_question
from handlers import (
    handle_custom_sql_question,
    handle_forecast_bom_requirements,
    handle_bom_for_item,
    handle_item_sales_month,
    handle_mrp_style_question,
    handle_raw_material_usage_month,
    handle_order_point_gap,
    handle_component_where_used,
    handle_top_selling_question,
    handle_mrp_planning,
)
from router import handle_question
from main_runner import main

__all__ = [
    # constants
    "APP_ROOT",
    "BILL_OF_MATERIALS_UI_REFERENCE",
    "CUSTOM_SQL_ALLOWED_TABLES",
    "CUSTOM_SQL_HINTS",
    "CUSTOM_SQL_MAX_ROWS",
    "FEW_SHOT_EXAMPLES",
    "ITEM_RESOURCE_PLANNING_UI_REFERENCE",
    "LOCAL_SECRETS_PATHS",
    "LOGGER",
    "OPENAI_BEST_MODEL",
    "OPENAI_CHAT_URL",
    "OPENAI_DEFAULT_MODEL",
    "OPENAI_TIMEOUT_SECONDS",
    "SQL_SCHEMA_CACHE",
    "SQL_SCHEMA_CACHE_KEY",
    "SQL_TABLE_TOKEN_PATTERN",
    # secrets/config
    "build_connection_string",
    "load_local_secret_section",
    "load_openai_settings",
    "load_project_secrets",
    "load_sql_secrets",
    # schema/context helpers
    "load_allowed_sql_schema",
    "summarize_schema_for_prompt",
    "summarize_sql_context",
    # parsing utils
    "decimal_or_zero",
    "extract_item_from_prompt",
    "normalize_item_for_bom",
    "parse_month_year_from_prompt",
    "parse_percent_increase",
    # sql helpers
    "extract_invalid_column_names",
    "extract_table_tokens",
    "format_sql_preview",
    "normalize_sql_params",
    "summarize_table_columns",
    "validate_custom_sql",
    # BOM + inventory helpers
    "build_bom_guidance",
    "fetch_on_hand_by_item",
    "fetch_open_po_supply",
    "fetch_mfg_bom_grouped_by_component",
    # OpenAI clients/heuristics
    "call_openai_general_response",
    "call_openai_question_router",
    "call_openai_sql_generator",
    "call_openai_small_talk",
    "enhance_prompt_for_complexity",
    "is_erp_intent",
    "is_likely_erp_question",
    # Web + routing + handlers
    "fetch_web_context",
    "handle_web_question",
    "handle_custom_sql_question",
    "handle_forecast_bom_requirements",
    "handle_bom_for_item",
    "handle_item_sales_month",
    "handle_mrp_style_question",
    "handle_raw_material_usage_month",
    "handle_order_point_gap",
    "handle_component_where_used",
    "handle_top_selling_question",
    "handle_mrp_planning",
    "handle_question",
    # entrypoint
    "main",
]


if __name__ == "__main__":
    main()
