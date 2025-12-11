import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyodbc

from context_utils import summarize_sql_context
from handlers import handle_custom_sql_question
from openai_clients import call_openai_data_narrative, call_openai_question_router, call_openai_small_talk, call_openai_deep_analyst
from rag_retrieval import build_rag_answer
from reasoning_coordinator import execute_reasoning_chain, should_use_reasoning_coordinator
from vector_store import VectorStore
from web_handlers import handle_web_question


def _should_show_sql(prompt: str, context: Optional[Dict[str, Any]]) -> bool:
    """Determine whether to show the SQL query in the response."""
    ctx = context or {}
    if ctx.get("debug") or ctx.get("show_sql") is True:
        return True
    if ctx.get("show_sql") is False:
        return False
    if not prompt:
        return True
    lower = prompt.lower()
    hide_tokens = (
        "hide sql",
        "don't show sql",
        "do not show sql",
        "no sql",
        "without sql",
        "skip the sql",
    )
    return not any(tok in lower for tok in hide_tokens)


def _is_numeric_cell(value: Any) -> bool:
    """Check for numeric-like values while avoiding bools."""
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _analyze_null_rows(rows: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Analyze result sets for null values to provide better user feedback and handle data gaps.
    """
    if not isinstance(rows, list):
        return None

    if not rows:
        return {
            "clean_rows": rows,
            "data_unavailable": True,
            "message": "The query completed but returned no data for the requested range.",
            "note": None,
            "filled_zero_columns": [],
            "null_columns": [],
        }

    dict_rows = [row for row in rows if isinstance(row, dict)]
    if not dict_rows:
        return {
            "clean_rows": rows,
            "data_unavailable": False,
            "message": None,
            "note": None,
            "filled_zero_columns": [],
            "null_columns": [],
        }

    total_cells, null_cells = 0, 0
    null_counts: Dict[str, int] = {}
    non_null_columns: set[str] = set()
    numeric_columns: set[str] = set()

    for row in dict_rows:
        for col, val in row.items():
            total_cells += 1
            if val is None:
                null_cells += 1
                null_counts[col] = null_counts.get(col, 0) + 1
            else:
                non_null_columns.add(col)
                if _is_numeric_cell(val):
                    numeric_columns.add(col)

    all_null_or_empty = dict_rows and all(not row or all(v is None for v in row.values()) for row in dict_rows)
    data_unavailable = (total_cells > 0 and null_cells == total_cells) or bool(all_null_or_empty)
    
    fill_zero_columns = {col for col in numeric_columns if null_counts.get(col, 0) > 0}
    clean_rows = rows
    if fill_zero_columns:
        clean_rows = [
            {k: (0 if k in fill_zero_columns and v is None else v) for k, v in row.items()}
            if isinstance(row, dict) else row for row in rows
        ]

    null_only_columns = [col for col, count in null_counts.items() if count == len(dict_rows) and col not in non_null_columns]
    
    note_parts = []
    if null_only_columns:
        note_parts.append(f"Columns with no data: {', '.join(sorted(null_only_columns))}.")
    if fill_zero_columns:
        note_parts.append(f"Nulls in numeric columns {', '.join(sorted(fill_zero_columns))} were treated as 0.")

    message = None
    if data_unavailable:
        message = "The query returned only NULL values for the requested range." if null_only_columns else "The query completed but returned no data for the requested range."

    return {
        "clean_rows": clean_rows,
        "data_unavailable": data_unavailable,
        "message": message,
        "note": " ".join(note_parts).strip() or None,
        "filled_zero_columns": sorted(list(fill_zero_columns)),
        "null_columns": sorted(null_only_columns),
    }


def _decorate_response(result: Dict[str, Any], prompt: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhance the response with presentation defaults, narrative generation, and feedback cues.
    """
    if not isinstance(result, dict):
        return result

    insights = result.get("insights", {})
    data_rows = result.get("data")

    # Analyze for nulls and adjust data and notes
    if (gap_analysis := _analyze_null_rows(data_rows)):
        result["data"] = data_rows = gap_analysis.get("clean_rows", data_rows)
        note_bits = [insights.get("note"), gap_analysis.get("note")]
        insights["note"] = " ".join(filter(None, note_bits)).strip()
        if gap_analysis.get("data_unavailable"):
            gap_summary = gap_analysis.get("message", "No data returned.")
            insights.update({"summary": gap_summary, "narrative": gap_summary, "skip_story": True})
        elif gap_analysis.get("filled_zero_columns"):
            insights["null_coalesced_columns"] = gap_analysis["filled_zero_columns"]

    # Handle SQL visibility
    if (sql_available := result.get("sql")):
        result["show_sql"] = _should_show_sql(prompt, context)
        if result["show_sql"]:
            insights.pop("sql_hidden", None)
        else:
            insights["sql_hidden"] = True
            insights.setdefault("note", "SQL hidden per request; say 'show SQL' to display it.")

    # Generate data narrative if needed
    route_hint = insights.get("route") or result.get("route")
    if route_hint == "erp_sql" and data_rows and not insights.get("narrative") and not insights.get("skip_story"):
        # Check for deep analysis intent
        lower_prompt = prompt.lower()
        analysis_keywords = ("analyze", "analysis", "why", "explain", "reason", "trend", "outlier", "correlation", "breakdown", "machine learning", "ml", "ai", "predict", "forecast")
        is_deep_analysis = any(k in lower_prompt for k in analysis_keywords)

        if is_deep_analysis:
            if (analysis := call_openai_deep_analyst(prompt, data_rows, insights.get("summary"), context, result.get("entities"))):
                if content := analysis.get("analysis"):
                    insights["narrative"] = insights["summary"] = content
                if usage := analysis.get("usage"):
                    result["usage"] = (result.get("usage", []) or []) + [{"label": "deep_analyst", **usage}]
        else:
            if (story := call_openai_data_narrative(prompt, data_rows, insights.get("summary"), context, result.get("entities"))):
                if narrative := story.get("narrative"):
                    insights["narrative"] = insights["summary"] = narrative
                if usage := story.get("usage"):
                    result["usage"] = (result.get("usage", []) or []) + [{"label": "story", **usage}]
    
    if narrative := insights.get("narrative"):
        insights["summary"] = narrative

    result["insights"] = insights
    return result


def _resolve_route(routing: Dict[str, Any]) -> str:
    """Determine the final route from the router's output."""
    route = str(routing.get("route", "erp_sql")).lower()
    allowed = {"erp_sql", "rag", "reject", "chat", "web", "nlp"}
    return route if route in allowed else "erp_sql"


def _attach_routing_metadata(result: Dict[str, Any], route: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    """Embed routing information into the response for transparency."""
    insights = result.get("insights", {})
    insights.setdefault("route", route)
    if trace:
        insights["routing_trace"] = trace
    result["insights"] = insights
    return result


def _handle_reject_route() -> Dict[str, Any]:
    """Handle the 'reject' route."""
    summary = (
        "I'm focused on your chemical manufacturing data in Dynamics GP: production, BOMs, inventory, purchasing, "
        "costing, and related analysis. Ask about those areas and I'll generate the SQL and results."
    )
    return {"data": [], "insights": {"summary": summary}, "entities": {}, "sql": None}


def _handle_chat_route(prompt: str, today: datetime.date) -> Dict[str, Any]:
    """Handle the 'chat' route."""
    chat_reply = call_openai_small_talk(prompt, today)
    summary = chat_reply.get("reply") if isinstance(chat_reply, dict) else "Hey there! I'm here to help."
    result = {"data": [], "insights": {"summary": summary}, "entities": {}, "sql": None}
    if isinstance(chat_reply, dict) and (usage := chat_reply.get("usage")):
        result["usage"] = [{"label": "chat", **usage}]
    return result


def _handle_rag_route(prompt: str, today: datetime.date, cursor: pyodbc.Cursor, context: Dict, history_hint: str) -> Dict[str, Any]:
    """Handle the 'rag' route, with a fallback to ERP SQL."""
    vector_store_path = Path("erp_vector_store.json")
    vector_store = VectorStore(vector_store_path)
    rag_result = build_rag_answer(prompt, today, vector_store)
    if rag_result:
        return rag_result
    # Fallback to ERP SQL if RAG returns nothing
    return handle_custom_sql_question(cursor, prompt, today, context, history_hint)


def handle_question(
    cursor: pyodbc.Cursor, prompt: str, today: datetime.date, context: Optional[Dict[str, Any]] = None, history_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route a question to the appropriate handler and return a decorated response.
    """
    context = context or {}
    context_hint = summarize_sql_context(context)
    
    # Check if this might be a multi-step question that needs reasoning coordination
    if should_use_reasoning_coordinator(prompt):
        reasoning_result = execute_reasoning_chain(cursor, prompt, today, context)
        if reasoning_result:
            # Multi-step reasoning was successful, decorate and return
            return _decorate_response(reasoning_result, prompt, context)
    
    # Get route from the question router
    routing = call_openai_question_router(prompt, today, context_hint, history_hint)
    route = _resolve_route(routing)
    
    # Prepare metadata for the response
    routing_trace = {"initial_route": route, "router_confidence": routing.get("confidence"), "router_reason": routing.get("rationale")}
    usage_events = [{"label": "router", **usage}] if (usage := routing.pop("usage", None)) else []

    # Route to the appropriate handler
    if route == "reject":
        result = _handle_reject_route()
    elif route == "chat":
        result = _handle_chat_route(prompt, today)
    elif route == "web":
        result = handle_web_question(prompt, today, context)
    elif route == "rag":
        result = _handle_rag_route(prompt, today, cursor, context, history_hint)
    else: # Default to 'erp_sql'
        result = handle_custom_sql_question(cursor, prompt, today, context, history_hint)

    # Combine usage data and attach metadata
    if result_usage := result.get("usage"):
        usage_events.extend(result_usage if isinstance(result_usage, list) else [result_usage])
    if usage_events:
        result["usage"] = usage_events
        
    # Attach routing metadata FIRST so decoration logic (narratives) knows the route
    result_with_metadata = _attach_routing_metadata(result, route, routing_trace)
    
    # Then decorate with narratives, etc.
    return _decorate_response(result_with_metadata, prompt, context)
