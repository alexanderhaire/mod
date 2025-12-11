"""
Utilities for capturing structured training data and response traces.
Writes compact JSONL files that can be used for evaluation or fine-tuning.
"""

import datetime
import json
import os
import time
import re
from decimal import Decimal
from typing import Any

TRAINING_EVENTS_FILE = "training_events.jsonl"
TRAINING_EXAMPLES_FILE = "training_examples.jsonl"
MAX_TEXT_LENGTH = 4000
MAX_DATA_PREVIEW_ROWS = 50
_PRONOUN_TOKENS = {
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "there",
    "same",
    "each",
    "either",
    "neither",
}
_MONTH_TOKENS = {
    "january",
    "jan",
    "february",
    "febuary",
    "feb",
    "march",
    "mar",
    "april",
    "apr",
    "may",
    "june",
    "jun",
    "july",
    "jul",
    "august",
    "aug",
    "september",
    "sept",
    "sep",
    "october",
    "oct",
    "november",
    "nov",
    "december",
    "dec",
}


def _has_anchor_terms(prompt: str) -> bool:
    """Detect whether a prompt stands on its own (months, years, item codes, or numbers)."""
    if not prompt:
        return False
    lower = prompt.lower()
    if any(tok in lower for tok in _MONTH_TOKENS):
        return True
    if re.search(r"\b(19|20)\d{2}\b", lower):
        return True
    if re.search(r"\d", lower):
        return True
    # Item codes tend to be 4+ characters and upper-case.
    if re.search(r"\b[A-Z0-9]{4,}\b", prompt):
        return True
    return False


def should_log_training_example(prompt: str | None) -> bool:
    """
    Skip context-dependent follow-ups (e.g., 'that', 'those months') that lack explicit anchors.
    Prevents contaminating training data with prompts that only make sense with chat history.
    """
    if not prompt or not isinstance(prompt, str):
        return False
    prompt = prompt.strip()
    tokens = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    anchor = _has_anchor_terms(prompt)

    if len(tokens) <= 3 and not anchor:
        return False

    lower = prompt.lower()
    if ("those months" in lower or "these months" in lower) and not anchor:
        return False

    if any(tok in _PRONOUN_TOKENS for tok in tokens) and not anchor:
        return False

    return True


def _truncate_text(text: str | None, limit: int = MAX_TEXT_LENGTH) -> str | None:
    if text is None:
        return None
    value = str(text)
    if len(value) <= limit:
        return value
    overflow = len(value) - limit
    return value[:limit] + f"... [truncated {overflow} chars]"


def _json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _preview_rows(rows: Any) -> list | None:
    """Limit row logging to a small preview to keep logs compact."""
    if not isinstance(rows, list):
        return None
    preview = []
    for row in rows[:MAX_DATA_PREVIEW_ROWS]:
        if isinstance(row, dict):
            preview.append({k: _json_safe(v) for k, v in row.items()})
        else:
            preview.append(_json_safe(row))
    return preview


def _write_jsonl(path: str, payload: dict) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def record_training_event(
    prompt: str,
    result: dict | None,
    *,
    user: str | None = None,
    chat_id: str | None = None,
    context: dict | None = None,
    history_hint: str | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Log a high-level response event for later analysis.
    Captures the prompt, route, summary, SQL preview, row counts, and any errors.
    """
    if not prompt or not isinstance(result, dict):
        return

    insights = result.get("insights") if isinstance(result.get("insights"), dict) else {}
    data_rows = result.get("data") if isinstance(result.get("data"), list) else None

    payload = {
        "timestamp": int(time.time()),
        "prompt": _truncate_text(prompt),
        "user": user,
        "chat_id": chat_id,
        "route": insights.get("route") or result.get("route"),
        "summary": _truncate_text(insights.get("summary")),
        "note": _truncate_text(insights.get("note")),
        "entities": _json_safe(result.get("entities")),
        "sql_preview": _truncate_text(result.get("sql")),
        "raw_sql": _truncate_text(result.get("raw_sql")),
        "params": _json_safe(result.get("params")),
        "row_count": insights.get("row_count") or (len(data_rows) if data_rows is not None else None),
        "truncated": insights.get("truncated"),
        "error": _truncate_text(result.get("error")),
        "usage": _json_safe(result.get("usage")),
        "context": _json_safe(context),
        "history_hint": _truncate_text(history_hint),
        "data_preview": _preview_rows(data_rows),
    }

    if metadata and isinstance(metadata, dict):
        payload["meta"] = _json_safe(metadata)

    try:
        _write_jsonl(TRAINING_EVENTS_FILE, payload)
    except Exception:
        # Swallow logging errors to avoid impacting the main app.
        return


def record_sql_example(
    prompt: str,
    summary: str | None,
    sql: str | None,
    params: list | tuple | None,
    entities: dict | None = None,
    *,
    route: str | None = None,
    source: str = "generated",
    metadata: dict | None = None,
) -> None:
    """
    Persist a prompt-to-SQL pair for fine-tuning or regression checks.
    """
    if not prompt or not sql:
        return
    if not should_log_training_example(prompt):
        return

    payload = {
        "timestamp": int(time.time()),
        "prompt": _truncate_text(prompt),
        "summary": _truncate_text(summary),
        "sql": _truncate_text(sql),
        "params": _json_safe(params),
        "entities": _json_safe(entities),
        "route": route,
        "source": source,
    }

    if metadata and isinstance(metadata, dict):
        payload["meta"] = _json_safe(metadata)

    try:
        _write_jsonl(TRAINING_EXAMPLES_FILE, payload)
    except Exception:
        return


def record_sql_example_from_response(
    prompt: str,
    response: dict | None,
    *,
    user: str | None = None,
    chat_id: str | None = None,
    source: str | None = None,
) -> None:
    """
    Capture SQL examples from any response so every account interaction feeds training data.
    Skips when no SQL is present or when a prior call marked the response as logged.
    """
    if not prompt or not isinstance(response, dict):
        return
    if not should_log_training_example(prompt):
        return
    if response.get("training_example_logged"):
        return

    insights = response.get("insights") if isinstance(response.get("insights"), dict) else {}
    sql_text = response.get("raw_sql") or response.get("sql")
    if not sql_text:
        return

    summary = insights.get("summary") or insights.get("narrative") or response.get("summary")
    params = response.get("params")
    entities = response.get("entities") if isinstance(response.get("entities"), dict) else None
    route = insights.get("route") or response.get("route")
    handler_name = response.get("handler_name") or insights.get("handler_name")

    meta = {
        "user": user,
        "chat_id": chat_id,
        "handler_name": handler_name,
        "route": route,
    }
    cleaned_meta = {k: v for k, v in meta.items() if v is not None}

    record_sql_example(
        prompt=prompt,
        summary=summary,
        sql=sql_text,
        params=params,
        entities=entities,
        route=route,
        source=source or handler_name or route or "response_logger",
        metadata=cleaned_meta or None,
    )
