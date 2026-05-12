"""Persistence helpers for vendor-quote data.

Storage shape: a JSON file whose top-level keys are ITEMNMBR strings and whose
values are lists of quote-row dicts. Append-only; the reducer
``latest_per_vendor_mode`` collapses rows to one per (vendor, mode) at read time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_quotes(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Return the full quote store. Empty dict if the file doesn't exist.

    Raises ``json.JSONDecodeError`` if the file exists but is not valid JSON
    (e.g., truncated by an interrupted write). Loud failure is intentional —
    silent recovery would discard historical quotes.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    if not raw.strip():
        return {}
    return json.loads(raw)


def append_quote(path: Path, item_number: str, row: dict[str, Any]) -> None:
    """Append a quote row under ``item_number``. Creates the file if missing.

    Uses write-temp-then-rename so an interrupted write can't leave the store
    truncated. ``os.replace`` is atomic on Windows since Python 3.3.
    """
    store = load_quotes(path)
    store.setdefault(item_number, []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def latest_per_vendor_mode(
    store: dict[str, list[dict[str, Any]]],
    item_number: str,
) -> list[dict[str, Any]]:
    """Return at most one row per (vendor, mode) — the row with the newest ``quote_date``.

    Assumes ``quote_date`` values are ISO 8601 ``YYYY-MM-DD`` strings; the
    comparison is lexicographic.
    """
    rows = store.get(item_number, [])
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("vendor", ""), row.get("mode", ""))
        existing = latest.get(key)
        if existing is None or row.get("quote_date", "") > existing.get("quote_date", ""):
            latest[key] = row
    return list(latest.values())
