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
    """Return the full quote store. Empty dict if the file doesn't exist."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    if not raw.strip():
        return {}
    return json.loads(raw)


def append_quote(path: Path, item_number: str, row: dict[str, Any]) -> None:
    """Append a quote row under ``item_number``. Creates the file if missing."""
    store = load_quotes(path)
    store.setdefault(item_number, []).append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def latest_per_vendor_mode(
    store: dict[str, list[dict[str, Any]]],
    item_number: str,
) -> list[dict[str, Any]]:
    """Return at most one row per (vendor, mode) — the row with the newest ``quote_date``."""
    rows = store.get(item_number, [])
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("vendor", ""), row.get("mode", ""))
        existing = latest.get(key)
        if existing is None or row.get("quote_date", "") > existing.get("quote_date", ""):
            latest[key] = row
    return list(latest.values())
