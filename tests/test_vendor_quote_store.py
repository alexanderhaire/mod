"""Tests for vendor_quote_store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vendor_quote_store import (
    append_quote,
    load_quotes,
    latest_per_vendor_mode,
)


def test_load_quotes_returns_empty_when_file_missing(tmp_path: Path):
    missing = tmp_path / "nope.json"
    assert load_quotes(missing) == {}


def test_load_quotes_returns_dict_keyed_by_item(write_store):
    path = write_store({
        "NPKU32": [{"vendor": "HELM", "price_per_ton": 475.0}],
        "NPKUREA": [{"vendor": "Nutrien", "price_per_ton": 510.0}],
    })
    result = load_quotes(path)
    assert set(result.keys()) == {"NPKU32", "NPKUREA"}
    assert result["NPKU32"][0]["vendor"] == "HELM"


def test_append_quote_creates_file_and_item_bucket(store_path: Path):
    row = {
        "vendor": "HELM",
        "quote_date": "2026-04-03",
        "price_per_ton": 475.0,
        "mode": "rail_delivered",
    }
    append_quote(store_path, "NPKU32", row)

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert payload == {"NPKU32": [row]}


def test_append_quote_preserves_existing_rows(write_store, store_path: Path):
    write_store({"NPKU32": [{"vendor": "HELM", "price_per_ton": 475.0}]})
    append_quote(store_path, "NPKU32", {"vendor": "Nutrien", "price_per_ton": 595.0})

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert len(payload["NPKU32"]) == 2
    assert payload["NPKU32"][0]["vendor"] == "HELM"
    assert payload["NPKU32"][1]["vendor"] == "Nutrien"


def test_latest_per_vendor_mode_picks_most_recent_quote_date(write_store):
    path = write_store({
        "NPKU32": [
            {"vendor": "HELM", "mode": "rail_delivered", "quote_date": "2026-01-10", "price_per_ton": 450.0},
            {"vendor": "HELM", "mode": "rail_delivered", "quote_date": "2026-04-03", "price_per_ton": 475.0},
            {"vendor": "HELM", "mode": "pickup",         "quote_date": "2026-03-15", "price_per_ton": 410.0},
            {"vendor": "Nutrien", "mode": "rail_delivered", "quote_date": "2026-04-22", "price_per_ton": 595.0},
        ]
    })
    result = latest_per_vendor_mode(load_quotes(path), "NPKU32")
    assert len(result) == 3
    helm_rail = next(r for r in result if r["vendor"] == "HELM" and r["mode"] == "rail_delivered")
    assert helm_rail["price_per_ton"] == 475.0
    assert helm_rail["quote_date"] == "2026-04-03"


def test_latest_per_vendor_mode_returns_empty_for_unknown_item(write_store):
    path = write_store({"NPKU32": [{"vendor": "HELM", "mode": "rail_delivered", "quote_date": "2026-04-03", "price_per_ton": 475.0}]})
    assert latest_per_vendor_mode(load_quotes(path), "NPKUREA") == []
