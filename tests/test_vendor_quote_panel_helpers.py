"""Tests for the pure helpers behind the Vendor Quotes panel.

The Streamlit rendering itself isn't tested here — only the joining and
aggregation logic that the panel relies on.
"""
from __future__ import annotations

import datetime as dt

import pytest

from market_insights import (
    join_quotes_to_receipts,
    compute_cheapest_current_quote,
)


SAMPLE_QUOTES = [
    {"vendor": "HELM", "mode": "rail_delivered", "quote_date": "2026-04-03",
     "price_per_ton": 475.0, "po_number": "431-8237", "confidence": "high"},
    {"vendor": "Nutrien", "mode": "rail_delivered", "quote_date": "2026-04-22",
     "price_per_ton": 595.0, "po_number": "431-8330", "confidence": "high"},
]
SAMPLE_RECEIPTS = [
    {"VendorName": "HELM", "TransactionDate": "2026-04-08", "AvgCost": 475.0, "PONUMBER": "431-8237"},
    {"VendorName": "Nutrien", "TransactionDate": "2026-02-10", "AvgCost": 480.0, "PONUMBER": "431-8102"},
    {"VendorName": "Trademark Nitrogen", "TransactionDate": "2025-12-15", "AvgCost": 411.0, "PONUMBER": "431-8066"},
]


def test_join_returns_one_row_per_vendor_mode_present_in_either():
    rows = join_quotes_to_receipts(SAMPLE_QUOTES, SAMPLE_RECEIPTS)
    vendors_modes = {(r["vendor"], r["mode"]) for r in rows}
    # HELM rail, Nutrien rail, Trademark <mode-unknown>
    assert ("HELM", "rail_delivered") in vendors_modes
    assert ("Nutrien", "rail_delivered") in vendors_modes
    assert any(r["vendor"] == "Trademark Nitrogen" for r in rows)


def test_join_marks_matched_po_with_indicator():
    rows = join_quotes_to_receipts(SAMPLE_QUOTES, SAMPLE_RECEIPTS)
    helm = next(r for r in rows if r["vendor"] == "HELM")
    assert helm["po_match"] is True

    nutrien = next(r for r in rows if r["vendor"] == "Nutrien")
    assert nutrien["po_match"] is False  # quote PO 8330, receipt PO 8102 differ


def test_join_computes_delta_when_both_sides_present():
    rows = join_quotes_to_receipts(SAMPLE_QUOTES, SAMPLE_RECEIPTS)
    nutrien = next(r for r in rows if r["vendor"] == "Nutrien")
    # quote 595 - receipt 480 = +115
    assert nutrien["delta_per_ton"] == pytest.approx(115.0)

    helm = next(r for r in rows if r["vendor"] == "HELM")
    assert helm["delta_per_ton"] == pytest.approx(0.0)


def test_join_leaves_delta_none_when_a_side_missing():
    rows = join_quotes_to_receipts(SAMPLE_QUOTES, SAMPLE_RECEIPTS)
    tm = next(r for r in rows if r["vendor"] == "Trademark Nitrogen")
    assert tm["delta_per_ton"] is None
    assert tm["quote_price_per_ton"] is None
    assert tm["receipt_price_per_ton"] == 411.0


def test_cheapest_current_picks_lowest_fresh_high_confidence_quote():
    today = dt.date(2026, 5, 12)
    cheapest = compute_cheapest_current_quote(SAMPLE_QUOTES, today=today, freshness_days=60)
    assert cheapest is not None
    assert cheapest["vendor"] == "HELM"
    assert cheapest["price_per_ton"] == 475.0


def test_cheapest_current_returns_none_when_all_stale():
    today = dt.date(2026, 12, 31)
    cheapest = compute_cheapest_current_quote(SAMPLE_QUOTES, today=today, freshness_days=60)
    assert cheapest is None


def test_cheapest_current_skips_low_confidence_rows():
    quotes = [
        {"vendor": "Nutrien", "mode": "rail_delivered", "quote_date": "2026-05-01",
         "price_per_ton": None, "confidence": "low", "po_number": "431-8330"},
        {"vendor": "HELM", "mode": "rail_delivered", "quote_date": "2026-04-03",
         "price_per_ton": 475.0, "confidence": "high", "po_number": "431-8237"},
    ]
    cheapest = compute_cheapest_current_quote(quotes, today=dt.date(2026, 5, 12), freshness_days=60)
    assert cheapest is not None
    assert cheapest["vendor"] == "HELM"
