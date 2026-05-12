"""Tests for vendor_quote_extractor."""
from __future__ import annotations

import json

import pytest

from vendor_quote_extractor import (
    build_prompt_messages,
    parse_extraction_response,
    extract_quotes_from_email,
    ExtractedRow,
)


ALIASES = {
    "NPKU32": ["U-32", "U32", "UAN 32"],
    "NPKUREA": ["urea 46"],
}


def test_build_prompt_messages_includes_aliases_and_email_body():
    msgs = build_prompt_messages(
        email_subject="RE: PO 431-8237",
        email_body="HELM is offering $475/ton rail delivered for U-32.",
        sender="todd.wilson@helm.com",
        aliases=ALIASES,
    )
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user_content = msgs[1]["content"]
    assert "U-32" in user_content
    assert "$475/ton" in user_content
    assert "todd.wilson@helm.com" in user_content


def test_parse_extraction_response_well_formed_single_row():
    raw = json.dumps({
        "quotes": [
            {
                "item_number": "NPKU32",
                "vendor": "HELM",
                "quote_date": "2026-04-03",
                "price": 475.0,
                "unit": "ton",
                "mode": "rail_delivered",
                "po_number": "431-8237",
                "notes": "rail delivered to Florida",
                "source_excerpt": "HELM is offering $475/ton rail delivered for U-32.",
                "confidence": "high"
            }
        ]
    })
    rows = parse_extraction_response(raw)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, ExtractedRow)
    assert row.item_number == "NPKU32"
    assert row.vendor == "HELM"
    assert row.price == 475.0
    assert row.confidence == "high"


def test_parse_extraction_response_multiple_rows():
    raw = json.dumps({
        "quotes": [
            {"item_number": "NPKU32", "vendor": "HELM", "quote_date": "2026-04-03",
             "price": 475.0, "unit": "ton", "mode": "rail_delivered",
             "po_number": "431-8237", "notes": "", "source_excerpt": "...", "confidence": "high"},
            {"item_number": "NPKUREA", "vendor": "Nutrien", "quote_date": "2026-04-22",
             "price": 520.0, "unit": "ton", "mode": "rail_delivered",
             "po_number": None, "notes": "", "source_excerpt": "...", "confidence": "high"},
        ]
    })
    rows = parse_extraction_response(raw)
    assert len(rows) == 2
    assert {r.vendor for r in rows} == {"HELM", "Nutrien"}


def test_parse_extraction_response_handles_no_quotes():
    raw = json.dumps({"quotes": []})
    assert parse_extraction_response(raw) == []


def test_parse_extraction_response_skips_malformed_rows():
    raw = json.dumps({
        "quotes": [
            {"item_number": "NPKU32", "vendor": "HELM", "quote_date": "2026-04-03",
             "price": 475.0, "unit": "ton", "mode": "rail_delivered",
             "po_number": "431-8237", "notes": "", "source_excerpt": "...", "confidence": "high"},
            {"this_is": "not a valid row"},
        ]
    })
    rows = parse_extraction_response(raw)
    assert len(rows) == 1
    assert rows[0].vendor == "HELM"


def test_parse_extraction_response_extracts_from_fenced_json():
    fenced = "Some chat preface\n```json\n" + json.dumps({"quotes": []}) + "\n```\nTrailing text"
    assert parse_extraction_response(fenced) == []


def test_parse_extraction_response_returns_empty_on_unparseable():
    assert parse_extraction_response("totally not json") == []


def test_extract_quotes_from_email_end_to_end_with_stub(stub_openai):
    stub_openai(json.dumps({
        "quotes": [
            {"item_number": "NPKU32", "vendor": "HELM", "quote_date": "2026-04-03",
             "price": 475.0, "unit": "ton", "mode": "rail_delivered",
             "po_number": "431-8237", "notes": "rail to FL",
             "source_excerpt": "HELM is offering $475/ton rail delivered for U-32.",
             "confidence": "high"}
        ]
    }))
    rows = extract_quotes_from_email(
        email_subject="RE: PO 431-8237",
        email_body="HELM is offering $475/ton rail delivered for U-32.",
        sender="todd.wilson@helm.com",
        aliases=ALIASES,
        api_key="stub",
        model="gpt-4o",
    )
    assert len(rows) == 1
    assert rows[0].vendor == "HELM"
