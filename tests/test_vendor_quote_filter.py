"""Tests for vendor_quote_filter."""
from __future__ import annotations

import pytest

from vendor_quote_filter import (
    load_vendor_domains,
    should_process_email,
    resolve_vendor_from_sender,
)


SAMPLE_DOMAINS = {
    "vendors": [
        {"name": "HELM", "domains": ["helm.com", "helmagro.com"]},
        {"name": "Nutrien", "domains": ["nutrien.com"]},
        {"name": "Trademark Nitrogen", "domains": ["trademarknitrogen.com"]},
    ]
}


def test_load_vendor_domains_returns_dict(tmp_path, monkeypatch):
    import json
    p = tmp_path / "vendor_domains.json"
    p.write_text(json.dumps(SAMPLE_DOMAINS), encoding="utf-8")
    result = load_vendor_domains(p)
    assert result == SAMPLE_DOMAINS


def test_should_process_email_matches_known_vendor_domain():
    assert should_process_email(
        sender="charles.nelson@nutrien.com",
        subject="Some unrelated subject",
        domains=SAMPLE_DOMAINS,
    ) is True


def test_should_process_email_matches_po_subject_even_with_unknown_sender():
    assert should_process_email(
        sender="random@example.com",
        subject="Fw: PO 431-8294 confirmation",
        domains=SAMPLE_DOMAINS,
    ) is True


def test_should_process_email_matches_product_keyword_subject():
    assert should_process_email(
        sender="random@example.com",
        subject="RE: U-32 pricing for May",
        domains=SAMPLE_DOMAINS,
    ) is True


def test_should_process_email_rejects_unknown_sender_and_unrelated_subject():
    assert should_process_email(
        sender="marketing@somesite.com",
        subject="Newsletter for May",
        domains=SAMPLE_DOMAINS,
    ) is False


def test_should_process_email_handles_missing_subject():
    assert should_process_email(
        sender="todd.wilson@helm.com",
        subject=None,
        domains=SAMPLE_DOMAINS,
    ) is True


def test_should_process_email_handles_missing_sender():
    assert should_process_email(
        sender=None,
        subject="PO 431-8330",
        domains=SAMPLE_DOMAINS,
    ) is True


def test_resolve_vendor_from_sender_known():
    assert resolve_vendor_from_sender("Todd.Wilson@HELM.com", SAMPLE_DOMAINS) == "HELM"


def test_resolve_vendor_from_sender_unknown_returns_none():
    assert resolve_vendor_from_sender("hi@example.org", SAMPLE_DOMAINS) is None


def test_resolve_vendor_handles_malformed_sender():
    assert resolve_vendor_from_sender("not-an-email", SAMPLE_DOMAINS) is None
    assert resolve_vendor_from_sender("", SAMPLE_DOMAINS) is None
    assert resolve_vendor_from_sender(None, SAMPLE_DOMAINS) is None
