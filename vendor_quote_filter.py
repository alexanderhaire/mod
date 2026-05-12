"""Decide whether an email is worth running through the quote extractor.

Recall over precision: include if EITHER the sender belongs to a known vendor
OR the subject contains a PO number / product keyword. False positives are
cheap (the extractor will return zero rows); false negatives mean a quote we
never see.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SUBJECT_PATTERNS = [
    re.compile(r"PO\s*431-\d+", re.IGNORECASE),
    re.compile(r"U-?32", re.IGNORECASE),
    re.compile(r"UAN", re.IGNORECASE),
    re.compile(r"urea", re.IGNORECASE),
    re.compile(r"phos(?:phoric|phate)?", re.IGNORECASE),
    re.compile(r"potash", re.IGNORECASE),
]


def load_vendor_domains(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _domain_of(email: str | None) -> str | None:
    if not email or "@" not in email:
        return None
    return email.rsplit("@", 1)[-1].strip().lower()


def resolve_vendor_from_sender(sender: str | None, domains: dict[str, Any]) -> str | None:
    """Return the vendor name for ``sender`` if its domain matches a known one."""
    domain = _domain_of(sender)
    if domain is None:
        return None
    for vendor in domains.get("vendors", []):
        for known in vendor.get("domains", []):
            if domain == known.lower():
                return vendor["name"]
    return None


def _subject_matches(subject: str | None) -> bool:
    if not subject:
        return False
    return any(pat.search(subject) for pat in SUBJECT_PATTERNS)


def should_process_email(
    sender: str | None,
    subject: str | None,
    domains: dict[str, Any],
) -> bool:
    """Return True if the message should be passed to the extractor."""
    if resolve_vendor_from_sender(sender, domains) is not None:
        return True
    return _subject_matches(subject)
