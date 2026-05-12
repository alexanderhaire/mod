# Vendor Quotes on Item Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface supplier price quotes ingested from Outlook (via Microsoft Graph + OpenAI extraction) alongside ERP-confirmed receipts on the Market Intelligence item detail page, so the user can see who's cheapest at quote-time.

**Architecture:** A scheduled Python script polls Outlook with delta queries, filters by sender/subject, extracts structured quote rows with an OpenAI structured-output call, and appends to `data/vendor_quotes.json`. The existing Market Intelligence product-insights view in `app.py` renders a new "Vendor Quotes & Receipts" panel that reads that JSON and joins each row to confirmed receipts from `POP30300`. A refresh button on the panel calls the same ingest function in-process.

**Tech Stack:** Python 3.12, Streamlit 1.51, `msal` (Microsoft auth), `requests` (Graph + OpenAI REST), `pyodbc` (existing), `pytest` (new — minimal scaffold for pure-function tests).

**Spec:** [`docs/superpowers/specs/2026-05-12-vendor-quotes-on-item-page-design.md`](../specs/2026-05-12-vendor-quotes-on-item-page-design.md)

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `vendor_quote_store.py` | Read/append/reduce `data/vendor_quotes.json`. No external I/O. |
| `vendor_quote_normalize.py` | Convert raw prices ($/lb, $/railcar, $/gallon) to $/ton. Flag ambiguous units. |
| `vendor_quote_filter.py` | Decide whether an email should be processed (sender domain OR subject pattern). |
| `vendor_quote_extractor.py` | Build OpenAI prompt with alias map; call OpenAI; validate and parse extraction result. |
| `graph_mail_client.py` | Wrap `msal` + Graph REST: certificate auth, delta query, message-body fetch. |
| `vendor_quote_ingest.py` | Orchestrator. Runs Graph → filter → extractor → store. CLI entry point and importable `run_ingest()`. |
| `data/vendor_domains.json` | Vendor → email-domain registry (seed). |
| `data/vendor_quote_aliases.json` | ITEMNMBR → list of natural-language aliases (seed). |
| `data/vendor_quotes.json` | Append-only store of extracted quotes (created at first ingest). |
| `data/vendor_quote_cursor.json` | Graph delta cursor (created at first ingest). |
| `data/vendor_quote_ingest.log` | Run summary log (created at first ingest). |
| `tests/__init__.py` | Empty, makes `tests/` importable. |
| `tests/conftest.py` | pytest fixtures (tmp_path-based store factory, stub OpenAI client). |
| `tests/test_vendor_quote_store.py` | TDD for store. |
| `tests/test_vendor_quote_normalize.py` | TDD for unit conversion. |
| `tests/test_vendor_quote_filter.py` | TDD for email filter. |
| `tests/test_vendor_quote_extractor.py` | TDD for prompt builder + result parsing. |
| `tests/test_vendor_quote_panel_helpers.py` | TDD for join + cheapest-current helpers. |
| `docs/superpowers/runbooks/vendor-quote-ingest-setup.md` | One-time setup: Azure app reg, certificate, secrets, Task Scheduler. |

### Modified files

| Path | Change |
|---|---|
| `requirements.txt` | Add `msal`, `cryptography`, `pytest`. |
| `constants.py` | Add Graph constants (auth URL, scope, endpoint). |
| `secrets_loader.py` | Add `load_graph_settings()`. |
| `market_insights.py` | Add `render_vendor_quotes_panel(...)` and supporting pure helpers. |
| `app.py` | Wire `render_vendor_quotes_panel()` into the `product_insights` branch (around line 2620). |

---

## Task 1: Add dependencies and minimal test scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add new dependencies to `requirements.txt`**

Append these three lines to `requirements.txt` (preserve existing content):

```
msal==1.31.0
cryptography==43.0.1
pytest==8.3.3
```

- [ ] **Step 2: Install the new deps**

Run: `.\.venv\Scripts\python.exe -m pip install msal==1.31.0 cryptography==43.0.1 pytest==8.3.3`

Expected: successful install, no errors. Verify with `.\.venv\Scripts\python.exe -c "import msal, cryptography, pytest; print('ok')"` → prints `ok`.

- [ ] **Step 3: Create empty `tests/__init__.py`**

Create `tests/__init__.py` with empty content (zero bytes is fine; an empty docstring works too).

- [ ] **Step 4: Create `tests/conftest.py`**

Create `tests/conftest.py` with:

```python
"""Shared pytest fixtures for vendor-quote tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Return a temp file path suitable for a vendor_quotes.json store."""
    return tmp_path / "vendor_quotes.json"


@pytest.fixture
def write_store(store_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Return a helper that writes a dict to the temp store and returns the path."""
    def _write(payload: dict[str, Any]) -> Path:
        store_path.write_text(json.dumps(payload), encoding="utf-8")
        return store_path
    return _write


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Replace the OpenAI HTTP call in vendor_quote_extractor with a stub returning a fixed string."""
    def _install(response_content: str) -> None:
        def _fake_call(_messages, _model=None, _api_key=None):
            return response_content
        import vendor_quote_extractor as vqe
        monkeypatch.setattr(vqe, "_call_openai_chat", _fake_call)
    return _install
```

- [ ] **Step 5: Verify pytest discovers the new tests dir**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q --collect-only`
Expected: no tests collected yet (0 items), no errors about import paths.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt tests/__init__.py tests/conftest.py
git commit -m @'
feat: add msal/cryptography/pytest deps and tests scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 2: `vendor_quote_store.py` (storage, TDD)

**Files:**
- Create: `tests/test_vendor_quote_store.py`
- Create: `vendor_quote_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vendor_quote_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_store.py -v`
Expected: collection error or `ModuleNotFoundError: No module named 'vendor_quote_store'`.

- [ ] **Step 3: Write `vendor_quote_store.py`**

Create `vendor_quote_store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```powershell
git add vendor_quote_store.py tests/test_vendor_quote_store.py
git commit -m @'
feat: add vendor_quote_store with append-only JSON persistence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 3: `vendor_quote_normalize.py` (price unit conversion, TDD)

**Files:**
- Create: `tests/test_vendor_quote_normalize.py`
- Create: `vendor_quote_normalize.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vendor_quote_normalize.py`:

```python
"""Tests for vendor_quote_normalize."""
from __future__ import annotations

import pytest

from vendor_quote_normalize import normalize_to_per_ton


def test_dollars_per_ton_passthrough():
    result = normalize_to_per_ton(price=475.0, unit="ton")
    assert result.price_per_ton == 475.0
    assert result.confidence == "high"
    assert result.warnings == []


def test_dollars_per_lb_converted():
    # 0.2055 $/lb * 2000 lb/ton = 411.00 $/ton
    result = normalize_to_per_ton(price=0.2055, unit="lb")
    assert result.price_per_ton == pytest.approx(411.0, rel=1e-3)
    assert result.confidence == "high"
    assert result.warnings == []


def test_railcar_flagged_as_low_confidence():
    # We can't know tons/railcar from the unit alone (25-30 typical).
    # Strategy: store the raw value, flag low-confidence, do NOT pick a number.
    result = normalize_to_per_ton(price=595.0, unit="railcar")
    assert result.price_per_ton is None
    assert result.confidence == "low"
    assert "unit_ambiguous_railcar" in result.warnings


def test_gallon_requires_weight_per_gallon():
    # Without lbs/gallon, can't normalize.
    result = normalize_to_per_ton(price=4.20, unit="gallon")
    assert result.price_per_ton is None
    assert result.confidence == "low"
    assert "unit_requires_weight_per_gallon" in result.warnings


def test_gallon_with_weight_per_gallon():
    # $4.20/gal * (2000 lb/ton ÷ 11.06 lb/gal) ≈ $759.5/ton
    result = normalize_to_per_ton(price=4.20, unit="gallon", weight_per_gallon=11.06)
    assert result.price_per_ton == pytest.approx(759.5, abs=1.0)
    assert result.confidence == "high"


def test_unknown_unit_is_low_confidence():
    result = normalize_to_per_ton(price=1.0, unit="bushel")
    assert result.price_per_ton is None
    assert result.confidence == "low"
    assert "unit_unknown" in result.warnings


def test_unit_case_insensitive():
    a = normalize_to_per_ton(price=475.0, unit="TON")
    b = normalize_to_per_ton(price=475.0, unit="Ton")
    c = normalize_to_per_ton(price=475.0, unit="t")
    assert a.price_per_ton == b.price_per_ton == c.price_per_ton == 475.0


def test_short_ton_alias():
    result = normalize_to_per_ton(price=475.0, unit="short_ton")
    assert result.price_per_ton == 475.0
    assert result.confidence == "high"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_normalize.py -v`
Expected: `ModuleNotFoundError: No module named 'vendor_quote_normalize'`.

- [ ] **Step 3: Write `vendor_quote_normalize.py`**

Create `vendor_quote_normalize.py`:

```python
"""Convert vendor-quoted prices to a canonical $/ton.

Why this exists: emails quote in $/lb, $/ton, $/railcar, $/gallon. Comparing
vendors requires one column. Anything we can't safely convert (e.g., "per
railcar" without knowing tons/car) stays as the raw value with a low-confidence
flag so the UI can prompt the user before treating it as comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LB_PER_TON = 2000.0

_TON_ALIASES = {"ton", "tons", "t", "short_ton", "shortton"}
_LB_ALIASES = {"lb", "lbs", "pound", "pounds"}
_GALLON_ALIASES = {"gallon", "gal", "gallons"}
_RAILCAR_ALIASES = {"railcar", "rail_car", "rc", "car"}


@dataclass
class NormalizedPrice:
    price_per_ton: float | None
    confidence: str  # "high" | "low"
    warnings: list[str] = field(default_factory=list)


def normalize_to_per_ton(
    price: float,
    unit: str,
    weight_per_gallon: float | None = None,
) -> NormalizedPrice:
    """Convert ``price`` (numeric) in ``unit`` to $/ton.

    Returns a ``NormalizedPrice`` whose ``price_per_ton`` is ``None`` when the
    conversion can't be done safely. Callers should preserve the raw quoted
    value separately and surface low-confidence rows for user confirmation.
    """
    u = (unit or "").strip().lower()

    if u in _TON_ALIASES:
        return NormalizedPrice(price_per_ton=float(price), confidence="high")

    if u in _LB_ALIASES:
        return NormalizedPrice(price_per_ton=float(price) * LB_PER_TON, confidence="high")

    if u in _GALLON_ALIASES:
        if weight_per_gallon and weight_per_gallon > 0:
            tons_per_gallon = weight_per_gallon / LB_PER_TON
            return NormalizedPrice(
                price_per_ton=float(price) / tons_per_gallon,
                confidence="high",
            )
        return NormalizedPrice(
            price_per_ton=None,
            confidence="low",
            warnings=["unit_requires_weight_per_gallon"],
        )

    if u in _RAILCAR_ALIASES:
        return NormalizedPrice(
            price_per_ton=None,
            confidence="low",
            warnings=["unit_ambiguous_railcar"],
        )

    return NormalizedPrice(
        price_per_ton=None,
        confidence="low",
        warnings=["unit_unknown"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_normalize.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```powershell
git add vendor_quote_normalize.py tests/test_vendor_quote_normalize.py
git commit -m @'
feat: add vendor_quote_normalize for $/ton unit conversion

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 4: `vendor_quote_filter.py` (email inclusion filter, TDD)

**Files:**
- Create: `tests/test_vendor_quote_filter.py`
- Create: `vendor_quote_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vendor_quote_filter.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_filter.py -v`
Expected: `ModuleNotFoundError: No module named 'vendor_quote_filter'`.

- [ ] **Step 3: Write `vendor_quote_filter.py`**

Create `vendor_quote_filter.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_filter.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```powershell
git add vendor_quote_filter.py tests/test_vendor_quote_filter.py
git commit -m @'
feat: add vendor_quote_filter for email inclusion logic

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 5: Seed `data/vendor_domains.json` and `data/vendor_quote_aliases.json`

**Files:**
- Create: `data/vendor_domains.json`
- Create: `data/vendor_quote_aliases.json`

- [ ] **Step 1: Create `data/vendor_domains.json`**

```json
{
  "vendors": [
    {"name": "HELM", "domains": ["helm.com", "helmagro.com", "helmag.com"]},
    {"name": "Nutrien", "domains": ["nutrien.com"]},
    {"name": "Trademark Nitrogen", "domains": ["trademarknitrogen.com"]}
  ]
}
```

- [ ] **Step 2: Create `data/vendor_quote_aliases.json`**

```json
{
  "NPKU32": ["U-32", "U32", "UAN 32", "UAN-32", "32% UAN", "urea ammonium nitrate 32", "urea ammonium nitrate solution 32"],
  "NPKUREA": ["urea 46", "urea prill", "granular urea", "urea 46%"],
  "NPKKCL62": ["potassium chloride 62", "potash 62", "MOP"],
  "NPKKNO3": ["potassium nitrate 13-0-45", "KNO3", "NOP"],
  "NPKPHOS75": ["phosphoric acid 75", "phosphoric acid 85", "phos acid"]
}
```

- [ ] **Step 3: Verify both files parse as JSON**

Run: `.\.venv\Scripts\python.exe -c "import json, pathlib; [json.loads(pathlib.Path(p).read_text()) for p in ['data/vendor_domains.json', 'data/vendor_quote_aliases.json']]; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```powershell
git add data/vendor_domains.json data/vendor_quote_aliases.json
git commit -m @'
feat: seed vendor-domain registry and item-alias map

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 6: `vendor_quote_extractor.py` (OpenAI extraction, TDD on parse/validate)

**Files:**
- Create: `tests/test_vendor_quote_extractor.py`
- Create: `vendor_quote_extractor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vendor_quote_extractor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_extractor.py -v`
Expected: `ModuleNotFoundError: No module named 'vendor_quote_extractor'`.

- [ ] **Step 3: Write `vendor_quote_extractor.py`**

Create `vendor_quote_extractor.py`:

```python
"""Extract structured vendor-quote rows from email bodies via OpenAI.

The flow: build a system+user prompt that names the item-alias map, the JSON
output schema, and the email content. Call OpenAI chat completions with
``response_format = {"type": "json_object"}``. Parse the response with a forgiving
``_extract_json_block`` so model-side wrapping (```json fences``, prose preface)
doesn't break us.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

LOGGER = logging.getLogger("vendor_quote_extractor")

_SYSTEM_PROMPT = """You extract supplier price quotes from procurement emails.

Return ONLY a JSON object with this shape:
{
  "quotes": [
    {
      "item_number": "<one of the ITEMNMBR keys from the alias map, or null if unknown>",
      "vendor": "<supplier name>",
      "quote_date": "<YYYY-MM-DD or null>",
      "price": <numeric>,
      "unit": "<ton|lb|gallon|railcar|...>",
      "mode": "<rail_delivered|pickup|truck_delivered|null>",
      "po_number": "<PO# string or null>",
      "notes": "<short free-text>",
      "source_excerpt": "<verbatim 1-2 sentences from the email>",
      "confidence": "<high|low>"
    }
  ]
}

Rules:
- Map natural-language item references to ITEMNMBR using the alias map provided in the user message.
- Return an empty quotes array if no clear price is stated.
- Set confidence to "low" when unit is ambiguous (e.g., "per railcar"), date is missing, or the item isn't resolvable.
- Do not invent values. If unsure, use null.
"""

_USER_TEMPLATE = """Item alias map:
{aliases_json}

Email metadata:
- From: {sender}
- Subject: {subject}

Email body:
\"\"\"
{body}
\"\"\"
"""


@dataclass
class ExtractedRow:
    item_number: str | None
    vendor: str | None
    quote_date: str | None
    price: float | None
    unit: str | None
    mode: str | None
    po_number: str | None
    notes: str
    source_excerpt: str
    confidence: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExtractedRow | None":
        required = {"item_number", "vendor", "price", "unit"}
        if not required.issubset(d.keys()):
            return None
        try:
            return cls(
                item_number=d.get("item_number"),
                vendor=d.get("vendor"),
                quote_date=d.get("quote_date"),
                price=float(d["price"]) if d.get("price") is not None else None,
                unit=d.get("unit"),
                mode=d.get("mode"),
                po_number=d.get("po_number"),
                notes=str(d.get("notes") or ""),
                source_excerpt=str(d.get("source_excerpt") or ""),
                confidence=str(d.get("confidence") or "high"),
            )
        except (TypeError, ValueError):
            return None


def build_prompt_messages(
    email_subject: str | None,
    email_body: str,
    sender: str | None,
    aliases: dict[str, list[str]],
) -> list[dict[str, str]]:
    user = _USER_TEMPLATE.format(
        aliases_json=json.dumps(aliases, indent=2),
        sender=sender or "<unknown>",
        subject=email_subject or "<no subject>",
        body=email_body or "",
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _coerce_to_json_object(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output, even if wrapped in prose or fences."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fenced = _JSON_FENCE_RE.search(raw)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    # Last-ditch: take from the first '{' to the last '}'
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(raw[first:last + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_extraction_response(raw: str) -> list[ExtractedRow]:
    obj = _coerce_to_json_object(raw)
    if not isinstance(obj, dict):
        return []
    quotes = obj.get("quotes")
    if not isinstance(quotes, list):
        return []
    rows: list[ExtractedRow] = []
    for entry in quotes:
        if not isinstance(entry, dict):
            continue
        row = ExtractedRow.from_dict(entry)
        if row is not None:
            rows.append(row)
    return rows


def _call_openai_chat(messages: list[dict[str, str]], model: str, api_key: str) -> str:
    """Call OpenAI chat completions and return the assistant content string.

    Retries up to 3 times with exponential backoff on transient errors
    (HTTP 429, 500, 502, 503, 504, or any network exception). Once the delta
    cursor advances past a message, that message will not be re-pulled, so
    we have one shot per cron run to extract its quote.
    """
    import time
    from constants import OPENAI_CHAT_URL, OPENAI_TIMEOUT_SECONDS

    transient_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 ** (2 * attempt))  # 0s, 4s, 16s
        try:
            response = requests.post(
                OPENAI_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                },
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
            if response.status_code in transient_statuses and attempt < 2:
                last_error = requests.HTTPError(f"transient {response.status_code}")
                continue
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            continue
    raise last_error or RuntimeError("OpenAI call failed without specific error")


def extract_quotes_from_email(
    email_subject: str | None,
    email_body: str,
    sender: str | None,
    aliases: dict[str, list[str]],
    api_key: str,
    model: str,
) -> list[ExtractedRow]:
    messages = build_prompt_messages(email_subject, email_body, sender, aliases)
    try:
        raw = _call_openai_chat(messages, model=model, api_key=api_key)
    except Exception:
        LOGGER.exception("OpenAI call failed for subject=%r sender=%r", email_subject, sender)
        return []
    return parse_extraction_response(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_extractor.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```powershell
git add vendor_quote_extractor.py tests/test_vendor_quote_extractor.py
git commit -m @'
feat: add vendor_quote_extractor for OpenAI structured-output extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 7: `graph_mail_client.py` (Microsoft Graph wrapper)

**Note:** No automated tests for this module — auth and Graph calls are hard to mock meaningfully. Verification is via a manual smoke step at the end.

**Files:**
- Modify: `constants.py`
- Modify: `secrets_loader.py`
- Create: `graph_mail_client.py`

- [ ] **Step 1: Add Graph constants to `constants.py`**

Append these lines to `constants.py` after the OpenAI constants (after line 130 or so):

```python
# --- Microsoft Graph ---
GRAPH_AUTHORITY_TEMPLATE = "https://login.microsoftonline.com/{tenant_id}"
GRAPH_SCOPE_DEFAULT = ["https://graph.microsoft.com/.default"]
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_TIMEOUT_SECONDS = 30
```

- [ ] **Step 2: Add `load_graph_settings()` to `secrets_loader.py`**

Append this function to the end of `secrets_loader.py`:

```python
def load_graph_settings() -> dict:
    """Load Microsoft Graph settings from the [graph] TOML section.

    Expected keys:
      tenant_id            - Azure tenant ID (GUID)
      client_id            - App registration client ID (GUID)
      certificate_path     - Absolute path to the X.509 cert (.pem with private key)
      certificate_thumbprint - SHA-1 thumbprint of the cert, hex
      mailbox              - UPN of the mailbox to poll
    """
    section = load_local_secret_section("graph")
    required = {"tenant_id", "client_id", "certificate_path", "certificate_thumbprint", "mailbox"}
    missing = required - set(section)
    if missing:
        return {}
    return dict(section)
```

- [ ] **Step 3: Create `graph_mail_client.py`**

Create `graph_mail_client.py`:

```python
"""Thin wrapper around Microsoft Graph: app-only auth + delta-query mail fetch.

This module assumes the secrets section ``[graph]`` is populated (see
``secrets_loader.load_graph_settings``). Use a real Outlook account; there is no
mocking layer here on purpose — failures should surface immediately during
manual setup.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import msal
import requests

from constants import (
    GRAPH_API_BASE,
    GRAPH_AUTHORITY_TEMPLATE,
    GRAPH_SCOPE_DEFAULT,
    GRAPH_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("graph_mail_client")


class GraphMailClient:
    def __init__(self, settings: dict[str, Any]):
        self._settings = settings
        self._token: str | None = None

    def _acquire_token(self) -> str:
        if self._token:
            return self._token
        cert_path = Path(self._settings["certificate_path"])
        cert_pem = cert_path.read_text(encoding="utf-8")
        app = msal.ConfidentialClientApplication(
            client_id=self._settings["client_id"],
            authority=GRAPH_AUTHORITY_TEMPLATE.format(tenant_id=self._settings["tenant_id"]),
            client_credential={
                "private_key": cert_pem,
                "thumbprint": self._settings["certificate_thumbprint"],
            },
        )
        result = app.acquire_token_for_client(scopes=GRAPH_SCOPE_DEFAULT)
        if "access_token" not in result:
            raise RuntimeError(f"Graph auth failed: {result.get('error_description', result)}")
        self._token = result["access_token"]
        return self._token

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        token = self._acquire_token()
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def iter_delta_messages(
        self,
        delta_link: str | None = None,
        backfill_days: int = 30,
    ) -> Iterator[tuple[dict[str, Any], str | None]]:
        """Yield (message, next_delta_link) pairs.

        Only the LAST tuple's ``next_delta_link`` is meaningful (the persisted cursor).
        Intermediate yields have ``next_delta_link = None``.
        """
        mailbox = self._settings["mailbox"]
        if delta_link:
            url = delta_link
            params = None
        else:
            url = f"{GRAPH_API_BASE}/users/{mailbox}/mailFolders/Inbox/messages/delta"
            params = {
                "$select": "id,subject,from,bodyPreview,body,receivedDateTime,internetMessageId",
            }

        while True:
            payload = self._get(url, params)
            params = None
            for msg in payload.get("value", []):
                yield msg, None
            next_link = payload.get("@odata.nextLink")
            delta_link_out = payload.get("@odata.deltaLink")
            if next_link:
                url = next_link
                continue
            if delta_link_out:
                yield {}, delta_link_out
                return
            return

    def fetch_message_body(self, message_id: str) -> str:
        mailbox = self._settings["mailbox"]
        data = self._get(f"{GRAPH_API_BASE}/users/{mailbox}/messages/{message_id}")
        body = data.get("body", {})
        return body.get("content", "") if isinstance(body, dict) else ""
```

- [ ] **Step 4: Verify the module imports cleanly**

Run: `.\.venv\Scripts\python.exe -c "import graph_mail_client; print('ok')"`
Expected: `ok`. (Real Graph calls are tested at end of plan.)

- [ ] **Step 5: Commit**

```powershell
git add constants.py secrets_loader.py graph_mail_client.py
git commit -m @'
feat: add Microsoft Graph mail client with cert-based app auth

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 8: `vendor_quote_ingest.py` (orchestrator script)

**Files:**
- Create: `vendor_quote_ingest.py`

This script has no unit tests — it composes already-tested modules. Verification is via the manual smoke step at the end of the plan.

- [ ] **Step 1: Create `vendor_quote_ingest.py`**

```python
"""Orchestrate vendor-quote ingestion: Graph → filter → extractor → store.

Usable two ways:
  * CLI: ``python vendor_quote_ingest.py`` (scheduled run)
  * Import: ``from vendor_quote_ingest import run_ingest`` (Streamlit refresh button)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from constants import OPENAI_DEFAULT_MODEL
from graph_mail_client import GraphMailClient
from secrets_loader import load_graph_settings, load_openai_settings
from vendor_quote_extractor import extract_quotes_from_email
from vendor_quote_filter import load_vendor_domains, resolve_vendor_from_sender, should_process_email
from vendor_quote_normalize import normalize_to_per_ton
from vendor_quote_store import append_quote

LOGGER = logging.getLogger("vendor_quote_ingest")

DATA_DIR = Path(__file__).parent / "data"
STORE_PATH = DATA_DIR / "vendor_quotes.json"
CURSOR_PATH = DATA_DIR / "vendor_quote_cursor.json"
LOCK_PATH = DATA_DIR / "vendor_quote_ingest.lock"
LOG_PATH = DATA_DIR / "vendor_quote_ingest.log"
DOMAINS_PATH = DATA_DIR / "vendor_domains.json"
ALIASES_PATH = DATA_DIR / "vendor_quote_aliases.json"


@dataclass
class IngestSummary:
    seen: int = 0
    matched: int = 0
    extracted_rows: int = 0
    low_confidence: int = 0
    errors: int = 0


@contextlib.contextmanager
def _lockfile(path: Path):
    """Simple advisory lock — exclusive create. Caller decides what to do on contention."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Lock present at {path} — another ingest already running") from exc
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _load_cursor() -> str | None:
    if not CURSOR_PATH.exists():
        return None
    try:
        return json.loads(CURSOR_PATH.read_text(encoding="utf-8")).get("delta_link")
    except (json.JSONDecodeError, OSError):
        return None


def _save_cursor(delta_link: str) -> None:
    CURSOR_PATH.write_text(json.dumps({"delta_link": delta_link}), encoding="utf-8")


def _extract_plain_text(html_or_text: str) -> str:
    """Strip rudimentary HTML so the extractor sees readable text."""
    import re
    no_tags = re.sub(r"<[^>]+>", " ", html_or_text or "")
    return re.sub(r"\s+", " ", no_tags).strip()


def run_ingest(
    item_filter: str | None = None,
    backfill_days: int = 30,
    dry_run: bool = False,
) -> IngestSummary:
    """Run one ingest pass. Returns a summary."""
    summary = IngestSummary()
    graph_settings = load_graph_settings()
    if not graph_settings:
        LOGGER.error("Graph settings missing or incomplete in secrets.toml. Aborting.")
        return summary
    openai_settings = load_openai_settings()
    if not openai_settings.get("api_key"):
        LOGGER.error("OpenAI api_key missing. Aborting.")
        return summary

    domains = load_vendor_domains(DOMAINS_PATH)
    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))

    with _lockfile(LOCK_PATH):
        client = GraphMailClient(graph_settings)
        delta_link = _load_cursor()
        next_delta: str | None = None

        for msg, maybe_next_delta in client.iter_delta_messages(
            delta_link=delta_link, backfill_days=backfill_days
        ):
            if maybe_next_delta:
                next_delta = maybe_next_delta
                break
            summary.seen += 1
            sender = (msg.get("from") or {}).get("emailAddress", {}).get("address")
            subject = msg.get("subject")
            if not should_process_email(sender, subject, domains):
                continue
            summary.matched += 1
            body_dict = msg.get("body") or {}
            body_raw = body_dict.get("content", "") or msg.get("bodyPreview", "")
            body_text = (
                _extract_plain_text(body_raw)
                if body_dict.get("contentType", "").lower() == "html"
                else body_raw
            )
            try:
                rows = extract_quotes_from_email(
                    email_subject=subject,
                    email_body=body_text,
                    sender=sender,
                    aliases=aliases,
                    api_key=openai_settings["api_key"],
                    model=openai_settings.get("model") or OPENAI_DEFAULT_MODEL,
                )
            except Exception:
                summary.errors += 1
                LOGGER.exception("Extractor crashed on msg id=%s", msg.get("id"))
                continue
            for row in rows:
                norm = normalize_to_per_ton(price=row.price or 0.0, unit=row.unit or "")
                warnings = list(norm.warnings)
                confidence = "low" if (row.confidence == "low" or norm.confidence == "low") else "high"
                if confidence == "low":
                    summary.low_confidence += 1
                item_number = row.item_number or "_unresolved"
                if item_filter and item_number != item_filter:
                    continue
                stored_row = {
                    "vendor": row.vendor or resolve_vendor_from_sender(sender, domains) or "unknown",
                    "quote_date": row.quote_date,
                    "price_per_ton": norm.price_per_ton,
                    "raw_price": f"${row.price}/{row.unit}" if row.price is not None else None,
                    "mode": row.mode,
                    "po_number": row.po_number,
                    "notes": row.notes,
                    "source_message_id": msg.get("id"),
                    "source_subject": subject,
                    "source_excerpt": row.source_excerpt,
                    "confidence": confidence,
                    "warnings": warnings,
                    "ingested_at": dt.datetime.utcnow().isoformat() + "Z",
                }
                if not dry_run:
                    append_quote(STORE_PATH, item_number, stored_row)
                summary.extracted_rows += 1
        if next_delta and not dry_run:
            _save_cursor(next_delta)

    _append_log(summary)
    return summary


def _append_log(summary: IngestSummary) -> None:
    line = (
        f"{dt.datetime.utcnow().isoformat()}Z "
        f"seen={summary.seen} matched={summary.matched} "
        f"rows={summary.extracted_rows} low_conf={summary.low_confidence} "
        f"errors={summary.errors}\n"
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest vendor quotes from Outlook via Graph.")
    parser.add_argument("--item", dest="item_filter", default=None,
                        help="If set, only persist rows for this ITEMNMBR.")
    parser.add_argument("--backfill-days", type=int, default=30,
                        help="If no cursor exists, how many days of history to pull.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process messages but do not write to the store or cursor.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_ingest(
            item_filter=args.item_filter,
            backfill_days=args.backfill_days,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2
    LOGGER.info(
        "ingest complete: seen=%d matched=%d rows=%d low_conf=%d errors=%d",
        summary.seen, summary.matched, summary.extracted_rows, summary.low_confidence, summary.errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `.\.venv\Scripts\python.exe -c "import vendor_quote_ingest; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Verify the CLI help works**

Run: `.\.venv\Scripts\python.exe vendor_quote_ingest.py --help`
Expected: argparse help text listing `--item`, `--backfill-days`, `--dry-run`, `--verbose`.

- [ ] **Step 4: Commit**

```powershell
git add vendor_quote_ingest.py
git commit -m @'
feat: add vendor_quote_ingest orchestrator (CLI + importable)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 9: Panel helpers in `market_insights.py` (TDD on pure logic)

**Files:**
- Create: `tests/test_vendor_quote_panel_helpers.py`
- Modify: `market_insights.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vendor_quote_panel_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_panel_helpers.py -v`
Expected: `ImportError: cannot import name 'join_quotes_to_receipts' from 'market_insights'`.

- [ ] **Step 3: Add helpers and panel renderer to `market_insights.py`**

Open `market_insights.py` and append at the end:

```python
# ---------------------------------------------------------------------------
# Vendor Quotes & Receipts panel — pure helpers + Streamlit renderer
# ---------------------------------------------------------------------------
import datetime as _vq_dt
from pathlib import Path as _vq_Path


def _vq_parse_date(value) -> _vq_dt.date | None:
    if not value:
        return None
    if isinstance(value, _vq_dt.date):
        return value
    s = str(value)[:10]
    try:
        return _vq_dt.date.fromisoformat(s)
    except ValueError:
        return None


def _vq_latest_receipt_per_vendor_mode(receipts: list[dict]) -> dict[tuple[str, str], dict]:
    """Reduce raw receipt rows to one-per-(vendor, mode) by latest TransactionDate.

    Mode defaults to "unknown" because POP30300 doesn't carry shipping mode.
    """
    out: dict[tuple[str, str], dict] = {}
    for r in receipts:
        vendor = (r.get("VendorName") or "").strip()
        if not vendor:
            continue
        mode = r.get("Mode") or "unknown"
        d = _vq_parse_date(r.get("TransactionDate"))
        key = (vendor, mode)
        existing = out.get(key)
        if existing is None or (d and (_vq_parse_date(existing.get("TransactionDate")) or _vq_dt.date.min) < d):
            out[key] = r
    return out


def join_quotes_to_receipts(quotes: list[dict], receipts: list[dict]) -> list[dict]:
    """Build per-(vendor, mode) rows that pair the latest quote with the latest receipt.

    A row is emitted for every (vendor, mode) appearing on either side. ``po_match``
    is True when the quote's PO and the receipt's PO are identical and non-empty.
    """
    latest_q = {(q.get("vendor", ""), q.get("mode") or "unknown"): q for q in quotes}
    latest_r = _vq_latest_receipt_per_vendor_mode(receipts)

    # Quotes can be more granular on mode than receipts. If a quote has a vendor
    # whose receipts only ever come back with mode="unknown", we treat the receipt
    # as the partner for any mode of that vendor.
    receipt_by_vendor: dict[str, dict] = {}
    for (vendor, _mode), receipt in latest_r.items():
        existing = receipt_by_vendor.get(vendor)
        if existing is None or (
            _vq_parse_date(receipt.get("TransactionDate")) or _vq_dt.date.min
        ) > (_vq_parse_date(existing.get("TransactionDate")) or _vq_dt.date.min):
            receipt_by_vendor[vendor] = receipt

    rows: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()

    for (vendor, mode), q in latest_q.items():
        receipt = receipt_by_vendor.get(vendor)
        q_price = q.get("price_per_ton")
        r_price = float(receipt["AvgCost"]) if receipt and receipt.get("AvgCost") is not None else None
        delta = None
        if q_price is not None and r_price is not None:
            delta = float(q_price) - r_price
        po_match = bool(
            q.get("po_number") and receipt and q.get("po_number") == receipt.get("PONUMBER")
        )
        rows.append({
            "vendor": vendor,
            "mode": mode,
            "quote_date": q.get("quote_date"),
            "quote_price_per_ton": q_price,
            "quote_po": q.get("po_number"),
            "receipt_date": receipt.get("TransactionDate") if receipt else None,
            "receipt_price_per_ton": r_price,
            "receipt_po": receipt.get("PONUMBER") if receipt else None,
            "delta_per_ton": delta,
            "po_match": po_match,
            "confidence": q.get("confidence", "high"),
        })
        seen_keys.add((vendor, mode))

    # Vendors with receipts but no quote
    for vendor, receipt in receipt_by_vendor.items():
        if any(v == vendor for v, _ in seen_keys):
            continue
        r_price = float(receipt["AvgCost"]) if receipt.get("AvgCost") is not None else None
        rows.append({
            "vendor": vendor,
            "mode": receipt.get("Mode") or "unknown",
            "quote_date": None,
            "quote_price_per_ton": None,
            "quote_po": None,
            "receipt_date": receipt.get("TransactionDate"),
            "receipt_price_per_ton": r_price,
            "receipt_po": receipt.get("PONUMBER"),
            "delta_per_ton": None,
            "po_match": False,
            "confidence": "high",
        })
    return rows


def compute_cheapest_current_quote(
    quotes: list[dict],
    today: _vq_dt.date | None = None,
    freshness_days: int = 60,
) -> dict | None:
    """Return the cheapest quote among rows that are fresh AND high confidence."""
    today = today or _vq_dt.date.today()
    candidates: list[dict] = []
    for q in quotes:
        if q.get("confidence") == "low":
            continue
        price = q.get("price_per_ton")
        if price is None:
            continue
        d = _vq_parse_date(q.get("quote_date"))
        if d is None:
            continue
        if (today - d).days > freshness_days:
            continue
        candidates.append(q)
    if not candidates:
        return None
    return min(candidates, key=lambda q: q["price_per_ton"])


def render_vendor_quotes_panel(cursor, item_number: str) -> None:
    """Render the Vendor Quotes & Receipts panel for one item.

    Imports of streamlit/pandas are kept local so this module stays importable
    in test contexts without those deps loaded eagerly.
    """
    import streamlit as st
    import pandas as pd
    from vendor_quote_store import load_quotes, latest_per_vendor_mode

    store_path = _vq_Path(__file__).parent / "data" / "vendor_quotes.json"
    store = load_quotes(store_path)
    quotes_latest = latest_per_vendor_mode(store, item_number)
    receipts = fetch_product_price_history(cursor, item_number, days=730)
    rows = join_quotes_to_receipts(quotes_latest, receipts)

    st.markdown("### Vendor Quotes & Receipts")
    refresh_col, status_col = st.columns([1, 6])
    with refresh_col:
        clicked = st.button("↻ Refresh", key=f"vq_refresh_{item_number}")
    if clicked:
        with st.spinner("Pulling latest quotes from Outlook..."):
            from vendor_quote_ingest import run_ingest
            try:
                summary = run_ingest(item_filter=item_number)
                with status_col:
                    st.caption(
                        f"Refreshed: {summary.matched} matched, "
                        f"{summary.extracted_rows} rows, "
                        f"{summary.low_confidence} low-confidence."
                    )
            except RuntimeError as exc:
                with status_col:
                    st.warning(f"Already running: {exc}")
        st.rerun()

    cheapest = compute_cheapest_current_quote(quotes_latest)
    if cheapest:
        st.success(
            f"★ Cheapest current quote (≤60d): {cheapest['vendor']} — "
            f"${cheapest['price_per_ton']:.2f}/ton {cheapest.get('mode', '')} "
            f"({cheapest.get('quote_date', '?')}, PO {cheapest.get('po_number') or '—'})"
        )
    else:
        st.info("No fresh confirmed quotes in the last 60 days. Click Refresh to pull from Outlook.")

    if not rows:
        st.caption("No vendor history for this item yet.")
        return

    today = _vq_dt.date.today()
    table_rows = []
    for r in rows:
        qd = _vq_parse_date(r.get("quote_date"))
        fresh = bool(qd and (today - qd).days <= 60)
        table_rows.append({
            "Vendor": r["vendor"],
            "Mode": r["mode"],
            "Last Quote": (
                f"${r['quote_price_per_ton']:.2f}/t  {r['quote_date'] or '?'}  #{r.get('quote_po') or '—'}"
                if r["quote_price_per_ton"] is not None else "—"
            ),
            "Last Receipt": (
                f"${r['receipt_price_per_ton']:.2f}/t  {str(r.get('receipt_date'))[:10] or '?'}  #{r.get('receipt_po') or '—'}"
                if r["receipt_price_per_ton"] is not None else "—"
            ),
            "Δ": f"${r['delta_per_ton']:+.2f}" if r["delta_per_ton"] is not None else "—",
            "Notes": (
                ("⚠ " if r.get("confidence") == "low" else "")
                + ("● " if fresh else "○ ")
                + ("= " if r.get("po_match") else "")
            ),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption("● Fresh   ○ Stale (>60d)   ⚠ Low-confidence   = Quote fulfilled by paired PO")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vendor_quote_panel_helpers.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests -q`
Expected: 39 passed (6 store + 8 normalize + 10 filter + 8 extractor + 7 panel).

- [ ] **Step 6: Commit**

```powershell
git add market_insights.py tests/test_vendor_quote_panel_helpers.py
git commit -m @'
feat: add vendor quotes panel helpers + renderer to market_insights

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 10: Wire `render_vendor_quotes_panel` into `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Import the new function**

In `app.py`, locate the existing import block from `market_insights` (starts at line 33: `from market_insights import (`). Add `render_vendor_quotes_panel,` to the list of imported names. Place it next to `fetch_product_price_history` for adjacency.

The added line should sit alongside other names in the existing `from market_insights import (...)` statement — do NOT create a new import statement.

- [ ] **Step 2: Insert the panel call in the product_insights branch**

In `app.py`, find the section starting with:

```python
elif st.session_state.current_page == "product_insights" and st.session_state.selected_product:
```

(currently around line 2608). Within this branch, after the header section and `if not details: ... else:` opening, but BEFORE the chatbot context construction or any chart rendering, insert:

```python
                # --- VENDOR QUOTES & RECEIPTS PANEL ---
                render_vendor_quotes_panel(cursor, product_item)
```

The exact insertion point: immediately after the `# Extract Data` block ends (after `price_hist = details.get('price_history', [])` near line 2643) and before the `# Fetch External Data` spinner. Use the same indentation level as the surrounding `# Extract Data` comments (two-space `else:` body).

- [ ] **Step 3: Smoke test — Streamlit launches**

Run: `.\.venv\Scripts\streamlit.exe run app.py --server.headless true` in one terminal.

In a second PowerShell:
```powershell
Start-Sleep -Seconds 8
Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health | Select-Object -ExpandProperty Content
```
Expected: response body contains `ok`.

Stop the Streamlit process (`Ctrl+C` in the first terminal, or `Stop-Process -Name streamlit`).

- [ ] **Step 4: Commit**

```powershell
git add app.py
git commit -m @'
feat: wire Vendor Quotes panel into Market Intelligence product view

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 11: Write the setup runbook

**Files:**
- Create: `docs/superpowers/runbooks/vendor-quote-ingest-setup.md`

- [ ] **Step 1: Create the runbook**

Create `docs/superpowers/runbooks/vendor-quote-ingest-setup.md`:

````markdown
# Vendor Quote Ingest — One-Time Setup

This runbook walks through the one-time work needed before `vendor_quote_ingest.py`
can pull mail from Outlook. After the seven steps below, the scheduled task
will run every 15 minutes unattended.

---

## 1. Generate a self-signed certificate (PowerShell, admin)

```powershell
$cert = New-SelfSignedCertificate `
    -Subject "CN=vendor-quote-ingest" `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -KeyExportPolicy Exportable `
    -KeySpec Signature `
    -KeyLength 2048 `
    -KeyAlgorithm RSA `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddYears(3)

$cert.Thumbprint    # copy — you'll need this
```

Export the public key for Azure upload:

```powershell
Export-Certificate -Cert $cert -FilePath "$HOME\vendor-quote-ingest.cer"
```

Export the private key (PEM, no password — protect the file with NTFS perms):

```powershell
$pwd = ConvertTo-SecureString -String "tmp" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath "$HOME\vendor-quote-ingest.pfx" -Password $pwd
# convert to PEM
& "C:\Program Files\Git\usr\bin\openssl.exe" pkcs12 -in "$HOME\vendor-quote-ingest.pfx" -nodes -nocerts -out "$HOME\vendor-quote-ingest.key" -passin pass:tmp
& "C:\Program Files\Git\usr\bin\openssl.exe" pkcs12 -in "$HOME\vendor-quote-ingest.pfx" -nokeys -clcerts -out "$HOME\vendor-quote-ingest-cert.pem" -passin pass:tmp
Get-Content "$HOME\vendor-quote-ingest.key", "$HOME\vendor-quote-ingest-cert.pem" | Set-Content "$HOME\vendor-quote-ingest.pem"
Remove-Item "$HOME\vendor-quote-ingest.pfx", "$HOME\vendor-quote-ingest.key", "$HOME\vendor-quote-ingest-cert.pem"
```

The combined `vendor-quote-ingest.pem` is what `secrets.toml` will reference.

## 2. Register the Azure AD app

1. Go to https://entra.microsoft.com → **Identity** → **Applications** → **App registrations** → **New registration**.
2. Name: `Vendor Quote Ingest`. Account types: *Accounts in this organizational directory only*. Redirect URI: leave blank.
3. After creation, copy **Application (client) ID** and **Directory (tenant) ID** — you'll need both.

## 3. Upload the cert to the app registration

1. App registration → **Certificates & secrets** → **Certificates** → **Upload certificate**.
2. Upload `vendor-quote-ingest.cer` from step 1.
3. Confirm the thumbprint matches the one you captured.

## 4. Grant Mail.Read

1. App registration → **API permissions** → **Add a permission** → **Microsoft Graph** → **Application permissions** → check **Mail.Read** → **Add**.
2. Click **Grant admin consent for <tenant>**. Status should turn green.

## 5. (Recommended) Scope to one mailbox

By default, **Mail.Read** is tenant-wide. To scope it to only the procurement
mailbox, use `New-ApplicationAccessPolicy` from Exchange Online PowerShell:

```powershell
Connect-ExchangeOnline -UserPrincipalName admin@yourdomain.com

New-DistributionGroup -Name "VendorQuoteIngestScope" -Members "procurement@yourdomain.com" -Type Security
New-ApplicationAccessPolicy -AppId "<client-id-from-step-2>" `
    -PolicyScopeGroupId "VendorQuoteIngestScope" -AccessRight RestrictAccess `
    -Description "Limit Vendor Quote Ingest app to procurement mailbox only"
```

Verify with: `Test-ApplicationAccessPolicy -Identity "procurement@yourdomain.com" -AppId "<client-id>"` — should return `AccessAllowed`.

## 6. Add the `[graph]` section to `secrets.toml`

Edit `.streamlit/secrets.toml` (or whichever path `LOCAL_SECRETS_PATHS` resolves to first). Add:

```toml
[graph]
tenant_id = "00000000-0000-0000-0000-000000000000"
client_id = "00000000-0000-0000-0000-000000000000"
certificate_path = "C:/Users/alexh/vendor-quote-ingest.pem"
certificate_thumbprint = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
mailbox = "procurement@yourdomain.com"
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -c "from secrets_loader import load_graph_settings; s = load_graph_settings(); print('ok' if s else 'missing keys')"
```
Expected: `ok`.

## 7. First manual run (backfill + verify)

```powershell
.\.venv\Scripts\python.exe vendor_quote_ingest.py --backfill-days 30 --verbose
```

Watch for:
- `seen=N matched=M rows=R` summary at the end.
- `data\vendor_quotes.json` created with at least one item key.
- `data\vendor_quote_cursor.json` written.

If you get `Graph auth failed`, recheck steps 2–4 (thumbprint, consent).

## 8. Register Task Scheduler entry

```powershell
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\alexh\Downloads\mod\.venv\Scripts\python.exe" `
    -Argument "C:\Users\alexh\Downloads\mod\vendor_quote_ingest.py" `
    -WorkingDirectory "C:\Users\alexh\Downloads\mod"

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 15)

$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U

Register-ScheduledTask `
    -TaskName "VendorQuoteIngest" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Description "Pull vendor quotes from Outlook every 15 minutes"
```

Verify with `Get-ScheduledTaskInfo -TaskName "VendorQuoteIngest"`. Tail
`data\vendor_quote_ingest.log` after 20 minutes to confirm runs are happening.

## Certificate rotation (every 2 years)

The cert from step 1 expires 3 years out, but rotate at the 2-year mark to
avoid a surprise outage. To rotate:

1. Repeat step 1 with a new subject (`CN=vendor-quote-ingest-2028`) to generate
   a fresh cert.
2. Upload the new `.cer` via step 3 (Azure now holds two valid certs).
3. Update `certificate_path` and `certificate_thumbprint` in `secrets.toml`.
4. Run `vendor_quote_ingest.py --backfill-days 0 --verbose` to confirm the new
   cert authenticates.
5. In Azure, delete the old cert.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Graph auth failed: AADSTS70011` | Missing admin consent | Re-do step 4. |
| `Graph auth failed: AADSTS700027` | Cert thumbprint mismatch | Verify step 1's thumbprint matches step 3's upload. |
| `Graph auth failed: AADSTS700016` | App registration not yet propagated | Wait 5 min after step 2 and retry. |
| `Lock present at ...` | Prior run crashed mid-flight | Delete `data\vendor_quote_ingest.lock`. |
| Cursor never updates | Filter rejecting everything | Run with `--verbose`; inspect which messages were `seen` vs `matched`. |
| OpenAI extractor returns 0 rows | Prompt missing aliases | Add missing aliases to `data\vendor_quote_aliases.json`. |
| Refresh button shows "Already running" | Scheduled task and manual click overlapped | Wait for cron run to finish (≤2 min) then retry. |
````

- [ ] **Step 2: Commit**

```powershell
git add docs/superpowers/runbooks/vendor-quote-ingest-setup.md
git commit -m @'
docs: add vendor-quote ingest one-time setup runbook

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

## Task 12: End-to-end smoke test

This task verifies the whole pipeline with real Outlook data. It assumes Task 11's
runbook steps 1–7 have been completed manually by the user. The plan executor
should NOT attempt to register the Azure app — that's the user's job.

**Files:** none (verification only)

- [ ] **Step 1: Confirm setup is done**

Ask the user: "Have you completed steps 1–7 of `docs/superpowers/runbooks/vendor-quote-ingest-setup.md`? Type `yes` to continue, or `no` to pause for setup."

If `no`, stop here and direct the user to the runbook.

- [ ] **Step 2: Run a dry-run ingest**

Run: `.\.venv\Scripts\python.exe vendor_quote_ingest.py --backfill-days 30 --dry-run --verbose`

Expected output ends with a line like:
```
ingest complete: seen=<N> matched=<M> rows=<R> low_conf=<L> errors=0
```

Verify: `data\vendor_quotes.json` does NOT exist yet (dry-run), `data\vendor_quote_cursor.json` does NOT exist yet.

- [ ] **Step 3: Run a real ingest, scoped to NPKU32**

Run: `.\.venv\Scripts\python.exe vendor_quote_ingest.py --backfill-days 30 --item NPKU32 --verbose`

Verify:
- `data\vendor_quotes.json` exists.
- `python -c "import json; print(list(json.load(open('data/vendor_quotes.json', encoding='utf-8')).keys()))"` includes `"NPKU32"`.
- `data\vendor_quote_cursor.json` exists and contains a `delta_link` URL.

- [ ] **Step 4: Launch Streamlit and verify the panel renders**

Run: `.\.venv\Scripts\streamlit.exe run app.py`

Navigate to the Market Intelligence section → select an item that contains `NPKU32`. Verify:
- The "Vendor Quotes & Receipts" section renders below the header.
- At least one row appears.
- The `★ Cheapest current quote` callout displays (if any fresh high-confidence quotes exist).

If a low-confidence row appears (e.g., the Nutrien `$595/railcar` case), confirm:
- It shows the `⚠` marker.
- It does NOT appear in the cheapest-current headline.

- [ ] **Step 5: Test the Refresh button**

In the same Streamlit session, click `↻ Refresh` on the NPKU32 panel. Verify:
- A spinner appears.
- The caption underneath updates with the new summary numbers.
- No errors in the Streamlit log.

- [ ] **Step 6: Register the scheduled task**

Follow step 8 of the runbook. Then verify scheduled task is registered:

```powershell
Get-ScheduledTaskInfo -TaskName "VendorQuoteIngest" | Select-Object NextRunTime, LastRunTime, LastTaskResult
```

- [ ] **Step 7: Final commit (only if any fixes were needed)**

If any code changes were made during smoke testing, commit them:

```powershell
git add -A
git commit -m @'
fix: smoke-test adjustments to vendor quote pipeline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

If no changes were needed, skip this step.

---

## Completion criteria

All 12 tasks done means:
- 39 unit tests pass (`pytest tests -q`).
- `vendor_quote_ingest.py` runs without error against the real inbox.
- Market Intelligence item detail shows the Vendor Quotes & Receipts panel for `NPKU32`.
- The Refresh button on the panel triggers an ingest and reflects updated data.
- Task Scheduler entry `VendorQuoteIngest` is registered and showing successful runs in `data\vendor_quote_ingest.log`.
