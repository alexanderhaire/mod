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
