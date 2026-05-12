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


def test_gallon_zero_weight_is_low_confidence():
    result = normalize_to_per_ton(price=4.20, unit="gallon", weight_per_gallon=0.0)
    assert result.price_per_ton is None
    assert result.confidence == "low"
    assert "unit_requires_weight_per_gallon" in result.warnings


def test_nan_price_downgraded_to_low_confidence():
    import math
    result = normalize_to_per_ton(price=math.nan, unit="ton")
    assert result.price_per_ton is None
    assert result.confidence == "low"
    assert "price_not_finite" in result.warnings


def test_negative_price_passes_through():
    # Negatives can be legitimate credits/adjustments.
    result = normalize_to_per_ton(price=-50.0, unit="ton")
    assert result.price_per_ton == -50.0
    assert result.confidence == "high"
