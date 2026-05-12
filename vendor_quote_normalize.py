"""Convert vendor-quoted prices to a canonical $/ton.

Why this exists: emails quote in $/lb, $/ton, $/railcar, $/gallon. Comparing
vendors requires one column. Anything we can't safely convert (e.g., "per
railcar" without knowing tons/car) stays as the raw value with a low-confidence
flag so the UI can prompt the user before treating it as comparable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

LB_PER_TON = 2000.0

_TON_ALIASES = {"ton", "tons", "t", "short_ton", "shortton"}
_LB_ALIASES = {"lb", "lbs", "pound", "pounds"}
_GALLON_ALIASES = {"gallon", "gal", "gallons"}
_RAILCAR_ALIASES = {"railcar", "rail_car", "rc", "car"}


@dataclass
class NormalizedPrice:
    price_per_ton: float | None
    confidence: Literal["high", "low"]
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

    Non-finite prices (NaN, ±inf — usually a sign the extractor returned
    garbage) are downgraded to low confidence with ``price_not_finite`` and
    ``price_per_ton=None`` so they can't poison the cheapest-current calc.
    Negative prices ARE passed through — they're legitimate for credits.
    """
    if not math.isfinite(price):
        return NormalizedPrice(
            price_per_ton=None,
            confidence="low",
            warnings=["price_not_finite"],
        )
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
