"""Priority scoring for order queue management.

Ported verbatim from c:/Users/alexh/Production/core/priority.py.
Used by kanban_reorder.py to rank open sales orders by urgency (the
"necessity" step in the necessity -> location -> kanban rule).
"""

from datetime import date


def calculate_priority_score(
    req_ship_date: date | None,
    today: date | None = None,
    total_qty: float = 0.0,
    manual_boost: int = 0,
    proximity_boost: float = 0.0,
) -> float:
    if today is None:
        today = date.today()

    if req_ship_date is None:
        date_points = 50.0
    else:
        days_until_due = (req_ship_date - today).days
        if days_until_due < 0:
            date_points = 100.0 + min(abs(days_until_due) * 10, 100)
        elif days_until_due == 0:
            date_points = 100.0
        elif days_until_due == 1:
            date_points = 80.0
        elif days_until_due <= 3:
            date_points = 60.0
        elif days_until_due <= 7:
            date_points = 40.0
        elif days_until_due <= 14:
            date_points = 20.0
        else:
            date_points = 10.0

    size_points = min(total_qty / 100.0, 10.0)
    boost = max(-50, min(manual_boost, 50))

    return round(date_points + size_points + boost + proximity_boost, 2)


def urgency_label(score: float) -> str:
    if score >= 100:
        return "CRITICAL"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def urgency_color(score: float) -> str:
    if score >= 100:
        return "#ff4444"
    elif score >= 60:
        return "#ff8c00"
    elif score >= 40:
        return "#ffd700"
    else:
        return "#44bb44"
