"""
Calendar Utilities

Centralized calendar and month-related utilities. This module consolidates
all calendar logic that was previously duplicated across app.py, handlers.py,
parsing_utils.py, and ui_utils.py.

Usage:
    from calendar_utils import (
        MONTH_ORDER,
        MONTH_LOOKUP,
        month_number,
        format_month_label,
        format_month_end_date,
        extract_months_from_prompt,
        extract_year_from_prompt,
        extract_month_year_pairs,
        get_prior_workday,
        get_week_start,
    )
"""

import calendar
import datetime
import re
from typing import Any

# Ordered list of full month names (January through December)
MONTH_ORDER = [calendar.month_name[i] for i in range(1, 13)]

# Comprehensive lookup mapping month names/abbreviations/typos to 1-12
# Includes common misspellings like "febuary"
MONTH_LOOKUP: dict[str, int] = {
    # Full names
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    # Standard abbreviations
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    # Common misspellings
    "febuary": 2,
}


def month_number(val: Any) -> int | None:
    """
    Parse a month value from number/name/abbreviation strings into 1-12.
    
    Accepts:
        - int: 1-12
        - float: 1.0-12.0 (will be converted to int)
        - str: "January", "jan", "1", "01", etc.
        
    Returns:
        int 1-12 if valid, None otherwise.
    """
    if val is None:
        return None
    
    # Handle numeric types
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        try:
            import pandas as pd
            if pd.isna(val):
                return None
        except ImportError:
            pass
        num = int(val)
        return num if 1 <= num <= 12 else None
    
    # Handle string types
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return None
        # Check if it's a numeric string
        if cleaned.isdigit():
            num = int(cleaned)
            return num if 1 <= num <= 12 else None
        # Check the lookup table
        return MONTH_LOOKUP.get(cleaned.lower())
    
    return None


def format_month_label(val: Any) -> str:
    """
    Return a human-friendly month label from various input formats.
    
    Examples:
        format_month_label(1) -> "January"
        format_month_label("jan") -> "January"
        format_month_label("February") -> "February"
        
    Falls back to str(val) if the value cannot be parsed as a month.
    """
    num = month_number(val)
    if num is not None:
        return calendar.month_name[num]
    return str(val)


def format_month_end_date(year: int, month: int) -> str:
    """
    Get the last day of a month as a date string (YYYY-MM-DD format).
    
    Example:
        format_month_end_date(2024, 2) -> "2024-02-29"
        format_month_end_date(2023, 2) -> "2023-02-28"
    """
    last_day = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-{last_day}"


def extract_months_from_prompt(prompt: str) -> list[int]:
    """
    Extract month numbers from a prompt in the order they appear.
    
    Handles full month names and common abbreviations/misspellings.
    
    Example:
        extract_months_from_prompt("Sales for January and March") -> [1, 3]
    """
    pattern = r"\b(" + "|".join(MONTH_LOOKUP.keys()) + r")\b"
    seen: list[int] = []
    for match in re.finditer(pattern, prompt.lower()):
        month_num = MONTH_LOOKUP.get(match.group(1))
        if month_num and month_num not in seen:
            seen.append(month_num)
    return seen


def extract_year_from_prompt(prompt: str) -> int | None:
    """
    Find a four-digit year (1900-2099) in the prompt.
    
    Example:
        extract_year_from_prompt("sales in 2024") -> 2024
        extract_year_from_prompt("no year here") -> None
    """
    match = re.search(r"\b(19|20)\d{2}\b", prompt)
    return int(match.group(0)) if match else None


def extract_month_year_pairs(
    prompt: str, 
    today: datetime.date
) -> list[tuple[int, int]]:
    """
    Extract (year, month) pairs from a prompt, handling year propagation
    across month sequences like "December 2023 to April 2024".
    
    Returns pairs in prompt order with duplicates removed.
    
    Example:
        prompt = "December 2023, January, February 2024"
        # Returns: [(2023, 12), (2024, 1), (2024, 2)]
    """
    lower = prompt.lower()
    pattern = r"(?P<month>" + "|".join(MONTH_LOOKUP.keys()) + r")\s*(?P<year>(?:19|20)\d{2})?"
    
    month_matches: list[dict] = []
    used_spans: list[tuple[int, int]] = []
    
    for m in re.finditer(pattern, lower):
        month_num = MONTH_LOOKUP.get(m.group("month"))
        if not month_num:
            continue
        year_val = int(m.group("year")) if m.group("year") else None
        month_matches.append({
            "month": month_num, 
            "year": year_val, 
            "pos": m.start(), 
            "end": m.end()
        })
        if year_val is not None:
            used_spans.append((m.start("year"), m.end("year")))
    
    if not month_matches:
        return []
    
    # Find standalone year tokens not already captured with a month
    year_tokens: list[tuple[int, int]] = []
    for y in re.finditer(r"(19|20)\d{2}", lower):
        span = (y.start(), y.end())
        # Skip if this year is already part of a month-year pair
        if any(not (span[1] <= us[0] or span[0] >= us[1]) for us in used_spans):
            continue
        year_tokens.append((int(y.group()), y.start()))
    year_tokens.sort(key=lambda t: t[1])
    
    # Build pairs with year propagation logic
    pairs: list[tuple[int, int]] = []
    current_year: int | None = None
    
    for match in month_matches:
        if match["year"]:
            current_year = match["year"]
            pairs.append((current_year, match["month"]))
            continue
        
        # If no year context yet, try to find one from standalone year tokens
        if current_year is None and year_tokens:
            # Pick the nearest future year token if available; otherwise the last year token
            future_year = next((yt[0] for yt in year_tokens if yt[1] >= match["pos"]), None)
            current_year = future_year if future_year else year_tokens[-1][0]
        
        if current_year is None:
            current_year = today.year
        
        pairs.append((current_year, match["month"]))
    
    # Remove duplicates while preserving order
    deduped: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for pair in pairs:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduped.append(pair)
    
    return deduped


def get_month_date_range(year: int, month: int) -> tuple[str, str]:
    """
    Get the start and end dates for a month as ISO date strings.
    
    Example:
        get_month_date_range(2024, 2) -> ("2024-02-01", "2024-02-29")
    """
    start_date = f"{year}-{month:02d}-01"
    end_date = format_month_end_date(year, month)
    return start_date, end_date


def get_prior_workday(date_obj: datetime.date) -> datetime.date:
    """
    Return the previous workday (Friday if Sat/Sun).
    Useful for getting the last valid trading or business day.
    """
    weekday = date_obj.weekday()
    if weekday == 5: # Saturday
        return date_obj - datetime.timedelta(days=1)
    elif weekday == 6: # Sunday
        return date_obj - datetime.timedelta(days=2)
    return date_obj


def get_week_start(date_obj: datetime.date) -> datetime.date:
    """Return the Monday of the current week."""
    return date_obj - datetime.timedelta(days=date_obj.weekday())
