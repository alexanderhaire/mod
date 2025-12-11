import calendar
import datetime
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation


def normalize_item_for_bom(item: str) -> str:
    """Map finished goods to their BOM parent by converting trailing digits to '00'."""
    if not isinstance(item, str):
        return item
    trimmed = item.strip()
    match = re.match(r"^(.*?)(\d+)$", trimmed)
    if not match:
        return trimmed
    prefix, _ = match.groups()
    return prefix + "00"


def decimal_or_zero(val) -> Decimal:
    """Convert a value to Decimal safely, defaulting to 0 on any parsing issue."""
    try:
        return Decimal(str(val))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def parse_month_year_from_prompt(
    prompt: str,
    today: datetime.date,
    preference: str = "past",
    prefer_same_year: bool = False,
) -> tuple[int | None, int | None]:
    """
    Extract month/year with contextual cues like 'last', 'next', 'this'.
    - preference: when no cues exist, 'past' biases toward past months, 'future' toward upcoming.
    - prefer_same_year: when true, default to the current calendar year unless explicit cues override.
    """
    prompt_lower = prompt.lower()
    
    # robust month map including common typos
    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2, "febuary": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    month = None
    # Check for full names first to avoid partial matches on abbr
    for name, idx in month_map.items():
        # Use simple word boundary check if needed, but 'in' is often enough for simple tokens
        if name in prompt_lower:
            month = idx
            break
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", prompt)
    year = int(year_match.group(1)) if year_match else None
    if month is None:
        return None, None

    last_tokens = ("last", "previous", "prior")
    next_tokens = ("next", "upcoming", "future")
    this_tokens = ("this", "current")
    has_last = any(tok in prompt_lower for tok in last_tokens)
    has_next = any(tok in prompt_lower for tok in next_tokens)
    has_this = any(tok in prompt_lower for tok in this_tokens)

    if year is not None:
        return month, year

    if prefer_same_year:
        if has_last:
            year = today.year - 1
        elif has_next:
            year = today.year + 1
        elif has_this:
            year = today.year
        else:
            year = today.year
        return month, year

    if has_last:
        year = today.year if today.month > month else today.year - 1
    elif has_next:
        year = today.year if today.month < month else today.year + 1
    elif has_this:
        year = today.year
    else:
        if preference == "future":
            year = today.year if today.month < month or today.month == month else today.year + 1
        else:
            year = today.year if today.month >= month else today.year - 1
    return month, year


def parse_percent_increase(prompt: str) -> float | None:
    """Extract a percentage as a decimal (e.g., 0.08 for 8%)."""
    prompt_lower = prompt.lower()
    patterns = [
        r"increase\s+in\s+.*\s+by\s+(\d+(?:\.\d+)?)\s*%",
        r"increase\s+by\s+(\d+(?:\.\d+)?)\s*%",
        r"rise\s+by\s+(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s*(?:increase|growth)",
        r"up\s+(\d+(?:\.\d+)?)\s*%",
    ]
    for pat in patterns:
        match = re.search(pat, prompt_lower)
        if match:
            try:
                return float(match.group(1)) / 100.0
            except ValueError:
                continue
    return None


def extract_lot_from_prompt(prompt: str) -> str | None:
    """Extract a lot number from the prompt, assuming it's a standalone alphanumeric token."""
    if not isinstance(prompt, str):
        return None
    # Simplistic lot number extraction: find a token that looks like a lot number.
    # This could be improved with more robust pattern matching.
    tokens = re.split(r'\s+|[.,;?]', prompt)
    for token in tokens:
        # Assuming lot numbers are primarily alphanumeric and of a certain length.
        # This is a heuristic and might need adjustment based on common lot number formats.
        if len(token) > 4 and re.match(r'^[A-Z0-9-]+$', token.upper()):
            return token.upper()
    return None


def extract_item_from_prompt(prompt: str) -> str | None:
    """Find a likely item number token (alphanumeric, >=5 chars). Prefer tokens containing digits; otherwise avoid obvious English words."""
    if not isinstance(prompt, str):
        return None
    matches = re.findall(r"\b[A-Z0-9]{5,}\b", prompt.upper())
    if not matches:
        return None

    # Prefer tokens that include digits (most item numbers) to avoid words like EXPECT/SALES.
    for token in matches:
        if any(ch.isdigit() for ch in token):
            return token

    english_words = {
        "ABOUT", "AFTER", "AGAIN", "ALWAYS", "AUGUST", "BEFORE", "BOOST", "BUY",
        "CHANGE", "CURRENT", "DECEMBER", "DEMAND", "EXPECT", "FUTURE", "GROWTH",
        "HISTORY", "INCREASE", "ITEM", "ITEMS", "JANUARY", "JULY", "JUNE", "LAST",
        "MONTH", "NEED", "NEXT", "NOVEMBER", "OCTOBER", "ORDER", "PLANNING",
        "PLAN", "QUESTIONS", "REPORT", "SALES", "SHIPMENTS", "SHOULD", "SHOW",
        "SINCE", "THIS", "TODAY", "USAGE", "WHAT", "WHERE", "WHY", "WILL", "YEAR",
    }

    for token in matches:
        if token not in english_words:
            return token
    return None
