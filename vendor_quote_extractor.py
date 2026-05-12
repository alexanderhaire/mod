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
        raw = _call_openai_chat(messages, model, api_key)
    except Exception:
        LOGGER.exception("OpenAI call failed for subject=%r sender=%r", email_subject, sender)
        return []
    return parse_extraction_response(raw)
