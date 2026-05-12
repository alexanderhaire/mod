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
