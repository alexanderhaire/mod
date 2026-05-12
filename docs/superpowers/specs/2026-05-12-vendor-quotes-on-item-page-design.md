# Vendor Quotes on Item Page — Design Spec

**Date:** 2026-05-12
**Author:** awhaire@gmail.com (via Claude)
**Status:** Draft for review

---

## 1. Problem

When the user is quoting a customer (or sourcing) on a raw material like U-32, deciding who the cheapest current vendor is requires sifting through email threads — supplier quotes, POs, and shipping-mode caveats are scattered across Outlook with no single view. The ERP (`POP30300`) holds *confirmed receipts* (what was actually paid), but not *quotes* (what suppliers are currently offering). Microsoft 365 Copilot is good at extracting this from email when asked, but its output lives in chat windows and disappears.

The user wants a per-item view that answers, at quote-time:

> **"Who's cheapest right now, and what did each vendor actually charge me last time?"**

## 2. Goal

Add a **Vendor Quotes & Receipts** panel to the Market Intelligence item detail page that:

- Shows each vendor's most recent quote (from email) next to its most recent confirmed receipt (from `POP30300`).
- Surfaces the cheapest current quote in a single headline.
- Ingests quote data automatically from Outlook via Microsoft Graph + OpenAI extraction, with no manual paste step required.

## 3. Non-Goals

- **Replacing Copilot's email Q&A.** Copilot remains useful for ad-hoc inbox questions; this feature does not try to be a chat interface.
- **Handling PDF attachments.** Many POs arrive as PDFs. v1 reads email *bodies* only. PDF extraction is a follow-on.
- **Sending quotes back to vendors.** Read-only ingest; no outbound email.
- **Sales-side quoting workflow** (`item_sales_history.py`). That page stays as-is.

## 4. Architecture

```
   Outlook inbox (M365)
          │
          │  Graph API delta query
          │  scope: Mail.Read (application permission)
          │  auth:  certificate-based, no client secret
          ▼
   vendor_quote_ingest.py
     1. Pull messages changed since last cursor
     2. Filter: sender ∈ vendor_domains.json
        OR subject ~ /PO 431-\d+|U-?32|UAN|urea|phos|potash/i
     3. For each survivor → OpenAI structured-output extract
        (returns 0..N rows, each with confidence flag)
     4. Normalize price to $/ton
     5. Append rows to vendor_quotes.json
     6. Persist new delta cursor
          │
          ▼
   data/vendor_quotes.json         (append-only, keyed by ITEMNMBR)
   data/vendor_domains.json        (seeded supplier registry; user-editable)
   data/vendor_quote_cursor.json   (Graph delta state)
          │
          ▼  read by Streamlit (no auth path; file I/O only)
   market_insights.py → Vendor Quotes & Receipts panel
   + [↻ Refresh] button on item page subprocesses the ingest script
   on demand

   Schedule: Windows Task Scheduler runs vendor_quote_ingest.py every 15 min
```

## 5. Components

### 5.1 `vendor_quote_ingest.py` (new)

A standalone Python script that runs unattended.

**Responsibilities:**
1. Authenticate to Microsoft Graph using app permission + certificate (via `msal`).
2. Issue a delta query against `/users/{mailbox}/messages` from the cached cursor.
3. For each new/changed message, apply the inclusion filter (sender domain OR subject pattern).
4. For each surviving message, call OpenAI with a structured-output schema to extract zero or more quote rows.
5. Normalize each row's price to `$/ton` based on its unit (`lb`, `ton`, `railcar`, `gallon`).
6. Append to `data/vendor_quotes.json` keyed by `ITEMNMBR`.
7. Persist the new delta token.
8. Log a summary (emails seen, emails matched, rows extracted, rows flagged) to `data/vendor_quote_ingest.log`.

**Exit codes:** `0` on success, non-zero on hard failures (auth, network, malformed config), so Task Scheduler can alert.

**Reentrancy:** safe to run on-demand and on schedule simultaneously — file writes are guarded by a lock file at `data/vendor_quote_ingest.lock`. A second instance exits cleanly with "already running" on the log.

### 5.2 OpenAI extractor

- Reuses the existing `openai_clients.py` wrapper.
- Uses **JSON-mode / structured outputs** so the response conforms to a fixed schema; no free-text parsing.
- The schema and the prompt template live in `vendor_quote_extractor.py` and are version-controlled.
- The prompt instructs the model to:
  - Map item descriptions (e.g., "U-32", "32% UAN", "urea ammonium nitrate solution 32") to the ERP `ITEMNMBR` using a small alias table loaded from `data/vendor_quote_aliases.json`.
  - Return `null` rather than guess when a field is unclear.
  - Flag `confidence: "low"` when the unit is ambiguous (e.g., "$595/railcar"), the date is missing, or the vendor isn't resolvable.

### 5.3 `data/vendor_quotes.json` (new)

Append-only JSON object keyed by `ITEMNMBR`. Each value is a list of quote rows:

```json
{
  "NPKU32": [
    {
      "vendor": "HELM",
      "quote_date": "2026-04-03",
      "price_per_ton": 475.00,
      "raw_price": "$475/ton",
      "mode": "rail_delivered",
      "po_number": "431-8237",
      "notes": "early April delivery, freight included",
      "source_message_id": "<Graph message id>",
      "source_subject": "RE: PO 431-8237",
      "source_excerpt": "<verbatim sentence(s) the model extracted from>",
      "confidence": "high",
      "warnings": [],
      "ingested_at": "2026-05-12T10:30:00Z"
    },
    {
      "vendor": "Nutrien",
      "quote_date": "2026-04-22",
      "price_per_ton": 595.00,
      "raw_price": "$595/railcar",
      "mode": "rail_delivered",
      "po_number": "431-8330",
      "notes": "Copilot's summary said 'per railcar' — almost certainly per ton; awaiting user confirmation",
      "source_message_id": "<Graph message id>",
      "source_subject": "Re: [EXT] PO 431-8330",
      "source_excerpt": "...",
      "confidence": "low",
      "warnings": ["unit_ambiguous"],
      "ingested_at": "..."
    }
  ]
}
```

**Why append-only:** every paste/ingest yields new rows even if the vendor is the same. The display layer reduces to "latest per (vendor, mode)" but the underlying history lets the user see drift over time (e.g., Trademark Dec → HELM Apr → Nutrien Apr).

**Why `source_excerpt` is kept:** if the extractor mis-parses, the verbatim source lets the user verify without going back to Outlook.

**Why `price_per_ton` is normalized at ingest, not at render:** display logic stays simple; one column to sort. Raw value is kept for audit.

### 5.4 `data/vendor_domains.json` (new)

```json
{
  "vendors": [
    {"name": "HELM",     "domains": ["helm.com", "helmagro.com"]},
    {"name": "Nutrien",  "domains": ["nutrien.com"]},
    {"name": "Trademark Nitrogen", "domains": ["trademarknitrogen.com"]}
  ]
}
```

Seeded with current known vendors. Editable from a small admin panel (not in v1 — direct file edit is fine to start).

### 5.5 `data/vendor_quote_cursor.json` (new)

Stores the Graph `@odata.deltaLink` so each run picks up where the prior run left off. Resets to a 30-day backfill if missing or stale.

### 5.6 Market Intelligence panel — display

A new section on the item detail view in `market_insights.py`, rendered between the existing item summary and the existing confirmed-price chart. The chart is preserved unchanged.

Layout:

```
═══════════════════════════════════════════════════════════════════
 NPKU32 — U-32 (UAN 32%)                              [↻ Refresh]
═══════════════════════════════════════════════════════════════════

 ★ Cheapest current quote (last 60d):
   HELM — $475/ton  rail delivered  (Apr 3, PO 431-8237)

 VENDOR QUOTES & RECEIPTS
 ┌──────────┬────────┬──────────────────────┬──────────────────────┬─────┐
 │ Vendor   │ Mode   │ Last Quote           │ Last Receipt         │  Δ  │
 ├──────────┼────────┼──────────────────────┼──────────────────────┼─────┤
 │ HELM     │ Rail   │ $475/t  Apr 3  #8237 │ $475/t  Apr 8  #8237 │ $0  │
 │ Nutrien  │ Rail   │ $595/t  Apr 22 #8330 │ $480/t  Feb 10 #8102 │+$115│
 │ Trade.   │ Pickup │ —                    │ $411/t  Dec 15 #8066 │  —  │
 └──────────┴────────┴──────────────────────┴──────────────────────┴─────┘
   ● Fresh   ○ Stale (>60d)   ⚠ Low-confidence   = Quote fulfilled by paired PO

 [existing confirmed-price chart continues below, unchanged]
```

**Rendering rules:**
- One row per `(vendor, mode)` showing the **latest quote** and the **latest receipt** for that combination — independently retrieved, not strictly PO-paired. This answers the user's actual question ("are they getting more expensive?") rather than the narrower "did they honor this specific PO."
- **Cheapest-current headline** considers only quotes with `quote_date` ≤ 60 days old AND `confidence != "low"`. If nothing qualifies, the headline says "No fresh confirmed quote — last extracted X days ago."
- **PO# match indicator.** When the quote's `po_number` equals the receipt's PO#, a `=` icon appears next to the Δ, signalling "this quote was fulfilled by this receipt." Otherwise the Δ stands alone as a movement signal.
- **Δ = quote_price_per_ton − latest_receipt_price_per_ton.**
  - `Δ > 0` → vendor is currently quoting **higher** than their last sale to us (price moving against us).
  - `Δ < 0` → vendor is currently quoting **lower** than their last sale (price moving in our favor).
  - `Δ = 0` → flat.
  - Shown as `—` if either side is missing.
- **Receipt lookup** reuses `fetch_product_price_history(item_number)` from `market_insights.py`; no new SQL. Mode and PO# are read from the same POP30300 rows when available.
- **Stale rows** stay visible, dimmed, with a `○` marker. Lets the user see "Trademark hasn't quoted since December."
- **Low-confidence rows** highlighted with `⚠` and a tooltip showing the `notes` and `source_excerpt`. Clicking the row opens an inline edit form so the user can confirm or correct the value.
- **[↻ Refresh]** button subprocesses `python vendor_quote_ingest.py --item NPKU32` (item-scoped to keep latency low), then `st.rerun()`s. Disabled while running; shows last-ingest timestamp.

**Scope:** the panel appears on any **purchased item** — defined as any `ITEMNMBR` with at least one row in `POP30300` (i.e., the item has been received at least once). Resale finished goods and never-purchased items skip the panel entirely. This mirrors commit `b144677`'s broadening of vendor info beyond raw materials.

## 6. Decisions and rationale

| # | Decision | Rationale |
|---|---|---|
| D1 | App permission + certificate auth | Unattended job needs to run without a user session. Certificate beats client-secret because no annual rotation pain. User is tenant admin → consent is one click. |
| D2 | Filter = sender domain **OR** subject pattern | Subject-only would miss vendor follow-ups on neutral subjects. Sender-only would miss POs forwarded internally. Either-match keeps recall high without bringing in obvious noise. |
| D3 | Delta queries, not webhook subscriptions | Webhooks need a public HTTPS endpoint; Streamlit isn't publicly exposed. 15-min polling is well within procurement-decision freshness needs. |
| D4 | Low-confidence ⚠ flag, not auto-reject | Drops would lose information; auto-accept would let the `$595/railcar` ambiguity poison "cheapest current." Flag + manual confirm threads the needle. |
| D5 | Append-only JSON, not SQL | Volume is small (~50 rows/month). Matches existing JSON-state patterns (`label_onhand_counts.json`, `raw_material_state.json`). SQL adds a migration step and a connector dependency without paying for itself yet. |
| D6 | Normalize `$/ton` at ingest | Display logic stays trivial; one comparable column. Raw value is kept in `raw_price` for audit. |
| D7 | Receipt data stays in SQL Server | Receipts already live in `POP30300`. We don't duplicate; we join in memory at render. |
| D8 | One row per `(vendor, mode)` in display | HELM quoting rail and pickup are different decisions with different prices. Collapsing would hide the cheaper option. |
| D9 | Item-scoped on-demand refresh | Refreshing all items would mean a full delta pull; the per-item refresh in the UI passes an `--item` filter so the script can early-exit on non-matches. |

## 7. Data model — item lookup notes

Copilot's natural-language references ("U-32", "32% UAN", "UAN-32") need to resolve to the ERP `ITEMNMBR` (`NPKU32`) for the row to land on the right page. v1 ships with a small alias table at `data/vendor_quote_aliases.json`:

```json
{
  "NPKU32": ["U-32", "U32", "UAN 32", "UAN-32", "urea ammonium nitrate 32", "32% UAN"],
  "NPKUREA": ["urea 46", "urea prill", "granular urea"],
  ...
}
```

Seeded for the known raw materials; the extractor receives this table in its prompt context. Quotes that can't be resolved to a known item get logged with `ITEMNMBR = "_unresolved"` and shown on an admin "Unresolved Quotes" page (out of v1 scope, but the storage shape supports it).

## 8. Failure modes and handling

| Failure | Detection | Behavior |
|---|---|---|
| Graph auth expires (cert rotation) | `msal` raises | Script exits non-zero; Task Scheduler surfaces via Windows event log. |
| OpenAI rate-limit | HTTP 429 | Exponential backoff inside the script; max 3 retries per message. |
| Malformed extractor JSON | Schema validation fails | Row skipped, message marked with `_extract_error` in cursor state, logged. Does not block other rows. |
| Vendor not in `vendor_domains.json` but subject matched | Normal path | Row saved with `vendor = <model's best guess>` and `warnings = ["unknown_vendor"]`. User confirms or corrects. |
| Item not resolvable | Alias miss | Saved under `_unresolved`. Doesn't pollute item pages. |
| Concurrent ingest runs (cron + manual refresh) | Lock file present | Second instance exits with "already running"; manual refresh button shows the in-flight indicator instead. |

## 9. Security

- **Certificate** is stored in Windows Certificate Store (current user); the script references it by thumbprint. Not in the repo.
- **Tenant ID, client ID, certificate thumbprint, target mailbox UPN** stored in environment variables, loaded by `secrets_loader.py` alongside existing SQL connection settings.
- **Mail.Read application permission** is tenant-wide by default. Recommended hardening: scope via `New-ApplicationAccessPolicy` to only the procurement mailbox. Documented in the implementation plan, not enforced in code.
- `data/vendor_quotes.json` may contain commercial pricing; it's already in the same `data/` directory as other business state and inherits the same access control as the rest of the repo.

## 10. Effort estimate

| Task | Effort |
|---|---|
| Azure AD app registration + cert provisioning + Mail.Read consent | 30 min |
| `vendor_quote_ingest.py` (Graph delta query, msal auth, filter, write loop) | 3–4 h |
| `vendor_quote_extractor.py` (OpenAI structured-output schema + prompt + alias map) | 1–2 h |
| Seed `vendor_domains.json` and `vendor_quote_aliases.json` | 30 min |
| Market Intelligence panel rendering + receipt join | 2–3 h |
| Refresh button wiring (`subprocess` + lock file + status indicator) | 1 h |
| Task Scheduler entry | 15 min |
| Manual smoke test with real inbox + drift fixes | 2 h |
| **Total** | **~10–12 hours focused work (≈ 1.5 days)** |

## 11. Out of scope (deferred)

- PDF attachment extraction.
- Auto-suggesting vendor for unresolved quotes via fuzzy match.
- Slack/Teams notification on quote drift (vendor walks price >5% week-over-week).
- Bulk historical backfill UI ("scan last 12 months").
- Multi-mailbox ingest (e.g., admin@ and purchasing@ both monitored).
- Strict PO-matched honesty view ("did this vendor honor quote #X with PO #X"). v1 surfaces the PO match as an indicator only; a dedicated honesty report is a follow-on.

## 12. Open questions

None at the time of writing. Decisions D1–D9 are locked. If new questions surface during planning, they'll be tracked on the implementation plan rather than this design doc.
