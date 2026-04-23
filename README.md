# Chemical Dynamics — Purchasing & Operations Platform

Streamlit-based operations dashboard for Chemical Dynamics. Pulls live data from
Dynamics GP (SQL Server) and surfaces purchasing, inventory, freight, and
margin analytics in one place for the purchasing and sales teams.

## What's inside

The main entry point is [app.py](app.py), which serves a natural-language
SQL chat interface over the GP database plus a collection of focused pages
in [pages/](pages/):

- **Purchasing** ([pages/purchasing.py](pages/purchasing.py)) — reorder
  recommendations, vendor coverage, and PO workflow.
- **Kanban Reorder** ([pages/kanban_reorder.py](pages/kanban_reorder.py)) —
  shop-floor kanban card view with reorder-point alerts.
- **Reorder Recommendations**
  ([pages/reorder_recommendations.py](pages/reorder_recommendations.py)) —
  data-driven buy suggestions with usage history.
- **Perpetual Inventory**
  ([pages/perpetual_inventory.py](pages/perpetual_inventory.py)) — live
  on-hand vs. allocated vs. available by site.
- **Inventory Snapshot**
  ([pages/inventory_snapshot.py](pages/inventory_snapshot.py)) — point-in-time
  inventory position for reporting.
- **True Margin** ([pages/true_margin.py](pages/true_margin.py)) — finished-
  good margin over time, net of raw-material cost drift.
- **Customer Sales History**
  ([pages/customer_sales_history.py](pages/customer_sales_history.py)) —
  customer-level sales trend and concentration view.
- **Nitrogen Sales** ([pages/nitrogen_sales.py](pages/nitrogen_sales.py)) —
  segment-specific sales analytics for nitrogen products.
- **Southwest Report**
  ([pages/southwest_report.py](pages/southwest_report.py)) — regional sales
  summary for the southwest territory.
- **Freight Analytics Map**
  ([pages/freight_analytics_map.py](pages/freight_analytics_map.py)) and
  **Delivery Cost per Mile**
  ([pages/delivery_cost_per_mile.py](pages/delivery_cost_per_mile.py)) —
  lane-level freight spend and $/mi visualization.
- **Label Tracking** ([pages/label_tracking.py](pages/label_tracking.py)) —
  finished-good label audit.
- **ML Buy Dashboard**
  ([pages/ml_buy_dashboard.py](pages/ml_buy_dashboard.py)) and
  **Brain Center** ([pages/brain_center.py](pages/brain_center.py)) —
  ML-assisted procurement and learning system views.

### REC prefix convention

Item codes prefixed with `REC-` (e.g. `REC-ABC123`) are receiving codes for
the same underlying material as the non-prefixed item. Treat them as the
same SKU when aggregating usage, cost, or on-hand.

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

Copy [.env.example](.env.example) to `.env` and fill in real values:

- **GP SQL Server** — `GP_SQL_SERVER`, `GP_SQL_DATABASE`, auth mode.
  Windows trusted connection is the default.
- **OpenAI** — `OPENAI_API_KEY` for the natural-language SQL chat.
- **Google Maps** — `GOOGLE_MAPS_API_KEY` for the freight map.
- **Alpaca** — paper-trading keys (optional; only used by the research/
  strategy pages, not shop-floor workflows).
- **Teams webhook** — `TEAMS_WEBHOOK_URL` for reorder-point alerts. See
  [TEAMS_SETUP.md](TEAMS_SETUP.md) for webhook setup steps.

`SHADOW_MODE=true` is the safe default — POs are logged to CSV instead of
emailed. Flip to `false` only when you're ready to send live emails.

### 3. Run

```bash
streamlit run app.py
```

By default binds to `127.0.0.1:8501`. Set `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
to expose behind a reverse proxy.

## Data sources

- **Dynamics GP (CDI database)** — authoritative source for items, vendors,
  POs, sales history, on-hand, and transactions.
- **CSV caches in [data/](data/)** — user profiles, events, and learning-
  system state for the ML features.
- **Alpaca** — market data for the research pages only.

## Repository layout

- [app.py](app.py) — Streamlit entry point and chat interface.
- [pages/](pages/) — individual Streamlit pages (see list above).
- `inventory_queries.py`, `market_insights.py`, `ml_engine.py` — shared
  query and analytics modules.
- `auth.py`, `vendor_portal.py`, `broker_portal.py` — authentication and
  external-facing portals.
- [data/](data/) — CSV/JSON caches for the learning system.
- [.streamlit/](.streamlit/) — Streamlit config and secrets.

## Notes

- Keep `.env` and `.streamlit/secrets.toml` out of version control — both
  are gitignored.
- The Teams webhook URL is sensitive; anyone with it can post to the
  channel.
