# Ops/Prioritization Dashboard — IT/CIO Tech Brief

**Owner:** Alex H.  |  **Date:** 2026-04-14  |  **Status:** Request for infrastructure support

## What this is
An internal web dashboard (Python/Streamlit) that reads **read-only** from our Dynamics GP 18.5.1661 SQL database and renders production-floor decisions: kanban reorder queue, perpetual inventory, label/lot tracking, reorder recommendations, and historical inventory snapshots. Target audience is the shop floor; the plan is to mount it on a 65" commercial touch panel as a kiosk replacing the current clipboard system.

Currently it runs on a developer laptop. This brief is the ask to move it to a supported 24/7 host.

## What we need from IT

### 1. Dedicated host
- **Preferred:** small Windows mini-PC or existing Windows VM.
- **Spec target:** 4-core CPU, 16 GB RAM, 256 GB SSD, Windows Server 2019+ or Windows 11 Pro.
- **Why Windows:** dashboard currently uses Windows trusted-connection auth to GP SQL; Windows host removes an AD/SQL-auth change from the critical path. Will also accept Linux if IT prefers — we'd switch to a SQL login (see §2).

### 2. SQL access
- **Read-only service account** on the GP SQL instance (suggested name: `svc_opsdash`).
- Scope: `SELECT` on production-relevant tables only — `IV00101`, `IV00102`, `IV00103`, `IV30300`, `SOP30300`, `SOP30200`, `POP10110`, `POP10100`, `PM00200`, `BM010115`, `MOP1016`, `GL00105`.
- **Preferred:** point at a reporting replica if one exists, to avoid load on the live OLTP DB. If not, we'll profile query cost and add caching.
- Connection: ODBC Driver 18 for SQL Server, Encrypt=yes, TrustServerCertificate=yes.

### 3. Network
- Host needs reachability to the GP SQL instance on TCP 1433 (or configured port).
- Host needs outbound HTTPS (443) to `api.openai.com` for the natural-language query page. This is an explicit product decision; happy to allowlist only that FQDN.
- Internal DNS name (suggested `opsdash.internal` or similar) pointing at the host.
- Internal-only binding. No public internet exposure.

### 4. TLS + reverse proxy
- Internal CA cert for the DNS name (wildcard is fine if one exists).
- Reverse proxy (IIS or nginx) in front of Streamlit on port 443 → 8501 loopback.

### 5. Secrets management
- Secrets (SQL creds for the service account, OpenAI API key, Google Maps key, Teams webhook) will be stored in a `.env` file on the host, not in source control. `.env` is gitignored.
- **Preferred:** Windows Credential Manager or an enterprise vault if available — we can adapt the loader in `config.py`.
- **Note for security review:** the current `.streamlit/secrets.toml` on the dev laptop contains live API keys that have existed in plaintext on that machine. Those keys should be **rotated** before the production host goes live. List of keys to rotate: OpenAI, Google Maps, Alpaca (paper), Teams webhook URL.

### 6. Ownership + handover
- Alex maintains the app code.
- IT owns: OS patching, backups of the host, cert rotation, DNS, service restart on failure.
- Runbook (`docs/RUNBOOK.md` in the repo) documents start/stop, log locations, common failure modes, and rollback.

## What we are NOT asking for
- No new database — reads existing GP SQL only, read-only.
- No writes back to GP. (PO emails, when re-enabled, go through a separate Teams webhook; currently gated by `SHADOW_MODE=true`.)
- No AD integration in v1 — shop-floor view is read-only kiosk, no login. Admin URL has file-backed auth.
- No internet exposure.

## Phased plan (2–3 weeks total)
1. **Phase 0 (this week, code-side, no IT dependency):** secrets scaffolding (`.env.example`, `config.py`), startup script (`run.ps1`), runbook. **Done.**
2. **Phase 1 (IT dependencies):** host provisioned, service account, DNS, cert, reverse proxy, firewall rules. Smoke test.
3. **Phase 2:** 65" commercial touch panel mounted, kiosk mode configured, touch calibrated.
4. **Phase 3:** handover; monitoring/alerting added.

## Contacts
- App owner: Alex H.
- Business sponsors: Ben, Mark

## Appendix — resource footprint
- Baseline RAM: ~500 MB
- Peak RAM: ~1.5 GB during ML retraining
- Disk: 3–5 GB (app + venv + logs)
- Network: continuous SQL polling (~1 query/min average), outbound OpenAI calls only when users use the AI query page
