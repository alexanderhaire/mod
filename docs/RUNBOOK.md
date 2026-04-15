# Ops Dashboard — Runbook

## Start / stop

**Manual (Windows):**
```powershell
cd C:\path\to\mod
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

**As a service (recommended):** install with [NSSM](https://nssm.cc/) pointing at `run.ps1`, or wrap as a Scheduled Task set to run at startup whether or not a user is logged on.

**Stop:** `Ctrl+C` if interactive; `Stop-Service opsdash` if installed via NSSM.

## Configuration

All runtime config lives in `.env` at the repo root (not in git). Template is `.env.example`. Edit `.env`, then restart the service.

Key settings:
- `GP_SQL_AUTH=windows` or `sql` — controls whether to use trusted connection or username/password
- `SHADOW_MODE=true` — POs logged to CSV instead of emailed. Flip to `false` only after live-email sign-off
- `STREAMLIT_SERVER_ADDRESS` — `127.0.0.1` behind a reverse proxy; `0.0.0.0` only if binding directly (not recommended)

## Logs

- Application logs: `logs/opsdash-<timestamp>.log` (one per start)
- Reorder monitor: `logs/reorder_monitor.log`
- Rotate manually or via a scheduled task; nothing auto-rotates today

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Page loads blank / "Connection refused" | Streamlit process not running | Restart service; check latest log |
| SQL errors on every page | GP SQL down, or service account locked | Verify from the host: `sqlcmd -S <server> -E` (Windows auth) or with `-U/-P` |
| AI query page 401 | OpenAI key invalid or rotated | Update `OPENAI_API_KEY` in `.env`, restart |
| Reorder emails not sending | `SHADOW_MODE=true` | Intentional until live-email sign-off |
| Touch screen not responding | Browser kiosk lost focus / screen asleep | Reboot display; verify kiosk shortcut autostarts |
| 500 on one page, others fine | Bad data row in GP (nulls, bad dates) | Check log; most pages have per-row try/except, but new queries may not |

## Health check

From any browser on the internal network:
- `https://opsdash.internal/` → should render the main page within 5s
- Look for a "SQL: connected" indicator in the sidebar (added in Phase 1)

## Rollback

App is stateless against GP; rollback = `git checkout <previous-sha> && restart`. Local JSON state files (`data/*.json`) are non-critical overrides; back them up before a major change but losing them does not break the app.

## Known items to clean up (Phase 1 work)

- Many `analyze_*.py` / `check_*.py` scripts still build ODBC conn strings inline. Migrate to `config.build_odbc_connection_string()` as they're touched.
- `auth.py` uses a file-backed `users.json`. Fine for current scale; evaluate AD integration if the user list grows.
- `.streamlit/secrets.toml` on the dev laptop contains live API keys. Rotate before production go-live: OpenAI, Google Maps, Alpaca, Teams webhook.

## Escalation

1. Check latest log under `logs/`
2. Restart the service
3. If SQL-related, check with IT that the GP SQL instance is up and `svc_opsdash` isn't locked
4. If still broken, roll back to previous git SHA and notify Alex
