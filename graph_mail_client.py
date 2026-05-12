"""Thin wrapper around Microsoft Graph: app-only auth + delta-query mail fetch.

This module assumes the secrets section ``[graph]`` is populated (see
``secrets_loader.load_graph_settings``). Use a real Outlook account; there is no
mocking layer here on purpose — failures should surface immediately during
manual setup.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any, Iterator

import msal
import requests

from constants import (
    GRAPH_API_BASE,
    GRAPH_AUTHORITY_TEMPLATE,
    GRAPH_SCOPE_DEFAULT,
    GRAPH_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("graph_mail_client")


class GraphMailClient:
    def __init__(self, settings: dict[str, Any]):
        self._settings = settings

    def _acquire_token(self) -> str:
        """Get an access token. Delegates expiry handling to msal's internal cache."""
        cert_path = Path(self._settings["certificate_path"])
        cert_pem = cert_path.read_text(encoding="utf-8")
        app = msal.ConfidentialClientApplication(
            client_id=self._settings["client_id"],
            authority=GRAPH_AUTHORITY_TEMPLATE.format(tenant_id=self._settings["tenant_id"]),
            client_credential={
                "private_key": cert_pem,
                "thumbprint": self._settings["certificate_thumbprint"],
            },
        )
        result = app.acquire_token_for_client(scopes=GRAPH_SCOPE_DEFAULT)
        if "access_token" not in result:
            raise RuntimeError(f"Graph auth failed: {result.get('error_description', result)}")
        return result["access_token"]

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        token = self._acquire_token()
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def iter_delta_messages(
        self,
        delta_link: str | None = None,
        backfill_days: int = 30,
    ) -> Iterator[tuple[dict[str, Any], str | None]]:
        """Yield ``(message, next_delta_link)`` pairs from the Inbox delta feed.

        On the *initial* call (no ``delta_link``), the delta is seeded with a
        ``receivedDateTime ge <now - backfill_days>`` filter so we capture the
        recent history. Subsequent runs use the persisted ``delta_link`` and the
        filter is sticky to the cursor — no need to re-pass it.

        Yield semantics: every real message comes back as ``(msg, None)``. The
        terminal yield, sent once Graph returns ``@odata.deltaLink``, is the
        sentinel ``({}, delta_link)`` — an empty dict signals "this is the
        cursor, not a message; persist the second element and stop iterating."
        """
        mailbox = self._settings["mailbox"]
        if delta_link:
            url = delta_link
            params = None
        else:
            cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=backfill_days)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            url = f"{GRAPH_API_BASE}/users/{mailbox}/mailFolders/Inbox/messages/delta"
            params = {
                "$select": "id,subject,from,bodyPreview,body,receivedDateTime,internetMessageId",
                "$filter": f"receivedDateTime ge {cutoff}",
            }

        while True:
            payload = self._get(url, params)
            params = None
            for msg in payload.get("value", []):
                yield msg, None
            next_link = payload.get("@odata.nextLink")
            delta_link_out = payload.get("@odata.deltaLink")
            if next_link:
                url = next_link
                continue
            if delta_link_out:
                yield {}, delta_link_out
                return
            return
