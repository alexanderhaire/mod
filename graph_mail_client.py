"""Thin wrapper around Microsoft Graph: app-only auth + delta-query mail fetch.

This module assumes the secrets section ``[graph]`` is populated (see
``secrets_loader.load_graph_settings``). Use a real Outlook account; there is no
mocking layer here on purpose — failures should surface immediately during
manual setup.
"""
from __future__ import annotations

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
        self._token: str | None = None

    def _acquire_token(self) -> str:
        if self._token:
            return self._token
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
        self._token = result["access_token"]
        return self._token

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
        """Yield (message, next_delta_link) pairs.

        Only the LAST tuple's ``next_delta_link`` is meaningful (the persisted cursor).
        Intermediate yields have ``next_delta_link = None``.
        """
        mailbox = self._settings["mailbox"]
        if delta_link:
            url = delta_link
            params = None
        else:
            url = f"{GRAPH_API_BASE}/users/{mailbox}/mailFolders/Inbox/messages/delta"
            params = {
                "$select": "id,subject,from,bodyPreview,body,receivedDateTime,internetMessageId",
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

    def fetch_message_body(self, message_id: str) -> str:
        mailbox = self._settings["mailbox"]
        data = self._get(f"{GRAPH_API_BASE}/users/{mailbox}/messages/{message_id}")
        body = data.get("body", {})
        return body.get("content", "") if isinstance(body, dict) else ""
