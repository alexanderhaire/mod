import datetime
from collections.abc import Mapping

import requests

from constants import LOGGER
from context_utils import summarize_sql_context
from openai_clients import call_openai_general_response


def fetch_web_context(query: str, max_snippets: int = 5, max_chars: int = 1200) -> tuple[str, list[str]]:
    """
    Fetch lightweight web context using DuckDuckGo's instant answer API.
    Returns a combined text blob and a list of source URLs.
    """
    if not query or not isinstance(query, str):
        return "", []

    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
        "skip_disambig": 1,
    }
    try:
        resp = requests.get("https://api.duckduckgo.com/", params=params, timeout=12)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as err:
        LOGGER.warning("Web lookup failed: %s", err)
        return "", []

    snippets: list[str] = []
    sources: list[str] = []

    if isinstance(payload, Mapping):
        abstract = payload.get("AbstractText")
        if abstract:
            snippets.append(str(abstract))
        abstract_url = payload.get("AbstractURL")
        if abstract_url:
            sources.append(str(abstract_url))

        related = payload.get("RelatedTopics") or []
        collected = 0

        def _collect(entry):
            nonlocal collected
            if collected >= max_snippets:
                return
            if not isinstance(entry, Mapping):
                return
            text = entry.get("Text")
            url = entry.get("FirstURL")
            if text:
                snippets.append(str(text))
            if url:
                sources.append(str(url))
            collected += 1

        for entry in related:
            if collected >= max_snippets:
                break
            if isinstance(entry, Mapping) and entry.get("Topics"):
                for sub in entry.get("Topics", []):
                    _collect(sub)
                    if collected >= max_snippets:
                        break
            else:
                _collect(entry)

    context_blob = "\n".join(snippets)
    if len(context_blob) > max_chars:
        context_blob = context_blob[:max_chars] + "..."

    return context_blob, sources[:max_snippets]


def handle_web_question(prompt: str, today: datetime.date, context: dict | None = None) -> dict:
    """Handle non-ERP questions by pulling lightweight web context and asking OpenAI to synthesize an answer."""
    context_hint = summarize_sql_context(context)
    web_context, web_sources = fetch_web_context(prompt)
    llm_answer = call_openai_general_response(prompt, today, web_context, context_hint)
    usage_snapshot = llm_answer.pop("usage", None) if llm_answer else None

    if llm_answer:
        answer = llm_answer.get("answer") or ""
        bullets = [b for b in llm_answer.get("bullets", []) if isinstance(b, str)]
        sources = [s for s in llm_answer.get("sources", []) if isinstance(s, str)] or web_sources
        summary_parts = [answer] if answer else []
        if bullets:
            summary_parts.append("\n".join(f"- {b}" for b in bullets))
        summary_text = "\n\n".join(part for part in summary_parts if part)
        if not summary_text:
            summary_text = "I wasn't able to compose a strong answer from the web context."
        insights = {"summary": summary_text, "sources": sources, "route": "web"}
        result = {"data": [], "insights": insights, "entities": {}, "sql": None}
        if isinstance(usage_snapshot, dict):
            result["usage"] = [{"label": "web_answer", **usage_snapshot}]
        return result

    fallback_summary = web_context or "No web context was available for this question."
    result = {"data": [], "insights": {"summary": fallback_summary, "sources": web_sources, "route": "web"}, "entities": {}, "sql": None}
    if isinstance(usage_snapshot, dict):
        result["usage"] = [{"label": "web_answer", **usage_snapshot}]
    return result
