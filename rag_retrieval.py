"""
Retrieval helpers to ground LLM answers in ERP data using a vector store.
"""

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict

from constants import LOGGER
from openai_clients import call_openai_rag_answer
from vector_store import VectorStore

# Prompts that are inherently numeric/transactional should bypass RAG and go straight to ERP SQL.
_SQL_FIRST_PATTERNS = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\busage\b",
        r"\buse\b\s+(by|per)\b",  # e.g., "use per day" but not "how do I use"
        r"\bdemand\b",
        r"\bsales?\b",
        r"\bbuy\b",
        r"\bpurchase\b",
        r"\breorder\b",
        r"\bshortage\b",
        r"\binventory\b",
        r"\bstock\b",
        r"\bcoverage\b",
        r"\bdays?\s+on\s+hand\b",
        r"\bforecast\b",
        r"\bplan\b",
        r"\border\b",
        r"\bpo\b",
        r"\bbom\b",
        r"\braw material\b",
        r"\brun\s+out\b",
        r"\bby\s+month\b",
        r"\bmonthly\b",
        r"\bweekly\b",
        r"\bdaily\b",
        r"\btrend\b",
        r"\bqty\b",
        r"\bquantity\b",
        r"\bunits?\b",
    )
)


def _should_skip_rag(prompt: str) -> bool:
    """Heuristic: if the question is clearly transactional/numeric, skip RAG and let ERP SQL handle it."""
    return any(p.search(prompt) for p in _SQL_FIRST_PATTERNS)


def _format_retrieved_docs_for_prompt(retrieved_docs: list) -> str:
    """Serialize retrieved documents into a compact text blob for the LLM prompt."""
    return "\n".join(json.dumps(doc) for doc in retrieved_docs)


def build_rag_answer(
    prompt: str,
    today: datetime.date,
    vector_store: VectorStore,
) -> dict | None:
    """
    Query the vector store, retrieve relevant documents, and ask the LLM to answer.
    Returns None when no relevant information is found.
    """
    if not vector_store or _should_skip_rag(prompt):
        return None

    retrieved_docs = vector_store.query(prompt, top_k=5)
    if not retrieved_docs:
        return None

    # The retrieved_docs are tuples of (text, score, metadata)
    # We'll format them for the prompt
    context_docs = [
        {"text": text, "score": score, "metadata": metadata}
        for text, score, metadata in retrieved_docs
    ]
    context_blob = _format_retrieved_docs_for_prompt(context_docs)

    rag_reply = call_openai_rag_answer(prompt, today, context_blob)
    
    narrative = None
    usage_snapshot = None
    if isinstance(rag_reply, dict):
        narrative = rag_reply.get("answer") or rag_reply.get("summary")
        usage_snapshot = rag_reply.get("usage")

    insights = {
        "summary": narrative or "Information retrieved from the knowledge base.",
        "narrative": narrative or "",
        "row_count": len(retrieved_docs),
        "note": "Grounded in ERP data from the vector store.",
        "rag": True,
    }

    # Since we are not executing a SQL query, we will return the retrieved data
    # in a format that can be displayed in the UI.
    data_rows = [
        {
            "retrieved_text": text,
            "similarity_score": score,
            **metadata,
        }
        for text, score, metadata in retrieved_docs
    ]

    result: Dict[str, Any] = {
        "data": data_rows,
        "sql": None,  # No SQL is generated in this new RAG flow
        "insights": insights,
        "entities": {},
    }
    if usage_snapshot:
        result["usage"] = {"label": "rag_answer", **usage_snapshot}
    
    return result
