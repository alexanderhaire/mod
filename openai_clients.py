import calendar
import datetime
import json
import re
from decimal import Decimal

import requests

from constants import (
    BILL_OF_MATERIALS_UI_REFERENCE,
    CUSTOM_SQL_ALLOWED_TABLES,
    CUSTOM_SQL_HINTS,
    CUSTOM_SQL_MAX_ROWS,
    DEFAULT_MODEL_CONTEXT_LIMIT,
    FEW_SHOT_EXAMPLES,
    ITEM_RESOURCE_PLANNING_UI_REFERENCE,
    LOGGER,
    OPENAI_EMBEDDING_MODEL,
    MODEL_CONTEXT_LIMITS,
    OPENAI_CHAT_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    PLATFORM_CAPABILITIES,
)
from secrets_loader import load_openai_settings
from sql_utils import _extract_json_block


def _coerce_json_value(value):
    """Convert common non-JSON-native types into serializable primitives."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _context_limit_for_model(model: str | None) -> int | None:
    if not model:
        return None
    normalized = model.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if normalized.startswith(key.lower()):
            return limit
    return DEFAULT_MODEL_CONTEXT_LIMIT


def _build_usage_snapshot(response_json: dict, model: str) -> dict | None:
    if not isinstance(response_json, dict):
        return None

    usage = response_json.get("usage")
    if not isinstance(usage, dict):
        return None

    try:
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    except (TypeError, ValueError):
        return None

    snapshot: dict[str, int | float | str] = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    context_limit = _context_limit_for_model(model)
    if context_limit:
        snapshot["context_limit"] = context_limit
        try:
            utilization_pct = (total_tokens / context_limit) * 100
        except ZeroDivisionError:
            utilization_pct = None
        if utilization_pct is not None:
            snapshot["context_utilization_pct"] = round(utilization_pct, 2)
            snapshot["tokens_remaining"] = max(context_limit - total_tokens, 0)

    return snapshot


def enhance_prompt_for_complexity(prompt: str, system_prompt: str) -> str:
    """Adds guidance to the system prompt for complex questions."""
    lower_prompt = prompt.lower()
    complex_keywords = (
        "compare",
        "vs",
        "versus",
        "average",
        "trend",
        "growth",
        "multiple",
        "breakdown",
        "per month",
        "per-item",
        "ranking",
        "top",
        "most used",
        "usage",
        "each",
    )
    long_prompt = len(prompt.split()) > 12
    has_complex_signal = any(keyword in lower_prompt for keyword in complex_keywords)
    if has_complex_signal or long_prompt:
        complexity_hint = (
            "\nThe question appears multi-step. Before returning JSON, silently plan:\n"
            "1) Restate the goal and outputs.\n"
            "2) Break the ask into ordered sub-calculations (e.g., compute usage per month, then compare or rank items).\n"
            "3) Pick tables/joins and map date phrases like 'last month' or 'last 90 days' to explicit ranges; prefer CTEs to stage logic.\n"
            "4) For comparisons or rankings, compute per-period/per-item metrics first, then aggregate and order.\n"
            "5) Verify '?' parameters align with filters and the count matches placeholders. Keep reasoning concise and, when helpful, include a short 'reasoning' array in the JSON."
        )
        return system_prompt + complexity_hint
    return system_prompt


def call_openai_sql_generator(
    prompt: str,
    today: datetime.date,
    schema_summary: str,
    context_hint: str | None = None,
    retry_reason: str | None = None,
    conversation_hint: str | None = None,
    previous_sql: str | None = None,
    previous_params: list | tuple | None = None,
    date_hint: str | None = None,
) -> dict | None:
    """Ask OpenAI to translate a free-form question into a safe, parameterized SQL query."""
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None
    model = settings.get("sql_model", OPENAI_DEFAULT_MODEL)
    system_prompt = (
        "You are a T-SQL expert for Microsoft Dynamics GP. You will be given a question and context, and you must generate a safe, read-only T-SQL query. "
        "You must respond with a JSON object with keys: 'sql', 'params', 'summary', and 'entities'. "
        "You may include a short 'reasoning' array (2-5 steps) showing how you decomposed the request, and an optional 'chart' object when the user asks for a chart/graph/plot or requests the data to be visualized. "
        "The chart object should follow: {\"type\":\"bar|line|area\",\"x\":\"<column>\",\"y\":\"<column>\",\"series\":\"<optional grouping column or null>\",\"title\":\"<short title>\"}. "
        "Use column names exactly as returned by the SQL, choose 'line' for time-series/trends and 'bar' for ranked comparisons.\n"
        "The 'sql' key must contain the SQL query as a string.\n"
        "The 'params' key must contain a list of parameters for the query.\n"
        "The 'summary' key must contain a short, human-readable description of what the query does.\n"
        "The 'entities' key must contain a JSON object of key entities found in the user question, such as 'item', 'year', or 'month'.\n"
        "If the user asks for a 'dashboard', 'report', or 'page' (e.g., 'generate a market dashboard'), you MUST include a 'report_structure' object.\n"
        "The 'report_structure' object should follow: {\"title\": \"<Page Title>\", \"sections\": [{\"type\": \"metric|chart|table\", \"title\": \"<Section Title>\", \"value\": \"<Metric Value>\", \"data_source\": \"sql_result\", \"x\": \"<x_col>\", \"y\": \"<y_col>\"}]}.\n"
        "For dashboards, prioritize showing MULTIPLE time-series charts (e.g., price vs time, quantity vs time) to give a comprehensive view.\n"
        "Ensure your SQL returns enough columns (e.g., Date, Cost, Quantity, Sales) to populate these multiple charts from a single query.\n"
        "Rules:\n"
        "- Only SELECT statements are allowed.\n"
        "- Do not use any DML or DDL.\n"
        "- Do not use temporary tables.\n"
        "- Only reference the tables provided in the schema.\n"
        "- Use '?' for parameters.\n"
        f"- Limit results to {CUSTOM_SQL_MAX_ROWS} rows.\n"
        "- If a chart is requested, ensure the SQL selects the columns named in chart.x and chart.y so they exist in the result.\n"
        "- If you include reasoning, keep it concise and focused on the calculation steps; do not include SQL in reasoning.\n"
        "- For analysis questions (e.g., 'compare', 'trend', 'ratio'), use CTEs to break down the logic and perform calculations in the final SELECT.\n"
        "- You now have access to Vendor (PM00200) and Paid Transaction History (PM30200) tables. Use them for vendor-related questions.\n"
        "- CRITICAL: When comparing two columns (e.g., WHERE STNDCOST < CURRCOST), DO NOT quote the column names. Quoting them makes them strings and causes type errors."
    )
    
    system_prompt = enhance_prompt_for_complexity(prompt, system_prompt)

    user_sections = [
        f"Current date: {today.isoformat()}",
        "Approved tables: " + ", ".join(CUSTOM_SQL_ALLOWED_TABLES),
        f"Schema reference:\n{schema_summary}",
        f"Dynamics GP modeling tips:\n" + "\n".join(f"- {h}" for h in CUSTOM_SQL_HINTS),
        f"UI reference:\n{ITEM_RESOURCE_PLANNING_UI_REFERENCE.strip()}",
        f"Bill of Materials reference:\n{BILL_OF_MATERIALS_UI_REFERENCE.strip()}",
        "Examples:\n" + FEW_SHOT_EXAMPLES,
    ]
    if context_hint:
        user_sections.append(f"Structured context: {context_hint}")
    if date_hint:
        user_sections.append(f"Date hint: {date_hint}")
    if conversation_hint:
        user_sections.append(f"Recent chat history:\n{conversation_hint}")

    retry_notes = []
    if retry_reason:
        retry_notes.append("The previous SQL attempt failed. Use this feedback to correct it:")
        retry_notes.append(str(retry_reason))
    if previous_sql:
        retry_notes.append(f"Previous SQL attempt:\n{previous_sql}")
    if previous_params:
        retry_notes.append(f"Previous parameters: {list(previous_params)}")
    if retry_notes:
        user_sections.append("\n".join(retry_notes))

    user_sections.append("User question:\n" + prompt.strip())
    user_sections.append("Respond with JSON only.")
    
    user_prompt = "\n\n".join(user_sections)
    
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI SQL generation failed: %s", err)
        return None


def is_erp_intent(prompt: str, context: dict | None = None) -> bool:
    """
    Heuristic override to keep manufacturing/ERP questions in-scope even if the router is uncertain.
    Acts as a guardrail against false rejects for short planning prompts like 'december 2024' or
    'what should I buy in december'.
    """
    if context and isinstance(context, dict) and context:
        return True
    if not isinstance(prompt, str):
        return False

    text = prompt.lower()
    erp_tokens = (
        "inventory", "stock", "raw material", "raw materials", "materials", "mrp",
        "bill of materials", "bom", "production", "batch", "blend", "batch ticket",
        "purchase", "po", "p.o.", "buy", "procure", "replenish", "plan", "planning",
        "dynamics", "gp", "great plains", "cost", "usage", "consume", "consumption",
        "sop", "pop", "iv", "gl", "variance", "forecast"
    )
    month_tokens = [m.lower() for m in calendar.month_name if m]

    if any(tok in text for tok in erp_tokens):
        return True

    has_month = any(m in text for m in month_tokens)
    has_timeframe = has_month or bool(re.search(r"\b20\d{2}\b", text))
    if has_timeframe and any(word in text for word in ("buy", "purchase", "plan", "use", "usage", "consume", "requirement")):
        return True

    return False


def call_openai_question_router(
    prompt: str,
    today: datetime.date,
    context_hint: str | None = None,
    conversation_hint: str | None = None,
) -> dict:
    """
    Ask OpenAI to decide whether a question is in-scope for ERP/SQL, RAG, or should be rejected.
    Falls back to ERP/SQL when routing fails or OpenAI is not configured.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return {"route": "erp_sql", "reason": "openai_not_configured"}

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    system_prompt = (
        "You are a routing model for a chemical manufacturing assistant. Your primary function is to route user questions to the appropriate tool."
        "You have the following tools available:\n"
        "- 'erp_sql': For questions that require generating a T-SQL query to be run against a Microsoft Dynamics GP ERP database. Use this for questions about production, bills of material, inventory, purchasing, costing, quality, finance, SQL/reporting, comparisons, or time-bound planning. IMPORTANT: Questions like 'What should I buy' or 'What do we need' are VALID ERP purchasing questions—route them here, not to reject.\n"
        "- 'rag': For questions that can be answered from a knowledge base of indexed documents. Use this for questions about item details, product specifications, or other information that is likely to be found in the indexed data.\n"
        "- 'nlp': For general NLP tasks that do not need ERP data (e.g., summarize this text, sentiment/tone analysis, classify/categorize, rewrite/translate, or math/reasoning on provided text/numbers).\n"
        "- 'chat': For brief greetings, pleasantries, or meta questions about the assistant with no data ask.\n"
        "- 'reject': For consumer advice, holidays, generic shopping, trivia, programming help, or anything unrelated to the plant's ERP data that should not be answered. Do NOT reject purchasing/planning related questions.\n"
        "Please deliberate before deciding. If a question could be answered by either `erp_sql` or `rag`, prefer `rag` if the question is about descriptive information of an entity, and `erp_sql` if it is about metrics, calculations or reports.\n"
        "Default to 'erp_sql' when unsure. Never suggest web lookups or hybrid answers.\n\n"
        f"{PLATFORM_CAPABILITIES}"
    )

    user_sections = [
        f"Current date: {today.isoformat()}",
        f"Question: {prompt.strip()}",
    ]
    if context_hint:
        user_sections.append(f"Structured context: {context_hint}")
    if conversation_hint:
        user_sections.append(f"Recent conversation:\n{conversation_hint}")
    user_sections.append(
        "Return JSON like {\"route\":\"erp_sql|rag|chat|reject|nlp\",\"rationale\":\"why\",\"confidence\":0.0-1.0,"
        "\"scores\":{\"erp_sql\":0-1,\"rag\":0-1,\"chat\":0-1,\"reject\":0-1,\"nlp\":0-1}}. "
        "Confidence should reflect how certain you are in the chosen route."
    )

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n\n".join(user_sections)}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        route = str(parsed.get("route", "erp_sql")).lower()
        if route not in {"erp_sql", "rag", "reject", "chat", "nlp"}:
            route = "erp_sql"
        parsed["route"] = route
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI routing failed: %s", err)
        return {"route": "erp_sql", "reason": "router_failed"}


def _detect_nlp_task(prompt: str) -> str:
    """
    Lightweight heuristic to categorize non-ERP NLP tasks so prompts can include sharper guidance.
    """
    if not prompt:
        return "qa"

    text = prompt.lower()
    if any(tok in text for tok in ("sentiment", "tone", "positive or negative", "emotion", "how do they feel")):
        return "sentiment"
    if any(tok in text for tok in ("summarize", "summary", "tl;dr", "condense", "brief")):
        return "summary"
    if any(tok in text for tok in ("classify", "category", "categorize", "label", "tag this", "bucket")):
        return "classification"
    if any(tok in text for tok in ("extract", "pull out", "list the", "find all", "identify the")):
        return "extraction"
    if any(tok in text for tok in ("calculate", "calc", "compute", "math", "difference", "ratio", "percentage", "percent", "how many", "total", "average", "mean", "sum", "variance")):
        return "reasoning"
    return "qa"


def call_openai_nlp_task(
    prompt: str,
    today: datetime.date,
    context_hint: str | None = None,
) -> dict | None:
    """
    Handle non-ERP NLP asks such as summaries, sentiment/tone, lightweight reasoning, or classification
    using the general OpenAI model with structured guidance and math double-checks.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    task = _detect_nlp_task(prompt)

    system_prompt = (
        "You are an NLP utility for a chemical manufacturing copilot. "
        "You handle summarization, sentiment/tone analysis, classification, extraction, and numerical reasoning that do not require SQL. "
        "Always return strict JSON with keys: "
        "'task' (string), "
        "'answer' (short string), "
        "'bullets' (list of brief strings), "
        "'sentiment' (object with 'label' and 'score' -1 to 1), "
        "'steps' (list of reasoning steps), "
        "'checked' (boolean that is true only after double-checking any calculations), "
        "'confidence' (0-1 float), "
        "'extractions' (list of strings or key-value pairs), "
        "'warnings' (list of strings). "
        "Rules:\n"
        "- For sentiment tasks, pick positive/neutral/negative and a score between -1 and 1; include a one-line rationale in 'answer'.\n"
        "- For summaries, include a 1-2 sentence TL;DR in 'answer' and 2-4 bullets in 'bullets'.\n"
        "- For classification, list top 1-3 labels in 'extractions' and set a confidence score.\n"
        "- For extraction, return the extracted nuggets in 'extractions' and a concise 'answer'.\n"
        "- For reasoning/math, show numbered steps, check arithmetic carefully, and set 'checked' to true only after verifying the math twice.\n"
        "- Stay grounded in the provided text; do not invent data or company-specific facts."
    )

    user_sections = [
        f"Current date: {today.isoformat()}",
        f"Detected task: {task}",
        "User request:",
        prompt.strip(),
    ]
    if context_hint:
        user_sections.append(f"Context: {context_hint}")
    user_sections.append("Respond with JSON only following the schema above.")

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n\n".join(user_sections)}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        parsed["task"] = parsed.get("task") or task
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI NLP task failed: %s", err)
        return None


def is_likely_erp_question(prompt: str) -> bool:
    """
    Lightweight heuristic to catch in-scope ERP questions the router might over-reject.
    """
    if not prompt:
        return False

    text = prompt.lower()
    keywords = (
        "inventory",
        "stock",
        "on hand",
        "quantity",
        "qty",
        "purchase",
        "purchasing",
        "buy",
        "po",
        "purchase order",
        "vendor",
        "supplier",
        "material",
        "materials",
        "raw material",
        "raw materials",
        "bill of material",
        "bom",
        "assembly",
        "component",
        "production",
        "manufactur",
        "work order",
        "batch",
        "lot",
        "cost",
        "costing",
        "gp",
        "dynamics gp",
        "demand",
        "forecast",
    )
    return any(term in text for term in keywords)


def call_openai_embedding(text: str, model: str | None = None) -> dict | None:
    """
    Generate an embedding vector for semantic similarity use cases (e.g., learned handler recall).
    Returns None when OpenAI is not configured or the request fails.
    """
    if not text or not isinstance(text, str):
        return None

    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    embed_model = model or settings.get("embedding_model") or OPENAI_EMBEDDING_MODEL
    payload = {"model": embed_model, "input": text}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        record = data["data"][0]
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            return None
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return {"embedding": embedding, "model": embed_model, "usage": usage}
    except (requests.RequestException, KeyError, TypeError, ValueError) as err:
        LOGGER.warning("OpenAI embedding failed: %s", err)
        return None


def call_openai_general_response(
    prompt: str,
    today: datetime.date,
    web_context: str = "",
    context_hint: str | None = None,
) -> dict | None:
    """Ask OpenAI to synthesize an answer using provided web snippets and optional ERP context."""
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    system_prompt = (
        "Answer concisely, cite sources when possible, and acknowledge freshness limits if context is thin. "
        "Think step by step before responding.\n\n"
        f"{PLATFORM_CAPABILITIES}"
    )
    user_sections = [
        f"Current date: {today.isoformat()}",
        f"Question: {prompt.strip()}",
    ]
    if context_hint:
        user_sections.append(f"ERP context: {context_hint}")
    if web_context:
        user_sections.append(f"Web snippets:\n{web_context}")
    else:
        user_sections.append("Web snippets: none found. Use general knowledge and mark potential staleness.")
    user_sections.append('Return JSON like {"answer": "...", "bullets": ["..."], "sources": ["url1", "..."]}.')

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n\n".join(user_sections)}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI general response failed: %s", err)
        return None


def call_openai_data_narrative(
    prompt: str,
    rows: list,
    summary_hint: str | None = None,
    context_hint: str | None = None,
    entities: dict | None = None,
    max_rows: int = 40,
) -> dict | None:
    """
    Ask OpenAI to turn tabular SQL results into a short, bolded story.
    Focus on peaks/valleys/trends and keep the response concise.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key or not rows:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)

    def _sample_rows(data: list, limit: int) -> list:
        sample = []
        for row in data[:limit]:
            if isinstance(row, dict):
                sample.append({str(k): _coerce_json_value(v) for k, v in row.items()})
            else:
                sample.append({"value": _coerce_json_value(row)})
        return sample

    sampled_rows = _sample_rows(rows, max_rows)
    rows_json = json.dumps(sampled_rows, default=_coerce_json_value)
    entities_json = json.dumps(entities, default=_coerce_json_value) if entities else ""

    system_prompt = (
        "You are a concise data storyteller for a chemical manufacturing assistant. "
        "Given a small JSON table excerpt, write 2-3 sentences that read like a mini story. "
        "Highlight peaks, dips, and overall direction. Treat negative usage values as consumption magnitudes. "
        "If a 'Month' column is numeric (1-12), convert to month names. "
        "Use Markdown **bold** for important months, items, or figures. "
        "Do not repeat the SQL or the raw JSON; only return the narrative."
    )

    user_sections = [
        f"Original question: {prompt.strip()}",
        f"Existing summary: {summary_hint or 'None provided.'}",
    ]
    if entities_json:
        user_sections.append(f"Entities: {entities_json}")
    if context_hint:
        user_sections.append(f"Context hint: {context_hint}")
    user_sections.append(f"Data sample (JSON, {len(sampled_rows)} rows max):\n{rows_json}")
    user_sections.append(
        "Respond with JSON like {\"narrative\": \"...\"}. "
        "Keep it tight (max ~70 words), make it read like a story, and bold the standout numbers or months."
    )

    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI data narrative failed: %s", err)
        return None


def call_openai_deep_analyst(
    prompt: str,
    rows: list,
    summary_hint: str | None = None,
    context_hint: str | None = None,
    entities: dict | None = None,
    max_rows: int = 100,
) -> dict | None:
    """
    Ask OpenAI to perform a deep analysis of the data, looking for trends, outliers, and correlations.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key or not rows:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)

    def _sample_rows(data: list, limit: int) -> list:
        sample = []
        for row in data[:limit]:
            if isinstance(row, dict):
                sample.append({str(k): _coerce_json_value(v) for k, v in row.items()})
            else:
                sample.append({"value": _coerce_json_value(row)})
        return sample

    sampled_rows = _sample_rows(rows, max_rows)
    rows_json = json.dumps(sampled_rows, default=_coerce_json_value)
    entities_json = json.dumps(entities, default=_coerce_json_value) if entities else ""

    system_prompt = (
        "You are a Senior Data Analyst for a chemical manufacturing company. "
        "Your goal is to provide deep, actionable insights based on the provided data. "
        "Do not just summarize the data; explain 'why' and 'what it means'. "
        "Look for:\n"
        "- Trends (up/down/seasonal)\n"
        "- Outliers (anomalies)\n"
        "- Correlations\n"
        "- Actionable recommendations (e.g., 'Stock up on X', 'Investigate Y')\n"
        "Use Markdown to format your response with headers, bullet points, and bold text. "
        "Keep it professional but engaging."
    )

    user_sections = [
        f"User Question: {prompt.strip()}",
        f"Data Summary: {summary_hint or 'None provided.'}",
    ]
    if entities_json:
        user_sections.append(f"Entities: {entities_json}")
    if context_hint:
        user_sections.append(f"Context: {context_hint}")
    user_sections.append(f"Data (JSON, {len(sampled_rows)} rows max):\n{rows_json}")
    user_sections.append(
        "Respond with JSON like {\"analysis\": \"...\"}. "
        "The 'analysis' field should contain the full Markdown formatted analysis."
    )

    payload = {
        "model": model,
        "temperature": 0.5,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI deep analyst failed: %s", err)
        return None


def call_openai_rag_answer(
    prompt: str,
    today: datetime.date,
    retrieval_context: str,
    entities: dict | None = None,
) -> dict | None:
    """
    Ask OpenAI to answer strictly from retrieved ERP facts (RAG) and avoid guessing.
    Returns None when OpenAI is not configured or no retrieval context is provided.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key or not retrieval_context:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    system_prompt = (
        "You are an ERP analyst answering only from the supplied ERP retrievals. "
        "Do not invent values beyond the provided facts. "
        "If the context is insufficient, say so clearly rather than guessing. "
        "Use concise language and bold key figures."
    )

    entities_blob = json.dumps(entities, default=_coerce_json_value) if entities else None
    user_sections = [
        f"Current date: {today.isoformat()}",
        f"Question: {prompt.strip()}",
        f"ERP retrievals (JSON):\n{retrieval_context}",
    ]
    if entities_blob:
        user_sections.insert(2, f"Entities/filters: {entities_blob}")
    user_sections.append(
        'Respond with JSON like {"answer": "...", "bullets": ["..."], "confidence": 0-1, "note": "..."} '
        "Base everything on the retrievals; if they lack the answer, say so explicitly."
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_sections)},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI RAG answer failed: %s", err)
        return None


def call_openai_small_talk(prompt: str, today: datetime.date) -> dict | None:
    """Generate a short, friendly reply for casual conversation without involving SQL."""
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    fallback_reply = "Hi there! I'm here to chat and help with your manufacturing data questions whenever you're ready."
    if not api_key:
        return {"reply": fallback_reply}

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    system_prompt = (
        "You are the friendly small-talk voice of a chemical manufacturing data copilot. "
        "Keep responses concise (1-3 sentences), upbeat, and workplace appropriate. "
        "Do not invent ERP data or SQL. If the user hints at data needs, invite them to ask a specific question you can look up.\n\n"
        f"{PLATFORM_CAPABILITIES}"
    )
    user_prompt = (
        f"Current date: {today.isoformat()}\n"
        f"User message: {prompt.strip()}\n"
        'Respond with JSON like {"reply": "<short, friendly response>"} only.'
    )
    payload = {
        "model": model,
        "temperature": 0.6,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        reply = parsed.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            parsed["reply"] = fallback_reply
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI small talk failed: %s", err)
        return {"reply": fallback_reply}


def call_openai_market_analyst(
    product_name: str,
    item_desc: str,
    price_history: list,
    usage_history: list,
    inventory_status: dict,
    external_market_context: str | None = None,
) -> dict | None:
    """
    Generate comprehensive market intelligence for a chemical product combining
    internal ERP data with external market context.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    
    def _coerce_json_value(value):
        """Convert non-JSON types to serializable primitives."""
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return value.isoformat()
        return value
    
    # Prepare data for LLM
    price_data = [{k: _coerce_json_value(v) for k, v in row.items()} for row in price_history[:50]]
    usage_data = [{k: _coerce_json_value(v) for k, v in row.items()} for row in usage_history[:50]]
    inventory_data = {k: _coerce_json_value(v) for k, v in inventory_status.items()}
    
    system_prompt = (
        "You are a Senior Market Analyst specializing in chemical manufacturing and agricultural inputs. "
        "Your role is to analyze product data and provide actionable market intelligence. "
        "You have access to:\n"
        "1. Internal ERP data: price history, usage patterns, inventory levels\n"
        "2. External market context: agricultural commodity trends, demand forecasts\n\n"
        "Provide a comprehensive analysis that includes:\n"
        "- Price trend analysis and volatility assessment\n"
        "- Usage patterns and seasonality detection\n"
        "- Correlation with external market factors\n"
        "- Inventory adequacy given demand trends\n"
        "- Actionable recommendations (pricing, purchasing, production)\n\n"
        "Format your response in Markdown with clear sections and bold key insights."
    )
    
    user_sections = [
        f"PRODUCT ANALYSIS REQUEST",
        f"Product: {product_name}",
        f"Description: {item_desc}",
        "",
        f"INTERNAL DATA:",
        f"Price History (last 12 months): {json.dumps(price_data, default=_coerce_json_value)}",
        f"Usage History (last 6 months): {json.dumps(usage_data, default=_coerce_json_value)}",
        f"Current Inventory Status: {json.dumps(inventory_data, default=_coerce_json_value)}",
    ]
    
    if external_market_context:
        user_sections.append("")
        user_sections.append("EXTERNAL MARKET CONTEXT:")
        user_sections.append(external_market_context)
    
    user_sections.append("")
    user_sections.append(
        'Respond with JSON: {"analysis": "...markdown content..."}'
    )
    
    payload = {
        "model": model,
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_sections)},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI market analyst failed: %s", err)
        return None


def call_openai_structured_market_data(
    category: str,
    today: datetime.date
) -> dict | None:
    """
    Generate structured market data for a given category using OpenAI.
    Returns a dict matching the schema required by external_data.py.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    model = settings.get("model", OPENAI_DEFAULT_MODEL)
    
    system_prompt = (
        "You are a market intelligence analyst. "
        "Generate realistic, grounded market data for the specified commodity category. "
        "Return strict JSON with the following structure:\n"
        "{\n"
        "  'commodity': 'specific commodity name',\n"
        "  'current_price_index': float (100-1000),\n"
        "  'trend': 'increasing'|'decreasing'|'stable',\n"
        "  'volatility': 'low'|'moderate'|'high',\n"
        "  'price_change_30d': float (percentage),\n"
        "  'price_change_90d': float (percentage),\n"
        "  'demand_level': 'low'|'moderate'|'high',\n"
        "  'demand_score': int (0-100),\n"
        "  'seasonal_pattern': 'strong'|'moderate'|'weak',\n"
        "  'forecast_next_30d': 'increasing'|'decreasing'|'stable'\n"
        "}\n"
        "Base estimates on real-world market conditions for this commodity."
    )

    user_prompt = f"Category: {category}\nDate: {today.isoformat()}"

    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI market data generation failed: %s", err)
        return None


def call_openai_vision_analyst(
    prompt: str,
    image_base64: str,
    today: datetime.date,
    context_hint: str | None = None,
) -> dict | None:
    """
    Ask OpenAI (Vision) to analyze an image.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return None

    # Use a vision-capable model, defaulting to gpt-4o
    model = settings.get("vision_model", "gpt-4o")
    
    system_prompt = (
        "You are a Visual Intelligence Analyst for a chemical manufacturing plant. "
        "You can see charts, screenshots, and diagrams provided by the user. "
        "Analyze them in the context of ERP/supply chain operations. "
        "Be concise, professional, and clear."
    )

    user_text = f"Current date: {today.isoformat()}\nQuestion: {prompt}"
    if context_hint:
        user_text += f"\nContext: {context_hint}"

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }
    
    # Vision responses are often just text, but we enforce JSON for consistency with our app
    payload["messages"][0]["content"] += " Respond with JSON like {\"answer\": \"Markdown text...\", \"insights\": [\"bullet 1\", \"bullet 2\"]}."
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
        response.raise_for_status()
        response_json = response.json()
        content = response_json["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if usage_snapshot := _build_usage_snapshot(response_json, model):
            parsed["usage"] = usage_snapshot
        return parsed
    except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
        LOGGER.warning("OpenAI Vision failed: %s", err)
        return {"answer": "I had trouble seeing that image. " + str(err)}