"""
Sandbox — an agentic AI workspace where the assistant builds the page itself.

The page is a vertical list of typed "blocks" (heading, markdown, code, chart,
table, metric). The user chats with the AI; the AI uses tools to query GP,
manipulate the canvas, and run Python — iterating until the result is built.
"""
from __future__ import annotations

import contextlib
import io
import json
import traceback
import uuid
from datetime import date

import pandas as pd
import requests
import streamlit as st

from constants import (
    CUSTOM_SQL_ALLOWED_TABLES,
    CUSTOM_SQL_HINTS,
    CUSTOM_SQL_MAX_ROWS,
    OPENAI_CHAT_URL,
    OPENAI_DEFAULT_MODEL,
    OPENAI_TIMEOUT_SECONDS,
)
from db_pool import get_connection
from secrets_loader import load_openai_settings
from sql_utils import validate_custom_sql


ALLOWED_BLOCK_TYPES = {"heading", "markdown", "code", "chart", "table", "metric"}
MAX_AGENT_TURNS = 24

DATA_DICTIONARY = """\
This is a chemical manufacturing ERP on Microsoft Dynamics GP. SQL is T-SQL,
read-only, parameterized with `?`. Key tables (only these are queryable):

- IV00101  - item master (ITEMNMBR, ITEMDESC, CURRCOST, ITMCLSCD)
- IV00102  - item per location (ITEMNMBR, LOCNCODE, QTYONHND, QTYONORD, ORDRPNTQTY)
- IV30200/IV30300 - inventory tx history (header has DOCDATE; lines have TRXQTY).
                    Join on DOCNUMBR + DOCTYPE. DOCTYPE=1 + TRXQTY<0 = consumption.
- SOP30200/SOP30300 - posted sales (header has DOCDATE; lines have QUANTITY,
                      XTNDPRCE). Join on SOPTYPE + SOPNUMBE. SOPTYPE=3 invoice,
                      4 return.
- SOP10100/SOP10200 - open sales orders (same join keys).
- POP30300/POP30310 - posted PO receipts (header: VENDORID, RECEIPTDATE; lines:
                      ITEMNMBR, UMQTYINB, UNITCOST). POPTYPE NOT IN (4,5),
                      VOIDSTTS=0.
- POP10100/POP10110 - open POs.
- PM00200 - vendor master (VENDORID, VENDNAME).
- PM30200 - paid AP transaction history (DOCAMNT, DOCDATE).
- BM00111 / BM010115 - bill of materials (parents -> components).

Quirks worth knowing:
{hints}

Default location is 'MAIN'. Always cap result sets ({max_rows} rows max).
""".format(
    hints="\n".join(f"- {h}" for h in CUSTOM_SQL_HINTS),
    max_rows=CUSTOM_SQL_MAX_ROWS,
)

SYSTEM_PROMPT = f"""\
You are a senior data analyst embedded inside a chemical manufacturing ERP. The
user talks to you through a chat box, and you build a live dashboard for them
by calling tools. The dashboard is a vertical list of typed blocks shown above
the chat.

You are AGENTIC: keep calling tools until the dashboard is complete, then send
a final plain-text reply to the user. Do not stop after one block.

When the user asks for a "dashboard", "interface", "report", or "view":
  - Produce 6-12 blocks: a heading, 2-4 KPI metrics, 2-4 charts (different
    angles), at least one data table, and a short markdown narrative
    summarizing what the data shows.
  - Pull REAL data with the `query_sql` tool. Do not invent numbers. If a
    query fails, read the error and try again with a fix.
  - Translate raw rows into clear chart/table/metric blocks - never paste raw
    JSON into a markdown block.

When the user asks a focused question, answer it directly with the smallest
useful set of blocks (often just one chart or table + a one-line takeaway).

WORKFLOW (important):
  1. Issue independent tool calls IN PARALLEL in the same turn whenever you
     can - e.g., 4 KPI queries at once, not 4 sequential turns. Each turn is
     a network round-trip; batching cuts latency dramatically.
  2. If a query fails, read the error and call `describe_table` to see real
     column names BEFORE retrying. Do not retry blindly.
  3. After you have data, batch all `add_block` calls into one turn.
  4. End with a plain-text reply summarizing what you built. Do not loop
     forever; if you can't finish, say what's blocking you.

Tools available:
- query_sql(sql, params)  : run a read-only SELECT/WITH against the ERP.
- describe_table(name)    : list real column names + types for a table.
- add_block(block)        : append a block to the canvas.
- update_block(id, block) : replace fields on an existing block.
- delete_block(id)        : remove a block.
- clear_canvas()          : remove all blocks (use only when the user asks).
- run_python(code)        : execute Python with pd, st, date available. Returns
                            stdout. Use sparingly - prefer SQL + chart blocks.

Block specs:
- heading  : {{"type":"heading","level":1|2|3,"text":"..."}}
- markdown : {{"type":"markdown","text":"..."}}
- metric   : {{"type":"metric","label":"...","value":"...","delta":"..."}}
- chart    : {{"type":"chart","kind":"line|bar|area","data":[{{...}}],
              "x":"col","y":"col","title":"..."}}
- table    : {{"type":"table","rows":[{{...}}],"caption":"..."}}
- code     : {{"type":"code","language":"python|sql|text","source":"..."}}

Today is {{today}}.

{DATA_DICTIONARY}
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_sql",
            "description": (
                "Run a read-only T-SQL SELECT against the ERP and return rows "
                "as a list of dicts. Use parameterized queries with '?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SELECT or WITH... SELECT statement."},
                    "params": {
                        "type": "array",
                        "items": {"type": ["string", "number", "boolean", "null"]},
                        "description": "Values for '?' placeholders, in order.",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Return real column names and types for a GP table. Use when a query failed with an Invalid column name error.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_block",
            "description": "Append a block to the bottom of the canvas.",
            "parameters": {
                "type": "object",
                "properties": {"block": {"type": "object"}},
                "required": ["block"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_block",
            "description": "Replace fields on an existing block by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "block": {"type": "object"},
                },
                "required": ["id", "block"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_block",
            "description": "Delete a block by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_canvas",
            "description": "Remove all blocks from the canvas.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python in a sandbox (pd, st, date, sql_query available). Returns stdout.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

STARTER_PROMPTS = [
    ("📊 Sales dashboard (last 12 months)",
     "Build a sales dashboard for the last 12 months: total revenue KPI, monthly revenue line chart, top 10 customers table, top 10 items by revenue bar chart, and a short narrative."),
    ("📦 Inventory health overview",
     "Build an inventory health overview: KPIs for total on-hand value and number of items, a chart of on-hand by item class, a table of items below their reorder point, and a short narrative."),
    ("🛒 Purchasing summary (last 90 days)",
     "Build a purchasing summary for the last 90 days: KPIs for total spend and PO count, a bar chart of spend by vendor, a line chart of weekly spend, and a table of the top 20 receipts."),
    ("🌱 Top consumed raw materials this fiscal year",
     "Show me the top 20 raw materials consumed this fiscal year (Jul-Jun), as a bar chart and a table."),
]


# --------------------------------------------------------------------------
# Block / canvas helpers
# --------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _init_state() -> None:
    st.session_state.setdefault("sandbox_blocks", [])
    st.session_state.setdefault("sandbox_chat", [])
    st.session_state.setdefault("sandbox_show_controls", False)
    st.session_state.setdefault("sandbox_pending_prompt", None)


def _summarize_blocks(blocks: list[dict]) -> list[dict]:
    summary = []
    for b in blocks:
        rest = {k: v for k, v in b.items() if k != "id"}
        preview = json.dumps(rest, default=str)
        if len(preview) > 200:
            preview = preview[:197] + "..."
        summary.append({"id": b["id"], "type": b.get("type"), "preview": preview})
    return summary


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------

def _tool_query_sql(args: dict) -> dict:
    sql = args.get("sql", "")
    params = args.get("params") or []
    ok, err = validate_custom_sql(sql, CUSTOM_SQL_ALLOWED_TABLES)
    if not ok:
        return {"error": err}
    try:
        with get_connection() as conn:
            df = pd.read_sql(sql, conn, params=params)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if len(df) > CUSTOM_SQL_MAX_ROWS:
        df = df.head(CUSTOM_SQL_MAX_ROWS)
    df = df.where(pd.notnull(df), None)
    return {"rows": df.to_dict(orient="records"), "row_count": len(df)}


def _tool_describe_table(args: dict) -> dict:
    name = (args.get("name") or "").strip().upper()
    if name not in {t.upper() for t in CUSTOM_SQL_ALLOWED_TABLES}:
        return {"error": f"table {name!r} is not in the allowlist"}
    try:
        with get_connection() as conn:
            df = pd.read_sql(
                "SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
                conn,
                params=[name],
            )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"columns": df.to_dict(orient="records")}


def _tool_add_block(args: dict) -> dict:
    block = dict(args.get("block") or {})
    if block.get("type") not in ALLOWED_BLOCK_TYPES:
        return {"error": f"unknown block type: {block.get('type')}"}
    block["id"] = block.get("id") or _new_id()
    st.session_state["sandbox_blocks"].append(block)
    return {"id": block["id"]}


def _tool_update_block(args: dict) -> dict:
    bid = args.get("id")
    spec = args.get("block") or {}
    for b in st.session_state["sandbox_blocks"]:
        if b["id"] == bid:
            b.update(spec)
            b["id"] = bid
            return {"ok": True}
    return {"error": f"no block with id {bid}"}


def _tool_delete_block(args: dict) -> dict:
    bid = args.get("id")
    blocks = st.session_state["sandbox_blocks"]
    new = [b for b in blocks if b["id"] != bid]
    if len(new) == len(blocks):
        return {"error": f"no block with id {bid}"}
    st.session_state["sandbox_blocks"] = new
    return {"ok": True}


def _tool_clear_canvas(_args: dict) -> dict:
    st.session_state["sandbox_blocks"] = []
    return {"ok": True}


def _tool_run_python(args: dict) -> dict:
    code = args.get("code", "")
    buf = io.StringIO()
    glb = {
        "pd": pd, "st": st, "date": date,
        "sql_query": lambda s, p=None: _tool_query_sql({"sql": s, "params": p or []}),
    }
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<sandbox>", "exec"), glb)
    except Exception:
        return {"stdout": buf.getvalue(), "error": traceback.format_exc()}
    return {"stdout": buf.getvalue()}


TOOL_DISPATCH = {
    "query_sql": _tool_query_sql,
    "describe_table": _tool_describe_table,
    "add_block": _tool_add_block,
    "update_block": _tool_update_block,
    "delete_block": _tool_delete_block,
    "clear_canvas": _tool_clear_canvas,
    "run_python": _tool_run_python,
}


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def _summarize_tool_result(result: dict) -> str:
    if not isinstance(result, dict):
        return str(result)[:120]
    if "error" in result:
        return f"❌ {str(result['error'])[:200]}"
    if "row_count" in result:
        return f"✅ {result['row_count']} rows"
    if "columns" in result:
        return f"✅ {len(result['columns'])} columns"
    if "id" in result:
        return f"✅ block `{result['id']}`"
    if result.get("ok"):
        return "✅ ok"
    return "✅"


def _run_agent(user_prompt: str, chat_history: list[dict], status, trace: list[dict]) -> str:
    """Drive the OpenAI tool-use loop. Returns the final assistant reply.

    `trace` is mutated in place — each entry is {"name", "args", "summary"}.
    """
    settings = load_openai_settings()
    api_key = settings.get("api_key")
    if not api_key:
        return "OpenAI is not configured. Add `[openai] api_key` to secrets.toml."
    model = settings.get("model", OPENAI_DEFAULT_MODEL)

    system = SYSTEM_PROMPT.replace("{today}", date.today().isoformat())
    state_blob = json.dumps(_summarize_blocks(st.session_state["sandbox_blocks"]), default=str)

    messages: list[dict] = [{"role": "system", "content": system}]
    for h in chat_history[-6:]:
        if h["role"] in ("user", "assistant") and isinstance(h.get("content"), str):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({
        "role": "user",
        "content": f"Current canvas state:\n{state_blob}\n\nUser request:\n{user_prompt}",
    })

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for turn in range(MAX_AGENT_TURNS):
        status.update(label=f"Thinking… (step {turn + 1}/{MAX_AGENT_TURNS})")
        payload = {
            "model": model,
            "temperature": 0.3,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        try:
            resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=OPENAI_TIMEOUT_SECONDS)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
        except (requests.RequestException, KeyError, ValueError) as err:
            return f"AI call failed: {err}"

        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return (msg.get("content") or "").strip() or "(done)"

        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": tool_calls,
        })

        names = [c.get("function", {}).get("name") for c in tool_calls]
        status.update(label=f"Step {turn + 1}: {', '.join(names)}")

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            handler = TOOL_DISPATCH.get(name)
            if handler is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = handler(args)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({
                "name": name,
                "args": args,
                "summary": _summarize_tool_result(result),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result, default=str)[:8000],
            })

    blocks_made = sum(1 for t in trace if t["name"] == "add_block")
    return (
        f"Hit the {MAX_AGENT_TURNS}-step limit without sending a final reply. "
        f"I made {blocks_made} block(s) and ran {len(trace)} tool call(s). "
        f"Open the trace below for details, then ask me to continue or simplify."
    )


# --------------------------------------------------------------------------
# Block rendering
# --------------------------------------------------------------------------

def _render_block(block: dict) -> None:
    t = block.get("type")
    if t == "heading":
        level = max(1, min(3, int(block.get("level", 2) or 2)))
        st.markdown(f"{'#' * level} {block.get('text', '')}")
    elif t == "markdown":
        st.markdown(block.get("text", ""))
    elif t == "metric":
        st.metric(
            block.get("label", ""),
            block.get("value", ""),
            block.get("delta") or None,
        )
    elif t == "table":
        rows = block.get("rows") or []
        caption = block.get("caption")
        if caption:
            st.markdown(f"**{caption}**")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("(no rows)")
    elif t == "chart":
        rows = block.get("data") or []
        title = block.get("title")
        if title:
            st.markdown(f"**{title}**")
        if not rows:
            st.info("(no data)")
            return
        df = pd.DataFrame(rows)
        x, y = block.get("x"), block.get("y")
        kind = block.get("kind", "line")
        if x and y and x in df.columns and y in df.columns:
            indexed = df.set_index(x)[[y]]
            if kind == "bar":
                st.bar_chart(indexed)
            elif kind == "area":
                st.area_chart(indexed)
            else:
                st.line_chart(indexed)
        else:
            st.dataframe(df, use_container_width=True)
    elif t == "code":
        st.code(block.get("source", ""), language=block.get("language", "python"))
    else:
        st.warning(f"Unknown block type: {t}")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def _handle_prompt(prompt: str) -> None:
    st.session_state["sandbox_chat"].append({"role": "user", "content": prompt})
    trace: list[dict] = []
    with st.status("Thinking…", expanded=True) as status:
        reply = _run_agent(prompt, st.session_state["sandbox_chat"], status, trace)
        status.update(label=f"Done · {len(trace)} tool call(s)", state="complete", expanded=False)
    st.session_state["sandbox_chat"].append({
        "role": "assistant",
        "content": reply,
        "trace": trace,
    })


def render_page() -> None:
    st.title("🧪 Sandbox")
    st.caption(
        "Tell the assistant what you want and it will build it for you using "
        "live ERP data. Try one of the starter ideas below or type your own."
    )
    _init_state()

    st.sidebar.header("Sandbox")
    st.session_state["sandbox_show_controls"] = st.sidebar.toggle(
        "Show block controls",
        value=st.session_state["sandbox_show_controls"],
        help="Show block IDs and ✕ delete buttons.",
    )
    if st.sidebar.button("🧹 Clear canvas", use_container_width=True):
        st.session_state["sandbox_blocks"] = []
        st.session_state["sandbox_chat"] = []
        st.rerun()
    if st.sidebar.button("🗑️ Clear chat only", use_container_width=True):
        st.session_state["sandbox_chat"] = []
        st.rerun()

    blocks: list[dict] = st.session_state["sandbox_blocks"]
    show_controls = st.session_state["sandbox_show_controls"]

    if not blocks:
        st.markdown("##### Try one of these:")
        cols = st.columns(2)
        for i, (label, prompt) in enumerate(STARTER_PROMPTS):
            with cols[i % 2]:
                if st.button(label, use_container_width=True, key=f"starter_{i}"):
                    st.session_state["sandbox_pending_prompt"] = prompt
                    st.rerun()
        st.caption("…or just type something below.")
    else:
        for block in blocks:
            with st.container(border=True):
                if show_controls:
                    cols = st.columns([0.92, 0.08])
                    with cols[0]:
                        _render_block(block)
                    with cols[1]:
                        st.caption(f"`{block['id']}`")
                        if st.button("✕", key=f"del_{block['id']}", help="Delete block"):
                            st.session_state["sandbox_blocks"] = [
                                b for b in blocks if b["id"] != block["id"]
                            ]
                            st.rerun()
                else:
                    _render_block(block)

    st.divider()
    st.subheader("💬 Ask the assistant")

    for msg in st.session_state["sandbox_chat"][-12:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            trace = msg.get("trace") or []
            if trace:
                with st.expander(f"Tool trace ({len(trace)} call{'s' if len(trace) != 1 else ''})"):
                    for i, t in enumerate(trace, 1):
                        args_preview = json.dumps(t["args"], default=str)
                        if len(args_preview) > 220:
                            args_preview = args_preview[:217] + "..."
                        st.markdown(
                            f"`{i:02d}` **{t['name']}** — {t['summary']}\n\n"
                            f"&nbsp;&nbsp;&nbsp;&nbsp;`{args_preview}`"
                        )

    pending = st.session_state.pop("sandbox_pending_prompt", None)
    if pending:
        _handle_prompt(pending)
        st.rerun()

    typed = st.chat_input("Ask for a chart, a dashboard, a number — anything…")
    if typed:
        _handle_prompt(typed)
        st.rerun()


render_page()
