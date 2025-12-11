"""A Streamlit chatbot application that translates natural language questions into SQL queries
and displays the results."""
import calendar
import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyodbc
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
except ImportError:
    alt = None
from auth import authenticate_user, ensure_user_store, is_admin, register_user
from constants import (
    LOGGER,
    RAW_MATERIAL_CLASS_CODES,
)
from context_utils import summarize_chat_history
from feedback import attachments_from_paste, log_feedback, save_feedback_attachments
from market_insights import (
    classify_item_segment,
    get_product_details, 
    calculate_buying_signals, 
    forecast_demand,
    fetch_monthly_price_trends,
    fetch_product_usage_history,
    calculate_inventory_runway,
    calculate_seasonal_burn_metrics,
    get_volatility_score,
    get_batch_volatility_scores,
    get_seasonal_pattern,
    get_priority_raw_materials,
    get_items_needing_attention,
    get_top_movers_raw_materials,
    get_raw_material_time_series,
    get_inventory_distribution
)
from external_data import (
    get_market_context, 
    fetch_agricultural_market_data, 
    get_usage_forecasts
)
from openai_clients import call_openai_market_analyst
from dynamic_handler import (
    delete_dynamic_handler,
    list_dynamic_handlers,
    record_handler_feedback,
    update_handler_status,
)
from router import handle_question
from secrets_loader import build_connection_string
from training_logger import record_sql_example_from_response, record_training_event

st.set_page_config(
    page_title="Chemical Dynamics",

    page_icon="CD",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.caption("System v1.5.1 [ML Logic Active]")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;700&display=swap');

        :root {
            --bg-terminal: #000000;
            --text-terminal: #ffb000; /* Amber */
            --text-dim: #b58900;
            --accent-red: #dc322f;
            --accent-green: #859900;
            --border-terminal: 1px solid #ffb000;
        }

        .stApp {
            background-color: var(--bg-terminal);
            color: var(--text-terminal);
            font-family: 'Inconsolata', monospace;
        }

        /* Force Sidebar to Right Side */
        [data-testid="stSidebar"] {
            right: 0;
            left: auto;
            border-left: var(--border-terminal);
            border-right: none;
            background-color: #0a0a0a;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            right: 0;
            left: auto;
        }

        /* Adjust Main Content to not overlap right sidebar */
        .main .block-container {
            padding-right: 350px; /* Approximate sidebar width */
            padding-left: 2rem;
            max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inconsolata', monospace;
            color: var(--text-terminal);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Terminal Card Style */
        .terminal-card {
            background: #050505;
            border: var(--border-terminal);
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0 10px rgba(255, 176, 0, 0.1);
        }

        .terminal-header {
            border-bottom: var(--border-terminal);
            margin-bottom: 0.5rem;
            padding-bottom: 0.25rem;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
        }

        /* Ticker Tape */
        .ticker-wrap {
            width: 100%;
            overflow: hidden;
            background-color: #111;
            border-bottom: var(--border-terminal);
            white-space: nowrap;
            box-sizing: border-box;
        }
        .ticker {
            display: inline-block;
            padding-left: 100%;
            animation: ticker 30s linear infinite;
        }
        .ticker-item {
            display: inline-block;
            padding: 0 2rem;
            font-size: 1.2rem;
            color: var(--text-terminal);
        }
        @keyframes ticker {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-100%, 0, 0); }
        }

        /* Chat Messages */
        div[data-testid="stChatMessage"] {
            background: transparent;
            border: none;
            padding: 0.5rem 0;
            font-family: 'Inconsolata', monospace;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            background: #111;
            border: 1px dashed var(--text-dim);
            padding: 10px;
            color: var(--text-terminal);
        }

        div[data-testid="stChatMessage"][data-testid="user"] [data-testid="stMarkdownContainer"] {
            background: #222;
            border-color: var(--text-terminal);
        }

        /* Input Field */
        div[data-baseweb="input"] {
            background: #000;
            border: var(--border-terminal);
            color: var(--text-terminal);
            font-family: 'Inconsolata', monospace;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: var(--border-terminal);
            font-family: 'Inconsolata', monospace;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ticker Placeholder (will be populated later)
st.markdown(
    """
    <div class="ticker-wrap">
        <div class="ticker">
            <div class="ticker-item">MARKET STATUS: OPEN</div>
            <div class="ticker-item">SYSTEM: ONLINE</div>
            <div class="ticker-item">DATA FEED: LIVE</div>
            <div class="ticker-item">MODE: TERMINAL</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title(">> CHEMICAL_MARKET_TERMINAL_V1")


# Configure logging


def _ensure_auth_state() -> None:
    """Initialize auth-related session state and ensure the user store exists."""
    ensure_user_store()
    if "user" not in st.session_state:
        st.session_state.user = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = []
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if st.session_state.user:
        st.session_state.is_admin = is_admin(st.session_state.user)
    else:
        st.session_state.is_admin = False


def _render_auth_gate() -> None:
    """Render sign-in and sign-up forms; halt the app until authenticated."""
    st.subheader("Welcome")
    st.markdown("Sign in to use the data copilot. Create an account if you do not have one.")

    login_tab, signup_tab = st.tabs(["Sign in", "Sign up"])

    with login_tab:
        login_user = st.text_input("Username", key="login_username")
        login_pass = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in"):
            ok, message = authenticate_user(login_user, login_pass)
            if ok:
                st.session_state.user = login_user.strip()
                st.session_state.is_admin = is_admin(st.session_state.user)
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with signup_tab:
        with st.form("signup_form", clear_on_submit=True):
            new_user = st.text_input("Username", key="signup_username")
            new_pass = st.text_input("Password", type="password", key="signup_password")
            confirm_pass = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create account")
            if submitted:
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 8:
                    st.error("Use at least 8 characters for your password.")
                else:
                    ok, message = register_user(new_user, new_pass)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

    st.stop()


def _render_handler_admin_panel() -> None:
    """Display a compact admin console for reviewing learned handlers."""
    if not st.session_state.get("is_admin"):
        return

    handlers = list_dynamic_handlers()
    pending_count = 0
    normalized: list[dict[str, Any]] = []
    for name, data in handlers.items():
        if not isinstance(data, dict):
            continue
        status = str(data.get("status", "approved")).lower()
        if status not in {"approved", "pending", "disabled"}:
            status = "approved"
        if status == "pending":
            pending_count += 1
        normalized.append(
            {
                "name": name,
                "status": status,
                "helpful": int(data.get("helpful_count", 0) or 0),
                "flagged": int(data.get("flagged_count", 0) or 0),
                "usage": int(data.get("usage_count", 0) or 0),
                "last_prompt": data.get("prompt") or data.get("last_prompt") or name,
                "sql": data.get("sql"),
                "review_note": data.get("review_note", ""),
                "keywords": data.get("keywords", []),
                "last_updated": data.get("last_updated") or data.get("created_at"),
            }
        )

    expanded = pending_count > 0
    with st.sidebar.expander(f"Admin: Learned Handlers ({len(normalized)})", expanded=expanded):
        if not normalized:
            st.caption("No learned handlers yet.")
            return

        st.caption(f"{pending_count} pending review | {len(normalized)} total")
        status_filter = st.multiselect(
            "Filter by status",
            options=["pending", "approved", "disabled"],
            default=["pending", "approved"],
            key="handler_status_filter",
        )
        filtered = [h for h in normalized if h["status"] in status_filter] if status_filter else normalized
        filtered = sorted(filtered, key=lambda h: (h["status"], -(h["last_updated"] or 0)))

        handler_labels = {h["name"]: f"{h['name']} · {h['status']} · 👍{h['helpful']}/👎{h['flagged']}" for h in filtered}
        if not handler_labels:
            st.caption("No handlers match this filter.")
            return

        default_index = 0
        if pending_count:
            for idx, h in enumerate(filtered):
                if h["status"] == "pending":
                    default_index = idx
                    break

        selected_name = st.selectbox(
            "Choose handler",
            options=list(handler_labels.keys()),
            index=default_index,
            format_func=lambda name: handler_labels.get(name, name),
            key="handler_select",
        )
        selected = next((h for h in filtered if h["name"] == selected_name), None)
        if not selected:
            return

        st.markdown(f"**Status:** {selected['status'].title()}  |  👍 {selected['helpful']}  👎 {selected['flagged']}  |  Used {selected['usage']}x")
        st.caption(f"Prompt: {selected['last_prompt']}")
        if selected.get("keywords"):
            st.caption("Keywords: " + ", ".join(selected["keywords"][:8]))
        if selected.get("sql"):
            st.code(selected["sql"], language="sql")

        with st.form(f"handler_status_form_{selected_name}", clear_on_submit=True):
            new_status = st.selectbox(
                "Set status",
                options=["approved", "pending", "disabled"],
                index=["approved", "pending", "disabled"].index(selected["status"]) if selected["status"] in {"approved", "pending", "disabled"} else 0,
            )
            note = st.text_input("Admin note (optional)", value=selected.get("review_note", ""))
            if st.form_submit_button("Save status"):
                updated = update_handler_status(selected_name, new_status, note)
                if updated:
                    st.success("Handler status updated.")
                    st.rerun()
                else:
                    st.error("Failed to update handler status.")

        confirm_delete = st.checkbox("Confirm delete", key=f"confirm_delete_{selected_name}")
        if st.button("Delete handler", key=f"delete_handler_{selected_name}", disabled=not confirm_delete):
            deleted = delete_dynamic_handler(selected_name)
            if deleted:
                st.warning(f"Deleted handler '{selected_name}'.")
                st.rerun()
            else:
                st.error("Unable to delete handler.")


def _ensure_chat_state() -> None:
    """Initialize multi-chat state and migrate any legacy single-chat history."""
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 1
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    legacy_messages = st.session_state.pop("messages", None)
    legacy_context = st.session_state.pop("chat_context", None)

    if not st.session_state.chats:
        chat_id = f"chat-{st.session_state.chat_counter}"
        st.session_state.chat_counter += 1
        st.session_state.chats.append(
            {
                "id": chat_id,
                "name": "Chat 1",
                "messages": legacy_messages if isinstance(legacy_messages, list) else [],
                "context": legacy_context if isinstance(legacy_context, dict) else {},
            }
        )
        st.session_state.active_chat_id = chat_id

    active_ids = {c["id"] for c in st.session_state.chats}
    if st.session_state.active_chat_id not in active_ids and st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]


def _get_chat(chat_id: str | None) -> dict | None:
    return next((chat for chat in st.session_state.chats if chat["id"] == chat_id), None)


def _ensure_new_chat(name: str | None = None) -> dict:
    """Create a new chat with a friendly default name."""
    base_name = name.strip() if name and name.strip() else f"Chat {st.session_state.chat_counter}"
    chat_id = f"chat-{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    chat = {"id": chat_id, "name": base_name, "messages": [], "context": {}}
    st.session_state.chats.append(chat)
    st.session_state.active_chat_id = chat_id
    return chat


def _match_column(df: pd.DataFrame, name: str | None) -> str | None:
    """Return the DataFrame column matching the provided name (case-insensitive)."""
    if df is None or name is None:
        return None
    target = str(name).strip().lower()
    for col in df.columns:
        if str(col).lower() == target:
            return col
    return None


MONTH_ORDER = [calendar.month_name[i] for i in range(1, 13)]
MONTH_LOOKUP = {
    name.lower(): idx for idx, name in enumerate(MONTH_ORDER, start=1)
    if name
} | {calendar.month_abbr[idx].lower(): idx for idx in range(1, 13)}

CLIPBOARD_PASTE_COMPONENT = components.declare_component(
    "feedback_paste_zone", path=str(Path(__file__).parent / "components" / "feedback_paste_zone")
)


def _month_number(val) -> int | None:
    """Parse a month value from number/name/abbr strings into 1-12."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        maybe = int(val)
        return maybe if 1 <= maybe <= 12 else None
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return None
        if cleaned.isdigit():
            maybe = int(cleaned)
            return maybe if 1 <= maybe <= 12 else None
        lower = cleaned.lower()
        return MONTH_LOOKUP.get(lower)
    return None


def _format_month_label(val) -> str:
    """Return a human-friendly month label, falling back to the raw value."""
    month_num = _month_number(val)
    if month_num:
        return calendar.month_name[month_num]
    return str(val)


def _get_market_segment(row: pd.Series) -> str:
    """
    Determine if an item is a Raw Material or Finished Good.
    Uses the same classifier as the market monitor (based on GP Item Class codes).
    """
    itm_class = str(row.get('ITMCLSCD', '') or '')
    return classify_item_segment(itm_class)



def _dense_month_series(df: pd.DataFrame, month_col: str, value_col: str, year_hint: int | None = None) -> pd.DataFrame:
    """
    Ensure the month/value columns are fully populated for a calendar year with zeros for missing months.
    Preserves a consistent month order for charts and tables.
    """
    working = df.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working["__month_num"] = working[month_col].apply(_month_number)

    if working["__month_num"].isna().any():
        working[month_col] = working[month_col].apply(_format_month_label)
        return working.drop(columns="__month_num")

    aggregated = working.groupby("__month_num", as_index=False)[value_col].sum()
    full = pd.DataFrame({"__month_num": range(1, 13)})
    dense = full.merge(aggregated, on="__month_num", how="left").fillna({value_col: 0})

    dense[month_col] = dense["__month_num"].apply(lambda m: calendar.month_name[m])

    static_cols = [c for c in working.columns if c not in {month_col, value_col, "__month_num"}]
    for col in static_cols:
        uniques = working[col].dropna().unique()
        if len(uniques) == 1:
            dense[col] = uniques[0]

    if year_hint and "Year" not in dense.columns:
        dense.insert(0, "Year", year_hint)

    dense[month_col] = pd.Categorical(dense[month_col], categories=MONTH_ORDER, ordered=True)
    dense = dense.sort_values(month_col)
    dense[month_col] = dense[month_col].astype(str)
    return dense.drop(columns="__month_num")


def _prepare_display_df(df: pd.DataFrame, chart_spec: dict | None, entities: dict | None) -> pd.DataFrame:
    """Normalize month labels, fill missing months for time series, and keep chart/table consistent."""
    if df is None or df.empty:
        return df

    prepared = df.copy()
    chart_spec = chart_spec or {}
    entities = entities or {}
    x_col = _match_column(prepared, chart_spec.get("x"))
    y_col = _match_column(prepared, chart_spec.get("y"))
    series_col = _match_column(prepared, chart_spec.get("series") or chart_spec.get("group"))

    month_col = _match_column(prepared, "Month")
    if month_col:
        prepared[month_col] = prepared[month_col].apply(_format_month_label)

    if x_col and y_col and series_col is None and (x_col.lower() == "month" or all(_month_number(v) for v in prepared[x_col])):
        year_hint = entities.get("year")
        prepared = _dense_month_series(prepared, x_col, y_col, year_hint)

    return prepared


def _render_chart(df: pd.DataFrame, chart_spec: dict | None) -> None:
    """Render a simple bar/line/area chart when the model asks for one and columns exist."""
    if df is None or df.empty or not isinstance(chart_spec, dict):
        return

    x_col = _match_column(df, chart_spec.get("x"))
    y_col = _match_column(df, chart_spec.get("y"))
    series_col = _match_column(df, chart_spec.get("series") or chart_spec.get("group"))
    chart_type = str(chart_spec.get("type") or "bar").lower()
    title = chart_spec.get("title")

    if not x_col or not y_col:
        return

    plot_df = df[[col for col in (x_col, y_col, series_col) if col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_col])

    month_sort = None
    x_dtype = plot_df[x_col].dtype
    if isinstance(x_dtype, pd.CategoricalDtype):
        month_sort = list(plot_df[x_col].cat.categories)
        plot_df = plot_df.sort_values(x_col)
    elif x_col.lower() == "month":
        month_sort = MONTH_ORDER
        plot_df[x_col] = pd.Categorical(plot_df[x_col].apply(_format_month_label), categories=MONTH_ORDER, ordered=True)
        plot_df = plot_df.sort_values(x_col)

    if alt:
        mark_method = {"line": "mark_line", "area": "mark_area"}.get(chart_type, "mark_bar")
        chart = getattr(alt.Chart(plot_df), mark_method)()
        x_encoding = alt.X(x_col, sort=month_sort or "ascending", title=chart_spec.get("x_label") or x_col)
        y_encoding = alt.Y(y_col, title=chart_spec.get("y_label") or y_col, scale=alt.Scale(zero=True))
        encoded = chart.encode(x=x_encoding, y=y_encoding)
        if series_col:
            encoded = encoded.encode(color=series_col)
        encoded = encoded.encode(tooltip=[c for c in plot_df.columns])
        if title:
            encoded = encoded.properties(title=title)
        st.altair_chart(encoded, width="stretch")
        return

    if month_sort:
        plot_df[x_col] = pd.Categorical(plot_df[x_col], categories=month_sort, ordered=True)
        plot_df = plot_df.sort_values(x_col)
    fallback_df = plot_df[[x_col, y_col]].set_index(x_col)
    if chart_type == "line":
        st.line_chart(fallback_df)
    else:
        st.bar_chart(fallback_df)


def _latest_question_answer(messages: list[dict]) -> tuple[int | None, dict | None, dict | None]:
    """Return the index and pair of the latest user/assistant messages."""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") == "assistant":
            # Find the nearest preceding user message.
            for j in range(idx - 1, -1, -1):
                prior = messages[j]
                if prior.get("role") == "user":
                    return idx, prior, msg
            return idx, None, msg
    return None, None, None


def _render_clipboard_paste(key: str, *, compact: bool = False, reset: bool = False) -> dict | None:
    """
    Render the clipboard paste capture component and return its payload.
    When compact=True, the UI stays hidden while still listening for global paste events.
    Use reset=True to clear any previously pasted images after they are consumed.
    """
    try:
        return CLIPBOARD_PASTE_COMPONENT(key=key, default=None, compact=compact, reset=reset)
    except Exception as err:  # noqa: BLE001
        LOGGER.warning("Clipboard paste component unavailable: %s", err)
        return None


def _render_feedback_form(chat_id: str, messages: list[dict]) -> None:
    """Collect feedback on the most recent assistant reply."""
    assistant_idx, user_msg, assistant_msg = _latest_question_answer(messages)
    if assistant_idx is None or not assistant_msg or not user_msg:
        return

    feedback_key = f"{chat_id}:{assistant_idx}"
    if feedback_key in st.session_state.feedback_submitted:
        return

    with st.expander("Rate this answer", expanded=False):
        with st.form(f"feedback_form_{feedback_key}", clear_on_submit=True):
            helpful_choice = st.radio("Was this response helpful?", ("Yes", "No"), horizontal=True)
            notes = st.text_area(
                "Feedback (optional)",
                help="You can paste screenshots with Ctrl/Cmd+V while typing; they'll attach automatically.",
            )
            attachments = st.file_uploader(
                "Attach screenshots (paste or drop images)",
                type=["png", "jpg", "jpeg", "gif", "webp"],
                accept_multiple_files=True,
                help="Add visual context by pasting from your clipboard while writing feedback or by dragging files.",
            )
            pasted_payload = _render_clipboard_paste(f"fb_paste_{feedback_key}", compact=True)
            submitted = st.form_submit_button("Send feedback")

            if submitted:
                pasted_files, paste_errors = attachments_from_paste(pasted_payload)
                combined_files = list(attachments) if attachments else []
                combined_files.extend(pasted_files)
                saved_paths, attachment_errors = save_feedback_attachments(combined_files)
                handler_name = assistant_msg.get("handler_name")
                helpful_flag = helpful_choice == "Yes"
                log_feedback(
                    {
                        "user": st.session_state.user,
                        "chat_id": chat_id,
                        "question": user_msg.get("content", ""),
                        "answer": assistant_msg.get("content", ""),
                        "sql": assistant_msg.get("sql"),
                        "helpful": helpful_flag,
                        "notes": notes or "",
                        "handler_name": handler_name,
                        "attachments": saved_paths,
                    }
                )
                if handler_name:
                    record_handler_feedback(handler_name, helpful_flag)
                st.session_state.feedback_submitted.append(feedback_key)
                all_attachment_errors = attachment_errors + paste_errors
                if all_attachment_errors:
                    st.warning("\n".join(all_attachment_errors))
                st.success("Thanks for the feedback! We will use it to improve future answers.")


def _fetch_market_data(cursor: pyodbc.Cursor) -> pd.DataFrame:
    """Fetch live item cost data for the market monitor with YoY price change from purchase receipts."""
    try:
        # Query to get items with year-over-year price change based on actual purchase receipts
        query = """
        WITH CurrentPrices AS (
            SELECT 
                l.ITEMNMBR, 
                AVG(l.UNITCOST) as CurrentAvgCost,
                COUNT(*) as RecentReceipts
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -3, GETDATE())
              AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR
        ),
        PriorPrices AS (
            SELECT 
                l.ITEMNMBR, 
                AVG(l.UNITCOST) as PriorAvgCost,
                COUNT(*) as PriorReceipts
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -15, GETDATE())
              AND h.RECEIPTDATE < DATEADD(month, -12, GETDATE())
              AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR
        )
        SELECT
            i.ITEMNMBR, 
            i.ITEMDESC, 
            i.ITEMTYPE,
            i.ITMCLSCD,
            i.STNDCOST, 
            i.CURRCOST, 
            i.USCATVLS_1 as Category,
            COALESCE(c.CurrentAvgCost, i.CURRCOST) as CurrentAvgCost,
            p.PriorAvgCost,
            COALESCE(c.CurrentAvgCost, i.CURRCOST) - COALESCE(p.PriorAvgCost, i.STNDCOST) as PriceChange,
            CASE 
                WHEN p.PriorAvgCost > 0 THEN 
                    ((COALESCE(c.CurrentAvgCost, i.CURRCOST) - p.PriorAvgCost) / p.PriorAvgCost) * 100 
                WHEN i.STNDCOST > 0 THEN
                    ((i.CURRCOST - i.STNDCOST) / i.STNDCOST) * 100
                ELSE 0 
            END as PctChange
        FROM IV00101 i
        LEFT JOIN CurrentPrices c ON i.ITEMNMBR = c.ITEMNMBR
        LEFT JOIN PriorPrices p ON i.ITEMNMBR = p.ITEMNMBR
        WHERE i.ITEMTYPE IN (0, 1, 2)
        ORDER BY i.ITEMNMBR
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame.from_records(data, columns=columns)
        return df
    except Exception as e:
        LOGGER.error(f"Market data fetch failed: {e}")
        return pd.DataFrame()

_ensure_auth_state()
if st.session_state.user is None:
    _render_auth_gate()

_ensure_chat_state()

active_chat = _get_chat(st.session_state.active_chat_id)
if not active_chat:
    active_chat = _ensure_new_chat()

# Initialize product selection state
if "selected_product" not in st.session_state:
    st.session_state.selected_product = None

# Initialize page routing
if "current_page" not in st.session_state:
    st.session_state.current_page = "market_overview"  # or "product_insights"

# --- MAIN DASHBOARD (MARKET MONITOR) ---
conn_str, server, db, auth = build_connection_string()
try:
    with pyodbc.connect(conn_str, autocommit=True) as conn:
        cursor = conn.cursor()
        
        # Fetch market data (needed for both pages)
        df_market = _fetch_market_data(cursor)
        
        # === PAGE ROUTING ===
        # GENERATIVE INSIGHTS PAGE
        if st.session_state.current_page == "generative_insights":
            report_data = st.session_state.get("generative_report_data", {})
            structure = report_data.get("structure", {})
            data = report_data.get("data", [])
            
            # Header
            st.markdown(f"## >> {structure.get('title', 'GENERATIVE INSIGHTS')}")
            if st.button("← BACK TO MONITOR"):
                st.session_state.current_page = "market_overview"
                st.rerun()
                
            # Dynamic Sections
            if data:
                df_report = pd.DataFrame(data)
                
                for section in structure.get("sections", []):
                    st.markdown("---")
                    st.subheader(f">> {section.get('title', 'SECTION')}")
                    
                    sec_type = section.get("type")
                    
                    if sec_type == "metric":
                        st.metric(label=section.get("title"), value=section.get("value"))
                        
                    elif sec_type == "chart":
                        # Use specific columns if provided, otherwise auto-detect
                        x_col = section.get("x")
                        y_col = section.get("y")
                        
                        if x_col and y_col and x_col in df_report.columns and y_col in df_report.columns:
                            st.line_chart(df_report.set_index(x_col)[y_col])
                        else:
                            # Auto-detect chart columns if not specified
                            numeric_cols = df_report.select_dtypes(include=['number']).columns.tolist()
                            categorical_cols = df_report.select_dtypes(include=['object']).columns.tolist()
                            
                            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                chart_data = df_report.set_index(categorical_cols[0])[numeric_cols[:1]]
                                st.bar_chart(chart_data)
                            else:
                                st.write("Insufficient data for chart.")
                            
                    elif sec_type == "table":
                        st.dataframe(df_report, width="stretch")
                        
            else:
                st.info("No data available for this report.")

        # MARKET OVERVIEW PAGE
        elif st.session_state.current_page == "market_overview":
            
            # 2. Ticker Update (Dynamic)
            if not df_market.empty:
                # Filter for Raw Materials for the ticker focus
                df_ticker = df_market.copy()
                df_ticker['Segment'] = df_ticker.apply(_get_market_segment, axis=1)
                df_ticker = df_ticker[df_ticker['Segment'] == "Raw Material"]
                
                ticker_items = []
                # If no raw materials found, fall back to top movers of any kind
                source_df = df_ticker if not df_ticker.empty else df_market
                
                for _, row in source_df.head(10).iterrows():
                    price_change = float(row.get('PriceChange', 0) or 0)
                    pct_change = float(row.get('PctChange', 0) or 0)
                    curr_cost = float(row.get('CURRCOST', 0) or 0)
                    symbol = "▲" if price_change >= 0 else "▼"
                    ticker_items.append(f"{row.get('ITEMNMBR', '???')} {curr_cost:.2f} {symbol}{abs(pct_change):.1f}%")
                ticker_html = f"""
                <div class="ticker-wrap">
                    <div class="ticker">
                        <div class="ticker-item">{'   ///   '.join(ticker_items)}</div>
                    </div>
                </div>
                """
                st.markdown(ticker_html, unsafe_allow_html=True)

            # 3. Market Monitor (Interactive List)
            col1, col2 = st.columns([2, 1])
            selected_segment_for_chart = "Raw Material"
            
            with col1:
                st.markdown("### >> LIVE_PRICE_MONITOR")
                
                if not df_market.empty:
                    # Calculate Segments
                    df_market['Segment'] = df_market.apply(_get_market_segment, axis=1)
                    
                    # View Selection
                    view_mode = st.radio("View Mode", ["Market Monitor", "Procurement Dashboard"], horizontal=True, label_visibility="collapsed", key="view_mode_radio")
                    
                    if view_mode == "Procurement Dashboard":
                        st.markdown("---")
                        st.markdown("### >> PROCUREMENT_INTELLIGENCE")
                        st.caption("Showing only active raw materials with usage/purchase history")
                        
                        # Get priority raw materials (items with actual activity)
                        df_priority = get_priority_raw_materials(cursor, limit=25)
                        
                        if df_priority.empty:
                            st.info("No active raw materials found. Items shown here have recent usage or purchase history.")
                        else:
                            # Progress bar for analysis
                            progress_bar = st.progress(0, text="Analyzing market signals...")
                            
                            # Analyze priority items
                            analysis_results = []
                            total_items = len(df_priority)
                            
                            for i, (_, row) in enumerate(df_priority.iterrows()):
                                item_num = row['ITEMNMBR']
                                # Calculate signals
                                signals = calculate_buying_signals(cursor, item_num)
                                forecast = forecast_demand(cursor, item_num)
                                runway = calculate_inventory_runway(cursor, item_num)
                                
                                analysis_results.append({
                                    'Item': item_num,
                                    'Description': row['ITEMDESC'],
                                    'Current Cost': row['CURRCOST'],
                                    'Signal': signals.get('signal', 'Unknown'),
                                    'Score': signals.get('score', 0),
                                    'Reason': signals.get('reason', ''),
                                    'Forecast (3mo)': forecast.get('forecast_next_3mo', 0),
                                    'Trend': forecast.get('trend', 'Unknown'),
                                    'Runway': runway.get('runway_days', 999),
                                    'Urgency': runway.get('urgency', 'OK'),
                                    'AnnualSpend': row.get('EstAnnualSpend', 0)
                                })
                                progress_bar.progress((i + 1) / total_items, text=f"Analyzing {item_num}...")
                            
                            progress_bar.empty()
                            
                            df_analysis = pd.DataFrame(analysis_results)
                            
                            # 0. Show Attention Items First (Most Important!)
                            st.subheader(">> ITEMS_REQUIRING_ATTENTION")
                            attention_items = get_items_needing_attention(cursor, df_priority)
                            
                            if attention_items:
                                for item in attention_items[:5]:  # Top 5 most urgent
                                    with st.container():
                                        alerts_html = "".join([
                                            f'<span style="margin-right: 10px;">{a["icon"]} <strong>{a["action"]}</strong>: {a["message"]}</span>'
                                            for a in item['Alerts']
                                        ])
                                        
                                        border_color = "#dc322f" if item['Priority'] >= 100 else "#b58900" if item['Priority'] >= 50 else "#859900"
                                        
                                        st.markdown(f"""
                                        <div style="border: 2px solid {border_color}; padding: 12px; margin-bottom: 10px; border-radius: 5px; background: #0a0a0a;">
                                            <h4 style="color: {border_color}; margin: 0;">{item['Item']}</h4>
                                            <p style="margin: 4px 0; color: #839496;">{item['Description']} | Cost: ${item['Cost']:.2f}</p>
                                            <div style="margin-top: 8px; font-size: 0.9em;">
                                                {alerts_html}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.success("✓ No items require immediate attention")
                            
                            # 1. Strategic Buying Opportunities
                            st.markdown("---")
                            st.subheader(">> STRATEGIC_BUY_SIGNALS")
                            buy_opps = df_analysis[df_analysis['Score'] >= 60].sort_values('Score', ascending=False)
                            
                            if not buy_opps.empty:
                                for _, row in buy_opps.head(5).iterrows():
                                    color = "#859900" if row['Score'] >= 80 else "#b58900"
                                    with st.container():
                                        st.markdown(f"""
                                        <div style="border: 1px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                                            <h4 style="color: {color}; margin: 0;">{row['Signal'].upper()} - {row['Item']} ({row['Score']}/100)</h4>
                                            <p style="margin: 5px 0;"><strong>{row['Description']}</strong> | Cost: ${row['Current Cost']:.2f}</p>
                                            <p style="margin: 5px 0; font-style: italic;">"{row['Reason']}"</p>
                                            <p style="margin: 5px 0;">Forecast Trend: <strong>{row['Trend']}</strong></p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info("No strong buying opportunities detected at this time.")
                            
                            # 2. Price Trend Charts
                            st.markdown("---")
                            st.subheader(">> PRICE_TREND_ANALYSIS")
                            
                            # Create two columns for charts
                            chart_col1, chart_col2 = st.columns(2)
                            def _render_info_card(message: str) -> None:
                                st.markdown(
                                    f"""
                                    <div style="background: #0c1f3a; border: 1px solid #14395f; color: #6fb5ff; padding: 14px; border-radius: 6px;">
                                        {message}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            
                            with chart_col1:
                                st.markdown("#### 12-MONTH PRICE TRENDS")
                                # Get top 5 items with highest score for charting
                                top_items = df_analysis.nlargest(5, 'Score')['Item'].tolist()[:3]
                                
                                if top_items and alt:
                                    all_trends = []
                                    for item in top_items:
                                        trend_df = fetch_monthly_price_trends(cursor, item, months=12)
                                        if not trend_df.empty:
                                            trend_df['Item'] = item
                                            all_trends.append(trend_df)
                                    
                                    if all_trends:
                                        combined_df = pd.concat(all_trends, ignore_index=True)
                                        
                                        # Create Altair line chart with terminal theme
                                        chart = alt.Chart(combined_df).mark_line(strokeWidth=2).encode(
                                            x=alt.X('Date:T', title='Month', axis=alt.Axis(
                                                labelColor='#ffb000', titleColor='#ffb000', gridColor='#333'
                                            )),
                                            y=alt.Y('AvgCost:Q', title='Avg Cost ($)', axis=alt.Axis(
                                                labelColor='#ffb000', titleColor='#ffb000', gridColor='#333'
                                            )),
                                            color=alt.Color('Item:N', scale=alt.Scale(
                                                range=['#ffb000', '#859900', '#268bd2']
                                            )),
                                            tooltip=['Item', 'MonthLabel', 'AvgCost']
                                        ).properties(
                                            height=300
                                        ).configure_axis(
                                            labelColor='#ffb000',
                                            titleColor='#ffb000',
                                            gridColor='#333',
                                            domainColor='#333',
                                            tickColor='#333'
                                        ).configure_legend(
                                            labelColor='#ffb000',
                                            titleColor='#ffb000'
                                        ).configure_view(
                                            stroke='#333',
                                            strokeWidth=1
                                        ).configure(background='#000000')
                                        
                                        st.altair_chart(chart, width="stretch")
                                    else:
                                        _render_info_card("No price history available for charting")
                                else:
                                    _render_info_card("Select items to view price trends")

                                # Usage map below price trends (line + points, styled like price chart)
                                st.markdown("#### 12-MONTH USAGE MAP")
                                if alt:
                                    usage_df = fetch_product_usage_history(cursor, product_item, days=365, location=None)
                                    if usage_df:
                                        dfu = pd.DataFrame(usage_df)
                                        if not dfu.empty:
                                            # Ensure no Decimal types leak into Altair
                                            from decimal import Decimal as _Decimal
                                            dfu = dfu.map(lambda v: float(v) if isinstance(v, _Decimal) else v)
                                            if "UsageQty" in dfu.columns:
                                                dfu["UsageQty"] = pd.to_numeric(dfu["UsageQty"], errors="coerce")
                                            dfu["Item"] = product_item
                                            dfu["MonthLabel"] = dfu.apply(
                                                lambda r: f"{int(r.get('Year', 0))}-{int(r.get('Month', 0)):02d}", axis=1
                                            )
                                            base = alt.Chart(dfu).encode(
                                                x=alt.X(
                                                    "MonthLabel:N",
                                                    sort=None,
                                                    title="Month",
                                                    axis=alt.Axis(labelAngle=-45, labelColor="#555", titleColor="#555"),
                                                )
                                            )
                                            bars = base.mark_bar(color="#859900", opacity=0.3).encode(
                                                y=alt.Y(
                                                    "UsageQty:Q",
                                                    title="Usage Qty",
                                                    axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"),
                                                    scale=alt.Scale(zero=True),
                                                )
                                            )
                                            line = base.mark_line(
                                                strokeWidth=2,
                                                color="#ffb000",
                                            ).encode(
                                                y=alt.Y(
                                                    "UsageQty:Q",
                                                    title="Usage Qty",
                                                    axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"),
                                                    scale=alt.Scale(zero=True),
                                                ),
                                                tooltip=["Item", "MonthName", "Year", "UsageQty"],
                                            )
                                            pts = line.mark_point(filled=True, size=70, color="#ffb000")
                                            usage_chart = (
                                                alt.layer(bars, line, pts)
                                                .resolve_scale(y="shared")
                                                .properties(height=260)
                                                .configure_axis(
                                                    gridColor="#e0e0e0",
                                                    domainColor="#cccccc",
                                                    tickColor="#cccccc",
                                                )
                                                .configure_view(stroke="#e0e0e0", strokeWidth=1)
                                                .configure(background="#ffffff")
                                        )
                                        st.altair_chart(usage_chart, width="stretch")
                                    else:
                                        _render_info_card("No usage history available for charting")
                                else:
                                    _render_info_card("No usage history available for charting")

                            with chart_col2:
                                st.markdown("#### INVENTORY RUNWAY (DAYS OF SUPPLY)")
                                
                                # Calculate runway for top items
                                runway_data = []
                                for item in df_analysis['Item'].head(10).tolist():
                                    runway = calculate_inventory_runway(cursor, item)
                                    runway_data.append({
                                        'Item': item,
                                        'Days': min(runway.get('runway_days', 0), 180),
                                        'Urgency': runway.get('urgency', 'UNKNOWN'),
                                        'Color': runway.get('color', '#839496')
                                    })
                                
                                if runway_data and alt:
                                    runway_df = pd.DataFrame(runway_data)
                                    runway_df = runway_df.sort_values('Days', ascending=False)
                                    
                                    # Horizontal bar chart for runway
                                    runway_chart = alt.Chart(runway_df).mark_bar().encode(
                                        y=alt.Y('Item:N', sort='-x', axis=alt.Axis(
                                            labelColor='#ffb000', titleColor='#ffb000'
                                        )),
                                        x=alt.X('Days:Q', title='Days of Supply', scale=alt.Scale(domain=[0, 180]), axis=alt.Axis(
                                            labelColor='#ffb000', titleColor='#ffb000', gridColor='#333'
                                        )),
                                        color=alt.Color('Urgency:N', scale=alt.Scale(
                                            domain=['CRITICAL', 'WARNING', 'OK'],
                                            range=['#dc322f', '#b58900', '#859900']
                                        ), legend=alt.Legend(title='Urgency', labelColor='#ffb000', titleColor='#ffb000')),
                                        tooltip=['Item', 'Days', 'Urgency']
                                    ).properties(
                                        height=300
                                    ).configure_axis(
                                        labelColor='#ffb000',
                                        titleColor='#ffb000',
                                        gridColor='#333',
                                        domainColor='#333',
                                        tickColor='#333'
                                    ).configure_view(
                                        stroke='#333',
                                        strokeWidth=1
                                    ).configure(background='#000000')
                                    
                                    st.altair_chart(runway_chart, width="stretch")
                                    
                                    # Add legend explanation
                                    st.markdown("""
                                    <div style="font-size: 0.8em; color: #839496;">
                                        🔴 CRITICAL: &lt;30 days | 🟡 WARNING: 30-60 days | 🟢 OK: &gt;60 days
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    _render_info_card("No runway data available for selected items")
                            
                            # 3. Seasonal Pattern Detection
                            st.markdown("---")
                            st.subheader(">> SEASONAL_PATTERNS")
                            
                            seasonal_col1, seasonal_col2, seasonal_col3 = st.columns(3)
                            
                            # Analyze seasonality for top items
                            seasonal_items = df_analysis['Item'].head(6).tolist()
                            
                            for i, item in enumerate(seasonal_items[:3]):
                                with [seasonal_col1, seasonal_col2, seasonal_col3][i]:
                                    pattern = get_seasonal_pattern(cursor, item)
                                    volatility = get_volatility_score(cursor, item)
                                    
                                    pattern_icon = "🌊" if pattern.get('has_pattern') else "➡️"
                                    vol_color = volatility.get('color', '#839496')
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; padding: 12px; border-radius: 5px; background: #050505;">
                                        <h5 style="color: #ffb000; margin: 0 0 8px 0;">{item}</h5>
                                        <p style="margin: 4px 0; font-size: 0.9em;">
                                            {pattern_icon} <strong>{pattern.get('pattern', 'Unknown')}</strong>
                                        </p>
                                        <p style="margin: 4px 0; font-size: 0.85em; color: #839496;">
                                            Peak: {pattern.get('peak_month', 'N/A')} | Low: {pattern.get('low_month', 'N/A')}
                                        </p>
                                        <p style="margin: 4px 0;">
                                            <span style="color: {vol_color};">● Volatility: {volatility.get('volatility_label', 'Unknown')}</span>
                                            <span style="color: #839496; font-size: 0.8em;"> ({volatility.get('volatility_score', 0)}%)</span>
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # 4. Market Analysis Table
                            st.markdown("---")
                            st.subheader(">> MARKET_ANALYSIS_TABLE")
                            st.dataframe(
                                df_analysis.style.map(lambda x: 'color: #859900' if x == 'Strong Buy' else 'color: #dc322f' if x == 'Wait' else '', subset=['Signal']),
                                width="stretch"
                            )
                    
                    else: # Market Monitor View
                        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
                        with f_col1:
                            # Segment Filter (Default to Raw Material)
                            segment_options = ["Raw Material", "Finished Good", "ALL"]
                            selected_segment = st.radio("MARKET_SEGMENT", segment_options, horizontal=True, label_visibility="collapsed")
                            selected_segment_for_chart = selected_segment
                        
                        # Apply Segment Filter first to populate categories correctly
                        if selected_segment != "ALL":
                            filtered_df = df_market[df_market['Segment'] == selected_segment]
                        else:
                            filtered_df = df_market.copy()
                        
                        with f_col2:
                            # Category Filter (Dynamic based on segment)
                            categories = sorted(list(set(filtered_df['Category'].dropna().unique())))
                            selected_category = st.selectbox("FILTER_BY_CATEGORY", ["ALL"] + categories)
                            
                        with f_col3:
                            # Search
                            search_term = st.text_input("SEARCH_PRODUCT", placeholder="ITEM NUMBER OR DESC...", label_visibility="collapsed")
                    
                        # If user is searching, query database directly for that item
                        if search_term and len(search_term) >= 2:
                            search_query = """
                            SELECT TOP 100
                                ITEMNMBR, 
                                ITEMDESC, 
                                ITEMTYPE,
                                ITMCLSCD,
                                STNDCOST, 
                                CURRCOST, 
                                USCATVLS_1 as Category,
                                (CURRCOST - STNDCOST) as PriceChange,
                                CASE WHEN STNDCOST > 0 THEN ((CURRCOST - STNDCOST) / STNDCOST) * 100 ELSE 0 END as PctChange
                            FROM IV00101 
                            WHERE ITEMTYPE IN (0, 1, 2)
                                AND (ITEMNMBR LIKE ? OR ITEMDESC LIKE ?)
                            ORDER BY ITEMNMBR
                            """
                            search_pattern = f'%{search_term}%'
                            cursor.execute(search_query, search_pattern, search_pattern)
                            search_columns = [column[0] for column in cursor.description]
                            search_data = cursor.fetchall()
                            search_df = pd.DataFrame.from_records(search_data, columns=search_columns)
                            
                            if not search_df.empty:
                                # Add segment classification
                                search_df['Segment'] = search_df.apply(_get_market_segment, axis=1)
                                # Apply segment filter if not ALL
                                if selected_segment != "ALL":
                                    filtered_df = search_df[search_df['Segment'] == selected_segment]
                                else:
                                    filtered_df = search_df
                            else:
                                filtered_df = pd.DataFrame()
                        else:
                            # Use pre-fetched market data
                            # Apply Segment Filter first to populate categories correctly
                            if selected_segment != "ALL":
                                filtered_df = df_market[df_market['Segment'] == selected_segment]
                            else:
                                filtered_df = df_market.copy()
                        
                        # Apply Remaining Filters
                        if not filtered_df.empty:
                            if selected_category != "ALL":
                                filtered_df = filtered_df[filtered_df['Category'].str.strip() == selected_category]
                        else:
                            filtered_df = pd.DataFrame()

                        if not filtered_df.empty:
                            # Header
                            h1, h2, h3, h4 = st.columns([2, 3, 1, 1])
                            h1.markdown("**ITEM**")
                            h2.markdown("**DESCRIPTION**")
                            h3.markdown("**COST**")
                            h4.markdown("**CHANGE**")
                            st.markdown("---")
                            
                            # Sort by % change: highest absolute change first, zeros at the bottom
                            filtered_df = filtered_df.copy()
                            filtered_df['AbsPctChange'] = filtered_df['PctChange'].abs()
                            filtered_df['IsZero'] = (filtered_df['PctChange'] == 0) | (filtered_df['PctChange'].isna())
                            filtered_df = filtered_df.sort_values(
                                by=['IsZero', 'AbsPctChange'], 
                                ascending=[True, False]  # Non-zeros first (IsZero=False), then by highest absolute change
                            )
                            
                            # Scrollable container for the list
                            with st.container(height=600):
                                for idx, row in filtered_df.iterrows():
                                    item_num = row['ITEMNMBR']
                                    item_desc = row['ITEMDESC']
                                    curr_cost = row.get('CURRCOST', 0)
                                    pct_change = row.get('PctChange', 0)
                                    
                                    # Determine color
                                    if pct_change > 0:
                                        color = "#859900" # Green
                                        arrow = "▲"
                                    elif pct_change < 0:
                                        color = "#dc322f" # Red
                                        arrow = "▼"
                                    else:
                                        color = "#839496" # Grey
                                        arrow = "▶"
                                        
                                    # Row Layout
                                    r1, r2, r3, r4 = st.columns([2, 3, 1, 1])
                                    
                                    with r1:
                                        if st.button(f"🔗 {item_num}", key=f"list_btn_{idx}_{item_num}"):
                                            st.session_state.selected_product = item_num
                                            st.session_state.current_page = "product_insights"
                                            st.rerun()
                                    
                                    with r2:
                                        st.markdown(f"<div style='padding-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{item_desc}</div>", unsafe_allow_html=True)
                                        
                                    with r3:
                                        st.markdown(f"<div style='padding-top: 5px;'>${curr_cost:.2f}</div>", unsafe_allow_html=True)
                                        
                                    with r4:
                                        st.markdown(f"<div style='color: {color}; padding-top: 5px;'>{arrow} {pct_change:+.1f}%</div>", unsafe_allow_html=True)
                                    
                                    st.markdown("<hr style='margin: 0.2em 0; opacity: 0.2;'>", unsafe_allow_html=True)
                        else:
                            if search_term:
                                st.info("No items match your search with the current filters.")
                            else:
                                st.warning("MARKET DATA OFFLINE")

            with col2:
                if not df_market.empty:
                    st.subheader(">> INVENTORY_VALUE_BY_CATEGORY")
                    volatility_rows = get_batch_volatility_scores(cursor, limit=1) # Minimal fetch to avoid error if reused elsewhere, but we don't use it here.
                        
                    # Fetch Inventory Distribution instead
                    inv_dist = get_inventory_distribution(cursor, limit=10, segment=selected_segment_for_chart)
                    
                    if inv_dist:
                        dist_df = pd.DataFrame(inv_dist)
                        
                        # Interactive Donut Chart for Portfolio Composition
                        base = alt.Chart(dist_df).encode(
                            theta=alt.Theta("Value", stack=True)
                        )
                        
                        pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                            color=alt.Color("Category", scale=alt.Scale(scheme='category20b'), legend=None),
                            order=alt.Order("Value", sort="descending"),
                            tooltip=["Category", alt.Tooltip("Value", format="$,.2f"), "ItemCount"]
                        )
                        
                        text = base.mark_text(radius=140).encode(
                            text="Category",
                            order=alt.Order("Value", sort="descending"),
                            color=alt.value("#ffb000")  # Terminal amber
                        )
                        
                        chart = (pie + text).properties(
                            height=300
                        ).configure_view(
                            strokeWidth=0
                        ).configure(
                            background='transparent'
                        )
                        
                        st.altair_chart(chart, width="stretch")
                        
                        # Add a "Total Value" metric in the middle or below
                        total_val = sum(d['Value'] for d in inv_dist)
                        st.markdown(f"#### Total Value: ${total_val:,.2f}")
                        
                    else:
                        st.info("No Inventory Value data available.")
                
                st.markdown("### >> SYSTEM_ALERTS")
                st.info("⚠️ SUPPLY CHAIN DISRUPTION DETECTED IN SECTOR 7G")
                st.info("ℹ️ NEW PRICING UPDATES AVAILABLE")
            
            # --- TIME SERIES ANALYTICS ---
            st.markdown("---")
            st.markdown("## 📈 RAW MATERIAL ANALYTICS")
            
            # Fetch time series data
            ts_data = get_raw_material_time_series(cursor, months=24)
            
            ts_col1, ts_col2 = st.columns(2)
            
            with ts_col1:
                st.markdown("### >> MONTHLY_PURCHASE_VOLUME")
                mv_df = ts_data.get('monthly_volume', pd.DataFrame())
                if not mv_df.empty:
                    mv_df = mv_df.set_index('Date')
                    mv_df['TotalSpend'] = pd.to_numeric(mv_df['TotalSpend'], errors='coerce')
                    st.area_chart(mv_df['TotalSpend'].astype(float), color='#ffb000')
                    st.caption("Total $ spent on Raw Materials per month")
                else:
                    st.info("No purchase volume data available.")
            
            with ts_col2:
                st.markdown("### >> WEIGHTED_AVG_COST_TREND")
                mc_df = ts_data.get('monthly_cost', pd.DataFrame())
                if not mc_df.empty:
                    mc_df = mc_df.set_index('Date')
                    mc_df['AvgCost'] = pd.to_numeric(mc_df['AvgCost'], errors='coerce')
                    st.line_chart(mc_df['AvgCost'].astype(float), color='#2aa198')
                    st.caption("Weighted average cost across all Raw Materials")
                else:
                    st.info("No cost trend data available.")
            
            # Cost Index Chart (multi-line)
            st.markdown("### >> COST_INDEX (TOP 5 RAW MATERIALS)")
            ci_df = ts_data.get('cost_index', pd.DataFrame())
            if not ci_df.empty:
                # Pivot for multi-line chart
                ci_pivot = ci_df.pivot(index='Date', columns='Item', values='Index')
                ci_pivot = ci_pivot.apply(pd.to_numeric, errors='coerce').astype(float)
                st.line_chart(ci_pivot)
                st.caption("Normalized cost index (base=100 at start of period). Shows relative price movements.")
            else:
                st.info("No cost index data available.")
            
            # --- PRODUCT SELECTION (Top Movers) ---
            st.markdown("---")
            st.markdown("## ⚡ TOP MOVERS (RAW MATERIALS - 1 YR CHANGE)")
        
            # Fetch Top Movers specific to Raw Materials
            df_movers = get_top_movers_raw_materials(cursor, limit=15)
            
            if not df_movers.empty:
                # Create grid of clickable product cards
                for idx, row in df_movers.iterrows():
                    item_num = row['ITEMNMBR']
                    item_desc = row['ITEMDESC']
                    curr_cost = row.get('CurrentCost', 0)
                    pct_change = row.get('PctChange', 0)
                    
                    # Color based on change
                    if pct_change > 0:
                        change_color = "#859900"  # Green
                        arrow = "▲"
                    elif pct_change < 0:
                        change_color = "#dc322f"  # Red
                        arrow = "▼"
                    else:
                        change_color = "#ffb000"  # Amber
                        arrow = "▶"
                    
                    # Create clickable card
                    card_col1, card_col2, card_col3 = st.columns([3, 2, 1])
                    
                    with card_col1:
                        # Make product number and description clickable
                        if st.button(
                            f"**{item_num}** • {item_desc[:40]}{'...' if len(str(item_desc)) > 40 else ''}",
                            key=f"select_top_{idx}_{item_num}"
                        ):
                            st.session_state.selected_product = item_num
                            st.session_state.current_page = "product_insights"  # Navigate to insights page
                            st.rerun()
                    
                    with card_col2:
                        st.markdown(f"**${curr_cost:.2f}**")
                    
                    with card_col3:
                        st.markdown(f"<span style='color:{change_color}; font-weight:bold'>{arrow} {pct_change:+.1f}%</span>", unsafe_allow_html=True)
            else:
                st.info("No significant price movements detected in Raw Materials.")
        
        # === PRODUCT INSIGHTS PAGE ===
        elif st.session_state.current_page == "product_insights" and st.session_state.selected_product:
            product_item = st.session_state.selected_product
            
            # Header with back button
            header_col1, header_col2 = st.columns([8, 1])
            with header_col1:
                st.markdown(f"# 🎯 MARKET_INTELLIGENCE: `{product_item}`")
            with header_col2:
                if st.button("← BACK", key="back_to_overview"):
                    st.session_state.current_page = "market_overview"
                    st.session_state.selected_product = None
                    st.rerun()

            # Fetch Data
            details = get_product_details(cursor, product_item)
            if not details:
                st.error("PRODUCT DATA NOT FOUND")
            else:
                # Extract Data
                category = details.get('category', 'default')
                inventory = details.get('inventory_status', {})
                price_hist = details.get('price_history', [])
                
                # Fetch External Data
                with st.spinner("● ANALYZING GLOBAL MARKETS..."):
                    ag_data = fetch_agricultural_market_data(category)
                    forecast = get_usage_forecasts(category)
                
                # --- INJECT CONTEXT FOR CHATBOT ---
                # Build a rich context summary for the Intelligence Unit
                import datetime as dt
                
                # Summarize price history
                price_summary = {}
                if price_hist:
                    hist_df = pd.DataFrame(price_hist)
                    if not hist_df.empty and 'AvgCost' in hist_df.columns:
                        hist_df['TransactionDate'] = pd.to_datetime(hist_df['TransactionDate'])
                        price_summary = {
                            'avg_cost': float(hist_df['AvgCost'].mean()),
                            'min_cost': float(hist_df['AvgCost'].min()),
                            'max_cost': float(hist_df['AvgCost'].max()),
                            'latest_cost': float(hist_df.iloc[-1]['AvgCost']) if len(hist_df) > 0 else 0,
                            'transaction_count': len(hist_df),
                            'date_range': f"{hist_df['TransactionDate'].min().strftime('%Y-%m-%d')} to {hist_df['TransactionDate'].max().strftime('%Y-%m-%d')}",
                        }
                        # Vendors
                        if 'VendorName' in hist_df.columns:
                            price_summary['vendors'] = [v for v in hist_df['VendorName'].unique().tolist() if v]
                        # Spend by month (current year)
                        if 'ExtendedCost' in hist_df.columns:
                            current_year = dt.date.today().year
                            year_df = hist_df[hist_df['TransactionDate'].dt.year == current_year]
                            if not year_df.empty:
                                monthly_spend = year_df.groupby(year_df['TransactionDate'].dt.month)['ExtendedCost'].sum().to_dict()
                                price_summary['monthly_spend_current_year'] = {calendar.month_abbr[m]: v for m, v in monthly_spend.items()}
                                price_summary['ytd_total_spend'] = float(year_df['ExtendedCost'].sum())
                
                # Summarize usage history
                # Summarize usage history
                usage_summary = {}
                usage_hist = details.get('usage_history', [])
                if usage_hist:
                    us_df = pd.DataFrame(usage_hist)
                    if not us_df.empty and 'UsageQty' in us_df.columns:
                        # Construct Date from Year/Month for filtering
                        if 'Year' in us_df.columns and 'Month' in us_df.columns:
                            us_df['Date'] = pd.to_datetime(us_df[['Year', 'Month']].assign(day=1))
                            
                            usage_summary = {
                                'total_usage_qty': float(us_df['UsageQty'].sum()),
                                'avg_usage_qty': float(us_df['UsageQty'].mean()),
                                'entries': len(us_df),
                                'usage_trend': 'increasing' if len(us_df) > 1 and us_df.iloc[-1]['UsageQty'] > us_df['UsageQty'].mean() else 'stable'
                            }
                            
                            # Monthly Usage Breakdown (Current Year)
                            current_year = dt.date.today().year
                            us_year_df = us_df[us_df['Date'].dt.year == current_year]
                            if not us_year_df.empty:
                                monthly_usage = us_year_df.groupby(us_year_df['Date'].dt.month)['UsageQty'].sum().to_dict()
                                usage_summary['monthly_usage_current_year'] = {calendar.month_abbr[m]: v for m, v in monthly_usage.items()}
                            
                            # Monthly Usage Breakdown (Last Year) for Seasonality
                            last_year = current_year - 1
                            us_last_year_df = us_df[us_df['Date'].dt.year == last_year]
                            if not us_last_year_df.empty:
                                last_year_usage = us_last_year_df.groupby(us_last_year_df['Date'].dt.month)['UsageQty'].sum().to_dict()
                                usage_summary['monthly_usage_last_year'] = {calendar.month_abbr[m]: v for m, v in last_year_usage.items()}

                burn_metrics = calculate_seasonal_burn_metrics(
                    usage_history=usage_hist or [],
                    on_hand=inventory.get('TotalOnHand', 0),
                    on_order=inventory.get('OnOrder', 0),
                    today=datetime.date.today(),
                )

                # Build chatbot context
                active_chat["context"] = {
                    'selected_item': product_item,
                    'item_description': details.get('description', ''),
                    'category': category,
                    'current_cost': inventory.get('CURRCOST', 0),
                    'standard_cost': inventory.get('STNDCOST', 0),
                    'stock_status': inventory.get('StockStatus', 'Unknown'),
                    'on_hand': inventory.get('TotalOnHand', 0),
                    'allocated': inventory.get('TotalAllocated', 0),
                    'available': inventory.get('Available', 0),
                    'on_order': inventory.get('OnOrder', 0),
                    'price_history_summary': price_summary,
                    'usage_history_summary': usage_summary,
                    'market_trend': ag_data.get('trend', 'stable'),
                    'demand_forecast': forecast.get('forecast_next_30days', 'stable'),
                    'commodity': ag_data.get('commodity', ''),
                    'price_index': ag_data.get('current_price_index', 0),
                    'volatility': ag_data.get('volatility', ''),
                }
                
                # --- TOP ROW: KPI CARDS ---
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                with kpi1:
                    st.markdown("### INTERNAL COST")
                    curr_cost = inventory.get('CURRCOST', 0)
                    st.markdown(f"## ${curr_cost:,.5f}")
                    
                with kpi2:
                    st.markdown("### STOCK STATUS")
                    status = inventory.get('StockStatus', 'Unknown')
                    color = "#859900" if status == "High" else "#dc322f" if status == "Low" else "#ffb000"
                    st.markdown(f"## <span style='color:{color}'>{status.upper()}</span>", unsafe_allow_html=True)
                    
                with kpi3:
                    st.markdown("### MARKET TREND")
                    trend = ag_data.get('trend', 'stable')
                    trend_color = "#dc322f" if trend == "increasing" else "#859900" # Cost increasing is bad usually? Or good for value? Assuming cost.
                    arrow = "▲" if trend == "increasing" else "▼"
                    st.markdown(f"## <span style='color:{trend_color}'>{arrow} {trend.upper()}</span>", unsafe_allow_html=True)

                with kpi4:
                    st.markdown("### DEMAND FORECAST")
                    demand = forecast.get('forecast_next_30days', 'stable')
                    st.markdown(f"## {demand.upper()}")

                st.markdown("---")

                # --- MIDDLE ROW: CHARTS & CONTEXT ---
                row2_col1, row2_col2 = st.columns([2, 1])
                
                with row2_col1:
                    st.markdown("### >> PRICE_HISTORY_ANALYSIS (INTERNAL)")
                    if price_hist:
                        hist_df = pd.DataFrame(price_hist)
                        if not hist_df.empty and 'TransactionDate' in hist_df.columns:
                            # Ensure proper date format
                            hist_df['TransactionDate'] = pd.to_datetime(hist_df['TransactionDate'])
                            
                            # --- SPEND ANALYTICS ---
                            if 'ExtendedCost' in hist_df.columns:
                                import datetime as dt
                                current_date = dt.date.today()
                                curr_month_name = current_date.strftime('%B')
                                
                                # Current Month Spend (This Year)
                                current_spend = hist_df[
                                    (hist_df['TransactionDate'].dt.month == current_date.month) & 
                                    (hist_df['TransactionDate'].dt.year == current_date.year)
                                ]['ExtendedCost'].sum()
                                
                                # Historical Avg Spend (Same Month, Prior Years)
                                prior_spend_df = hist_df[
                                    (hist_df['TransactionDate'].dt.month == current_date.month) & 
                                    (hist_df['TransactionDate'].dt.year < current_date.year)
                                ]
                                if not prior_spend_df.empty:
                                    avg_hist_spend = prior_spend_df.groupby(prior_spend_df['TransactionDate'].dt.year)['ExtendedCost'].sum().mean()
                                else:
                                    avg_hist_spend = 0
                                    
                                # Display Spend Metrics
                                st.markdown("#### 💰 SPEND ANALYSIS")
                                sa1, sa2 = st.columns(2)
                                sa1.metric(f"Current Spend ({curr_month_name})", f"${current_spend:,.2f}")
                                delta_val = current_spend - avg_hist_spend
                                sa2.metric(f"Avg Hist. Spend ({curr_month_name})", f"${avg_hist_spend:,.2f}", 
                                          delta=f"${delta_val:+,.2f}" if avg_hist_spend > 0 else None, 
                                          delta_color="inverse")
                                st.markdown("---")
                            
                            # Helper to render chart + usage map
                            def render_price_chart(data_df):
                                if data_df.empty:
                                    st.info("No data available.")
                                    return

                                # Aggregate by date to handle multiple transactions per day
                                # For RM: Aggregates raw receipts. For FG: Re-aggregates daily summaries (no-op mostly)
                                chart_data = data_df.groupby('TransactionDate').agg({
                                    'AvgCost': 'mean',
                                    'TransactionCount': 'sum'
                                }).reset_index()

                                # Cast to float to avoid Altair Decimal warnings
                                chart_data['AvgCost'] = chart_data['AvgCost'].astype(float)
                                chart_data['TransactionCount'] = chart_data['TransactionCount'].astype(float)

                                # Base chart with zoom/pan interaction
                                base = alt.Chart(chart_data).encode(
                                    x=alt.X('TransactionDate:T', 
                                           title='Date',
                                           axis=alt.Axis(labelAngle=-45, format='%b %Y'))
                                )
                                
                                # Cost line (left axis) - Primary metric
                                cost_line = base.mark_line(
                                    color='#ffb000',
                                    strokeWidth=3,
                                    point=alt.OverlayMarkDef(
                                        filled=True,
                                        size=80,
                                        color='#ffb000'
                                    )
                                ).encode(
                                    y=alt.Y('AvgCost:Q', 
                                           title='Average Cost ($)',
                                           scale=alt.Scale(zero=False)),
                                    tooltip=[
                                        alt.Tooltip('TransactionDate:T', title='Date', format='%Y-%m-%d'),
                                        alt.Tooltip('AvgCost:Q', title='Avg Cost', format='$,.2f'),
                                        alt.Tooltip('TransactionCount:Q', title='Transactions', format=',d')
                                    ]
                                )
                                
                                # Transaction volume bars (right axis) - Secondary metric
                                if 'TransactionCount' in chart_data.columns:
                                    volume_bars = base.mark_bar(
                                        opacity=0.3,
                                        color='#859900'
                                    ).encode(
                                        y=alt.Y('TransactionCount:Q',
                                               title='Transaction Count',
                                               axis=alt.Axis(titleColor='#859900'))
                                    )
                                    
                                    # Combine with dual axis + interactivity
                                    chart = alt.layer(
                                        volume_bars,
                                        cost_line
                                    ).resolve_scale(
                                        y='independent'  # Independent Y-axes
                                    ).properties(
                                        height=300
                                    ).interactive()
                                else:
                                    chart = cost_line.properties(height=300).interactive()
                                
                                st.altair_chart(chart, width="stretch")

                                # Usage map below price chart (12 months) - styled to match pricing chart
                                price_start = pd.to_datetime(data_df["TransactionDate"].min()).normalize() if "TransactionDate" in data_df.columns else pd.NaT
                                price_end = pd.to_datetime(data_df["TransactionDate"].max()).normalize() if "TransactionDate" in data_df.columns else pd.NaT
                                lookback_days = 365
                                if pd.notna(price_start):
                                    today = pd.Timestamp.today().normalize()
                                    lookback_days = max(365, int((today - price_start).days) + 31)

                                usage_rows = fetch_product_usage_history(cursor, product_item, days=lookback_days, location=None)
                                if usage_rows and alt:
                                    udf = pd.DataFrame(usage_rows)
                                    if not udf.empty:
                                        from decimal import Decimal as _Decimal
                                        udf = udf.map(lambda v: float(v) if isinstance(v, _Decimal) else v)
                                        if "UsageQty" in udf.columns:
                                            udf["UsageQty"] = pd.to_numeric(udf["UsageQty"], errors="coerce")
                                        udf["UsageDate"] = pd.to_datetime(
                                            udf[["Year", "Month"]].assign(Day=1), errors="coerce"
                                        )
                                        udf = udf.dropna(subset=["UsageDate"])

                                        if pd.notna(price_start) and pd.notna(price_end):
                                            udf = udf[udf["UsageDate"].between(price_start, price_end)]

                                        if udf.empty:
                                            st.info("No usage history available.")
                                            return

                                        x_scale = alt.Scale(domain=[price_start, price_end]) if pd.notna(price_start) and pd.notna(price_end) else alt.Scale()

                                        base = alt.Chart(udf).encode(
                                            x=alt.X(
                                                "UsageDate:T",
                                                title="Date",
                                                scale=x_scale,
                                                axis=alt.Axis(labelAngle=-45, labelColor="#555", titleColor="#555", format="%b %Y"),
                                            )
                                        )
                                        bars = base.mark_bar(color="#859900", opacity=0.35).encode(
                                            y=alt.Y(
                                                "UsageQty:Q",
                                                title="Usage Qty",
                                                axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"),
                                                scale=alt.Scale(zero=True),
                                            )
                                        )
                                        line = base.mark_line(
                                            strokeWidth=3,
                                            color="#ffb000",
                                        ).encode(
                                            y=alt.Y(
                                                "UsageQty:Q",
                                                title="Usage Qty",
                                                axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"),
                                                scale=alt.Scale(zero=True),
                                            ),
                                            tooltip=[
                                                alt.Tooltip("UsageDate:T", title="Date", format="%b %Y"),
                                                alt.Tooltip("UsageQty:Q", title="Usage Qty", format=",.0f"),
                                            ],
                                        )
                                        pts = line.mark_point(filled=True, size=80, color="#ffb000")
                                        usage_chart = (
                                            alt.layer(bars, line, pts)
                                            .resolve_scale(y="shared")
                                            .properties(height=260, title="12-Month Usage")
                                            .configure_axis(
                                                gridColor="#e0e0e0",
                                                domainColor="#cccccc",
                                                tickColor="#cccccc",
                                            )
                                            .configure_title(color="#555")
                                            .configure_view(stroke="#e0e0e0", strokeWidth=1)
                                            .configure(background="#ffffff")
                                        )
                                        st.altair_chart(usage_chart, width="stretch")
                                    else:
                                        st.info("No usage history available.")
                                else:
                                    st.info("No usage history available.")

                            # Vendor Tabs Logic
                            if 'VendorName' in hist_df.columns:
                                # Get unique vendors (exclude empty/None)
                                vendors = [v for v in sorted(hist_df['VendorName'].unique().tolist()) if v and str(v).strip()]
                                
                                if vendors:
                                    # Create tabs: ALL + Vendors
                                    tabs = st.tabs(["ALL"] + vendors)
                                    
                                    # Render for ALL
                                    with tabs[0]:
                                        render_price_chart(hist_df)
                                        
                                    # Render for each Vendor
                                    for i, vendor in enumerate(vendors):
                                        with tabs[i+1]:
                                            vendor_df = hist_df[hist_df['VendorName'] == vendor]
                                            render_price_chart(vendor_df)
                                else:
                                    render_price_chart(hist_df)
                            else:
                                render_price_chart(hist_df)
                    else:
                        st.info("NO HISTORICAL DATA AVAILABLE")

                with row2_col2:
                    st.markdown("### >> EXTERNAL_MARKET_CONTEXT")
                    st.markdown(f"""
                    <div class="terminal-card">
                        <div style="color: #859900; font-weight: bold; margin-bottom: 10px;">COMMODITY: {ag_data.get('commodity', '').upper()}</div>
                        <div>PRICE INDEX: {ag_data.get('current_price_index', 0):.2f}</div>
                        <div>VOLATILITY: {ag_data.get('volatility', '').upper()}</div>
                        <br>
                        <div style="color: #b58900; font-weight: bold; margin-bottom: 10px;">DEMAND SIGNALS</div>
                        <div>CURRENT SCORE: {forecast.get('demand_score', 0)}/100</div>
                        <div>SEASONAL PATTERN: {forecast.get('seasonal_pattern', '').upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### >> AI_INSIGHT")
                    st.info(f"Market analysis suggests {trend} costs for {category} products. Recommended action: {'STOCK UP' if trend == 'increasing' else 'MONITOR'}.")

                # --- BOTTOM ROW: INVENTORY DETAILS ---
                st.markdown("### >> INVENTORY_LOGISTICS")
                inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
                with inv_col1:
                    st.metric("ON HAND", f"{inventory.get('TotalOnHand', 0):,.0f}")
                with inv_col2:
                    st.metric("ALLOCATED", f"{inventory.get('TotalAllocated', 0):,.0f}")
                with inv_col3:
                    st.metric("AVAILABLE", f"{inventory.get('Available', 0):,.0f}")
                with inv_col4:
                    st.metric("ON ORDER", f"{inventory.get('OnOrder', 0):,.0f}")

                burn_col1, burn_col2, burn_col3 = st.columns(3)
                seasonal_burn = burn_metrics.get('seasonal_burn_rate', 0) if burn_metrics else 0
                seasonal_factor = burn_metrics.get('seasonal_factor', 1) if burn_metrics else 1
                decayed_daily_usage = burn_metrics.get('decayed_daily_usage', 0) if burn_metrics else 0
                base_daily_usage = burn_metrics.get('avg_daily_usage', 0) if burn_metrics else 0
                coverage_days = burn_metrics.get('days_of_coverage') if burn_metrics else None
                coverage_display = "N/A" if coverage_days is None else f"{coverage_days:,.1f}"
                available_stock = burn_metrics.get('available_stock')
                if available_stock is None:
                    available_stock = float(inventory.get('TotalOnHand', 0) or 0) + float(inventory.get('OnOrder', 0) or 0)

                with burn_col1:
                    st.metric("BURN RATE (SEASONAL)", f"{seasonal_burn:,.1f}/day", delta=f"{seasonal_factor:.2f}x seasonality")
                with burn_col2:
                    st.metric("AVG USAGE/DAY", f"{decayed_daily_usage:,.1f}", delta=f"{base_daily_usage:,.1f} raw avg")
                with burn_col3:
                    st.metric("DAYS OF USAGE LEFT", coverage_display, delta=f"Stock: {available_stock:,.0f}")
                st.caption("Burn rate uses on-hand + on-order, weighted for current season with decayed material usage.")


        # --- SIDEBAR CHAT INTERFACE ---
        with st.sidebar:
            st.markdown("## >> INTELLIGENCE_UNIT")

            if st.session_state.current_page == "product_insights":
                if st.button("← BACK TO DASHBOARD", key="sidebar_back"):
                    st.session_state.current_page = "market_overview"
                    st.session_state.selected_product = None
                    st.rerun()
                st.markdown("---")
            
            # Chat History
            chat_container = st.container(height=600)
            with chat_container:
                for message in active_chat["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message.get("sql"):
                            st.code(message["sql"], language="sql")
                        if isinstance(message.get("df"), pd.DataFrame) and not message["df"].empty:
                            st.dataframe(message["df"], hide_index=True)
            
            # Chat Input
            if prompt := st.chat_input("QUERY INTELLIGENCE UNIT..."):
                active_chat["messages"].append({"role": "user", "content": prompt})
                with chat_container:
                    st.chat_message("user").markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("PROCESSING..."):
                            today = datetime.date.today()
                            history_hint = summarize_chat_history(active_chat["messages"])
                            response = handle_question(cursor, prompt, today, active_chat["context"], history_hint)
                            
                            # Process Response
                            if response.get("error"):
                                st.error(response["error"])
                            else:
                                content = response.get("insights", {}).get("summary", "DATA RETRIEVED.")
                                st.markdown(content)
                            
                            if sql := response.get("sql"):
                                st.code(sql, language="sql")
                                
                            df_result = pd.DataFrame(response.get("data", []))
                            if not df_result.empty:
                                st.dataframe(df_result, hide_index=True)
                                
                            # Save to history
                            active_chat["messages"].append({
                                "role": "assistant",
                                "content": content,
                                "sql": sql,
                                "df": df_result
                            })

            # Feedback Form
            _render_feedback_form(active_chat["id"], active_chat["messages"])

except Exception as e:
    st.error(f"SYSTEM FAILURE: {e}")
