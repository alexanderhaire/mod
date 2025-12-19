"""A Streamlit chatbot application that translates natural language questions into SQL queries
and displays the results."""
import calendar
import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pyodbc
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
except ImportError:
    alt = None
from auth import authenticate_user, authenticate_vendor, authenticate_broker, ensure_user_store, is_admin, register_user
from vendor_portal import render_vendor_portal
from broker_portal import render_broker_portal
from procurement_optimizer import render_procurement_cockpit
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
    fetch_product_price_history,
    fetch_product_usage_history,
    calculate_inventory_runway,
    calculate_seasonal_burn_metrics,
    recommend_optimal_buy_window,
    get_volatility_score,
    get_batch_volatility_scores,
    get_seasonal_pattern,
    get_priority_raw_materials,
    get_items_needing_attention,
    get_top_movers_raw_materials,
    get_raw_material_time_series,
    get_raw_material_time_series,
    get_inventory_distribution,
    calculate_hedge_metrics,
    find_optimal_hedging_asset,
    fetch_product_inventory_trends,
    simulate_portfolio_variance,
    calculate_black_scholes_put,
    optimize_capital_allocation,
    validate_price_history_quality
)
from ml_engine import OnlineLinearRegressor, PortfolioOptimizer
from portfolio_manager import PortfolioManager
import fifo_tracker
from wizard_ui import WizardFlow
from external_data import (
    get_market_context,
    fetch_agricultural_market_data,
    fetch_market_data_pool,
    get_usage_forecasts
)
from constants import FUTURES_UNIVERSE
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



def verify_vendor_identity(user_input):
    """Check if the input matches a valid Vendor ID or Name in PM00200."""
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Check ID match first
        cursor.execute("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDORID = ?", user_input)
        row = cursor.fetchone()
        if row:
            return True, row.VENDORID.strip(), row.VENDNAME.strip()
            
        # Check Name match (fuzzy or exact)
        cursor.execute("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDNAME LIKE ?", f"%{user_input}%")
        row = cursor.fetchone()
        if row:
            return True, row.VENDORID.strip(), row.VENDNAME.strip()
            
        return False, None, None
    except Exception as e:
        LOGGER.error(f"Vendor verification failed: {e}")
        return False, None, None

def _render_auth_gate() -> None:
    """Render sign-in and sign-up forms; halt the app until authenticated."""
    st.subheader("Welcome")
    st.markdown("Sign in to use the data copilot. Create an account if you do not have one.")

    employee_tab, vendor_tab, broker_tab, signup_tab = st.tabs(["CDI Employee login", "Vendor Login", "Freight broker login", "Sign up"])

    with employee_tab:
        login_user = st.text_input("Username", key="login_username")
        login_pass = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign in"):
            ok, message = authenticate_user(login_user, login_pass)
            if ok:
                st.session_state.user = login_user.strip()
                st.session_state.is_admin = is_admin(st.session_state.user)
                st.session_state.is_vendor = False
                st.session_state.is_broker = False
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    with vendor_tab:
        st.caption("Restricted area for approved vendors.")
        v_user = st.text_input("Vendor ID", key="vendor_user")
        v_pass = st.text_input("Access Key", type="password", key="vendor_pass")
        if st.button("Vendor Sign In"):
            ok, message, linked_id = authenticate_vendor(v_user, v_pass)
            if ok:
                # If account has a linked ID, use it directly (Verified Identity)
                if linked_id:
                    st.session_state.user = linked_id
                    st.session_state.user_name = v_user # Fallback name
                    # Try to fetch real name
                    _, _, real_name = verify_vendor_identity(linked_id)
                    if real_name:
                        st.session_state.user_name = real_name
                        
                    st.session_state.is_admin = False
                    st.session_state.is_vendor = True
                    st.session_state.is_broker = False
                    st.success(f"Welcome, {st.session_state.user_name}!")
                    st.rerun()
                # Legacy Fallback or New User without Link
                else:
                    # VERIFY IDENTITY
                    is_valid, clean_id, clean_name = verify_vendor_identity(v_user)
                    if is_valid:
                        st.session_state.user = clean_id
                        st.session_state.user_name = clean_name # Store friendly name
                        st.session_state.is_admin = False
                        st.session_state.is_vendor = True
                        st.session_state.is_broker = False
                        st.success(f"Welcome, {clean_name}!")
                        st.rerun()
                    else:
                        st.error("Authentication successful, but Vendor ID not found in Master File (PM00200).")
            else:
                st.error(message)

    with broker_tab:
        st.caption("Freight Operations Login")
        b_user = st.text_input("Broker ID", key="broker_user")
        b_pass = st.text_input("Broker Key", type="password", key="broker_pass")
        if st.button("Broker Sign In"):
            ok, message, linked_id = authenticate_broker(b_user, b_pass)
            if ok:
                # If account has a linked ID, use it directly
                if linked_id:
                    st.session_state.user = linked_id
                    st.session_state.user_name = b_user
                    _, _, real_name = verify_vendor_identity(linked_id)
                    if real_name:
                        st.session_state.user_name = real_name

                    st.session_state.is_admin = False
                    st.session_state.is_vendor = False
                    st.session_state.is_broker = True
                    st.success(f"Welcome, {st.session_state.user_name}!")
                    st.rerun()
                # Legacy Fallback
                else:
                    # VERIFY IDENTITY
                    is_valid, clean_id, clean_name = verify_vendor_identity(b_user)
                    if is_valid:
                        st.session_state.user = clean_id
                        st.session_state.user_name = clean_name
                        st.session_state.is_admin = False
                        st.session_state.is_vendor = False
                        st.session_state.is_broker = True
                        st.success(f"Welcome, {clean_name}!")
                        st.rerun()
                    else:
                        st.error("Authentication successful, but Broker ID not found in Vendor Master.")
            else:
                st.error(message)

    with signup_tab:
        st.subheader("Create New Account")
        # Move radio outside form to allow immediate rerun/UI update when changed
        account_type = st.radio("I am a:", ["CDI Employee", "Vendor", "Freight Broker"])
        
        # Identity Link Selection
        linked_identity_id = None
        if account_type in ["Vendor", "Freight Broker"]:
            st.markdown("#### Link to Existing Company")
            st.caption("Select your company to automatically verified access. If new, select 'New Entity'.")
            
            # Fetch active vendors/brokers for dropdown
            conn_str, _, _, _ = build_connection_string()
            try:
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                # Fetch Name and ID
                cursor.execute("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDSTTS = 1 ORDER BY VENDNAME")
                rows = cursor.fetchall()
                company_map = {f"{r.VENDNAME.strip()} ({r.VENDORID.strip()})": r.VENDORID.strip() for r in rows}
                
                # Add "New" option
                options = ["(New Entity / Not Listed)"] + list(company_map.keys())
                selection = st.selectbox("Select Your Company", options)
                
                if selection != "(New Entity / Not Listed)":
                    linked_identity_id = company_map[selection]
                    st.success(f"Account will be linked to: {linked_identity_id}")
            except Exception as e:
                st.warning("Could not load company list. Proceeding as new entity.")

        with st.form("signup_form", clear_on_submit=False):
            new_user = st.text_input("Username", key="signup_username")
            new_pass = st.text_input("Password", type="password", key="signup_password")
            confirm_pass = st.text_input("Confirm password", type="password", key="signup_confirm")
            
            access_key = ""
            if account_type == "CDI Employee":
                access_key = st.text_input("CDI Employee Access Key", type="password", help="Required for employee registration")

            submitted = st.form_submit_button("Create account")
            
            if submitted:
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 8:
                    st.error("Use at least 8 characters for your password.")
                else:
                    # Role validation logic
                    is_vendor = (account_type == "Vendor")
                    is_broker = (account_type == "Freight Broker")
                    
                    if account_type == "CDI Employee":
                        if access_key != "CDI2025":
                            st.error("Invalid CDI Employee Access Key.")
                        else:
                            ok, message = register_user(new_user, new_pass, is_vendor=False, is_broker=False)
                            if ok:
                                st.success(message)
                            else:
                                st.error(message)
                    else:
                        # Vendors and Brokers register freely (or per business logic, but for now open)
                        ok, message = register_user(new_user, new_pass, is_vendor=is_vendor, is_broker=is_broker, linked_id=linked_identity_id)
                        if ok:
                            st.success(f"Account created for {account_type}. pending approval.") # Optional message enhancement
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


def _add_margin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add dollar and percent margin columns comparing standard vs current cost."""
    if df is None or df.empty:
        return df

    df = df.copy()
    std_cost = pd.to_numeric(df.get("STNDCOST"), errors="coerce")
    cur_cost = pd.to_numeric(df.get("CURRCOST"), errors="coerce")

    df["STNDCOST"] = std_cost
    df["CURRCOST"] = cur_cost
    df["Margin"] = (std_cost - cur_cost).fillna(0)
    df["MarginPct"] = (
        df["Margin"]
        .div(std_cost.replace({0: pd.NA}))
        .mul(100)
        .fillna(0)
    )
    return df


def _buy_calendar_cache_key(df_priority: pd.DataFrame, today: datetime.date) -> str:
    """Generate a stable cache key for the buy calendar so chat reruns do not rebuild it unnecessarily."""
    if df_priority.empty:
        return f"{today.isoformat()}-empty"

    cols = [col for col in ("ITEMNMBR", "ITEMDESC", "CURRCOST", "Segment") if col in df_priority.columns]
    snapshot = df_priority[cols].copy() if cols else df_priority.copy()
    if "ITEMNMBR" in snapshot.columns:
        snapshot = snapshot.sort_values("ITEMNMBR")
    snapshot = snapshot.fillna("")
    digest_source = snapshot.to_json(date_format="iso", orient="split", double_precision=6)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()
    return f"{today.isoformat()}-{digest}"


def _build_buy_calendar(cursor: pyodbc.Cursor, df_priority: pd.DataFrame, today: datetime.date, progress_cb=None) -> pd.DataFrame:
    """Build the buy calendar DataFrame, optionally updating a progress callback as items are scheduled."""
    if df_priority.empty:
        return pd.DataFrame()

    calendar_rows = []
    total_items = len(df_priority)

    for i, (_, row) in enumerate(df_priority.iterrows()):
        item_num = row["ITEMNMBR"]
        item_desc = row["ITEMDESC"]

        runway = calculate_inventory_runway(cursor, item_num)
        usage_hist = fetch_product_usage_history(cursor, item_num, days=365)
        if not usage_hist:
            usage_hist = fetch_product_usage_history(cursor, item_num, days=365, location=None)

        burn_metrics = calculate_seasonal_burn_metrics(
            usage_history=usage_hist or [],
            on_hand=runway.get("on_hand", 0),
            on_order=runway.get("on_order", 0),
            today=today,
        )

        price_hist = fetch_product_price_history(cursor, item_num, days=730)

        optimal_buy = recommend_optimal_buy_window(
            price_history=price_hist,
            usage_history=usage_hist or [],
            coverage_days=burn_metrics.get("days_of_coverage") if burn_metrics else None,
            available_stock=burn_metrics.get("available_stock") if burn_metrics else runway.get("available"),
            on_order=runway.get("on_order"),
            today=today,
        )

        coverage_days = burn_metrics.get("days_of_coverage") if burn_metrics else None
        runway_days = runway.get("runway_days")
        fallback_days = float(coverage_days) if coverage_days is not None else float(runway_days or 0)

        if optimal_buy.get("status") == "ok":
            buy_in_days = int(max(0, optimal_buy.get("days_from_now", 0)))
            latest_safe = int(max(0, optimal_buy.get("latest_safe_day", buy_in_days)))
        else:
            buy_in_days = int(max(0, round(fallback_days - 7))) if fallback_days else 0
            latest_safe = int(max(buy_in_days, fallback_days)) if fallback_days else buy_in_days

        buy_date = today + datetime.timedelta(days=buy_in_days)
        latest_date = today + datetime.timedelta(days=latest_safe)

        calendar_rows.append(
            {
                "Item": item_num,
                "Description": item_desc,
                "BuyDate": buy_date,
                "LatestSafe": latest_date,
                "RunwayDays": runway_days,
                "Urgency": runway.get("urgency", "UNKNOWN"),
                "CoverageDays": coverage_days,
                "PriceSignal": optimal_buy.get("trend_direction", "flat"),
                "EstPrice": optimal_buy.get("expected_price", float(row.get("CURRCOST", 0) or 0)),
                "PriceDelta": optimal_buy.get("price_delta", 0.0),
                "Confidence": optimal_buy.get("confidence", 0.0),
                "Reason": optimal_buy.get("reason", "Scheduled using coverage and price trend."),
            }
        )

        if progress_cb and total_items:
            progress_cb(i + 1, total_items, item_num)

    cal_df = pd.DataFrame(calendar_rows)
    if cal_df.empty:
        return cal_df

    cal_df["BuyDate"] = pd.to_datetime(cal_df["BuyDate"])
    cal_df["LatestSafe"] = pd.to_datetime(cal_df["LatestSafe"])
    cal_df["RunwayDays"] = pd.to_numeric(cal_df["RunwayDays"], errors="coerce").fillna(0)
    cal_df["CoverageDays"] = pd.to_numeric(cal_df["CoverageDays"], errors="coerce")
    cal_df["ConfidencePct"] = (pd.to_numeric(cal_df["Confidence"], errors="coerce").fillna(0) * 100).clip(lower=0, upper=100)
    cal_df["Week"] = cal_df["BuyDate"].dt.to_period("W").apply(lambda r: r.start_time)
    cal_df = cal_df.sort_values(["BuyDate", "Urgency"])
    return cal_df

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

# Cache for the Buy Calendar so chat reruns don't trigger a rebuild
if "buy_calendar_cache" not in st.session_state:
    st.session_state.buy_calendar_cache = {}

# --- MAIN DASHBOARD (MARKET MONITOR) ---
conn_str, server, db, auth = build_connection_string()
try:

    with pyodbc.connect(conn_str, autocommit=True) as conn:
        cursor = conn.cursor()
        
        # Vendor Portal Redirection (Now with access to cursor)
        if st.session_state.get("is_vendor"):
            render_vendor_portal(cursor)
            st.stop()
            
        # Broker Portal Redirection
        if st.session_state.get("is_broker"):
            render_broker_portal(cursor)
            st.stop()
        
        # Fetch market data (needed for both pages)
        df_market = _add_margin_metrics(_fetch_market_data(cursor))
        
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
                        series_col = section.get("series")
                        
                        if x_col and y_col and x_col in df_report.columns and y_col in df_report.columns:
                            if series_col and series_col in df_report.columns:
                                # Use Altair for multi-series/scatter
                                c = alt.Chart(df_report).mark_circle(size=100).encode(
                                    x=alt.X(x_col, scale=alt.Scale(zero=False)),
                                    y=alt.Y(y_col, scale=alt.Scale(zero=False)),
                                    color=series_col,
                                    tooltip=[x_col, y_col, series_col]
                                ).interactive()
                                st.altair_chart(c, use_container_width=True)
                            else:
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
                    curr_cost = float(row.get('CURRCOST', 0) or 0)
                    margin = float(row.get('Margin', 0) or 0)
                    margin_pct = float(row.get('MarginPct', 0) or 0)
                    ticker_items.append(
                        f"{row.get('ITEMNMBR', '???')} ${curr_cost:.2f} margin {margin:+.2f} ({margin_pct:+.1f}%)"
                    )
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
                    view_mode = st.radio(
                        "View Mode",
                        ["Market Monitor", "Command Center", "FIFO Tracker", "Buy Calendar", "Hedge Optimizer"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key="view_mode_radio",
                    )
                    
                    if view_mode == "Command Center":
                        # Updated to Command Center
                        render_procurement_cockpit(cursor)
                    
                    elif view_mode == "Buy Calendar":
                        st.markdown("---")
                        st.markdown("### >> BUY_CALENDAR")
                        st.caption("Calendar of recommended buys using coverage, runway, and the optimal price window logic.")

                        # Initialize wizard for Buy Calendar
                        buy_wizard = WizardFlow(
                            "buy_calendar",
                            steps=["Configure", "Build", "Results"]
                        )
                        
                        # Render step progress
                        buy_wizard.render_progress()
                        
                        # Fetch raw materials (used across steps)
                        df_priority = get_priority_raw_materials(cursor, limit=5000, require_purchase_history=True)
                        
                        if not df_priority.empty:
                            if 'ITMCLSCD' in df_priority.columns:
                                df_priority['Segment'] = df_priority.apply(_get_market_segment, axis=1)
                            elif 'Segment' not in df_priority.columns:
                                df_priority['Segment'] = "Raw Material"
                            df_priority['ITEMNMBR'] = df_priority['ITEMNMBR'].astype(str).str.strip()
                            df_priority = df_priority[df_priority['Segment'] == "Raw Material"]

                        if df_priority.empty:
                            st.info("No active raw materials found to schedule buy windows.")
                        else:
                            # === STEP 1: CONFIGURE ===
                            if buy_wizard.current_step == 0:
                                st.markdown("#### ⚙️ Configure Buy Calendar")
                                
                                # Item limit selection
                                item_count = len(df_priority)
                                item_limit = st.slider(
                                    "Items to Analyze",
                                    min_value=10,
                                    max_value=min(500, item_count),
                                    value=min(100, item_count),
                                    step=10,
                                    help="More items = longer build time"
                                )
                                buy_wizard.set_data("item_limit", item_limit)
                                
                                # Urgency filter
                                urgency_filter = st.multiselect(
                                    "Show Urgency Levels",
                                    options=["CRITICAL", "WARNING", "OK"],
                                    default=["CRITICAL", "WARNING", "OK"]
                                )
                                buy_wizard.set_data("urgency_filter", urgency_filter)
                                
                                st.markdown("---")
                                st.markdown("**Preview:**")
                                st.markdown(f"- Analyzing **{item_limit}** raw materials")
                                st.markdown(f"- Showing urgency: **{', '.join(urgency_filter) if urgency_filter else 'All'}**")
                                
                                buy_wizard.render_navigation(next_label="Build Calendar →")
                            
                            # === STEP 2: BUILD CALENDAR ===
                            elif buy_wizard.current_step == 1:
                                st.markdown("#### 🔄 Building Buy Calendar...")
                                
                                item_limit = buy_wizard.data.get("item_limit", 100)
                                urgency_filter = buy_wizard.data.get("urgency_filter", ["CRITICAL", "WARNING", "OK"])
                                
                                today = datetime.date.today()
                                df_limited = df_priority.head(item_limit)
                                
                                progress_bar = st.progress(0, text="Building buy calendar...")
                                
                                def _update_progress(done, total, item):
                                    progress_bar.progress(done / total, text=f"Scheduling {item}...")
                                
                                cal_df = _build_buy_calendar(cursor, df_limited, today, progress_cb=_update_progress)
                                progress_bar.empty()
                                
                                # Apply urgency filter
                                if cal_df is not None and not cal_df.empty and urgency_filter:
                                    cal_df = cal_df[cal_df['Urgency'].str.upper().isin([u.upper() for u in urgency_filter])]
                                
                                # Store results
                                buy_wizard.set_data("cal_df", cal_df)
                                buy_wizard.set_data("build_complete", True)
                                buy_wizard.set_data("built_at", datetime.datetime.utcnow().isoformat())
                                
                                if cal_df is not None and not cal_df.empty:
                                    st.success(f"✅ Built schedule with **{len(cal_df)}** buy recommendations!")
                                else:
                                    st.warning("No buy windows could be scheduled.")
                                
                                buy_wizard.render_navigation(next_label="View Results →")
                            
                            # === STEP 3: VIEW RESULTS ===
                            elif buy_wizard.current_step == 2:
                                cal_df = buy_wizard.data.get("cal_df")
                                built_at = buy_wizard.data.get("built_at", "")
                                
                                # Header with actions
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    if built_at:
                                        st.caption(f"Built at {built_at} UTC")
                                with col2:
                                    if st.button("🔄 Rebuild", key="buy_cal_restart"):
                                        buy_wizard.reset()
                                        st.rerun()
                                
                                if cal_df is None or cal_df.empty:
                                    st.info("No buy windows available. Adjust filters and rebuild.")
                                else:
                                    # CSV Export button
                                    csv_data = cal_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download CSV",
                                        data=csv_data,
                                        file_name=f"buy_calendar_{datetime.date.today()}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    st.markdown("#### UPCOMING BUY SCHEDULE")
                                    
                                    urgency_colors = {
                                        'CRITICAL': '#dc322f',
                                        'WARNING': '#b58900',
                                        'OK': '#859900'
                                    }
                                    
                                    for week_start, group in cal_df.groupby('Week'):
                                        week_label = pd.to_datetime(week_start).strftime("Week of %b %d")
                                        st.markdown(f"**{week_label}**")
                                        
                                        for _, rec in group.iterrows():
                                            color = urgency_colors.get(str(rec['Urgency']).upper(), '#839496')
                                            st.markdown(f"""
                                            <div style="border-left: 4px solid {color}; padding: 8px 12px; margin-bottom: 8px; background: #0a0a0a;">
                                                <div style="color:#ffb000; font-weight:bold;">{rec['Item']} · {rec['BuyDate'].strftime('%b %d')}</div>
                                                <div style="color:#839496; font-size:0.9em;">{rec['Description']}</div>
                                                <div style="color:{color}; font-size:0.85em;">Runway {rec['RunwayDays']:.1f}d | Latest safe {rec['LatestSafe'].strftime('%b %d')}</div>
                                                <div style="color:#839496; font-size:0.85em;">Price window: {rec['PriceSignal']} ${float(rec['EstPrice']):,.2f} ({float(rec['PriceDelta']):+,.2f}) · Confidence {rec['ConfidencePct']:.0f}%</div>
                                                <div style="color:#586e75; font-size:0.8em;">{rec['Reason']}</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                    st.markdown("#### FULL BUY LIST")
                                    display_df = cal_df[['BuyDate', 'LatestSafe', 'Item', 'Description', 'Urgency', 'RunwayDays', 'CoverageDays', 'PriceSignal', 'EstPrice', 'PriceDelta', 'ConfidencePct']].copy()
                                    display_df['BuyDate'] = display_df['BuyDate'].dt.strftime('%b %d')
                                    display_df['LatestSafe'] = display_df['LatestSafe'].dt.strftime('%b %d')
                                    display_df['RunwayDays'] = display_df['RunwayDays'].round(1)
                                    display_df['CoverageDays'] = display_df['CoverageDays'].round(1)
                                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                    
                    elif view_mode == "Hedge Optimizer":
                        st.markdown("---")
                        st.markdown("### >> HEDGE_OPTIMIZER_BOARD")
                        st.caption("Identify opportunities to improve Sharpe Ratio by hedging top raw material positions.")

                        # Initialize wizard for Hedge Optimizer
                        hedge_wizard = WizardFlow(
                            "hedge_optimizer",
                            steps=["Select Scan", "Configure", "Analyze", "Results"]
                        )
                        
                        # Render step progress
                        hedge_wizard.render_progress()
                        
                        # Fetch raw materials once (used across steps)
                        # Relaxed constraint: Includes items with usage even if not bought recently
                        df_target = get_priority_raw_materials(cursor, limit=5000, require_purchase_history=False)
                        
                        if not df_target.empty:
                            if 'ITMCLSCD' in df_target.columns:
                                df_target['Segment'] = df_target.apply(_get_market_segment, axis=1)
                            elif 'Segment' not in df_target.columns:
                                df_target['Segment'] = "Raw Material"
                            df_target['ITEMNMBR'] = df_target['ITEMNMBR'].astype(str).str.strip()
                            df_target = df_target[df_target['Segment'] == "Raw Material"]
                        
                        if df_target.empty:
                            st.info("No active raw materials found for analysis.")
                        else:
                            # === STEP 1: SELECT SCAN TYPE ===
                            if hedge_wizard.current_step == 0:
                                st.markdown("#### 🎯 Choose Your Analysis Mode")
                                st.markdown("Select how you want to analyze hedging opportunities:")
                                
                                scan_type = st.radio(
                                    "Scan Type",
                                    options=["market_scan", "portfolio_scan", "pure_allocator"],
                                    format_func=lambda x: {
                                        "market_scan": "📊 Market Scan — Broad opportunity discovery",
                                        "portfolio_scan": "🏛️ $10,000 Portfolio — Optimize current inventory",
                                        "pure_allocator": "🧪 Pure Allocator — Theoretical best opportunities"
                                    }.get(x, x),
                                    label_visibility="collapsed",
                                    key="hedge_scan_type"
                                )
                                hedge_wizard.set_data("scan_type", scan_type)
                                
                                # Description based on selection
                                descriptions = {
                                    "market_scan": "Scans your top 50 raw materials to find hedging opportunities. Best for **discovering** what's available.",
                                    "portfolio_scan": "Focuses on your existing inventory and allocates a $10,000 budget for maximum Sharpe ratio improvement.",
                                    "pure_allocator": "Ignores current stock levels. Finds the mathematically optimal hedges across 150 items."
                                }
                                st.info(descriptions.get(scan_type, ""))
                                
                                hedge_wizard.render_navigation(next_label="Configure →")
                            
                            # === STEP 2: CONFIGURE PARAMETERS ===
                            elif hedge_wizard.current_step == 1:
                                st.markdown("#### ⚙️ Configure Analysis Parameters")
                                
                                scan_type = hedge_wizard.data.get("scan_type", "market_scan")
                                
                                # Pure Allocator uses ALL items with quality validation (no limit)
                                if scan_type == "pure_allocator":
                                    st.info("🧪 **Pure Allocator Mode**: Will analyze ALL items with reliable price history (quality-validated).")
                                    item_limit = len(df_target)
                                    hedge_wizard.set_data("item_limit", item_limit)  # No limit
                                    hedge_wizard.set_data("use_quality_filter", True)
                                else:
                                    # Item limit based on scan type
                                    default_limit = {"market_scan": 50, "portfolio_scan": 100}.get(scan_type, 50)
                                    
                                    item_limit = st.slider(
                                        "Items to Analyze",
                                        min_value=10,
                                        max_value=min(200, len(df_target)),
                                        value=min(default_limit, len(df_target)),
                                        step=10,
                                        help="More items = longer analysis time but more opportunities"
                                    )
                                    hedge_wizard.set_data("item_limit", item_limit)
                                    hedge_wizard.set_data("use_quality_filter", False)
                                
                                # Sharpe threshold
                                default_threshold = 2.0 if scan_type in ["portfolio_scan", "pure_allocator"] else 0.0
                                sharpe_threshold = st.slider(
                                    "Minimum Sharpe Gain Threshold",
                                    min_value=0.0,
                                    max_value=3.0,
                                    value=default_threshold,
                                    step=0.1,
                                    help="Higher threshold = fewer but higher-quality opportunities"
                                )
                                hedge_wizard.set_data("sharpe_threshold", sharpe_threshold)
                                
                                # Budget (for portfolio modes)
                                if scan_type in ["portfolio_scan", "pure_allocator"]:
                                    capital = st.number_input(
                                        "Investment Budget ($)",
                                        min_value=1000,
                                        max_value=100000,
                                        value=10000,
                                        step=1000
                                    )
                                    hedge_wizard.set_data("capital", capital)

                                    # --- PORTFOLIO MANAGER INTEGRATION ---
                                    pm = PortfolioManager()
                                    p_summary = pm.get_portfolio_summary()
                                    
                                    st.markdown("### 💰 Capital Management")
                                    with st.expander(f"Current Portfolio Value: ${p_summary['Total Value']:,.2f}", expanded=False):
                                        st.metric("Available Cash", f"${p_summary['Cash']:,.2f}")
                                        st.metric("Invested Assets", f"${p_summary['Invested']:,.2f}")
                                        
                                        dep_col1, dep_col2 = st.columns([2, 1])
                                        with dep_col1:
                                            deposit_amt = st.number_input("Deposit Amount ($)", min_value=0.0, step=1000.0, key="pm_deposit")
                                        with dep_col2:
                                            if st.button("Add Funds"):
                                                if pm.deposit_capital(deposit_amt):
                                                    st.success(f"Deposited ${deposit_amt:,.2f}!")
                                                    st.rerun()
                                        
                                        if st.button("Use Available Cash for Budget"):
                                            # Update session budget to match available cash
                                             hedge_wizard.set_data("capital", float(p_summary['Cash']))
                                             st.rerun()
                                
                                st.markdown("---")
                                st.markdown("**Summary:**")
                                st.markdown(f"- Analyzing **{item_limit}** items")
                                st.markdown(f"- Min Sharpe threshold: **{sharpe_threshold}**")
                                if scan_type in ["portfolio_scan", "pure_allocator"]:
                                    st.markdown(f"- Budget: **${hedge_wizard.data.get('capital', 10000):,}**")
                                
                                hedge_wizard.render_navigation(next_label="Run Analysis →")
                            
                            # === STEP 3: RUN ANALYSIS ===
                            elif hedge_wizard.current_step == 2:
                                st.markdown("#### 🔄 Running Analysis...")
                                
                                # Get config from previous steps
                                scan_type = hedge_wizard.data.get("scan_type", "market_scan")
                                item_limit = hedge_wizard.data.get("item_limit", 50)
                                sharpe_threshold = hedge_wizard.data.get("sharpe_threshold", 0.0)
                                capital = hedge_wizard.data.get("capital", 10000.0)
                                use_quality_filter = hedge_wizard.data.get("use_quality_filter", False)
                                
                                
                                # Pure Allocator: analyze global futures universe directly
                                if scan_type == "pure_allocator":
                                    st.info("🧪 **Pure Allocator Mode**: Analyzing global correlation matrix for Max Sharpe...")
                                    df_analysis = pd.DataFrame() # Skip raw material loop
                                else:
                                    df_analysis = df_target.head(item_limit)
                                
                                candidates = []
                                skipped_items = []  # Track items skipped due to quality filter
                                progress_bar = st.progress(0, text="Initializing Market Scanner...")
                                status_text = st.empty()
                                
                                # --- BRANCH: PURE ALLOCATOR (Global Optimization) ---
                                if scan_type == "pure_allocator":
                                    status_text.text("Fetching Global Futures Universe...")
                                    # Use the predefined universe from constants
                                    # Fetch 2 years of data for robust correlation/volatility
                                    pool_data = fetch_market_data_pool(FUTURES_UNIVERSE, timeframe='2y')
                                    
                                    if pool_data:
                                        status_text.text("Calculating Correlation Matrix & Efficient Frontier...")
                                        
                                        # 1. Clean Data & Calculate Returns
                                        prices_df = pd.DataFrame(pool_data).fillna(method='ffill').fillna(method='bfill')
                                        # Ensure we have enough history
                                        if len(prices_df) > 50:
                                            returns_df = prices_df.pct_change().dropna()
                                            
                                            # 2. Run Optimizer
                                            optimizer = PortfolioOptimizer(returns_df)
                                            result = optimizer.optimize_max_sharpe_ratio()
                                            
                                            # 3. Format Candidates
                                            # We convert the portfolio weights directly into "candidates"
                                            p_sharpe = result.get('Sharpe', 0.0)
                                            p_ret = result.get('Return', 0.0)
                                            p_vol = result.get('Volatility', 0.0)
                                            
                                            for ticker, weight in result.get('Weights', {}).items():
                                                if weight < 0.01: continue # Ignore small dust
                                                
                                                candidates.append({
                                                    "Item": ticker,
                                                    "Description": "Global Market Instrument", # Placeholder
                                                    "Best Hedge": ticker,
                                                    "Correlation": 1.0,
                                                    "Hedge Ratio": 1.0,
                                                    "Current Sharpe": 0.0, # N/A
                                                    "Optimal Sharpe": p_sharpe,
                                                    "Sharpe Gain": p_sharpe, # Pure gain from 0
                                                    "Vol Reduction": 0.0,
                                                    "Allocation %": weight, # Pre-calculated exact weight
                                                    "Score": weight * 100 
                                                })
                                    progress_bar.progress(100, text="Optimization Complete.")
                                
                                # --- BRANCH: STANDARD SCANS (Item-by-Item Analysis) ---
                                else:  
                                    total_items = len(df_analysis)
                                
                                # Pre-fetch futures pool
                                with st.spinner("Fetching global futures data..."):
                                    pool_data = fetch_market_data_pool(FUTURES_UNIVERSE, timeframe='1y')
                                
                                total_items = len(df_analysis)
                                for idx, (_, row) in enumerate(df_analysis.iterrows()):
                                    item_num = str(row['ITEMNMBR']).strip()
                                    item_desc = str(row.get('ITEMDESC', ''))
                                    
                                    progress_bar.progress((idx + 1) / total_items, text=f"Analyzing {item_num}...")
                                    
                                    history_list = fetch_product_price_history(cursor, item_num, days=365)
                                    
                                    # Pure Allocator: Apply quality validation
                                    if use_quality_filter:
                                        quality = validate_price_history_quality(history_list, min_records=6)
                                        if not quality['is_reliable']:
                                            skipped_items.append(item_num)
                                            continue  # Skip unreliable items
                                        data_confidence = quality['confidence']
                                    else:
                                        data_confidence = 1.0  # Default for non-quality-filtered modes
                                    
                                    if history_list:
                                        best_fit = find_optimal_hedging_asset(history_list, pool_data)
                                        
                                        if best_fit and best_fit.get('metrics'):
                                            metrics = best_fit['metrics']
                                            curr_sharpe = metrics.get('current_sharpe', 0)
                                            opt_sharpe = metrics.get('optimal_sharpe', 0)
                                            sharpe_gap = opt_sharpe - curr_sharpe
                                            
                                            if sharpe_gap > 0:
                                                inv_data = fetch_product_inventory_trends(cursor, item_num)
                                                inv_value = inv_data.get('InventoryValue', 0) or 0
                                                hedge_ratio = metrics.get('optimal_hedge_ratio', 0)
                                                
                                                candidates.append({
                                                    "Item": item_num,
                                                    "Description": item_desc,
                                                    "Inventory Value": inv_value,
                                                    "Best Hedge": best_fit['best_asset'],
                                                    "Correlation": metrics.get('correlation', 0),
                                                    "Hedge Ratio": f"{hedge_ratio*100:.0f}%",
                                                    "Short Amount": inv_value * abs(hedge_ratio),
                                                    "Current Sharpe": curr_sharpe,
                                                    "Optimal Sharpe": opt_sharpe,
                                                    "Sharpe Gain": sharpe_gap,
                                                    "Vol Reduction": metrics.get('volatility_reduction', 0)
                                                })
                                
                                progress_bar.empty()
                                
                                # Store results in wizard data
                                hedge_wizard.set_data("candidates", candidates)
                                hedge_wizard.set_data("analysis_complete", True)
                                
                                if candidates:
                                    st.success(f"✅ Analysis complete! Found **{len(candidates)}** potential opportunities.")
                                    st.markdown("Click **View Results** to see the detailed breakdown.")
                                else:
                                    st.warning("No hedging opportunities found matching your criteria.")
                                
                                hedge_wizard.render_navigation(next_label="View Results →")
                            
                            # === STEP 4: RESULTS ===
                            elif hedge_wizard.current_step == 3:
                                candidates = hedge_wizard.data.get("candidates", [])
                                scan_type = hedge_wizard.data.get("scan_type", "market_scan")
                                sharpe_threshold = hedge_wizard.data.get("sharpe_threshold", 0.0)
                                capital = hedge_wizard.data.get("capital", 10000.0)
                                
                                # Reset button
                                if st.button("🔄 Start New Analysis", key="hedge_reset"):
                                    hedge_wizard.reset()
                                    st.rerun()
                                
                                if not candidates:
                                    st.info("No opportunities found. Try adjusting your parameters.")
                                else:
                                    # Build results DataFrame
                                    results_df = pd.DataFrame(candidates)
                                    results_df = results_df[results_df['Sharpe Gain'] >= sharpe_threshold]
                                    results_df = results_df.sort_values("Sharpe Gain", ascending=False)
                                    
                                    if results_df.empty:
                                        st.warning(f"No opportunities with Sharpe Gain ≥ {sharpe_threshold}. Adjust threshold in Step 2.")
                                    else:
                                        st.success(f"Found **{len(results_df)}** hedging opportunities!")
                                        
                                        # Format for display
                                        display_df = results_df.copy()
                                        display_df['Inventory Value'] = display_df['Inventory Value'].apply(lambda x: f"${x:,.0f}")
                                        display_df['Short Amount'] = display_df['Short Amount'].apply(lambda x: f"${x:,.0f}")
                                        display_df['Vol Reduction'] = display_df['Vol Reduction'].apply(lambda x: f"{x:.1f}%")
                                        display_df['Current Sharpe'] = display_df['Current Sharpe'].apply(lambda x: f"{x:.2f}")
                                        display_df['Optimal Sharpe'] = display_df['Optimal Sharpe'].apply(lambda x: f"{x:.2f}")
                                        
                                        # Display based on scan type
                                        if scan_type == "portfolio_scan":
                                            st.markdown("### 🏛️ CAPITAL BLUEPRINT (PORTFOLIO WEIGHTED)")
                                            blueprint_df = optimize_capital_allocation(candidates, total_capital=capital)
                                            if not blueprint_df.empty:
                                                st.success(f"Generated Max Sharpe Strategy for ${capital:,.0f} Budget")
                                                col_b1, col_b2 = st.columns([2, 1])
                                                with col_b1:
                                                    bp_display = blueprint_df.copy()
                                                    bp_display['Allocated Capital'] = bp_display['Allocated Capital'].apply(lambda x: f"${x:,.2f}")
                                                    bp_display['Allocation %'] = bp_display['Allocation %'].apply(lambda x: f"{x:.1%}")
                                                    bp_display['Sharpe Gain'] = bp_display['Sharpe Gain'].apply(lambda x: f"+{x:.2f}")
                                                    st.dataframe(bp_display, use_container_width=True, hide_index=True)
                                                with col_b2:
                                                    st.info("Optimized for your CURRENT inventory risks.")
                                            
                                            st.markdown("### 💎 HIGH CONVICTION OPPORTUNITIES")
                                            st.dataframe(
                                                display_df.style.format({"Sharpe Gain": "{:+.2f}"}).background_gradient(subset=["Sharpe Gain"], cmap="Greens"),
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        elif scan_type == "pure_allocator":
                                            st.markdown("### 🧪 PURE ALLOCATOR (THEORETICAL MODEL)")
                                            
                                            if candidates:
                                                blueprint_df = pd.DataFrame(candidates)
                                                # Calculate capital
                                                blueprint_df['Allocated Capital'] = blueprint_df['Allocation %'] * capital
                                                
                                                # Sort by weight
                                                blueprint_df = blueprint_df.sort_values('Allocated Capital', ascending=False)
                                                
                                                st.success(f"Generated Theoretical Max Sharpe Portfolio for ${capital:,.0f}")
                                                
                                                col_b1, col_b2 = st.columns([2, 1])
                                                with col_b1:
                                                    bp_display = blueprint_df.copy()
                                                    bp_display = bp_display[['Item', 'Allocation %', 'Allocated Capital']]
                                                    bp_display['Allocated Capital'] = bp_display['Allocated Capital'].apply(lambda x: f"${x:,.2f}")
                                                    bp_display['Allocation %'] = bp_display['Allocation %'].apply(lambda x: f"{x:.1%}")
                                                    # bp_display['Sharpe Gain'] = bp_display['Sharpe Gain'].apply(lambda x: f"+{x:.2f}")
                                                    st.dataframe(bp_display, use_container_width=True, hide_index=True)
                                                    
                                                with col_b2:
                                                    st.info("Ignores current stock — pure math optimization.")
                                                    st.markdown(f"**Est. Sharpe Ratio:** {candidates[0]['Optimal Sharpe']:.2f}")
                                                
                                                # EXECUTION BUTTON
                                                st.divider()
                                                pm = PortfolioManager()
                                                cash_avail = pm.state['cash']
                                                
                                                e_c1, e_c2 = st.columns([3, 1])
                                                with e_c1:
                                                    st.markdown(f"**Available for Deployment:** ${cash_avail:,.2f}")
                                                    if cash_avail < capital * 0.9:
                                                        st.warning(f"⚠️ Insufficient Cash! You need ${capital:,.2f} but have ${cash_avail:,.2f}. Trades will be scaled down.")
                                                
                                                with e_c2:
                                                    if st.button("🚀 EXECUTE TRADES", type="primary"):
                                                        if pm.execute_rebalancing(blueprint_df):
                                                            st.balloons()
                                                            st.success("✅ TRADES EXECUTED! Portfolio updated.")
                                                            st.caption("See 'Capital Management' tab for new holdings.")
                                                            
                                            else:
                                                st.warning("Optimization failed to find a valid portfolio.")
                                            
                                            st.markdown("---")
                                            st.markdown("### 🏆 THE IDEAL HEDGE PORTFOLIO (AGGREGATED)")
                                            st.caption("Consolidated trade instructions to hedge all identified risks.")
                                            
                                            # Aggregate recommendations by Hedge Asset
                                            portfolio_agg = results_df.groupby("Best Hedge")[["Short Amount"]].sum().reset_index()
                                            portfolio_agg = portfolio_agg.sort_values("Short Amount", ascending=False)
                                            total_short = portfolio_agg["Short Amount"].sum()
                                            
                                            if total_short > 0:
                                                portfolio_agg["Allocation %"] = portfolio_agg["Short Amount"] / total_short
                                                
                                                # Format for display
                                                agg_display = portfolio_agg.copy()
                                                agg_display["Action"] = "SHORT"  # Hedging implies shorting the correlate
                                                agg_display["Short Amount"] = agg_display["Short Amount"].apply(lambda x: f"${x:,.2f}")
                                                agg_display["Allocation %"] = agg_display["Allocation %"].apply(lambda x: f"{x:.1%}")
                                                
                                                # Reorder columns
                                                agg_display = agg_display[["Best Hedge", "Action", "Short Amount", "Allocation %"]]
                                                
                                                st.dataframe(agg_display, use_container_width=True, hide_index=True)
                                                
                                                # CSV Download
                                                csv = agg_display.to_csv(index=False).encode('utf-8')
                                                st.download_button(
                                                    "📥 Download Trade List (CSV)",
                                                    csv,
                                                    "ideal_hedge_portfolio.csv",
                                                    "text/csv",
                                                    key='download-hedge-csv'
                                                )
                                            


                                            # === ML OPTIMIZATION (Frontier) ===
                                            st.markdown("---")
                                            st.markdown("### 🧠 ML PORTFOLIO OPTIMIZATION")
                                            st.caption("Use Monte Carlo simulation to find the mathematical 'Efficient Frontier' of your hedge portfolio.")
                                            
                                            if st.button("🚀 Run Efficient Frontier Simulation", key="run_frontier"):
                                                with st.spinner("Initializing Math Engine & Simulating 5,000 Portfolios..."):
                                                    # 1. Get unique assets to optimize
                                                    unique_hedges = results_df['Best Hedge'].unique().tolist()
                                                    
                                                    # 2. Fetch market data for these assets (Re-using external data fetcher)
                                                    # We need price history to calculate covariance
                                                    market_data_map = fetch_market_data_pool(unique_hedges, timeframe="1y")
                                                    
                                                    # 3. Construct Returns DataFrame
                                                    price_data = {}
                                                    for asset in unique_hedges:
                                                        if asset in market_data_map:
                                                            m_data = market_data_map[asset].get('data', [])
                                                            if m_data:
                                                                df_m = pd.DataFrame(m_data)
                                                                df_m['date'] = pd.to_datetime(df_m['date'])
                                                                df_m = df_m.set_index('date').sort_index()
                                                                # Daily/Weekly freq
                                                                price_data[asset] = df_m['price_index']
                                                    
                                                    if price_data:
                                                        prices_df = pd.DataFrame(price_data).fillna(method='ffill').fillna(method='bfill')
                                                        returns_df = prices_df.pct_change().dropna()
                                                        
                                                        if not returns_df.empty:
                                                            # 4. Run Optimizer
                                                            optimizer = PortfolioOptimizer(returns_df)
                                                            opt_results = optimizer.simulate_frontier(n_portfolios=5000)
                                                            
                                                            # 5. Display Efficient Frontier Chart
                                                            frontier_df = opt_results.get('frontier_data')
                                                            max_sharpe = opt_results.get('max_sharpe')
                                                            
                                                            if frontier_df is not None:
                                                                # Base Chart (Scatter)
                                                                base = alt.Chart(frontier_df).mark_circle(size=60).encode(
                                                                    x=alt.X('Volatility', title='Risk (Volatility)'),
                                                                    y=alt.Y('Return', title='Expected Return'),
                                                                    color=alt.Color('Sharpe', scale=alt.Scale(scheme='viridis')),
                                                                    tooltip=['Return', 'Volatility', 'Sharpe']
                                                                ).properties(
                                                                    title='Efficient Frontier (5,000 Scenarios)',
                                                                    height=400
                                                                )
                                                                
                                                                # Optimal Point
                                                                opt_pt = pd.DataFrame([max_sharpe])
                                                                pt_chart = alt.Chart(opt_pt).mark_point(
                                                                    shape='diamond', size=200, color='red', fill='red'
                                                                ).encode(
                                                                    x='Volatility',
                                                                    y='Return',
                                                                    tooltip=alt.Tooltip('Sharpe', title='Max Sharpe')
                                                                )
                                                                
                                                                st.altair_chart(base + pt_chart, use_container_width=True)
                                                                
                                                                # 6. Trade Balance (Rebalancing)
                                                                st.markdown("#### ⚖️ OPTIMIZED REBALANCING")
                                                                st.caption("Adjust your current plan to match the Max Sharpe portfolio weights.")
                                                                
                                                                # Current allocations from the "Ideal Portfolio"
                                                                current_amts = dict(zip(portfolio_agg['Best Hedge'], portfolio_agg['Short Amount'].replace(r'[\$,]', '', regex=True).astype(float)))
                                                                total_cap = sum(current_amts.values())
                                                                
                                                                # Target allocations
                                                                target_wts = max_sharpe['Weights']
                                                                
                                                                # Calculate Trades
                                                                rebalance_df = optimizer.calculate_rebalancing_trades(
                                                                    current_holdings=current_amts,
                                                                    target_allocation=target_wts,
                                                                    total_capital=total_cap
                                                                )
                                                                
                                                                if not rebalance_df.empty:
                                                                    rebalance_df['Trade Value'] = rebalance_df['Trade Value'].apply(lambda x: f"${x:,.2f}")
                                                                    rebalance_df['Current Value'] = rebalance_df['Current Value'].apply(lambda x: f"${x:,.2f}")
                                                                    rebalance_df['Target Value'] = rebalance_df['Target Value'].apply(lambda x: f"${x:,.2f}")
                                                                    rebalance_df['Pct Portfolio'] = rebalance_df['Pct Portfolio'].apply(lambda x: f"{x:.1%}")
                                                                    
                                                                    st.dataframe(rebalance_df, use_container_width=True, hide_index=True)
                                                                else:
                                                                    st.success("Your portfolio is already perfectly optimized!")

                                            st.markdown("---")
                                            st.markdown("### 💎 TOP OPPORTUNITIES")
                                            st.dataframe(
                                                display_df.style.format({"Sharpe Gain": "{:+.2f}"}).background_gradient(subset=["Sharpe Gain"], cmap="Greens"),
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        else:  # market_scan
                                            # Metric Summary
                                            if not results_df.empty:
                                                top_gain = results_df.iloc[0]
                                                total_hedge_opp = results_df['Short Amount'].sum()
                                                
                                                m1, m2, m3, m4 = st.columns(4)
                                                m1.metric("Highest Opportunity", top_gain['Item'], f"+{top_gain['Sharpe Gain']:.2f} Sharpe")
                                                m2.metric("Most Common Hedge", results_df['Best Hedge'].mode()[0] if len(results_df) > 0 else "N/A")
                                                m3.metric("Total Hedge Value", f"${total_hedge_opp:,.0f}")
                                                m4.metric("Avg Vol Reduction", f"-{results_df['Vol Reduction'].mean():.1f}%")
                                            
                                            st.markdown("### OPPORTUNITY LIST")
                                            st.dataframe(
                                                display_df.style.format({"Sharpe Gain": "{:+.2f}"}).background_gradient(subset=["Sharpe Gain"], cmap="Greens"),
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                        
                                        # Monte Carlo Simulation (all modes)
                                        st.markdown("---")
                                        st.markdown("### 🎲 HEDGE IMPACT SIMULATION (6 MONTHS)")
                                        
                                        total_portfolio_value = results_df['Inventory Value'].sum()
                                        if total_portfolio_value > 0:
                                            unhedged_vol_assumed = 0.25
                                            total_weight = results_df['Inventory Value'].sum()
                                            
                                            if total_weight > 0:
                                                weighted_vol_red = (results_df['Vol Reduction'] * results_df['Inventory Value']).sum() / total_weight
                                                weighted_sharpe_opt = (results_df['Optimal Sharpe'] * results_df['Inventory Value']).sum() / total_weight
                                            else:
                                                weighted_vol_red = results_df['Vol Reduction'].mean()
                                                weighted_sharpe_opt = results_df['Optimal Sharpe'].mean()
                                            
                                            hedged_vol_sim = unhedged_vol_assumed * (1 - (weighted_vol_red / 100.0))
                                            hedged_drift_assumed = min(weighted_sharpe_opt * hedged_vol_sim, 0.50)
                                            unhedged_drift_assumed = 0.05
                                            
                                            st.caption(f"Simulating 1,000 scenarios. Baseline Vol: {unhedged_vol_assumed*100:.1f}% → Hedged Vol: {hedged_vol_sim*100:.1f}%")
                                            
                                            sim_results = simulate_portfolio_variance(
                                                current_value=total_portfolio_value,
                                                unhedged_vol=unhedged_vol_assumed,
                                                hedged_vol=hedged_vol_sim,
                                                mu=hedged_drift_assumed,
                                                months=6
                                            )
                                            
                                            if sim_results:
                                                dist_df = sim_results['distribution']
                                                d_chart = alt.Chart(dist_df).transform_density(
                                                    'Value', groupby=['Scenario'], as_=['Value', 'Density']
                                                ).mark_area(opacity=0.5).encode(
                                                    alt.X('Value:Q', title='Projected Portfolio Value ($)'),
                                                    alt.Y('Density:Q', title='Probability Density'),
                                                    alt.Color('Scenario:N', scale=alt.Scale(domain=['Unhedged', 'Hedged'], range=['#ff4b4b', '#00c853']))
                                                ).properties(width='container', height=300, title="Risk Distribution: Unhedged vs Hedged")
                                                
                                                ts_df = sim_results['timeseries']
                                                ts_base = alt.Chart(ts_df).encode(x=alt.X('Day:Q', title='Days'))
                                                lines = ts_base.mark_line().encode(
                                                    y=alt.Y('Mean:Q', title='Value ($)', scale=alt.Scale(zero=False)),
                                                    color=alt.Color('Scenario:N', scale=alt.Scale(domain=['Unhedged', 'Hedged'], range=['#ff4b4b', '#00c853']))
                                                )
                                                area = ts_base.mark_area(opacity=0.3).encode(y='Lower:Q', y2='Upper:Q', color='Scenario:N')
                                                ts_chart = (area + lines).properties(width='container', height=300, title="Projected Growth (90% CI)").interactive()
                                                
                                                c1, c2 = st.columns(2)
                                                with c1:
                                                    st.altair_chart(d_chart, use_container_width=True)
                                                with c2:
                                                    st.altair_chart(ts_chart, use_container_width=True)
                                                
                                                st.markdown("### 📈 RISK & RETURN ANALYSIS")
                                                final_day = ts_df['Day'].max()
                                                final_stats = ts_df[ts_df['Day'] == final_day]
                                                
                                                uh_row = final_stats[final_stats['Scenario'] == 'Unhedged'].iloc[0]
                                                h_row = final_stats[final_stats['Scenario'] == 'Hedged'].iloc[0]
                                                floor_gain = h_row['Lower'] - uh_row['Lower']
                                                
                                                sharpe_u = unhedged_drift_assumed / unhedged_vol_assumed if unhedged_vol_assumed > 0 else 0
                                                sharpe_h = hedged_drift_assumed / hedged_vol_sim if hedged_vol_sim > 0 else 0
                                                sharpe_delta = sharpe_h - sharpe_u
                                                
                                                m1, m2, m3, m4 = st.columns(4)
                                                m1.metric("Unhedged Sharpe", f"{sharpe_u:.2f}")
                                                m2.metric("Hedged Sharpe", f"{sharpe_h:.2f}", delta=f"+{sharpe_delta:.2f}")
                                                m3.metric("Vol Reduction", f"{weighted_vol_red:.1f}%")
                                                m4.metric("Risk Floor Gain", f"+${floor_gain:,.0f}")
                    
                    
                    elif view_mode == "FIFO Tracker":
                        # Delegate to the FIFO Tracker module
                        fifo_tracker.render_fifo_dashboard(cursor)
                    
                    elif view_mode == "Market Monitor":
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
                            search_df = _add_margin_metrics(search_df)
                            
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
                            h4.markdown("**MARGIN**")
                            st.markdown("---")
                            
                            # Sort by margin dollars: highest margin first
                            filtered_df = filtered_df.copy()
                            filtered_df['Margin'] = pd.to_numeric(filtered_df.get('Margin'), errors='coerce').fillna(0)
                            filtered_df = filtered_df.sort_values(by=['Margin'], ascending=True)
                            
                            # Scrollable container for the list
                            with st.container(height=600):
                                for idx, row in filtered_df.iterrows():
                                    item_num = row['ITEMNMBR']
                                    item_desc = row['ITEMDESC']
                                    curr_cost = row.get('CURRCOST', 0)
                                    margin = row.get('Margin', 0)
                                    margin_pct = row.get('MarginPct', 0)
                                    
                                    # Determine color
                                    if margin > 0:
                                        color = "#859900" # Green
                                    elif margin < 0:
                                        color = "#dc322f" # Red
                                    else:
                                        color = "#839496" # Grey
                                        
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
                                        st.markdown(
                                            f"<div style='color: {color}; padding-top: 5px;'>{margin:+.2f} ({margin_pct:+.1f}%)</div>",
                                            unsafe_allow_html=True,
                                        )
                                    
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
            
            # Fetch Data
            details = get_product_details(cursor, product_item)
            item_desc = details.get('item_desc', '') if details else ''

            # Header with back button
            header_col1, header_col2 = st.columns([8, 1])
            with header_col1:
                st.markdown(f"# 🎯 MARKET_INTELLIGENCE: `{product_item}`")
                if item_desc:
                    st.markdown(f"### {item_desc}") 
            with header_col2:
                if st.button("← BACK", key="back_to_overview"):
                    st.session_state.current_page = "market_overview"
                    st.session_state.selected_product = None
                    st.rerun()

            if not details:
                st.error("PRODUCT DATA NOT FOUND")
            else:
                # Extract Data
                category = details.get('category', 'default')
                inventory = details.get('inventory_status', {})
                price_hist = details.get('price_history', [])
                
                # Fetch External Data
                with st.spinner("● LOADING MARKET CONTEXT..."):
                    pool_data = None # Defer heavy load
                    
                    # Fallback context for narrative
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
                optimal_buy = recommend_optimal_buy_window(
                    price_history=price_hist,
                    usage_history=usage_hist or [],
                    coverage_days=burn_metrics.get('days_of_coverage') if burn_metrics else None,
                    available_stock=burn_metrics.get('available_stock') if burn_metrics else None,
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
                
                # --- KPI CARDS ---
                curr_cost = float(inventory.get('CURRCOST', 0) or 0)
                std_cost = float(inventory.get('STNDCOST', 0) or 0)
                margin_delta = std_cost - curr_cost
                margin_pct = (margin_delta / std_cost * 100) if std_cost else 0
                margin_color = "#859900" if margin_delta >= 0 else "#dc322f"

                on_order = float(inventory.get('OnOrder', 0) or 0)
                on_hand = float(inventory.get('TotalOnHand', 0) or 0)
                inv_value = on_hand * curr_cost

                trend = ag_data.get('trend', 'stable')
                trend_color = "#dc322f" if trend == "increasing" else "#859900" if trend == "decreasing" else "#ffb000"
                trend_arrow = "▲" if trend == "increasing" else "▼" if trend == "decreasing" else "▶"

                demand = forecast.get('forecast_next_30days', 'stable')

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.markdown("### INTERNAL COST")
                    st.markdown(f"## ${curr_cost:,.5f}")

                with kpi2:
                    st.markdown("### STANDARD COST")
                    st.markdown(f"## ${std_cost:,.5f}")

                with kpi3:
                    st.markdown("### MARGIN (STD-CURR)")
                    st.markdown(
                        f"## <span style='color:{margin_color}'>{margin_delta:+,.5f} ({margin_pct:+.1f}%)</span>",
                        unsafe_allow_html=True,
                    )

                with kpi4:
                    st.markdown("### MARKET TREND")
                    st.markdown(f"## <span style='color:{trend_color}'>{trend_arrow} {trend.upper()}</span>", unsafe_allow_html=True)

                # Smart Correlation Calculation - DEFERRED
                # We will calculate best_fit later to speed up initial render
                best_fit = None
                hedge_metrics = {}

                kpi5, kpi6, kpi7, kpi8 = st.columns(4)
                with kpi5:
                    st.markdown("### DEMAND FORECAST")
                    st.markdown(f"## {str(demand).upper()}")

                with kpi6:
                    st.markdown("### SUPPLY POSITION")
                    st.markdown(f"**On Order:** {on_order:,.0f}")
                    
                    po_vendor_infos = inventory.get('OnOrderVendorInfos', [])
                    # Fallback to plain list if new structure missing (backward compat)
                    if not po_vendor_infos and 'OnOrderVendors' in inventory:
                        po_vendor_infos = [{'name': v, 'phone': None} for v in inventory.get('OnOrderVendors', [])]
                        
                    if on_order > 0 and po_vendor_infos:
                        for v_info in po_vendor_infos:
                            v_name = v_info.get('name', 'Unknown')
                            v_phone = v_info.get('phone')
                            
                            st.markdown(f"**Vendor:** {v_name}")
                            if v_phone:
                                st.markdown(f"**Phone:** {v_phone}")
                            else:
                                st.markdown(f"**Phone:** No phone number")
                        
                    st.markdown(f"**Inventory Value:** ${inv_value:,.2f}")

                with kpi7:
                    hedge_placeholder = st.empty()
                    with hedge_placeholder.container():
                         st.markdown("### OPTIMAL HEDGE RATIO")
                         st.caption("⏳ Analyzing correlations...")

                with kpi8:
                    gap_placeholder = st.empty()
                    with gap_placeholder.container():
                         st.markdown("### SHARPE GAP")
                         st.caption("⏳ ...")

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

                            # Date range selector (drives both charts below)
                            min_txn_date = hist_df['TransactionDate'].min()
                            if pd.isna(min_txn_date):
                                min_txn_date = dt.date.today() - dt.timedelta(days=365)
                            else:
                                min_txn_date = min_txn_date.date()
                            
                            max_txn_date_val = hist_df['TransactionDate'].max()
                            max_txn_date = max_txn_date_val.date() if pd.notna(max_txn_date_val) else dt.date.today()
                            
                            today = dt.date.today()
                            # Use tomorrow as cap to avoid timezone/midnight edge cases
                            capped_max_date = max(min_txn_date, today + dt.timedelta(days=1))
                            
                            # --- SPECIAL OVERRIDE FOR PHOS 75 ---
                            # User requested explicit 5-year logic for this item to ensure usage reports are visible
                            if "PHOS" in str(product_item).upper() and "75" in str(product_item):
                                five_years_ago = today - dt.timedelta(days=1825)
                                min_txn_date = five_years_ago
                                capped_max_date = today + dt.timedelta(days=1)
                                
                            # --- ROBUST DATE SELECTION ---
                            # Global Bounds
                            min_dt = min_txn_date
                            max_dt = capped_max_date
                            
                            range_key = f"date_range_v2_{product_item}"
                            range_usage_key = f"{range_key}_usage"
                            range_usage_state_key = f"{range_key}_usage_state"
                            range_usage_override_key = f"{range_key}_usage_override"
                            
                            # Default Range: Full history if not too long, else last year
                            default_start = min_dt
                            default_end = max_dt
                            
                            if range_key not in st.session_state:
                                st.session_state[range_key] = (default_start, default_end)
                            
                            # Validation Step: Ensure retained state is valid for CURRENT product bounds
                            # This fixes the "crash on product switch" bug
                            current_val = st.session_state[range_key]
                            
                            # Ensure it's a tuple of 2
                            if not isinstance(current_val, (tuple, list)) or len(current_val) != 2:
                                current_val = (default_start, default_end)
                                
                            start_val, end_val = current_val
                            
                            # Safe Clamp Function
                            def safe_clamp(val, min_v, max_v):
                                if pd.isna(val): return min_v
                                if val < min_v: return min_v
                                if val > max_v: return max_v
                                return val

                            # Re-clamp values to current product's valid range
                            clean_start = safe_clamp(start_val, min_dt, max_dt)
                            clean_end = safe_clamp(end_val, min_dt, max_dt)
                            
                            if clean_start > clean_end:
                                clean_start = clean_end # classic swap fix
                                
                            # Update session state if we had to fix it (silent fix, no rerun needed usually unless strict)
                            # We pass the CLEAN values to the widget
                            
                            new_range = st.date_input(
                                "Adjust date range",
                                value=(clean_start, clean_end),
                                min_value=min_dt,
                                max_value=max_dt,
                                key=f"widget_{range_key}" # Widget key different from storage key to allow manual sync if needed, but here we just read return
                            )
                            
                            # Update Storage
                            if new_range != st.session_state[range_key]:
                                # Only update if it's a valid complete range (Streamlit returns tuple of 1 during selection)
                                if isinstance(new_range, (tuple, list)) and len(new_range) == 2:
                                    st.session_state[range_key] = new_range


                            
                            # Final values for filtering
                            if isinstance(new_range, (tuple, list)) and len(new_range) == 2:
                                start_date, end_date = new_range
                            else:
                                start_date, end_date = clean_start, clean_end

                            # --- USAGE OVERRIDE HELPERS ---
                            # These are now simpler and robust, checking keys specifically passed
                            def _ensure_usage_state(key, default_val):
                                if key not in st.session_state:
                                    st.session_state[key] = default_val
                                    
                            def _usage_override_enabled(key, override_key=None):
                                flag_key = override_key or f"{key}_is_overridden"
                                return st.session_state.get(flag_key, False)

                            def _set_usage_override(key, override_key=None, value=True):
                                flag_key = override_key or f"{key}_is_overridden"
                                st.session_state[flag_key] = value

                            filtered_hist_df = hist_df[
                                (hist_df['TransactionDate'].dt.date >= start_date)
                                & (hist_df['TransactionDate'].dt.date <= end_date)
                            ]
                            
                            # Helper to render chart + usage map
                            def render_price_chart(
                                data_df,
                                get_safe_clamp_func, # Pass safe_clamp or closure
                                start_date=None,
                                end_date=None,
                                date_bounds=None,
                                usage_state_key=None,
                                usage_input_key=None,
                                usage_override_key=None,
                                **kwargs
                            ):
                                if data_df.empty:
                                    st.info("No data available.")
                                    return

                                # robust clamp from outer scope
                                safe_clamp = get_safe_clamp_func 
                                
                                usage_view_key = f"{usage_input_key}_view_mode" if usage_input_key else "usage_view_mode" 

                                usage_widget_key = f"{usage_input_key}_widget" if usage_input_key else None
                                override_widget_key = f"{usage_override_key}_widget" if usage_override_key else None

                                usage_start_dt = pd.to_datetime(start_date) if start_date else pd.NaT
                                usage_end_dt = pd.to_datetime(end_date) if end_date else pd.NaT

                                auto_usage_range = None
                                if "TransactionDate" in data_df.columns:
                                    tx_dates = pd.to_datetime(data_df["TransactionDate"])
                                    if not tx_dates.empty:
                                        raw_min = tx_dates.min().date()
                                        raw_max = tx_dates.max().date()
                                        
                                        # Clamp auto range to global bounds if provided
                                        if date_bounds:
                                            # safe_clamp expects (val, min, max)
                                            c_min = safe_clamp(raw_min, date_bounds[0], date_bounds[1])
                                            c_max = safe_clamp(raw_max, date_bounds[0], date_bounds[1])
                                            if c_min > c_max:
                                                c_min = c_max
                                            auto_usage_range = (c_min, c_max)
                                        else:
                                            auto_usage_range = (raw_min, raw_max)

                                # Aggregate logic (Same as before)
                                chart_data = data_df.groupby('TransactionDate').agg({
                                    'AvgCost': 'mean',
                                    'LandedCost': 'mean',
                                    'TransactionCount': 'sum',
                                    'Quantity': 'sum'
                                }).reset_index()

                                chart_data['AvgCost'] = chart_data['AvgCost'].astype(float)
                                chart_data['LandedCost'] = chart_data['LandedCost'].astype(float)
                                chart_data['TransactionCount'] = chart_data['TransactionCount'].astype(float)
                                chart_data['Quantity'] = chart_data['Quantity'].astype(float)

                                # Base chart
                                base = alt.Chart(chart_data).encode(
                                    x=alt.X('TransactionDate:T', title='Date', axis=alt.Axis(labelAngle=-45, format='%b %Y'))
                                )
                                # Cost Layer (Line + Points)
                                cost_base = base.encode(
                                    y=alt.Y('AvgCost:Q', title='Average Cost ($)', scale=alt.Scale(zero=False)),
                                    tooltip=[
                                        alt.Tooltip('TransactionDate', title='Date', format='%Y-%m-%d'),
                                        alt.Tooltip('AvgCost', title='Delivered Cost', format='$,.4f'),
                                        alt.Tooltip('LandedCost', title='Landed Cost', format='$,.4f'),
                                        alt.Tooltip('Quantity', title='Quantity', format=',.0f')
                                    ]
                                )
                                line = cost_base.mark_line(color='#ffb000', strokeWidth=3)
                                points = cost_base.mark_point(color='#ffb000', size=80, filled=True)
                                cost_line = line + points
                                final_chart = cost_line
                                if 'Quantity' in chart_data.columns:
                                    bars = base.mark_bar(opacity=0.3, color='#859900').encode(
                                        y=alt.Y('Quantity:Q', title='Quantity', axis=alt.Axis(titleColor='#859900')),
                                        tooltip=[
                                            alt.Tooltip('TransactionDate', title='Date', format='%Y-%m-%d'),
                                            alt.Tooltip('AvgCost', title='Delivered Cost', format='$,.4f'),
                                            alt.Tooltip('LandedCost', title='Landed Cost', format='$,.4f'),
                                            alt.Tooltip('Quantity', title='Quantity', format=',.0f')
                                        ]
                                    )
                                    final_chart = alt.layer(bars, cost_line).resolve_scale(y='independent')
                                
                                # Layout: Chart (Left) + Stats (Right)
                                col_chart, col_stats = st.columns([3, 1])
                                
                                with col_chart:
                                    st.altair_chart(final_chart.properties(height=300).interactive(), width="stretch")
                                
                                with col_stats:
                                    st.markdown("##### 📊 Summary")
                                    # Calculate totals from the aggregated chart_data to match what's visual
                                    total_qty = chart_data['Quantity'].sum() if 'Quantity' in chart_data.columns else 0
                                    
                                    # Weighted Avg Cost = Sum(AvgCost * Qty) / Total Qty
                                    # Note: This is an approx based on daily avg. Ideally we'd sum ExtendedCost from raw df.
                                    # Let's try to get it from data_df if possible for accuracy
                                    if not data_df.empty and 'ExtendedCost' in data_df.columns:
                                        total_spend = data_df['ExtendedCost'].sum()
                                        transaction_count = len(data_df)
                                    else:
                                        # Fallback to approx
                                        total_spend = (chart_data['AvgCost'] * chart_data['Quantity']).sum() if 'Quantity' in chart_data.columns else 0
                                        transaction_count = chart_data['TransactionCount'].sum()

                                    if total_qty > 0:
                                        w_avg_cost = total_spend / total_qty
                                    else:
                                        # Fallback to simple average if no quantity (e.g. only price records or filtered out)
                                        w_avg_cost = chart_data['AvgCost'].mean() if not chart_data.empty else 0
                                    
                                    st.metric("Total Spend", f"${total_spend:,.2f}")
                                    st.metric("Total Qty", f"{total_qty:,.0f}")
                                    st.metric("Avg Price", f"${w_avg_cost:,.4f}")
                                    st.caption(f"{int(transaction_count)} transactions")

                                # Usage Date Logic (Cleaned Up)
                                # Usage Date Logic (Cleaned Up)
                                if usage_state_key and date_bounds:
                                    # Ensure we have a valid default
                                    default_u = auto_usage_range or date_bounds
                                    if usage_state_key not in st.session_state:
                                        st.session_state[usage_state_key] = default_u
                                    
                                    # Validate existing state
                                    curr_u = st.session_state[usage_state_key]
                                    if not isinstance(curr_u, (tuple, list)) or len(curr_u) != 2:
                                        curr_u = default_u
                                    
                                    # Clamp existing state
                                    c_start = safe_clamp(curr_u[0], date_bounds[0], date_bounds[1])
                                    c_end = safe_clamp(curr_u[1], date_bounds[0], date_bounds[1])
                                    if c_start > c_end: c_start = c_end
                                    
                                    # Update state if clamped
                                    clamped_range = (c_start, c_end)
                                    st.session_state[usage_state_key] = clamped_range

                                    # Detect if the user already changed the widget this run (widget key is separate)
                                    pending_user_change = bool(
                                        usage_widget_key
                                        and usage_widget_key in st.session_state
                                        and st.session_state[usage_widget_key] != st.session_state[usage_state_key]
                                    )

                                    # Override Logic
                                    # If override is NOT active and the user hasn't changed the widget, sync to auto range
                                    if auto_usage_range and not _usage_override_enabled(usage_state_key, usage_override_key) and not pending_user_change:
                                        st.session_state[usage_state_key] = auto_usage_range
                                    
                                    st.markdown("#### Adjust usage date range")
                                    
                                    current_usage_range = st.session_state[usage_state_key]
                                    date_input_kwargs = {
                                        "label": "Usage date range",
                                        "min_value": date_bounds[0],
                                        "max_value": date_bounds[1],
                                        "key": usage_widget_key,
                                        "label_visibility": "collapsed",
                                    }
                                    if not usage_widget_key or usage_widget_key not in st.session_state:
                                        date_input_kwargs["value"] = current_usage_range
                                    new_u_range = st.date_input(**date_input_kwargs)

                                    # Allow explicitly locking the usage range away from the purchase filter
                                    if usage_override_key:
                                        checkbox_kwargs = {
                                            "label": "Lock usage date range (independent from purchase filter)",
                                            "key": override_widget_key,
                                            "help": "When locked, changes to the purchase date filter will not reset this usage chart.",
                                        }
                                        if not override_widget_key or override_widget_key not in st.session_state:
                                            checkbox_kwargs["value"] = _usage_override_enabled(usage_state_key, usage_override_key)
                                        locked_usage = st.checkbox(**checkbox_kwargs)
                                        _set_usage_override(usage_state_key, usage_override_key, locked_usage)
                                    
                                    # Detect User Change
                                    if new_u_range != st.session_state[usage_state_key]:
                                        if isinstance(new_u_range, (tuple, list)) and len(new_u_range) == 2:
                                            st.session_state[usage_state_key] = new_u_range
                                            _set_usage_override(usage_state_key, usage_override_key, True) # User touched it -> Lock it


                                    
                                    u_start, u_end = st.session_state[usage_state_key]
                                    start_dt = pd.to_datetime(u_start)
                                    end_dt = pd.to_datetime(u_end)
                                elif auto_usage_range:
                                    start_dt = pd.to_datetime(auto_usage_range[0])
                                    end_dt = pd.to_datetime(auto_usage_range[1])


                                usage_start = start_dt.normalize() if pd.notna(start_dt) else (
                                    pd.to_datetime(data_df["TransactionDate"].min()).normalize() if "TransactionDate" in data_df.columns else pd.NaT
                                )
                                usage_end = end_dt.normalize() if pd.notna(end_dt) else (
                                    pd.to_datetime(data_df["TransactionDate"].max()).normalize() if "TransactionDate" in data_df.columns else pd.NaT
                                )
                                # Fetch a long window so sparse purchase history doesn't hide usage
                                base_horizon_days = 1825  # ~5 years
                                if pd.notna(usage_start):
                                    today = pd.Timestamp.today().normalize()
                                    days_back = int((today - usage_start).days)
                                    lookback_days = max(base_horizon_days, days_back + 31)
                                else:
                                    lookback_days = base_horizon_days

                                usage_rows = fetch_product_usage_history(
                                    cursor,
                                    product_item,
                                    days=lookback_days,
                                    location=None,
                                    group_by="day",
                                )
                                if usage_rows and alt:
                                    udf = pd.DataFrame(usage_rows)
                                    if not udf.empty:
                                        from decimal import Decimal as _Decimal
                                        udf = udf.map(lambda v: float(v) if isinstance(v, _Decimal) else v)
                                        if "UsageQty" in udf.columns:
                                            udf["UsageQty"] = pd.to_numeric(udf["UsageQty"], errors="coerce")
                                        if "UsageDate" in udf.columns:
                                            udf["UsageDate"] = pd.to_datetime(udf["UsageDate"], errors="coerce")
                                        else:
                                            udf["UsageDate"] = pd.to_datetime(
                                                udf[["Year", "Month"]].assign(Day=1), errors="coerce"
                                            )
                                        udf = udf.dropna(subset=["UsageDate"]).sort_values("UsageDate")

                                        monthly_udf = (
                                            udf.assign(Year=udf["UsageDate"].dt.year, Month=udf["UsageDate"].dt.month)
                                            .groupby(["Year", "Month"], as_index=False)["UsageQty"]
                                            .sum()
                                        )
                                        monthly_udf["UsageDate"] = pd.to_datetime(
                                            monthly_udf[["Year", "Month"]].assign(Day=1), errors="coerce"
                                        )
                                        monthly_udf = monthly_udf.dropna(subset=["UsageDate"]).sort_values("UsageDate")

                                        header_col, toggle_col = st.columns([5, 2])
                                        with header_col:
                                            st.markdown("#### Usage History")
                                        with toggle_col:
                                            # Robust Checkbox Logic: Decouple persistence from widget existence
                                            # This fixes issues where the widget resets if it temporarily disappears (e.g. valid checks)
                                            # or if Streamlit reruns cause state desync.
                                            view_mode_persistent_key = f"{usage_view_key}_persistent"
                                            
                                            # 1. Ensure persistent store exists
                                            if view_mode_persistent_key not in st.session_state:
                                                st.session_state[view_mode_persistent_key] = "Monthly"
                                            
                                            # 2. Sync Widget to Persistent State (if widget not yet initialized for this run)
                                            # This acts as the "default" but stronger - we force the widget to match our memory
                                            if usage_view_key not in st.session_state:
                                                st.session_state[usage_view_key] = st.session_state[view_mode_persistent_key]
                                            
                                            # 3. Render Widget
                                            st.radio(
                                                "Usage view",
                                                ["Monthly", "All"],
                                                horizontal=True,
                                                key=usage_view_key,
                                                label_visibility="collapsed",
                                            )
                                            
                                            # 4. Sync Persistent State to Widget (Capture User Change)
                                            # If user changed it just now, widget state is new. Update persistent.
                                            if st.session_state[usage_view_key] != st.session_state[view_mode_persistent_key]:
                                                st.session_state[view_mode_persistent_key] = st.session_state[usage_view_key]


                                                
                                            usage_view_mode = st.session_state[view_mode_persistent_key]

                                        if usage_view_mode == "Monthly":
                                            range_notice = None
                                            filtered_udf = monthly_udf
                                            monthly_start = usage_start.replace(day=1) if pd.notna(usage_start) else usage_start
                                            monthly_end = (usage_end + pd.offsets.MonthEnd(0)) if pd.notna(usage_end) else usage_end
                                            if pd.notna(monthly_start) and pd.notna(monthly_end):
                                                filtered_udf = monthly_udf[monthly_udf["UsageDate"].between(monthly_start, monthly_end)]
                                                if filtered_udf.empty:
                                                    filtered_udf = monthly_udf
                                                    range_notice = "No usage in the selected range; showing all available usage."
                                            elif pd.notna(monthly_start):
                                                filtered_udf = monthly_udf[monthly_udf["UsageDate"] >= monthly_start]
                                                if filtered_udf.empty:
                                                    filtered_udf = monthly_udf
                                                    range_notice = "No usage after the selected start; showing all available usage."
                                            elif pd.notna(monthly_end):
                                                filtered_udf = monthly_udf[monthly_udf["UsageDate"] <= monthly_end]
                                                if filtered_udf.empty:
                                                    filtered_udf = monthly_udf
                                                    range_notice = "No usage before the selected end; showing all available usage."

                                            if filtered_udf.empty:
                                                st.info("No usage history available.")
                                                return

                                            x_domain = None
                                            if range_notice is None and pd.notna(monthly_start) and pd.notna(monthly_end):
                                                x_domain = [monthly_start, monthly_end]
                                            elif not filtered_udf.empty:
                                                x_domain = [
                                                    filtered_udf["UsageDate"].min(),
                                                    filtered_udf["UsageDate"].max(),
                                                ]
                                            x_scale = alt.Scale(domain=x_domain) if x_domain else alt.Scale()

                                            base = alt.Chart(filtered_udf).encode(
                                                x=alt.X("UsageDate:T", title="Date", scale=x_scale, axis=alt.Axis(labelAngle=-45, labelColor="#555", titleColor="#555", format="%b %Y"),)
                                            )
                                            bars = base.mark_bar(color="#859900", opacity=0.35).encode(
                                                y=alt.Y("UsageQty:Q", title="Usage Qty", axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"), scale=alt.Scale(zero=True),)
                                            )
                                            line = base.mark_line(strokeWidth=3, color="#ffb000").encode(
                                                y=alt.Y("UsageQty:Q", title="Usage Qty", axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"), scale=alt.Scale(zero=True),),
                                                tooltip=[
                                                    alt.Tooltip("UsageDate:T", title="Date", format="%b %Y"),
                                                    alt.Tooltip("UsageQty:Q", title="Usage Qty", format=",.0f"),
                                                ],
                                            )
                                            pts = line.mark_point(filled=True, size=80, color="#ffb000")
                                            usage_chart = (
                                                alt.layer(bars, line, pts)
                                                .resolve_scale(y="shared")
                                                .properties(height=260, title="Monthly Usage")
                                                .configure_axis(
                                                    gridColor="#e0e0e0",
                                                    domainColor="#cccccc",
                                                    tickColor="#cccccc",
                                                )
                                                .configure_title(color="#555")
                                                .configure_view(stroke="#e0e0e0", strokeWidth=1)
                                                .configure(background="#ffffff")
                                                .interactive()
                                            )
                                            if range_notice:
                                                st.caption(range_notice)
                                            # Layout: Chart (Left) + Stats (Right)
                                            u_col_chart, u_col_stats = st.columns([3, 1])
                                            with u_col_chart:
                                                st.altair_chart(usage_chart, width="stretch")
                                            
                                            with u_col_stats:
                                                st.markdown("##### 📦 Usage Summary")
                                                total_usage = filtered_udf['UsageQty'].sum() if not filtered_udf.empty else 0
                                                st.metric("Total Usage", f"{total_usage:,.0f}")
                                                st.caption(f"Over {len(filtered_udf)} months")
                                        else:
                                            range_notice = None
                                            filtered_daily = udf
                                            if pd.notna(usage_start) and pd.notna(usage_end):
                                                ranged = udf[udf["UsageDate"].between(usage_start, usage_end)]
                                                if not ranged.empty:
                                                    filtered_daily = ranged
                                                else:
                                                    range_notice = "No usage in the selected range; showing all available usage."
                                            elif pd.notna(usage_start):
                                                ranged = udf[udf["UsageDate"] >= usage_start]
                                                if not ranged.empty:
                                                    filtered_daily = ranged
                                                else:
                                                    range_notice = "No usage after the selected start; showing all available usage."
                                            elif pd.notna(usage_end):
                                                ranged = udf[udf["UsageDate"] <= usage_end]
                                                if not ranged.empty:
                                                    filtered_daily = ranged
                                                else:
                                                    range_notice = "No usage before the selected end; showing all available usage."

                                            if filtered_daily.empty:
                                                st.info("No usage history available.")
                                                return

                                            x_domain = [
                                                filtered_daily["UsageDate"].min(),
                                                filtered_daily["UsageDate"].max(),
                                            ]
                                            x_scale = alt.Scale(domain=x_domain) if x_domain else alt.Scale()

                                            base = alt.Chart(filtered_daily).encode(
                                                x=alt.X("UsageDate:T", title="Date", scale=x_scale, axis=alt.Axis(labelAngle=-45, labelColor="#555", titleColor="#555", format="%b %d, %Y"),)
                                            )
                                            bars = base.mark_bar(color="#859900", opacity=0.25).encode(
                                                y=alt.Y("UsageQty:Q", title="Usage Qty", axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"), scale=alt.Scale(zero=True),)
                                            )
                                            line = base.mark_line(strokeWidth=2, color="#ffb000").encode(
                                                y=alt.Y("UsageQty:Q", title="Usage Qty", axis=alt.Axis(format=",.0f", labelColor="#555", titleColor="#555"), scale=alt.Scale(zero=True),),
                                                tooltip=[
                                                    alt.Tooltip("UsageDate:T", title="Date", format="%Y-%m-%d"),
                                                    alt.Tooltip("UsageQty:Q", title="Usage Qty", format=",.0f"),
                                                ],
                                            )
                                            pts = line.mark_point(filled=True, size=55, color="#ffb000")
                                            usage_chart = (
                                                alt.layer(bars, line, pts)
                                                .resolve_scale(y="shared")
                                                .properties(height=260, title="All Usage (Daily)")
                                                .configure_axis(
                                                    gridColor="#e0e0e0",
                                                    domainColor="#cccccc",
                                                    tickColor="#cccccc",
                                                )
                                                .configure_title(color="#555")
                                                .configure_view(stroke="#e0e0e0", strokeWidth=1)
                                                .configure(background="#ffffff")
                                                .interactive()
                                            )
                                            if range_notice:
                                                st.caption(range_notice)
                                            # Layout: Chart (Left) + Stats (Right)
                                            u_col_chart, u_col_stats = st.columns([3, 1])
                                            with u_col_chart:
                                                st.altair_chart(usage_chart, width="stretch")
                                            
                                            with u_col_stats:
                                                st.markdown("##### 📦 Usage Summary")
                                                total_usage = filtered_daily['UsageQty'].sum() if not filtered_daily.empty else 0
                                                st.metric("Total Usage", f"{total_usage:,.0f}")
                                                st.caption(f"Over {len(filtered_daily)} days")
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
                                        render_price_chart(
                                            filtered_hist_df,
                                            safe_clamp,
                                            start_date,
                                            end_date,
                                            date_bounds=(min_dt, max_dt),
                                            usage_state_key=range_usage_state_key,
                                            usage_override_key=range_usage_override_key,
                                            usage_input_key=range_usage_key,
                                        )
                                        
                                    # Render for each Vendor
                                    for i, vendor in enumerate(vendors):
                                        with tabs[i+1]:
                                            vendor_df = filtered_hist_df[filtered_hist_df['VendorName'] == vendor]
                                            
                                            # Display Vendor Phone if available
                                            if not vendor_df.empty and 'VendorPhone' in vendor_df.columns:
                                                phones = [p for p in vendor_df['VendorPhone'].unique() if p and str(p).strip()]
                                                phone_raw = str(phones[0]).strip() if phones else ""
                                                
                                                if phone_raw:
                                                    # Format Phone: 80023860750000 -> (800) 238-6075 Ext. 0000
                                                    # GP often stores as 14 chars (Area Code 3 + Prefix 3 + Number 4 + Ext 4)
                                                    formatted_phone = phone_raw
                                                    digits = "".join(filter(str.isdigit, phone_raw))
                                                    
                                                    if len(digits) == 14:
                                                        formatted_phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:10]} Ext. {digits[10:]}"
                                                    elif len(digits) == 10:
                                                        formatted_phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                                                    elif len(digits) == 7:
                                                        formatted_phone = f"{digits[:3]}-{digits[3:]}"
                                                        
                                                    st.text_input("Phone 1", value=formatted_phone, disabled=True, key=f"phone_{i}_{str(vendor)[:10]}")
                                                else:
                                                    st.text_input("Phone 1", value="No phone number", disabled=True, key=f"phone_{i}_{str(vendor)[:10]}")

                                            render_price_chart(
                                            vendor_df,
                                            safe_clamp,
                                            start_date,
                                            end_date,
                                            date_bounds=(min_dt, max_dt),
                                            usage_state_key=f"{range_usage_state_key}_{vendor}",
                                            usage_override_key=f"{range_usage_override_key}_{vendor}",
                                            usage_input_key=f"{range_usage_key}_{vendor}",
                                        )
                                else:
                                    render_price_chart(
                                        filtered_hist_df,
                                        safe_clamp,
                                        start_date,
                                        end_date,
                                        date_bounds=(min_dt, max_dt),
                                        usage_state_key=range_usage_state_key,
                                        usage_override_key=range_usage_override_key,
                                        usage_input_key=range_usage_key,
                                    )
                            else:
                                render_price_chart(
                                    filtered_hist_df,
                                    safe_clamp,
                                    start_date,
                                    end_date,
                                    date_bounds=(min_dt, max_dt),
                                    usage_state_key=range_usage_state_key,
                                    usage_override_key=range_usage_override_key,
                                    usage_input_key=range_usage_key,
                                )
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

                burn_col1, burn_col2, burn_col3, burn_col4 = st.columns(4)
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
                buy_caption = None
                with burn_col4:
                    if optimal_buy and optimal_buy.get('status') != 'insufficient':
                        buy_days = int(optimal_buy.get('days_from_now', 0))
                        buy_date = optimal_buy.get('buy_date')
                        date_label = buy_date.strftime("%b %d") if isinstance(buy_date, (datetime.date, datetime.datetime)) else f"In {buy_days}d"
                        expected_price = float(optimal_buy.get('expected_price', 0))
                        price_delta = float(optimal_buy.get('price_delta', 0))
                        st.metric(
                            "OPTIMAL BUY WINDOW",
                            f"{date_label} ({buy_days}d)",
                            delta=f"${expected_price:,.2f} est ({price_delta:+,.2f} vs now)"
                        )
                        buy_caption = optimal_buy.get('reason')
                    else:
                        st.metric("OPTIMAL BUY WINDOW", "N/A", delta="Need more history")
                        buy_caption = optimal_buy.get('reason') if optimal_buy else None

                if buy_caption:
                    st.caption(f"Buy timing model: {buy_caption}")
                st.caption("Burn rate uses on-hand + on-order, weighted for current season with decayed material usage.")

                # --- DEFERRED HEAVY CALCULATIONS ---
                if price_hist:
                     # This runs AFTER the page has rendered the charts, giving a "progressive load" feel
                     with st.spinner("SCANNING FUTURES UNIVERSE FOR BEST HEDGE MATCH..."):
                          pool_data = fetch_market_data_pool(FUTURES_UNIVERSE, timeframe='1y')
                          best_fit = find_optimal_hedging_asset(price_hist, pool_data)
                          
                          if best_fit and best_fit.get('metrics'):
                                metrics = best_fit['metrics']
                                ag_data_best = best_fit['data']
                                
                                # Update Hedge Placeholder
                                with hedge_placeholder.container():
                                    opt_hedge = metrics.get('optimal_hedge_ratio', 0) * 100
                                    if opt_hedge == 0:
                                        hedge_status = "NO HEDGE"
                                        color = "#839496"
                                    elif opt_hedge < 100:
                                        hedge_status = "PARTIAL HEDGE"
                                        color = "#b58900"
                                    else:
                                        hedge_status = "OVER-HEDGE"
                                        color = "#dc322f"

                                    st.markdown("### OPTIMAL HEDGE RATIO")
                                    st.markdown(f"## {opt_hedge:.0f}% <span style='font-size:0.6em; color:{color}'>({hedge_status})</span>", unsafe_allow_html=True)
                                    
                                    target_short_val = inv_value * (opt_hedge / 100)
                                    comm_name = ag_data_best.get('commodity', 'FUTURES')
                                    
                                    st.caption(
                                        f"**STRATEGY:** Short **${target_short_val:,.2f}** of **{comm_name}**.", 
                                        help=f"Best Fit: {comm_name} (r={metrics.get('correlation',0):.2f})."
                                    )
                                
                                # Update Gap Placeholder
                                with gap_placeholder.container():
                                    curr_sharpe = metrics.get('current_sharpe', 0)
                                    opt_sharpe = metrics.get('optimal_sharpe', 0)
                                    gap = opt_sharpe - curr_sharpe
                                    color = "#859900" if gap > 0 else "#839496"
                                    st.markdown("### SHARPE GAP")
                                    st.markdown(f"## <span style='color:{color}'>+{gap:.2f}</span>", unsafe_allow_html=True)
                                    st.caption(f"**Efficiency:** {curr_sharpe:.2f} → {opt_sharpe:.2f}")
                          else:
                                with hedge_placeholder.container():
                                     st.markdown("### OPTIMAL HEDGE RATIO")
                                     st.markdown("## N/A")
                                with gap_placeholder.container():
                                     st.markdown("### SHARPE GAP")
                                     st.markdown("## N/A")


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
                            
                            # Check for Structured Report (e.g. Hedging Analysis)
                            if "report_structure" in response:
                                st.session_state.generative_report_data = {
                                    "structure": response["report_structure"],
                                    "data": response.get("data", [])
                                }
                                st.session_state.current_page = "generative_insights"
                                st.rerun()

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
