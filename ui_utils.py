"""
UI Utility Functions

Common UI rendering functions extracted from app.py for reuse across pages.
"""

import calendar
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import altair as alt
except ImportError:
    alt = None

# Import calendar utilities from centralized module
from calendar_utils import (
    MONTH_ORDER,
    MONTH_LOOKUP,
    month_number,
    format_month_label,
)

LOGGER = logging.getLogger(__name__)


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency with commas and dollar sign."""
    if value is None:
        return "$0.00"
    return f"${value:,.{decimals}f}"


def render_kpi_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> None:
    """Render a styled KPI card using Streamlit metrics."""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def render_trend_line(data: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> None:
    """Render a simple trend line chart."""
    if data is None or data.empty:
        st.caption("No data available")
        return
    
    if alt:
        chart = alt.Chart(data).mark_line(strokeWidth=2).encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[x_col, y_col]
        ).properties(title=title).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(data.set_index(x_col)[y_col])


def render_dataframe_with_selection(df: pd.DataFrame, key: str = "df_select") -> pd.DataFrame | None:
    """Render a dataframe with row selection capability."""
    if df is None or df.empty:
        st.caption("No data to display")
        return None
    
    # Use Streamlit's native dataframe with selection
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key
    )
    
    if event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        return df.iloc[[selected_idx]]
    return None


def match_column(df: pd.DataFrame, name: str | None) -> str | None:
    """Return the DataFrame column matching the provided name (case-insensitive)."""
    if df is None or name is None:
        return None
    target = str(name).strip().lower()
    for col in df.columns:
        if str(col).lower() == target:
            return col
    return None


def dense_month_series(
    df: pd.DataFrame, 
    month_col: str, 
    value_col: str, 
    year_hint: int | None = None
) -> pd.DataFrame:
    """
    Ensure the month/value columns are fully populated for a calendar year with zeros for missing months.
    Preserves a consistent month order for charts and tables.
    """
    if df.empty:
        return df
    
    working = df.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working["__month_num"] = working[month_col].apply(month_number)
    
    # If any months can't be parsed, just format existing data and return
    if working["__month_num"].isna().any():
        working[month_col] = working[month_col].apply(format_month_label)
        return working.drop(columns="__month_num")
    
    # Sum values per month
    aggregated = working.groupby("__month_num", as_index=False)[value_col].sum()
    
    # Fill missing months with zeros
    full = pd.DataFrame({"__month_num": range(1, 13)})
    dense = full.merge(aggregated, on="__month_num", how="left").fillna({value_col: 0})
    dense[month_col] = dense["__month_num"].apply(lambda m: calendar.month_name[m])
    
    # Preserve static columns that have a single unique value
    static_cols = [c for c in working.columns if c not in {month_col, value_col, "__month_num"}]
    for col in static_cols:
        uniques = working[col].dropna().unique()
        if len(uniques) == 1:
            dense[col] = uniques[0]
    
    # Add year column if hinted and not already present
    if year_hint and "Year" not in dense.columns:
        dense.insert(0, "Year", year_hint)
    
    # Sort by proper month order
    dense[month_col] = pd.Categorical(dense[month_col], categories=MONTH_ORDER, ordered=True)
    dense = dense.sort_values(month_col)
    dense[month_col] = dense[month_col].astype(str)
    
    return dense.drop(columns="__month_num")


def prepare_display_df(
    df: pd.DataFrame, 
    chart_spec: dict | None, 
    entities: dict | None
) -> pd.DataFrame:
    """Normalize month labels, fill missing months for time series, and keep chart/table consistent."""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Check for month column
    month_col = match_column(df, "Month")
    if month_col:
        # Check for a value column
        value_candidates = ["Usage", "Quantity", "Amount", "Value", "Total", "Sales"]
        value_col = None
        for vc in value_candidates:
            found = match_column(df, vc)
            if found:
                value_col = found
                break
        
        if value_col:
            df = dense_month_series(df, month_col, value_col)
    
    return df


def render_chart(df: pd.DataFrame, chart_spec: dict | None) -> None:
    """Render a simple bar/line/area chart when the model asks for one and columns exist."""
    if chart_spec is None or df is None or df.empty:
        return
    
    chart_type = str(chart_spec.get("type", "bar")).lower()
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    title = chart_spec.get("title", "")
    
    # Find matching columns
    x_match = match_column(df, x_col)
    y_match = match_column(df, y_col)
    
    if not x_match or not y_match:
        LOGGER.debug(f"Chart columns not found: x={x_col} -> {x_match}, y={y_col} -> {y_match}")
        return
    
    if title:
        st.caption(title)
    
    chart_df = df[[x_match, y_match]].copy()
    chart_df = chart_df.set_index(x_match)
    
    try:
        if chart_type == "line":
            st.line_chart(chart_df)
        elif chart_type == "area":
            st.area_chart(chart_df)
        else:
            st.bar_chart(chart_df)
    except Exception as e:
        LOGGER.warning(f"Failed to render chart: {e}")


def latest_question_answer(messages: list[dict]) -> tuple[int | None, dict | None]:
    """Return the index and pair of the latest user/assistant messages."""
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break
    
    if user_idx is None or user_idx + 1 >= len(messages):
        return None, None
    
    assistant_msg = messages[user_idx + 1]
    if assistant_msg.get("role") != "assistant":
        return None, None
    
    return user_idx, {
        "question": messages[user_idx].get("content", ""),
        "answer": assistant_msg.get("content", ""),
    }


def add_margin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add dollar and percent margin columns comparing standard vs current cost."""
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    std_col = match_column(df, "STNDCOST")
    cur_col = match_column(df, "CURRCOST")
    
    if std_col and cur_col:
        df["MarginDollar"] = df[cur_col] - df[std_col]
        df["MarginPercent"] = df.apply(
            lambda row: (row["MarginDollar"] / row[std_col] * 100) if row[std_col] > 0 else 0,
            axis=1
        )
    
    return df


def render_pulse_header(user_id: str = "default") -> None:
    """
    Render the system 'Pulse' header - a unifying heartbeat for the app.
    Shows System Status, Brain Health, Market Mood, and Time.
    
    Brain Health is now REAL - it reflects how well the system knows the user.
    """
    import datetime
    import random
    
    # Get REAL brain health
    try:
        from user_brain import get_brain
        brain = get_brain(user_id)
        confidence_score = brain.get_brain_health()
    except Exception:
        confidence_score = 0.10  # Fallback
    
    # Generate dynamic vibe
    vibes = [
        "🐻 Bearish but hopeful", "🐂 Bullish momentum detected", 
        "🦀 Sideways drift", "⚡ High volatility alert",
        "🧘 Zen status: Accumulating", "🧠 Deep learning active",
        "🤖 AI Reasoning: Optimal", "🌊 Liquidity pools: Deep"
    ]
    # In a real app, this would be derived from market_insights.get_market_regime()
    # For now, we simulate "Aliveness" with a consistent seed based on hour
    random.seed(datetime.datetime.now().hour)
    market_vibe = random.choice(vibes)
    
    # Time
    now = datetime.datetime.now().strftime("%H:%M EST")
    
    # CSS for the Pulse Bar
    st.markdown("""
    <style>
    .pulse-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #002b36; /* Solarized Base03 */
        border-bottom: 2px solid #2aa198; /* Cyan accent */
        padding: 10px 20px;
        margin-bottom: 20px;
        border-radius: 8px;
        font-family: 'Inconsolata', monospace;
        color: #839496; /* Base0 */
    }
    .pulse-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .pulse-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #586e75; /* Base01 */
    }
    .pulse-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #fdf6e3; /* Base3 */
    }
    .pulse-value.health { color: #859900; } /* Green */
    .pulse-value.time { color: #268bd2; } /* Blue */
    .pulse-value.vibe { color: #b58900; } /* Yellow */
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="pulse-container">
        <div class="pulse-item">
            <span class="pulse-label">System Status</span>
            <span class="pulse-value health">● ONLINE</span>
        </div>
        <div class="pulse-item">
            <span class="pulse-label">Brain Health</span>
            <span class="pulse-value">{confidence_score*100:.0f}%</span>
        </div>
        <div class="pulse-item">
            <span class="pulse-label">Market Vibe</span>
            <span class="pulse-value vibe">{market_vibe}</span>
        </div>
        <div class="pulse-item">
            <span class="pulse-label">System Time</span>
            <span class="pulse-value time">{now}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# DIGITAL BODY LANGUAGE SENSORS 👁️
# =============================================================================

def init_decision_timer():
    """Start the clock for this page view."""
    import time
    if "decision_start_time" not in st.session_state:
        st.session_state.decision_start_time = time.time()

def render_sensor_button(label: str, key: str, context: str, item: str = None) -> bool:
    """
    A button that tracks how long it took you to click it.
    Returns True if clicked.
    """
    import time
    from user_brain import get_brain
    
    clicked = st.button(label, key=key)
    
    if clicked:
        start_time = st.session_state.get("decision_start_time", time.time())
        elapsed = time.time() - start_time
        
        # Log the "Digital Body Language"
        user_id = st.session_state.get("user", "default")
        brain = get_brain(user_id)
        brain.log_decision(context=context, seconds=elapsed, item=item)
        
        # Reset timer for next action
        st.session_state.decision_start_time = time.time()
        
    return clicked
