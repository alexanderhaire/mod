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

LOGGER = logging.getLogger(__name__)

# Month lookup utilities
MONTH_ORDER = [calendar.month_name[i] for i in range(1, 13)]
MONTH_LOOKUP = {
    name.lower(): idx for idx, name in enumerate(MONTH_ORDER, start=1)
    if name
} | {calendar.month_abbr[idx].lower(): idx for idx in range(1, 13)}


def month_number(val) -> int | None:
    """Parse a month value from number/name/abbr strings into 1-12."""
    if val is None:
        return None
    if isinstance(val, int):
        return val if 1 <= val <= 12 else None
    s = str(val).strip().lower()
    if s.isdigit():
        num = int(s)
        return num if 1 <= num <= 12 else None
    return MONTH_LOOKUP.get(s)


def format_month_label(val) -> str:
    """Return a human-friendly month label, falling back to the raw value."""
    num = month_number(val)
    if num is not None:
        return calendar.month_name[num]
    return str(val)


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
    
    df = df.copy()
    df["_month_num"] = df[month_col].apply(month_number)
    df = df.dropna(subset=["_month_num"])
    df["_month_num"] = df["_month_num"].astype(int)
    
    # Sum values per month
    grouped = df.groupby("_month_num")[value_col].sum().reset_index()
    
    # Fill missing months
    all_months = pd.DataFrame({"_month_num": range(1, 13)})
    merged = all_months.merge(grouped, on="_month_num", how="left").fillna(0)
    merged[month_col] = merged["_month_num"].apply(lambda m: calendar.month_name[m])
    
    return merged[[month_col, value_col]]


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
