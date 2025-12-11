from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is expected but guard against missing dep
    pd = None


def summarize_sql_context(context: dict | None) -> str:
    """Summarize context for the LLM, including time-series and market data when available."""
    if not isinstance(context, dict):
        return ""
    pieces = []
    
    # Basic context
    if intent := context.get("intent"): pieces.append(f"intent={intent}")
    if item := context.get("item"): pieces.append(f"item={item}")
    if month := context.get("month"): pieces.append(f"month={month}")
    if year := context.get("year"): pieces.append(f"year={year}")
    if notes := context.get("notes"): pieces.append(f"notes={notes}")
    
    # Product Insights context (time-series data)
    if selected := context.get("selected_item"):
        pieces.append(f"VIEWING_ITEM={selected}")
        if desc := context.get("item_description"):
            pieces.append(f"description={desc[:50]}")
        if cat := context.get("category"):
            pieces.append(f"category={cat}")
        
        # Cost/Inventory
        if (curr := context.get("current_cost")) is not None:
            pieces.append(f"current_cost=${curr:.2f}")
        if status := context.get("stock_status"):
            pieces.append(f"stock_status={status}")
        if (on_hand := context.get("on_hand")) is not None:
            pieces.append(f"on_hand={on_hand:,.0f}")
        if (available := context.get("available")) is not None:
            pieces.append(f"available={available:,.0f}")
        if (on_order := context.get("on_order")) is not None:
            pieces.append(f"on_order={on_order:,.0f}")
        
        # Price History Summary
        if price_sum := context.get("price_history_summary"):
            if isinstance(price_sum, dict):
                if (avg := price_sum.get("avg_cost")) is not None:
                    pieces.append(f"avg_purchase_cost=${avg:.4f}")
                if (min_c := price_sum.get("min_cost")) is not None:
                    pieces.append(f"min_cost=${min_c:.4f}")
                if (max_c := price_sum.get("max_cost")) is not None:
                    pieces.append(f"max_cost=${max_c:.4f}")
                if vendors := price_sum.get("vendors"):
                    pieces.append(f"vendors={','.join(vendors[:5])}")
                if (ytd := price_sum.get("ytd_total_spend")) is not None:
                    pieces.append(f"ytd_spend=${ytd:,.2f}")
                if monthly := price_sum.get("monthly_spend_current_year"):
                    spend_str = ", ".join(f"{m}:${v:,.0f}" for m, v in list(monthly.items())[:6])
                    pieces.append(f"monthly_spend=[{spend_str}]")
        
        # Market context
        if trend := context.get("market_trend"):
            pieces.append(f"market_trend={trend}")
        if forecast := context.get("demand_forecast"):
            pieces.append(f"demand_forecast={forecast}")
            
        # Usage Context
        if usage_sum := context.get("usage_history_summary"):
            if isinstance(usage_sum, dict):
                if (avg_u := usage_sum.get("avg_usage_qty")) is not None:
                    pieces.append(f"avg_usage_qty={avg_u:,.0f}")
                if (total_u := usage_sum.get("total_usage_qty")) is not None:
                    pieces.append(f"total_usage_180d={total_u:,.0f}")
                if trend_u := usage_sum.get("usage_trend"):
                    pieces.append(f"usage_trend={trend_u}")
                if monthly_u := usage_sum.get("monthly_usage_current_year"):
                    u_str = ", ".join(f"{m}:{v:,.0f}" for m, v in list(monthly_u.items())[:6])
                    pieces.append(f"monthly_usage_current=[{u_str}]")
                if monthly_last := usage_sum.get("monthly_usage_last_year"):
                    l_str = ", ".join(f"{m}:{v:,.0f}" for m, v in list(monthly_last.items())[:6])
                    pieces.append(f"monthly_usage_last_year=[{l_str}]")
    
    return "; ".join(pieces)


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def summarize_chat_history(messages: list | None, max_messages: int = 6, max_chars: int = 320) -> str:
    """
    Condense the last few user/assistant messages so the model can keep context without large payloads.
    Includes SQL snippets when present and trims overly long content.
    """
    if not messages:
        return ""

    history_lines = []
    allowed_roles = {"user", "assistant"}
    recent_messages = [m for m in messages if isinstance(m, dict) and m.get("role") in allowed_roles]
    for msg in recent_messages[-max_messages:]:
        role = msg.get("role", "").capitalize()
        text = str(msg.get("content", "")).strip()
        if not text:
            continue

        if sql := msg.get("sql"):
            sql_snippet = str(sql).replace("\n", " ").strip()
            sql_snippet = _truncate_text(sql_snippet, 120)
            text = f"{text} [SQL: {sql_snippet}]"

        text = _truncate_text(text, max_chars)
        history_lines.append(f"{role}: {text}")

    return "\n".join(history_lines)


def _summarize_dataframe(df: Any, max_rows: int = 3, max_cols: int = 4, max_chars: int = 260) -> str:
    """Return a compact preview of a dataframe for working-memory context."""
    if pd is None:
        return ""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    try:
        preview = df.iloc[:max_rows, :max_cols]
        columns = [str(col) for col in preview.columns]
        records = preview.to_dict(orient="records")
        samples: list[str] = []
        for rec in records:
            bits = []
            for col in columns:
                val = rec.get(col)
                if isinstance(val, float):
                    val = round(val, 4)
                bits.append(f"{col}={val}")
            samples.append(", ".join(bits))
        details = f"{len(df)} rows x {len(df.columns)} cols. Sample: " + " | ".join(samples)
        return _truncate_text(details, max_chars)
    except Exception:
        return ""


def summarize_memory_buffer(
    messages: list | None,
    max_items: int = 3,
    max_rows: int = 3,
    max_chars: int = 900,
) -> str:
    """
    Surface the last few assistant answers (data, SQL, chart spec, entities) as a working-memory hint.
    This gives the LLM the intermediate results it may need to answer follow-ups like "chart it".
    """
    if not messages:
        return ""

    snippets: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        parts: list[str] = []
        content = msg.get("content")
        if content:
            parts.append(_truncate_text(str(content), 220))

        entities = msg.get("entities") if isinstance(msg.get("entities"), dict) else {}
        if entities:
            entity_bits = [f"{k}={v}" for k, v in entities.items() if v not in (None, "")]
            if entity_bits:
                parts.append("entities: " + ", ".join(entity_bits[:6]))

        if sql := msg.get("sql"):
            sql_snippet = str(sql).replace("\n", " ").strip()
            parts.append("sql: " + _truncate_text(sql_snippet, 180))

        df_summary = _summarize_dataframe(msg.get("df"), max_rows=max_rows)
        if df_summary:
            parts.append("data: " + df_summary)

        chart = msg.get("chart") if isinstance(msg.get("chart"), dict) else None
        if chart:
            chart_bits = []
            if chart.get("type"):
                chart_bits.append(f"type={chart['type']}")
            if chart.get("x"):
                chart_bits.append(f"x={chart['x']}")
            if chart.get("y"):
                chart_bits.append(f"y={chart['y']}")
            parts.append("chart: " + ", ".join(chart_bits) if chart_bits else "chart: provided")

        if not parts:
            continue

        snippets.append(" | ".join(parts))
        if len(snippets) >= max_items:
            break

    if not snippets:
        return ""

    snippets.reverse()
    combined = "\n".join(f"{idx + 1}) {snippet}" for idx, snippet in enumerate(snippets))
    return _truncate_text("Working memory:\n" + combined, max_chars)


def build_conversation_hint(messages: list | None, max_history: int = 6) -> str:
    """Combine recent history and working memory into a single hint for model calls."""
    history = summarize_chat_history(messages, max_messages=max_history)
    memory = summarize_memory_buffer(messages)
    parts = [p for p in (history, memory) if p]
    return "\n\n".join(parts)
