"""
Reorder Recommendations Page

Streamlit page that displays data-driven reorder recommendations
based on actual usage patterns instead of manual eyeballing.
"""

import datetime
import io
import logging

import pandas as pd
import streamlit as st

from db_pool import get_connection as get_pooled_connection
from reorder_math import get_reorder_recommendations, calculate_average_daily_usage
from constants import RAW_MATERIAL_CLASS_CODES

LOGGER = logging.getLogger(__name__)


def render_reorder_recommendations():
    """Render the reorder recommendations dashboard."""
    st.header("📦 Reorder Recommendations")
    st.caption("Data-driven ordering based on actual usage patterns — no more eyeballing!")
    
    # Configuration sidebar
    with st.sidebar:
        st.subheader("⚙️ Settings")
        
        lookback_days = st.selectbox(
            "Usage Lookback Period",
            options=[30, 60, 90, 180],
            index=2,  # Default 90 days
            help="Days of history to calculate average usage"
        )
        
        safety_days = st.slider(
            "Safety Buffer (days)",
            min_value=3,
            max_value=21,
            value=7,
            help="Extra days of stock to maintain as safety buffer"
        )
        
        show_all = st.checkbox(
            "Show all items",
            value=False,
            help="Include items not below reorder point"
        )
        
        urgency_filter = st.multiselect(
            "Filter by Urgency",
            options=["Critical", "Soon", "OK"],
            default=["Critical", "Soon"] if not show_all else ["Critical", "Soon", "OK"]
        )
    
    # Load data
    with st.spinner("Calculating recommendations from usage history..."):
        try:
            with get_pooled_connection() as conn:
                cursor = conn.cursor()
                df = get_reorder_recommendations(
                    cursor,
                    lookback_days=lookback_days,
                    safety_days=safety_days,
                    only_below_rop=not show_all
                )
        except Exception as e:
            st.error(f"Error loading data: {e}")
            LOGGER.error(f"Reorder recommendations error: {e}")
            return
    
    if df.empty:
        st.info("No items require ordering at this time. 🎉")
        return
    
    # Apply urgency filter
    if urgency_filter:
        df = df[df["urgency"].isin(urgency_filter)]
    
    if df.empty:
        st.info("No items match the selected filters.")
        return
    
    # Summary KPIs
    _render_summary_kpis(df)
    
    st.divider()
    
    # Main recommendations table
    _render_recommendations_table(df)
    
    st.divider()
    
    # Export option
    _render_export_section(df)


def _render_summary_kpis(df: pd.DataFrame):
    """Render summary KPI cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    critical_count = len(df[df["urgency"] == "Critical"])
    soon_count = len(df[df["urgency"] == "Soon"])
    ok_count = len(df[df["urgency"] == "OK"])
    total_items = len(df)
    
    with col1:
        st.metric(
            "🚨 Critical",
            critical_count,
            help="Items past due or must order today"
        )
    
    with col2:
        st.metric(
            "⚠️ Order Soon",
            soon_count,
            help="Items to order within 7 days"
        )
    
    with col3:
        st.metric(
            "✅ OK for Now",
            ok_count,
            help="Items with sufficient coverage"
        )
    
    with col4:
        st.metric(
            "📋 Total Items",
            total_items,
            help="Total items in analysis"
        )


def _render_recommendations_table(df: pd.DataFrame):
    """Render the main recommendations table."""
    st.subheader("📊 Order Recommendations")
    
    # Prepare display dataframe
    display_df = df[[
        "item_number",
        "item_description",
        "qty_on_hand",
        "qty_on_order",
        "avg_daily_usage",
        "lead_time_days",
        "lead_time_source",
        "days_of_coverage",
        "calculated_rop",
        "suggested_order_qty",
        "must_order_by",
        "urgency",
        "vendor_name"
    ]].copy()
    
    # Combine lead time with source for display
    display_df["Lead Time"] = display_df.apply(
        lambda r: f"{int(r['lead_time_days'])}d ({r['lead_time_source'][:4]})",
        axis=1
    )
    display_df = display_df.drop(columns=["lead_time_days", "lead_time_source"])
    
    # Format columns
    display_df.columns = [
        "Item",
        "Description",
        "On Hand",
        "On Order",
        "Avg Daily Usage",
        "Days Coverage",
        "Reorder Point",
        "Order Qty",
        "Must Order By",
        "Urgency",
        "Vendor",
        "Lead Time"
    ]
    
    # Reorder columns to put Lead Time after Avg Daily Usage
    display_df = display_df[["Item", "Description", "On Hand", "On Order", "Avg Daily Usage", "Lead Time", "Days Coverage", "Reorder Point", "Order Qty", "Must Order By", "Urgency", "Vendor"]]
    
    # Round numeric columns
    for col in ["On Hand", "On Order", "Avg Daily Usage", "Days Coverage", "Reorder Point", "Order Qty"]:
        display_df[col] = display_df[col].round(1)
    
    # Add styling for urgency
    def style_urgency(val):
        if val == "Critical":
            return "background-color: #ff4b4b; color: white; font-weight: bold"
        elif val == "Soon":
            return "background-color: #ffa500; color: white"
        return ""
    
    styled_df = display_df.style.applymap(
        style_urgency,
        subset=["Urgency"]
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Show formula explanation
    with st.expander("📐 How these numbers are calculated"):
        st.markdown("""
        **Average Daily Usage** = Total usage over lookback period ÷ Number of days
        
        **Lead Time** = Average days from PO placement to receipt (from historical PO data)
        - "Hist" = Calculated from actual PO history
        - "Conf" = Fallback to GP's configured value (no PO history found)
        
        **Reorder Point (ROP)** = (Avg Daily Usage × Lead Time) + Safety Stock
        - Safety Stock = Avg Daily Usage × Safety Days
        
        **Days of Coverage** = (On Hand + On Order) ÷ Avg Daily Usage
        
        **Order Quantity** = Order Up To Qty − On Hand − On Order
        
        **Must Order By** = Today + Days of Coverage − Lead Time − Safety Days
        """)


def _render_export_section(df: pd.DataFrame):
    """Render export buttons."""
    st.subheader("📥 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Excel export
        export_df = df[[
            "item_number",
            "item_description",
            "item_class",
            "qty_on_hand",
            "qty_on_order",
            "qty_available",
            "avg_daily_usage",
            "days_of_coverage",
            "gp_order_point",
            "calculated_rop",
            "suggested_order_qty",
            "must_order_by",
            "urgency",
            "vendor_id",
            "vendor_name",
            "lead_time_days",
            "lead_time_source",
            "lead_time_samples",
            "safety_days"
        ]].copy()
        
        export_df.columns = [
            "Item Number",
            "Description",
            "Item Class",
            "Qty On Hand",
            "Qty On Order",
            "Qty Available",
            "Avg Daily Usage",
            "Days of Coverage",
            "GP Order Point",
            "Calculated ROP",
            "Suggested Order Qty",
            "Must Order By",
            "Urgency",
            "Vendor ID",
            "Vendor Name",
            "Lead Time (Days)",
            "Lead Time Source",
            "Lead Time Samples",
            "Safety Buffer (Days)"
        ]
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Reorder Recommendations")
        buffer.seek(0)
        
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        st.download_button(
            label="📥 Download Excel",
            data=buffer,
            file_name=f"reorder_recommendations_{today_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV export
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"reorder_recommendations_{today_str}.csv",
            mime="text/csv"
        )


# Entry point for standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="Reorder Recommendations",
        page_icon="📦",
        layout="wide"
    )
    render_reorder_recommendations()
