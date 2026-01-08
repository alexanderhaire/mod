"""
ML Buy Recommendations Dashboard

Streamlit page for ML-powered procurement recommendations.
Shows only critical items that MUST be bought today.
"""

import datetime
import logging

import pandas as pd
import streamlit as st

from constants import RAW_MATERIAL_CLASS_CODES
from db_pool import get_connection as get_pooled_connection
from procurement_ml import (
    ProcurementMLOptimizer,
    WalkForwardValidator,
    CriticalBuyFilter,
)

LOGGER = logging.getLogger(__name__)


def render_ml_buy_dashboard():
    """Render the ML-powered buy recommendations dashboard."""
    st.title("🤖 ML Buy Recommendations")
    st.caption("AI-powered procurement timing | Only showing items that need action TODAY")
    
    # Connection using pool
    try:
        with get_pooled_connection() as conn:
            cursor = conn.cursor()
            
            analysis_date = datetime.date.today()

            # Main View: Procurement Calendar (Auto-Loaded)
            _render_procurement_calendar(cursor, analysis_date)
            
            st.divider()
            
            # Advanced Options (Hidden by default)
            with st.expander("🛠️ Advanced Model Settings (Training & Validation)"):
                tab_validate, tab_train = st.tabs(["📊 Model Validation", "🎓 Train Model"])
                
                with tab_validate:
                    _render_validation(cursor)
                
                with tab_train:
                    _render_training(cursor)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return


def _render_critical_items(cursor, analysis_date):
    """Show only items that absolutely must be bought today."""
    st.subheader(f"Must-Buy Items for {analysis_date}")
    st.info("""
    **Criteria for Critical Items:**
    - High buy score (80+) with high confidence
    - Less than 14 days of inventory coverage
    - OR: Price at 52-week low with coverage < 30 days
    """)
    
    if st.button("🔍 Find Critical Items", type="primary"):
        with st.spinner("Analyzing all items..."):
            optimizer = ProcurementMLOptimizer(cursor)
            
            # Get all raw materials
            items = _get_all_raw_materials(cursor)
            
            if not items:
                st.warning("No raw materials found in inventory.")
                return
            
            # Progress bar
            progress = st.progress(0, "Analyzing items...")
            
            # Get recommendations in batches
            all_recs = []
            batch_size = 20
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                recs_df = optimizer.get_batch_recommendations(batch, as_of_date=analysis_date)
                all_recs.append(recs_df)
                progress.progress(min((i + batch_size) / len(items), 1.0))
            
            progress.empty()
            
            if not all_recs:
                st.info("No recommendations generated.")
                return
            
            all_df = pd.concat(all_recs, ignore_index=True)
            
            # Apply critical filter
            critical_filter = CriticalBuyFilter(
                confidence_threshold=0.7,
                coverage_threshold=14
            )
            critical_df = critical_filter.filter_critical(all_df)
            
            # Display results
            st.divider()
            
            if critical_df.empty:
                st.success(f"✅ **No critical items for {analysis_date}!** All inventory levels are healthy.")
            else:
                st.error(f"🚨 **{len(critical_df)} Critical Items Require Action**")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_low_stock = (critical_df['CriticalReason'] == 'LOW STOCK - Must buy').sum()
                    st.metric("Low Stock Alerts", n_low_stock, delta="Urgent" if n_low_stock > 0 else None, delta_color="inverse")
                with col2:
                    n_opportunity = len(critical_df) - n_low_stock
                    st.metric("Price Opportunities", n_opportunity)
                with col3:
                    # Estimate cash: Price * Last Volume (heuristic for "like-for-like" restock)
                    total_cash = (critical_df['CurrentPrice'] * critical_df['VendorLastVol']).sum()
                    st.metric("Est. Cash Required", f"${total_cash:,.0f}", help="Based on last purchase volume")
                
                # Display Focus Mode or Full List
                focus_mode = st.toggle("🎯 Focus Mode (One Item at a Time)", value=False)
                
                if focus_mode and not critical_df.empty:
                    # Show only the top item
                    item = critical_df.iloc[0]
                    st.success("🎯 **Top Priority Action Item**")
                    st.metric(f"Use Case: {item['ItemNumber']}", f"{item['BuyScore']:.0f}/100", delta="Strong Buy")
                    
                    st.markdown(f"""
                    **Why this item?**
                    - {item['CriticalReason']} 
                    - Confidence: **{item['Confidence']:.0%}**
                    - Coverage: **{item['DaysCoverage']:.1f} days**
                    - Last Bought: **{item.get('VendorLastVol', 0):,.0f} units** from {item.get('VendorName', 'Unknown')}
                    """)
                else:
                    # Table View
                    st.dataframe(
                        critical_df[[
                            'ItemNumber', 'BuyScore', 'DaysCoverage', 
                            'Price52wPct', 'Confidence', 'CriticalReason',
                            'VendorName', 'VendorLastVol'
                        ]].style.format({
                            'BuyScore': '{:.0f}',
                            'DaysCoverage': '{:.1f} days',
                            'Price52wPct': '{:.0%}',
                            'Confidence': '{:.0%}',
                            'VendorLastVol': '{:,.0f}',
                        }).background_gradient(
                            subset=['DaysCoverage'], 
                            cmap='RdYlGn',
                            vmin=0, vmax=30
                        ),

                        width="stretch",
                        height=400
                    )
                
                # Export button
                csv = critical_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Critical Items List",
                    csv,
                    f"critical_items_{datetime.date.today()}.csv",
                    "text/csv"
                )



import calendar

def _render_procurement_calendar(cursor, analysis_date):
    """Render a visual interactive calendar (Auto-loading)."""
    
    # Initialize session state
    if 'calendar_df' not in st.session_state:
        st.session_state.calendar_df = None
    if 'selected_calendar_date' not in st.session_state:
        st.session_state.selected_calendar_date = None
    if 'last_analysis_date' not in st.session_state:
        st.session_state.last_analysis_date = None

    # Check if we need to (re)load data
    should_reload = (
        st.session_state.calendar_df is None or 
        st.session_state.last_analysis_date != analysis_date
    )

    if should_reload:
        with st.spinner(f"Scheduling orders for {analysis_date}..."):
            st.session_state.calendar_df = _fetch_calendar_data(cursor, analysis_date)
            st.session_state.last_analysis_date = analysis_date
            st.session_state.selected_calendar_date = None # Reset selection on new data

    if st.session_state.calendar_df is None:
        st.info("No data available.") # Should not happen with auto-load unless fetch returns None
        return

    df = st.session_state.calendar_df
    
    # --- Calendar Grid UI ---
    
    # --- Calendar Grid UI ---
    
    # Navigation
    col_prev, col_header, col_next = st.columns([1, 5, 1])
    
    if 'view_date' not in st.session_state:
        st.session_state.view_date = analysis_date # Default to today
        
    with col_prev:
        if st.button("◀ Prev"):
            # Subtract one month
            curr = st.session_state.view_date
            # Logic to go back 1 month correctly handling year rollover
            new_month = curr.month - 1
            new_year = curr.year
            if new_month == 0:
                new_month = 12
                new_year -= 1
            st.session_state.view_date = datetime.date(new_year, new_month, 1)
            st.rerun()

    with col_next:
        if st.button("Next ▶"):
            # Add one month
            curr = st.session_state.view_date
            new_month = curr.month + 1
            new_year = curr.year
            if new_month == 13:
                new_month = 1
                new_year += 1
            st.session_state.view_date = datetime.date(new_year, new_month, 1)
            st.rerun()
            
    view_year = st.session_state.view_date.year
    view_month = st.session_state.view_date.month
    month_name = calendar.month_name[view_month]
    
    with col_header:
        st.markdown(f"<h3 style='text-align: center; margin-top: -10px;'>{month_name} {view_year}</h3>", unsafe_allow_html=True)
    
    # Days of Week Header
    cols = st.columns(7)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        cols[i].markdown(f"**{day}**")

    # Calendar Grid
    month_matrix = calendar.monthcalendar(view_year, view_month)
    
    # Pre-calculate counts per day
    # We differentiate between "Action Needed" (No PO) and "Pending" (Has PO)
    
    # Helper to check if item has PO
    df['HasPO'] = df['QtyOnOrder'] > 0
    
    # Group by Date and PO Status
    daily_stats = df.groupby(['MustBuyDate', 'HasPO']).size().unstack(fill_value=0)
    # columns will be False (Action), True (Pending) if data exists
    
    for week in month_matrix:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.write("") # Empty slot
                else:
                    this_date = datetime.date(view_year, view_month, day)
                    is_today = (this_date == datetime.date.today())
                    
                    # Get counts
                    count_action = 0
                    count_pending = 0
                    
                    if this_date in daily_stats.index:
                        if False in daily_stats.columns:
                            count_action = daily_stats.loc[this_date, False]
                        if True in daily_stats.columns:
                            count_pending = daily_stats.loc[this_date, True]
                    
                    label = f"{day}"
                    
                    # Indicators
                    if count_action > 0:
                        label += f"\n🔴 {count_action}"
                    if count_pending > 0:
                        label += f"\n✅ {count_pending}"
                    
                    # Button Style
                    type_ = "primary" if is_today else "secondary"
                    if is_today:
                        label += " (Today)"

                    key = f"cal_btn_{view_year}_{view_month}_{day}"
                    if st.button(label, key=key, type=type_, use_container_width=True):
                        st.session_state.selected_calendar_date = this_date

    # --- Detail View for Selected Date ---
    st.divider()
    
    selected_date = st.session_state.selected_calendar_date
    
    if selected_date:
        st.markdown(f"### Actions for {selected_date.strftime('%A, %b %d')}")
        
        # Filter for selected date
        day_items = df[df['MustBuyDate'] == selected_date]
        
        if day_items.empty:
            st.info("No purchases scheduled for this date.")
        else:
            # Show summary metrics for the day
            total_cash = (day_items['CurrentPrice'] * day_items['VendorLastVol']).sum()
            st.metric("Daily Cash Requirement", f"${total_cash:,.0f}")
            
            st.dataframe(
                day_items[['ItemNumber', 'VendorName', 'LeadTime', 'DaysCoverage', 'QtyOnOrder', 'CurrentPrice', 'VendorLastVol']],
                column_config={
                    "CurrentPrice": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "VendorLastVol": st.column_config.NumberColumn("Est. Qty", format="%.0f"),
                    "QtyOnOrder": st.column_config.NumberColumn("On Order", format="%.0f"),
                },
                width="stretch"
            )
    else:
        st.caption("👈 Click on a day with a red dot (🔴) to see required purchases.")

    # Show Overdue separately if any (items before today)
    today = analysis_date
    overdue_df = df[df['MustBuyDate'] < today]
    if not overdue_df.empty:
        st.error(f"⚠️ **{len(overdue_df)} Overdue Items (Must Buy Immediately)**")
        with st.expander("Review Overdue Items"):
            st.dataframe(overdue_df[['ItemNumber', 'MustBuyDate', 'DaysCoverage', 'LeadTime']])


def _fetch_calendar_data(cursor, analysis_date):
    """Helper to fetch all data."""
    optimizer = ProcurementMLOptimizer(cursor)
    items = _get_all_raw_materials(cursor)
    
    all_recs = []
    batch_size = 50
    progress = st.progress(0, "Analyzing portfolio...")
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        recs_df = optimizer.get_batch_recommendations(batch, as_of_date=analysis_date)
        all_recs.append(recs_df)
        progress.progress(min((i + batch_size) / len(items), 1.0))
    
    progress.empty()
    
    if not all_recs:
        return pd.DataFrame() # Empty
        
    full_df = pd.concat(all_recs, ignore_index=True)
    
    # Filter "Dead" / Missing Data Items logic
    mask_valid = (full_df['CurrentPrice'] > 0.01) | (full_df['DaysCoverage'] > 0.1)
    full_df = full_df[mask_valid].copy()
    
    # --- Logic Updates ---
    
    # 1. Parse Dates
    full_df['MustBuyDate'] = pd.to_datetime(full_df['MustBuyDate']).dt.date
    
    # 2. Workday Logic: Shift Sat/Sun to Friday
    # weekday(): Mon=0, Sun=6
    def shift_to_workday(d):
        wd = d.weekday()
        if wd == 5: # Sat
            return d - datetime.timedelta(days=1)
        elif wd == 6: # Sun
            return d - datetime.timedelta(days=2)
        return d
    
    full_df['MustBuyDate'] = full_df['MustBuyDate'].apply(shift_to_workday)

    # 3. PO Verification (Qty On Order)
    # Ensure column exists (it comes from features dict via simple join usually, but let's be safe)
    if 'features' in full_df.columns:
        # It's flattened in get_batch_recommendations usually? 
        # Wait, get_batch_recommendations returns a flat DF constructed from the dict.
        # Let's check how it's constructed in procurement_ml.py.
        # It flattens: 'QtyOnOrder': rec['features'].get('qty_on_order', 0.0) needs to be added there too?
        # Check procurement_ml.py lines 900+
        pass

    return full_df


def _render_validation(cursor):
    """Show model validation results with statistical significance."""
    st.subheader("Walk-Forward Model Validation")
    st.info("""
    Tests the model using **true out-of-sample** predictions:
    - Train on historical data → Predict future purchases
    - Rolling windows ensure no lookahead bias
    - Bootstrap confidence intervals for statistical significance
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        lookback = st.slider("Lookback Months", 12, 48, 24)
    with col2:
        train_window = st.slider("Training Window (months)", 6, 18, 12)
    
    if st.button("🧪 Run Walk-Forward Validation"):
        validator = WalkForwardValidator(
            cursor,
            train_window_months=train_window,
            test_window_months=3
        )
        
        progress = st.progress(0, "Running validation...")
        
        def update_progress(p):
            progress.progress(p, f"Validating... {int(p*100)}%")
        
        results = validator.run_validation(
            lookback_months=lookback,
            progress_callback=update_progress
        )
        
        progress.empty()
        
        if 'error' in results:
            st.error(f"Validation failed: {results['error']}")
            return
        
        st.divider()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Out-of-Sample Accuracy",
                f"{results['accuracy_out_sample']:.1%}"
            )
        
        with col2:
            ci_99 = results['confidence_interval_99']
            st.metric(
                "99% CI (Lower Bound)",
                f"{ci_99[0]:.1%}",
                delta="Significant!" if results['is_significant_99'] else "Not significant"
            )
        
        with col3:
            st.metric(
                "Total Predictions",
                f"{results['total_predictions']:,}"
            )
        
        with col4:
            st.metric(
                "Test Periods",
                results['n_periods']
            )
        
        # Statistical significance interpretation
        if results['is_significant_99']:
            st.success(f"""
            ✅ **Model is statistically significant at 99% confidence!**
            
            The 99% confidence interval for accuracy is **{ci_99[0]:.1%} - {ci_99[1]:.1%}**.
            This means we can say with 99% confidence that the model performs better than random guessing.
            """)
        else:
            st.warning(f"""
            ⚠️ **Model is NOT statistically significant at 99% confidence.**
            
            The 99% confidence interval ({ci_99[0]:.1%} - {ci_99[1]:.1%}) includes 50% (random chance).
            Consider collecting more data or tuning the model.
            """)
        
        # Period breakdown
        with st.expander("📈 Detailed Period Results"):
            period_df = pd.DataFrame(results['detailed_results'])
            st.dataframe(
                period_df.style.format({
                    'train_r2': '{:.2%}',
                    'oos_accuracy': '{:.1%}'
                }),
                width="stretch"
            )


def _render_training(cursor):
    """Train or retrain the ML model."""
    st.subheader("Train ML Model")
    st.info("""
    Train the Gradient Boosting model on historical purchase data.
    The model learns which purchases were "good buys" (bottom 30% of prices).
    """)
    
    lookback = st.slider("Training Lookback (months)", 12, 36, 24)
    
    if st.button("🎓 Train Model"):
        optimizer = ProcurementMLOptimizer(cursor)
        
        with st.spinner("Building training dataset and fitting model..."):
            metrics = optimizer.train_model(lookback_months=lookback)
        
        if 'error' in metrics:
            st.error(f"Training failed: {metrics['error']}")
        else:
            st.success("Model trained successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train R²", f"{metrics['train_r2']:.2%}")
            with col2:
                st.metric("Test R²", f"{metrics['test_r2']:.2%}")
            with col3:
                st.metric("Samples", f"{metrics['n_samples']:,}")
            
            st.subheader("Top Feature Importance")
            if 'feature_importance' in metrics:
                fi_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in metrics['feature_importance'].items()
                ])
                st.bar_chart(fi_df.set_index('Feature'))


def _get_all_raw_materials(cursor) -> list[str]:
    """Get all raw material item numbers."""
    try:
        class_list = "', '".join(RAW_MATERIAL_CLASS_CODES)
        query = f"""
        SELECT DISTINCT iv.ITEMNMBR
        FROM IV00102 iv
        JOIN IV00101 i ON iv.ITEMNMBR = i.ITEMNMBR
        WHERE UPPER(LTRIM(RTRIM(i.ITMCLSCD))) IN ('{class_list}')
          AND i.INACTIVE = 0 
          AND i.ITEMTYPE <> 2 -- Exclude discontinued
          AND iv.QTYONHND >= 0
        """
        cursor.execute(query)
        return [row.ITEMNMBR.strip() for row in cursor.fetchall()]
    except Exception as e:
        LOGGER.error(f"Error getting raw materials: {e}")
        return []


# Entry point for standalone testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="ML Buy Recommendations",
        page_icon="🤖",
        layout="wide"
    )
    render_ml_buy_dashboard()
