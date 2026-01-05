"""
ML Buy Recommendations Dashboard

Streamlit page for ML-powered procurement recommendations.
Shows only critical items that MUST be bought today.
"""

import datetime
import logging

import pandas as pd
import pyodbc
import streamlit as st

from constants import RAW_MATERIAL_CLASS_CODES
from procurement_ml import (
    ProcurementMLOptimizer,
    WalkForwardValidator,
    CriticalBuyFilter,
)
from secrets_loader import build_connection_string

LOGGER = logging.getLogger(__name__)


def render_ml_buy_dashboard():
    """Render the ML-powered buy recommendations dashboard."""
    st.title("🤖 ML Buy Recommendations")
    st.caption("AI-powered procurement timing | Only showing items that need action TODAY")
    
    # Connection
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return
    
    # Tabs for different functions
    tab_critical, tab_validate, tab_train = st.tabs([
        "🚨 Critical Items",
        "📊 Model Validation",
        "🎓 Train Model"
    ])
    
    with tab_critical:
        _render_critical_items(cursor)
    
    with tab_validate:
        _render_validation(cursor)
    
    with tab_train:
        _render_training(cursor)
    
    conn.close()


def _render_critical_items(cursor):
    """Show only items that absolutely must be bought today."""
    st.subheader("Must-Buy Items for Today")
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
                recs_df = optimizer.get_batch_recommendations(batch)
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
                st.success("✅ **No critical items today!** All inventory levels are healthy.")
            else:
                st.error(f"🚨 **{len(critical_df)} Critical Items Require Action**")
                
                # Summary by reason
                col1, col2 = st.columns(2)
                with col1:
                    n_low_stock = (critical_df['CriticalReason'] == 'LOW STOCK - Must buy').sum()
                    st.metric("Low Stock", n_low_stock, delta="urgent" if n_low_stock > 0 else None, delta_color="inverse")
                with col2:
                    n_opportunity = len(critical_df) - n_low_stock
                    st.metric("Price Opportunity", n_opportunity)
                
                # Table
                st.dataframe(
                    critical_df[[
                        'ItemNumber', 'BuyScore', 'DaysCoverage', 
                        'Price52wPct', 'Confidence', 'CriticalReason'
                    ]].style.format({
                        'BuyScore': '{:.0f}',
                        'DaysCoverage': '{:.1f} days',
                        'Price52wPct': '{:.0%}',
                        'Confidence': '{:.0%}',
                    }).background_gradient(
                        subset=['DaysCoverage'], 
                        cmap='RdYlGn',
                        vmin=0, vmax=30
                    ),
                    use_container_width=True,
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
                use_container_width=True
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
