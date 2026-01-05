"""
Quant Dashboard Page

CDI HEDGE FUND - ALGO TRADER interface for optimizing cross-asset allocations.
Extracted from app.py for better modularity.
"""

import time
from typing import Any

import pandas as pd
import pyodbc
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None

from auto_trader import AutoTrader
from constants import FUTURES_UNIVERSE
from external_data import fetch_market_data_pool, TICKER_MAP
from market_insights import (
    get_batch_price_history_for_optimization,
    merge_erp_and_futures_data,
)
from ml_engine import PortfolioOptimizer, Backtester
from portfolio_manager import PortfolioManager
from secrets_loader import build_connection_string

import logging
LOGGER = logging.getLogger(__name__)


def render_quant_dashboard() -> None:
    """
    CDI HEDGE FUND - ALGO TRADER
    Interface for optimizing cross-asset allocations (ERP Items + Global Futures).
    """
    st.title("🏦 CDI HEDGE FUND - ALGO TRADER")
    st.caption("PROPRIETARY TRADING TERMINAL - CROSS-ASSET QUANTITATIVE DEPLOYMENT")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info("🎯 **Hedge Fund Objective:** Maximize Sharpe Ratio across internal inventory costs and external futures headers.")
    with col2:
        if st.button("🔌 Disconnect Terminal"):
            st.session_state.is_quant_mode = False
            st.session_state.user = None
            st.rerun()

    # --- PORTFOLIO DASHBOARD ---
    
    # Initialize Auto-Trader in Session State
    if "auto_trader" not in st.session_state:
        st.session_state.auto_trader = AutoTrader(mode="paper")
    
    trader = st.session_state.auto_trader
    
    pm = PortfolioManager(mode="paper")
    summary = pm.get_portfolio_summary()
    total_equity = summary["Total Value"]
    
    # LIVE TRADER CONTROL PANEL
    st.divider()
    st.subheader(f"🟢 LIVE AUTO-TRADER [{trader.get_execution_mode_label()}]")
    
    t_col1, t_col2, t_col3 = st.columns([1, 1, 2])
    
    with t_col1:
        is_running = st.toggle("ACTIVATE TRADING LOOP", key="trader_switch")
    
    with t_col2:
        st.metric("Heartbeat", f"{trader.iteration_count} ticks")
        
    with t_col3:
        if is_running:
            st.info("⚡ Neural Loop Active: Scanning Cross-Asset Correlations...")
            # THE LOOP
            placeholder = st.empty()
            
            # Run a micro-loop (e.g. 5 steps) to simulate activity without blocking forever
            # In a real deployed app, this would be a background thread.
            for _ in range(5):
                tick_result = trader.heart_beat()
                
                with placeholder.container():
                    st.json(tick_result)
                    
                    # Updates dynamic chart
                    if "equity" in tick_result:
                        st.caption(f"Simulated Equity: ${tick_result['equity']:,.2f}")
                        
                time.sleep(0.5)
            
            st.rerun() # Refresh to show new logs/state
            
        else:
            st.caption("System Standby. Activate to begin automated execution.")
    
    st.divider()
    
    # Portfolio Header
    p_col1, p_col2, p_col3 = st.columns([2, 1, 1])
    with p_col1:
        st.metric("💰 Total AUM (Assets Under Management)", f"${total_equity:,.2f}", 
                 delta=f"{summary['Total By Asset'] if 'Total By Asset' in summary else 0:,.2f} Uninvested" if total_equity > 0 else None)
    with p_col2:
        st.metric("💵 Cash Balance", f"${summary['Cash']:,.2f}")
    with p_col3:
        if st.button("💸 Deposit Capital"):
            deposit_amount = 1000.0 if total_equity < 100 else 10000.0
            pm.deposit_capital(deposit_amount)
            st.toast(f"Deposited ${deposit_amount:,.0f}!", icon="🤑")
            st.rerun()

    # Holdings View
    if summary["Holdings"]:
        with st.expander("📂 Current Portfolio Holdings", expanded=False):
            holdings_df = pd.DataFrame([
                {"Asset": k, "Value": v, "Weight": v/total_equity if total_equity > 0 else 0} 
                for k, v in summary["Holdings"].items() if v > 0.01
            ])
            if not holdings_df.empty:
                st.dataframe(holdings_df.sort_values("Value", ascending=False).style.format({
                    "Value": "${:,.2f}",
                    "Weight": "{:.1%}"
                }), use_container_width=True)
    
    if total_equity < 10:
        st.warning("⚠️ Portfolio is empty. Please Deposit Capital to begin trading.")
        
    st.divider()

    # 1. Configuration
    with st.expander("🛠️ UNIVERSE CONFIGURATION", expanded=True):
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            erp_limit = st.slider("ERP Item Depth (Top N by Spend)", 5, 100, 20)
        with u_col2:
            futures_count = st.slider("Futures Universe Size", 10, len(FUTURES_UNIVERSE), 50)
            selected_futures = FUTURES_UNIVERSE[:futures_count]

    # 2. Fetch Data
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Parallel Load Bar
        progress_bar = st.progress(0, text="Initializing Quant Data Channels...")
        
        # Step A: ERP Data
        progress_bar.progress(0.1, text="Fetching ERP Batch Price History (QTYINVCD logic)...")
        df_erp = get_batch_price_history_for_optimization(cursor, limit=erp_limit)
        
        # Step B: Futures Data
        progress_bar.progress(0.4, text=f"Fetching Futures Universe ({len(selected_futures)} tickers)...")
        
        # Custom callback for fetcher to update bar
        def _update_progress(p):
            progress_bar.progress(0.4 + (p * 0.5), text=f"Syncing Market Data: {int(p*100)}%")
        
        pool_data = fetch_market_data_pool(selected_futures, _progress_callback=_update_progress)
        
        # Format Futures DF
        all_series = []
        for ticker, bundle in pool_data.items():
            if 'data' in bundle and bundle['data']:
                df_t = pd.DataFrame(bundle['data'])
                # FIX: Canonicalize to UTC then remove timezone to ensure naive-naive consistency
                df_t['date'] = pd.to_datetime(df_t['date'], utc=True).dt.tz_convert(None)
                df_t = df_t.rename(columns={'price_index': ticker})
                df_t = df_t.set_index('date')[[ticker]]
                all_series.append(df_t)
        
        df_futures = pd.concat(all_series, axis=1) if all_series else pd.DataFrame()
        
        # DEBUG: Universe Expansion Check
        with st.expander("🔍 Dataset Diagnostics", expanded=True):
             st.write(f"**ERP Assets:** {len(df_erp.columns)} items | Rows: {len(df_erp)}")
             st.write(f"**Futures Assets:** {len(df_futures.columns)} items | Rows: {len(df_futures)}")
             if df_futures.empty:
                 st.error("⚠️ Futures Data is Empty. Check Internet Connection or YFinance.")
             
        # Step C: Merge
        progress_bar.progress(0.95, text="Aligning Temporal Matrices...")
        df_combined = merge_erp_and_futures_data(df_erp, df_futures)
        progress_bar.progress(1.0, text="Data Stream Synced.")
        
        if df_combined.empty:
             st.error(f"Merged Data is Empty. ERP Range: {df_erp.index.min()} to {df_erp.index.max()}. Futures Range: {df_futures.index.min() if not df_futures.empty else 'N/A'} to {df_futures.index.max() if not df_futures.empty else 'N/A'}")
             return

        # Calculate Returns
        # Use fillna(0) instead of dropna() to prevent dropping all rows if one asset has a missing day (e.g. holidays)
        df_returns = df_combined.pct_change(fill_method=None).fillna(0)

        # ---------------------------------------------------------
        # TRADABLE UNIVERSE FILTER
        # ---------------------------------------------------------
        # We separate "Signal Assets" (Indices, Futures, Forex) from "Tradable Assets" (Stocks, ETFs, Crypto)
        # to ensure the Execution Engine allows receives valid orders.
        tradable_subset = []
        for asset in FUTURES_UNIVERSE:
            try:
                # 1. Get mapped ticker (or use asset itself if cleaned)
                cleaned = asset.split(' ')[0].split('(')[0].strip()
                if len(cleaned) >= 6 and cleaned.endswith("USD") and not cleaned.startswith("USDOLLAR"):
                        cleaned = cleaned.replace("USD", "-USD")
                
                ticker = TICKER_MAP.get(asset, cleaned)
                
                # 2. Check overlap with Alpaca constraints
                # ^ = Index (e.g. ^GSPC), = = Forex/Future (e.g. EUR=X, CL=F)
                if ticker and ("^" in ticker or "=" in ticker):
                    continue # Skip Non-Tradable Signal Asset
                
                tradable_subset.append(asset)
            except:
                pass
        
        if df_returns.empty or len(df_returns) < 10:
             st.error(f"Insufficient historical data points for optimization. Rows: {len(df_returns)}")
             return
        
        # Display Correlation Heatmap
        st.subheader("🕸️ Cross-Asset Correlation Network")
        corr_matrix = df_returns.corr()
        
        if alt:
            # Safe melt for Altair
            corr_reset = corr_matrix.reset_index()
            # The index column (Asset A) is now the first column
            id_col = corr_reset.columns[0]
            
            corr_melt = corr_reset.melt(id_vars=id_col, var_name='Target', value_name='Correlation')
            
            heatmap = alt.Chart(corr_melt).mark_rect().encode(
                x=alt.X(f'{id_col}:N', title="Asset A"),
                y=alt.Y('Target:N', title="Asset B"),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
                tooltip=[id_col, 'Target', 'Correlation']
            ).properties(width=800, height=800)
            
            st.altair_chart(heatmap, use_container_width=True)
            
        # 3. MPT Global Solver
        st.subheader("⚡ GLOBAL PORTFOLIO SOLVER")
        sol_col1, sol_col2 = st.columns([2, 1])
        
        with sol_col1:
            # User Overwrite: Fixed $2000 Capital
            default_cap = 2000.0

            # --- ML BACKTESTING SECTION ---
            st.divider()
            st.subheader("🤖 Neural Alpha Engine (RLS-Recursive Least Squares)")
            st.caption("Train a predictive model on historical data to dynamically adjust weights.")
            
            if st.button("🧪 Train & Backtest Strategy"):
                with st.spinner("Training models and simulating historical performance..."):
                    # 1. Initialize Backtester
                    backtester = Backtester(initial_capital=default_cap, transaction_cost_pct=0.001)
                    
                    # Check data quality
                    if df_combined.empty:
                         st.error("No data available for backtesting.")
                    else:
                        bt_progress_bar = st.progress(0, text="Backtesting Strategy...")
                        
                        def _bt_update_progress(p):
                            bt_progress_bar.progress(p, text=f"Backtesting... {int(p*100)}%")
                            
                        results = backtester.run(
                            df_combined, 
                            window_size=50, 
                            progress_callback=_bt_update_progress, 
                            checkpoint_path="ml_checkpoint.pkl",
                            tradable_assets=tradable_subset
                        )
                        bt_progress_bar.progress(1.0, text="Backtest Complete!")
                        bt_progress_bar.empty()
                        
                        if "error" in results:
                            st.error(f"Backtest Failed: {results['error']}")
                        else:
                            st.success("Optimization Complete!")
                            
                            # Metrics Display
                            m = results["metrics"]
                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            m_col1.metric("Total Return", f"{m['Total Return']:.1%}")
                            m_col2.metric("Annual CAGR", f"{m['CAGR']:.1%}")
                            m_col3.metric("Sharpe Ratio", f"{m['Sharpe']:.2f}")
                            m_col4.metric("Volatility", f"{m['Volatility']:.1%}")
                            
                            # Equity Curve Plot
                            st.line_chart(results["equity_curve"])
                            
                            # Show Current Recommended Allocation (Final Weights)
                            st.subheader("🔮 Recommended Allocation (Next Period)")
                            final_weights = results.get("final_weights", {})
                            
                            # Sort by weight
                            sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
                            
                            alloc_df = pd.DataFrame([
                                {"Asset": k, "Suggested Weight": v, "Allocated Capital": v * total_equity} 
                                for k, v in sorted_weights if v > 0.001
                            ])
                            
                            st.dataframe(alloc_df.style.format({
                                "Suggested Weight": "{:.1%}", 
                                "Allocated Capital": "${:,.2f}"
                            }))
                            
                            if st.button("🚀 Execute Recommendations (Rebalance Portfolio)"):
                                pm = PortfolioManager(mode=st.session_state.get("execution_mode", "paper"))
                                pm.execute_predictive_rebalancing(final_weights, {})
                                st.success("Portfolio rebalanced based on ML predictions!")
                                st.rerun()

            st.divider()
            
            # --- END ML SECTION ---

            capital = st.number_input("Hedge Fund Capital Allocation ($)", value=default_cap, step=1000.0)
            risk_free = st.slider("Risk-Free Rate (Annual)", 0.0, 0.10, 0.045)
            
        if st.button("🔥 EXECUTE QUANT SOLVER", type="primary"):
            # DEBUG: Show the inputs the optimizer is seeing
            # Data is MONTHLY resolution, so we must annualize by 12, not 252.
            annual_ret = df_returns.mean() * 252
            annual_vol = df_returns.std() * (252 ** 0.5)
            sharpe = (annual_ret - risk_free) / annual_vol
            
            debug_df = pd.DataFrame({
                "Exp. Return": annual_ret,
                "Volatility": annual_vol,
                "Sharpe": sharpe,
                "Data Points": df_returns.count()
            }).sort_values("Sharpe", ascending=False)
            
            # Show top candidates
            st.caption("🔬 Top Candidates Detected (Pre-Optimization):")
            st.dataframe(debug_df.head(10).style.format({
                "Exp. Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Data Points": "{:.0f}"
            }), height=200)

            # CRITICAL FIX: The optimizer struggles with 100+ assets if many are low-quality/diluted.
            # We strictly filter to the Top 50 candidates before solving to ensure convergence.
            # We also filter out "Perfect" assets (Vol < 1%) which break the covariance matrix.
            valid_candidates = debug_df[debug_df['Volatility'] > 0.01].head(50).index.tolist()
            
            if not valid_candidates:
                 st.error("No assets met the minimum volatility threshold (1%).")
                 st.stop()
                 
            df_optimized_universe = df_returns[valid_candidates]

            optimizer = PortfolioOptimizer(returns_df=df_optimized_universe, risk_free_rate=risk_free)
            result = optimizer.optimize_max_sharpe_ratio(tradable_assets=tradable_subset)
            
            st.success(f"**Optimization Complete!** Model Sharpe: {result['Sharpe']:.2f}")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Expected Fund Return (Annual)", f"{result['Return']:.1%}")
            with res_col2:
                st.metric("Portfolio Volatility (Total)", f"{result['Volatility']:.1%}")
                
            # Blueprint
            weights = result['Weights']
            blueprint = []
            for asset, weight in weights.items():
                if weight > 0.001: # Filter crumbs
                    blueprint.append({
                        "Asset": asset,
                        "Sector": "ERP Inventory" if asset in df_erp.columns else "Futures/Hedge",
                        "Weight": weight,
                        "Allocated Capital": capital * weight
                    })
            
            df_blue = pd.DataFrame(blueprint).sort_values("Weight", ascending=False)
            st.dataframe(df_blue.style.format({"Weight": "{:.1%}", "Allocated Capital": "${:,.2f}"}), use_container_width=True)
            
            if st.button("🚀 SUBMIT TRADES TO BROKER/ERP"):
                pm = PortfolioManager(mode=st.session_state.get("execution_mode", "paper"))
                # Unified rebalancing
                pm.execute_rebalancing(df_blue)
                st.toast("Hedge Fund Portfolio Updated Successfully.", icon="📈")
                
    except Exception as e:
        LOGGER.exception("Hedge Fund Engine Error")
        st.error(f"Quant Engine Failure: {e}")
