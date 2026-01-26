import streamlit as st
import pandas as pd
import time
from autonomous_fund.core.market_data import MarketDataProvider
from autonomous_fund.strategies.smart_bond import SmartBondStrategy
from autonomous_fund.config import HIGH_PROB_FLOOR

st.set_page_config(page_title="Eagle Strategy Live Monitor", layout="wide")

st.title("🦅 Eagle Strategy: Live Opportunity Monitor")
st.markdown(f"**Configuration:** Probability Floor > {HIGH_PROB_FLOOR*100:.0f}% | Min Liquidity > $5,000")

if st.button("Scan Live Markets"):
    progress_bar = st.progress(0)
    st.info("Scanning Polymarket... specific checking top ~50 candidates...")
    
    try:
        # Initialize
        md = MarketDataProvider()
        strat = SmartBondStrategy(portfolio_size=1000)
        
        # 1. Fetch Candidates (This handles the API calls)
        # Using the Deep Scan logic implicitly if implemented, or standard
        markets_df = md.get_live_markets()
        
        progress_bar.progress(50)
        
        if markets_df.empty:
            st.warning("No active market candidates returned from API scan.")
        else:
            # 2. Apply Strategy Filters (Price, Time, Liquidity, Slippage)
            st.write(f"Analyzing {len(markets_df)} candidates for safety...")
            opportunities = strat.scan_markets(markets_df)
            
            progress_bar.progress(100)
            
            if not opportunities.empty:
                st.success(f"Found {len(opportunities)} 'Fort Knox' Opportunities!")
                
                # Format for display
                display_df = opportunities.copy()
                display_df['Yield (%)'] = display_df['apy'].apply(lambda x: f"{x:.2f}%")
                display_df['Price'] = display_df['price'].apply(lambda x: f"${x:.3f}")
                display_df['Max Safe Bet'] = display_df['max_safe_capital'].apply(lambda x: f"${x:.2f}")
                display_df['Time Left'] = display_df['days_left'].apply(lambda x: f"{x:.1f} Days")
                
                st.dataframe(
                    display_df[['question', 'Price', 'Yield (%)', 'Time Left', 'Max Safe Bet', 'market_id']],
                    use_container_width=True
                )
            else:
                st.info("Zero opportunities met the strict 'Fort Knox' safety criteria right now.")
                st.markdown("""
                **Why?**
                - Prices might be too low (<$0.90)
                - Liquidity might be too thin (Risk of Slippage)
                - Duration might be too long (>60 Days)
                """)
                
                st.subheader("Raw Candidates (Filtered Out)")
                st.dataframe(markets_df.head(10))

    except Exception as e:
        st.error(f"Scan failed: {e}")
