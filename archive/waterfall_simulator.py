
import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from autonomous_fund.core.waterfall_allocator import WaterfallAllocator, StrategyBucket

st.set_page_config(page_title="Waterfall Strategy Simulator", layout="wide")

st.title("🌊 Waterfall Capital Allocator")
st.markdown("""
**The Philosophy**: Prioritize capital into high-certainty "Tier 1" assets first, then spill over into scalable "Tier 2" growth.
""")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Configuration")

try:
    # 1. Capital & Time
    st.sidebar.subheader("Portfolio Parameters")
    start_capital = st.sidebar.number_input("Starting Capital ($)", value=200, step=100)

    col_t1, col_t2 = st.sidebar.columns(2)
    start_year = col_t1.number_input("Start Year", 2005, 2025, 2005)
    end_year = col_t2.number_input("End Year", 2006, 2030, 2026)

    if start_year >= end_year:
        st.error("End Year must be after Start Year")
        st.stop()

    # 2. Smart Bond Settings (The Rocket)
    st.sidebar.subheader("Tier 1: Smart Bonds")
    bond_source = st.sidebar.radio("Yield Source", ["Static APY (Hypothetical)", "Real US Treasury Yields (1-Year)"])
    
    bond_apy = 3.00 # Default
    if bond_source == "Static APY (Hypothetical)":
        bond_apy = st.sidebar.slider("Bond APY (%)", 1.0, 500.0, 300.0, step=1.0) / 100.0
    else:
        st.sidebar.info("Using historical 1-Year US Treasury Rates.")
        
    bond_cap_start = st.sidebar.number_input("Liquidity Cap ($)", value=50_000, step=10_000, help="Max capital this strategy can take")
    bond_growth_rate = st.sidebar.slider("Cap Growth Rate (%)", 0.0, 20.0, 10.0, step=1.0) / 100.0

    # RISK MODULE
    st.sidebar.subheader("⚠️ Risk Simulation (Black Swans)")
    enable_risk = st.sidebar.checkbox("Enable Specific Risks")
    risk_prob = 0.0
    risk_severity = 0.0
    if enable_risk:
        risk_type = st.sidebar.selectbox("Risk Type", ["Smart Contract Exploit (-100%)", "Platform Insolvency (-50%)", "Stablecoin Depeg (-20%)", "Custom"])
        if risk_type == "Smart Contract Exploit (-100%)":
            risk_severity = 1.0
            risk_prob = 0.02 # 2% annual
        elif risk_type == "Platform Insolvency (-50%)":
            risk_severity = 0.5
            risk_prob = 0.05
        elif risk_type == "Stablecoin Depeg (-20%)":
            risk_severity = 0.2
            risk_prob = 0.10
        else:
            risk_severity = st.sidebar.slider("Loss Severity (%)", 0, 100, 100) / 100.0
            risk_prob = st.sidebar.slider("Annual Probability (%)", 0.0, 50.0, 5.0, step=0.5) / 100.0
        
        st.sidebar.caption(f"Simulating a **{risk_prob*100:.1f}%** annual chance of losing **{risk_severity*100:.0f}%** of the Bond Bucket.")

    # 3. Market Mode
    st.sidebar.subheader("Tier 2: Structural Alpha")
    sim_mode = st.sidebar.radio("Simulation Mode", ["Real Market History (S&P 500)", "Fixed CAGR Model"])
    
    alpha_premium = 0.0
    fixed_cagr = 0.25
    if sim_mode == "Real Market History (S&P 500)":
        alpha_premium = st.sidebar.slider("Alpha over S&P (%)", 0.0, 20.0, 5.0, step=1.0) / 100.0
    else:
        fixed_cagr = st.sidebar.slider("Fixed CAGR (%)", 10.0, 100.0, 25.0, step=1.0) / 100.0

    # ACTION BUTTON
    run_anim = st.sidebar.button("▶️ Run Live Simulation", type="primary")

    # ==========================================
    # SIMULATION LOGIC
    # ==========================================
    def run_simulation():
        real_sp500 = {
            2005: 0.049, 2006: 0.158, 2007: 0.055, 2008: -0.385, 2009: 0.265,
            2010: 0.151, 2011: 0.021, 2012: 0.160, 2013: 0.324, 2014: 0.137,
            2015: 0.014, 2016: 0.120, 2017: 0.218, 2018: -0.044, 2019: 0.315, 
            2020: 0.184, 2021: 0.287, 2022: -0.181, 2023: 0.263, 2024: 0.240,
            2025: 0.100, 2026: 0.050, 2027: 0.080, 2028: 0.080, 2029: 0.080, 2030: 0.080
        }
        
        real_treasury = {
            2005: 0.0362, 2006: 0.0494, 2007: 0.0453, 2008: 0.0183, 2009: 0.0047,
            2010: 0.0032, 2011: 0.0018, 2012: 0.0017, 2013: 0.0013, 2014: 0.0012,
            2015: 0.0032, 2016: 0.0061, 2017: 0.0120, 2018: 0.0233, 2019: 0.0206,
            2020: 0.0036, 2021: 0.0010, 2022: 0.0213, 2023: 0.0460, 2024: 0.0450,
            2025: 0.0400, 2026: 0.0350
        }

        dates = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-01-01', freq='B')
        
        init_yield = bond_apy
        if bond_source == "Real US Treasury Yields (1-Year)":
            init_yield = real_treasury.get(start_year, 0.03)

        bond_bucket = StrategyBucket("Smart Bonds", init_yield, bond_cap_start, 99.0)
        alpha_bucket = StrategyBucket("Structural Alpha", 0.10, -1, 1.0)
        
        allocator = WaterfallAllocator([bond_bucket, alpha_bucket])
        
        capital = start_capital
        benchmark = start_capital
        history = []
        
        # Risk Event Tracking
        daily_risk_prob = 1 - (1 - risk_prob)**(1/252) # Convert annual to daily
        np.random.seed(42) # Fixed seed for reproducibility
        
        for date in dates:
            year = date.year
            
            # --- 1. SET YIELDS ---
            current_bond_apy = bond_apy
            if bond_source == "Real US Treasury Yields (1-Year)":
                current_bond_apy = real_treasury.get(year, 0.03)
            
            daily_bond_ret = (1 + current_bond_apy)**(1/252) - 1
            
            if sim_mode == "Real Market History (S&P 500)":
                annual_ret = real_sp500.get(year, 0.08)
                daily_market = (1 + annual_ret)**(1/252) - 1
                daily_alpha = daily_market + (alpha_premium / 252)
            else:
                daily_alpha = (1 + fixed_cagr)**(1/252) - 1
                daily_market = (1 + 0.10)**(1/252) - 1

            # --- 2. UPDATE CAPACITY ---
            years_passed = year - start_year
            current_bond_cap = bond_cap_start * ((1 + bond_growth_rate) ** years_passed)
            bond_bucket.update_capacity(current_bond_cap)
            
            # --- 3. ALLOCATE ---
            alloc = allocator.allocate(capital)
            b = alloc["Smart Bonds"]
            a = alloc["Structural Alpha"]
            
            # --- 4. RISK EVENT ---
            risk_loss = 0.0
            risk_hit = False
            if enable_risk and b > 0:
                if np.random.rand() < daily_risk_prob:
                    risk_hit = True
                    loss = b * risk_severity
                    b -= loss # Wipe out portion of bond bucket
                    pnl_impact = -loss
                    risk_loss = loss
            
            # --- 5. COMPOUND ---
            # Returns on surviving capital
            pnl = (b * daily_bond_ret) + (a * daily_alpha) - risk_loss
            capital += pnl
            benchmark *= (1 + daily_market)
            
            history.append({
                'Date': date,
                'Portfolio': capital,
                'Benchmark': benchmark,
                'Smart Bond Alloc': b,
                'Alpha Alloc': a,
                'Risk Event': risk_hit,
                'Bond Capacity': current_bond_cap,
                'Bond APY': current_bond_apy
            })
            
        return pd.DataFrame(history)

    df = run_simulation()

    if df.empty:
        st.warning("No data generated. Check date range.")
        st.stop()

    # ==========================================
    # VISUALIZATION
    # ==========================================
    m_col1, m_col2, m_col3 = st.columns(3)
    metric_ph1 = m_col1.empty()
    metric_ph2 = m_col2.empty()
    metric_ph3 = m_col3.empty()
    chart_ph = st.empty()
    
    # Check for risk events
    risk_events = df[df['Risk Event'] == True]
    if not risk_events.empty:
        st.error(f"⚠️ **RISK EVENT DETECTED**: {len(risk_events)} Simulation day(s) experienced a loss event!")
        st.dataframe(risk_events[['Date', 'Portfolio', 'Smart Bond Alloc']])

    if run_anim:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        N = len(df)
        steps = 50
        step_size = max(1, N // steps)
        
        for i in range(step_size, N + step_size, step_size):
            idx = min(i, N - 1)
            current_slice = df.iloc[:idx]
            row = current_slice.iloc[-1]
            
            end_val = row['Portfolio']
            bench_val = row['Benchmark']
            mult = end_val / bench_val if bench_val > 0 else 0
            
            metric_ph1.metric("Portfolio Value", f"${end_val:,.0f}", delta=f"{((end_val/start_capital)-1)*100:.0f}%")
            metric_ph2.metric("S&P 500", f"${bench_val:,.0f}")
            metric_ph3.metric("Alpha", f"{mult:.1f}x")
            
            progress_bar.progress(idx / N)
            
            msg = f"Simulating: {row['Date'].date()}"
            if row['Risk Event']:
                msg += " 💥 RISK EVENT! 💥"
            
            status_text.text(msg)
            
            with chart_ph.container():
                st.caption("Growth Trajectory")
                st.line_chart(current_slice[['Date', 'Portfolio', 'Benchmark']].set_index('Date'))
                st.caption("Capital Allocation")
                st.area_chart(current_slice[['Date', 'Smart Bond Alloc', 'Alpha Alloc']].set_index('Date'), color=['#00BFFF', '#FFD700'])
            
            time.sleep(0.05)
            
        status_text.text("Simulation Complete!")
        
    else:
        # Static
        row = df.iloc[-1]
        end_val = row['Portfolio']
        bench_val = row['Benchmark']
        mult = end_val / bench_val if bench_val > 0 else 0
        
        metric_ph1.metric("Portfolio Value", f"${end_val:,.0f}", delta=f"{((end_val/start_capital)-1)*100:.0f}%")
        metric_ph2.metric("S&P 500", f"${bench_val:,.0f}")
        metric_ph3.metric("Alpha", f"{mult:.1f}x")
        
        with chart_ph.container():
            st.subheader("Performance Overview")
            
            # Combine lines + Points for Risk Events
            base = alt.Chart(df).encode(x='Date')
            l1 = base.mark_line(color='#00ff00').encode(y=alt.Y('Portfolio', scale=alt.Scale(type='log')))
            l2 = base.mark_line(color='gray', strokeDash=[5,5]).encode(y='Benchmark')
            
            # Red dots for risk events
            risk_points = alt.Chart(risk_events).mark_point(color='red', size=100, shape='cross').encode(
                x='Date', y='Portfolio', tooltip=['Date', 'Portfolio']
            )
            
            st.altair_chart((l1+l2+risk_points).properties(width=700, height=300).interactive(), use_container_width=True)
            
            st.subheader("Allocation Waterfall")
            st.area_chart(df[['Date', 'Smart Bond Alloc', 'Alpha Alloc']].set_index('Date'), color=['#00BFFF', '#FFD700'])

except Exception as e:
    st.error(f"Simulation Error: {str(e)}")

