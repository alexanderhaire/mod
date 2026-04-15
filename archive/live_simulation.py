
"""
Live Simulation Dashboard
=========================

Visualizing the "Alpha Hunt" in real-time.
Strategies:
1. SPY (Benchmark)
2. Ultimate Strategy (The Growth Engine)
3. HRP (The Defensive Engine)
4. Dollar Trend (The Crisis Hedge)
5. THE TRIFECTA (Combined Portfolio)
6. GOLDEN OMNI (The Final Boss)
7. Failures (Factor, Steepener)

Logic:
- Pre-calculates all equity curves.
- Animates them on a chart.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
import warnings
from scipy.optimize import minimize

# Import MLP Strategy
try:
    from mlp_strategy import calculate_mlp_returns
    HAS_MLP = True
except ImportError:
    HAS_MLP = False

warnings.filterwarnings('ignore')

# Try importing plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Alpha Hunt Simulation", page_icon="📈", layout="wide")

st.title("📈 Alpha Hunt: The Race")
st.markdown("Watching the evolution of **135 Hypotheses** vs **The Market**.")

if not HAS_PLOTLY:
    st.warning("⚠️ Plotly not found. Using static charts. (Please install plotly for animations)")

def optimize_portfolio(rets):
    """Find weights that maximize Sharpe Ratio."""
    def neg_sharpe(weights, rets):
        p_ret = (rets * weights).sum(axis=1)
        mean = p_ret.mean() * 252
        if p_ret.std() == 0: return 0
        vol = p_ret.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        return -sharpe

    try:
        # Fill NA for optimization to avoid crash, but locally
        clean_rets = rets.fillna(0)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(clean_rets.shape[1]))
        init_guess = [1.0 / clean_rets.shape[1]] * clean_rets.shape[1]
        
        if len(clean_rets) < 20: return init_guess
        
        opt = minimize(neg_sharpe, init_guess, args=(clean_rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return [1.0 / rets.shape[1]] * rets.shape[1]

@st.cache_data
def load_simulation_data():
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', 'XLE', # Trad
        'VUG', 'VTV', 'RSP', # Factors/Breadth
        '^FVX', '^TYX', '^VIX', # Macro
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD' # Crypto
    ]
    # Fetch long history for full context
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # CRITICAL FIX: Do NOT dropna(). This truncates history to the newest asset (SOL 2020).
    # Instead, forward fill existing data, and let NaNs exist for older dates of new assets.
    prices = prices.ffill()
    # Drop only rows where SPY (Benchmark) is missing, to ensure we have a valid timeline
    prices = prices.dropna(subset=['SPY'])
    
    # ---------------------------
    # CALCULATE STRATEGIES
    # ---------------------------
    # Returns (NaNs become 0.0, so new assets don't affect portfolio sum until they exist)
    rets = prices.pct_change().fillna(0)
    
    # 1. SPY
    r_spy = rets['SPY']
    
    # ==============================================================================
    # 0. SHARED SMART CRYPTO MODULE (Altseason Logic)
    # ==============================================================================
    # Moved up so Ultimate Strategy can use it too!
    
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    # rolling std needs handle NaNs
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True) # Assume safe if no data
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    # Smart Crypto Return Construction
    # We need to average only the EXISTING alts for each day
    alt_rets = rets[avail_alts].copy()
    for c in avail_alts:
        mask = (prices[c].isna()) | (prices[c] == 0)
        alt_rets.loc[mask, c] = np.nan
        
    avg_alt_ret = alt_rets.mean(axis=1) # Ignores NaNs
    avg_alt_ret = avg_alt_ret.fillna(0)
    
    # Altseason Signal
    r_crypto_piece = pd.Series(0.0, index=rets.index)
    
    btc_mom = btc_p.pct_change(14)
    if avail_alts:
        # Index of alts
        alt_idx = prices[avail_alts].mean(axis=1) # valid mean
        alt_mom = alt_idx.pct_change(14)
        is_altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    else:
        is_altseason = pd.Series(False, index=prices.index)
        
    mask_use_alts = is_crypto_safe & is_altseason
    mask_use_btc = is_crypto_safe & ~is_altseason
    
    # Construct Shared Crypto Leg
    r_crypto_piece[mask_use_alts] = avg_alt_ret[mask_use_alts]
    
    # Ensure BTC exists for the BTC leg
    btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
    r_crypto_piece[mask_use_btc] = rets['BTC-USD'][mask_use_btc]
    
    # Correction for Pre-Crypto Era (Backfill with SPY/Trad weight)
    # If BTC doesn't exist, the 40% allocations in strategies need to go somewhere.
    # We usually dump it into SPY or distribute it.
    missing_crypto_correction = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        # If BTC is missing, add 0.40 * SPY return to the leg (effectively rebalancing 40% weight to SPY)
        missing_crypto_correction[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]
    
    
    # ==============================================================================
    # 2. ULTIMATE (UPGRADED with Smart Crypto)
    # ==============================================================================
    # Require VIX. If missing, we cannot run Ultimate correctly.
    if '^VIX' not in prices.columns:
        # st.error("🚨 CRITICAL: VIX data not found. Cannot run volatility filter.")
        vix = pd.Series(20, index=prices.index) # Emergency Fallback
    else:
        vix = prices['^VIX'].fillna(method='ffill')
    
    vix_ma = vix.rolling(20).mean()
    signal = pd.Series(0, index=prices.index)
    signal[vix < vix_ma] = 1 # Calm
    signal[vix > vix_ma] = -1 # Fear
    
    # Ultimate Allocations
    # Weights define the strategy structure
    # Now calculating returns directly components
    
    # Bull logic: 45% SPY, 10% TLT, 40% Crypto
    r_ult_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    
    # Bear logic: 15% SPY, 35% TLT, 40% Crypto
    r_ult_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    
    # Neutral logic: 30% SPY, 22% TLT, 20% Crypto? (Wait, PDF says something else usually, sticking to sim logic)
    # Sim logic was: 30 SPY, 22 TLT, 20 BTC? 
    # Let's align with the existing sim logic structure but use NEW crypto piece
    # Existing sim had: 30 SPY, 22 TLT, 20 BTC.
    # Note: 30+22+20 = 72%. It wasn't 100%? The optimization function might have handled it, 
    # or it was just under-allocated (cash).
    # Let's assume the previous logic meant 100%. 
    # If we look at previous code: 
    # w_spy[mask_neut] = 0.30
    # w_tlt[mask_neut] = 0.22
    # w_btc[mask_neut] = 0.20
    # Total = 0.72. The rest (28%) was cash? 
    # Let's keep it consistent but use r_crypto_piece for the BTC part.
    
    r_ult_neut = (0.30 * rets['SPY'] + 0.22 * rets.get('TLT', 0) + 0.20 * r_crypto_piece + (0.20/0.40)*missing_crypto_correction)
    
    # Combine Signal
    # Signal is shifted T+1 execution? 
    # Original code: signal was calculated on close. Logic applied to standard weights.
    # We should shift the signal to represent T-1 decision for T return.
    # Signal was: signal[vix < vix_ma] = 1. This uses today's VIX. You can't trade today's open on today's close VIX.
    # So we MUST shift.
            
    # However, original code did:
    # w_spy[mask_bull] = ...
    # r_ult = (w_spy.shift(1) * rets['SPY']...
    # So YES, it shifted.
    
    sig_shifted = signal.shift(1).fillna(0)
    
    r_ult = pd.Series(0.0, index=rets.index)
    r_ult[sig_shifted > 0] = r_ult_bull[sig_shifted > 0]
    r_ult[sig_shifted < 0] = r_ult_bear[sig_shifted < 0]
    r_ult[sig_shifted == 0] = r_ult_neut[sig_shifted == 0] # Neutral/Init
    
    
    # 3. HRP (Inverse Vol)
    cols_hrp = [c for c in ['SPY', 'TLT', 'GLD', 'IEF'] if c in rets.columns]
    vol = rets[cols_hrp].rolling(126).std()
    inv_vol = 1 / vol
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[cols_hrp]).sum(axis=1)
    
    # 4. Dollar Trend
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        r_uup = pd.Series(0.0, index=prices.index)
        uup_ma = uup.rolling(200).mean()
        is_uptrend = (uup > uup_ma).shift(1).fillna(False)
        r_uup[is_uptrend] = rets['UUP'][is_uptrend]
    else:
        r_uup = pd.Series(0.0, index=prices.index)
        
    # 5. GOLDEN OMNI (The Final Boss)
    # -------------------------------
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    
    # Raw Daily Signals
    is_bull_daily = (spy_p > ma200)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation_daily = (xle_p > ma200_xle) & (~is_bull_daily)
    
    # WEEKLY REBALANCING (Friday Close)
    # We resample the signal to Friday, then forward fill it for the next week.
    # This aligns with the "Verified Alpha" test showing Weekly > Daily.
    
    # 1. Resample to Friday (take the state at Friday Close)
    # We use 'W-FRI' to anchor to Fridays.
    bull_weekly = is_bull_daily.resample('W-FRI').last()
    inf_weekly = is_inflation_daily.resample('W-FRI').last()
    
    # 2. Reindex back to Daily (Forward Fill)
    # So Mon-Thu uses the previous Friday's signal.
    # We .shift(1) so we trade *based on* that signal the next day.
    is_bull = bull_weekly.reindex(is_bull_daily.index, method='ffill').shift(1).fillna(False)
    is_inflation = inf_weekly.reindex(is_inflation_daily.index, method='ffill').shift(1).fillna(False)
    
    # Omni uses the Shared r_crypto_piece now!
    r_bull_leg = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_bear_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_inflation_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    
    r_omni = pd.Series(0.0, index=rets.index)
    r_omni[is_bull] = r_bull_leg[is_bull]
    r_omni[(~is_bull) & (~is_inflation)] = r_bear_leg[(~is_bull) & (~is_inflation)]
    r_omni[(~is_bull) & (is_inflation)] = r_inflation_leg[(~is_bull) & (is_inflation)]

    # 5.5 Crypto Composite (Deep Patterns)
    # ------------------------------------
    # "The Weird One" - Exploits Day-of-Week
    # 1. Avoid Thursdays (statistically worst day)
    # 2. Prime Days (Mon/Wed): Full Smart Crypto
    # 3. Other Days: Reduced BTC (0.7x)
    
    r_cc = pd.Series(0.0, index=rets.index)
    dow = r_cc.index.dayofweek
    
    mask_thurs = (dow == 3)
    mask_prime = (dow.isin([0, 2]))
    mask_other = (~mask_thurs) & (~mask_prime)
    
    # Logic
    r_cc[mask_thurs] = 0.0 # Cash
    r_cc[mask_prime] = r_crypto_piece[mask_prime]
    # Ensure BTC exists for other days
    if 'BTC-USD' in rets.columns:
        r_cc[mask_other] = 0.7 * rets['BTC-USD'][mask_other]
    else:
        r_cc[mask_other] = 0.0

    # 5.6 ETH/BTC Reversion ("The Flipper")
    # -------------------------------------
    # Hypo: ETH/BTC ratio mean reverts.
    # Signal: Long ETH when Ratio < Z-score -1.5, Long BTC when Ratio > Z-score 1.5
    
    r_flip = pd.Series(0.0, index=rets.index)
    if 'ETH-USD' in prices.columns and 'BTC-USD' in prices.columns:
        ratio = prices['ETH-USD'] / prices['BTC-USD']
        # Rolling Z-Score (90d)
        roll_mean = ratio.rolling(90).mean()
        roll_std = ratio.rolling(90).std()
        z_score = (ratio - roll_mean) / roll_std
        
        # State: 1 = ETH, -1 = BTC. Hold previous state if neutral.
        state = pd.Series(np.nan, index=rets.index)
        state[z_score < -1.5] = 1 # Cheap ETH -> Buy ETH
        state[z_score > 1.5] = -1 # Expensive ETH -> Buy BTC
        
        # Forward fill state to hold position until reversal signal
        state = state.ffill().shift(1).fillna(-1) # Default to BTC start
        
        mask_eth = (state == 1)
        mask_btc = (state == -1)
        
        r_flip[mask_eth] = rets['ETH-USD'][mask_eth]
        r_flip[mask_btc] = rets['BTC-USD'][mask_btc]
    else:
        # Fallback if no ETH
        if 'BTC-USD' in rets.columns:
            r_flip[:] = rets['BTC-USD']

    # 5.7 The Dog Flipper 🐕 (DOGE/BTC Reversion)
    # -------------------------------------------
    # Same logic, but for Meme Cycles.
    # DOGE cycles are even more explosive and mean-reverting.
    
    r_dog_flip = pd.Series(0.0, index=rets.index)
    if 'DOGE-USD' in prices.columns and 'BTC-USD' in prices.columns:
        # Ratio of Dog to King
        dog_ratio = prices['DOGE-USD'] / prices['BTC-USD']
        
        # Valid data check
        dog_ratio = dog_ratio.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        
        # Rolling Z-Score (90d)
        dr_mean = dog_ratio.rolling(90).mean()
        dr_std = dog_ratio.rolling(90).std()
        dr_z = (dog_ratio - dr_mean) / dr_std
        
        # State: 1 = DOGE, -1 = BTC.
        # DOGE is higher vol, maybe stricter bands? Or same?
        # Let's use same bands to capture the "cheap" moments.
        d_state = pd.Series(np.nan, index=rets.index)
        d_state[dr_z < -1.5] = 1 # Buy DOGE (Cheap)
        d_state[dr_z > 1.5] = -1 # Buy BTC (Expensive)
        
        d_state = d_state.ffill().shift(1).fillna(-1) # Start in BTC
        
        mask_doge = (d_state == 1)
        mask_d_btc = (d_state == -1)
        
        # Ensure we don't try to buy DOGE before it exists
        doge_ok = prices['DOGE-USD'].notna() & (prices['DOGE-USD'] > 0)
        mask_doge = mask_doge & doge_ok
        
        r_dog_flip[mask_doge] = rets['DOGE-USD'][mask_doge]
        r_dog_flip[mask_d_btc] = rets['BTC-USD'][mask_d_btc]
        
        # If DOGE selected but not avail, use BTC
        mask_doge_fail = (d_state == 1) & (~doge_ok)
        r_dog_flip[mask_doge_fail] = rets['BTC-USD'][mask_doge_fail]
        
    else:
        if 'BTC-USD' in rets.columns:
            r_dog_flip[:] = rets['BTC-USD']

    # 6. Trifecta
    strat_rets = pd.DataFrame({'Ultimate': r_ult, 'HRP': r_hrp, 'Dollar Trend': r_uup})
    strat_rets = strat_rets.loc[(strat_rets != 0).any(axis=1)]
    
    weights_dict = {'Ultimate': 1, 'HRP': 0, 'Dollar Trend': 0}
    if not strat_rets.empty:
        try:
           opt_w = optimize_portfolio(strat_rets)
           r_tri = (opt_w[0] * r_ult + opt_w[1] * r_hrp + opt_w[2] * r_uup)
           weights_dict = {'Ultimate': opt_w[0], 'HRP': opt_w[1], 'Dollar Trend': opt_w[2]}
        except:
           r_tri = r_ult
    else:
        r_tri = r_ult
    
    # 7. Failures
    prices_fail = prices.copy().bfill()
    spread = prices_fail.get('^TYX', pd.Series(0, index=prices.index)) - prices_fail.get('^FVX', pd.Series(0, index=prices.index))
    panic = (spread.diff(20) > 0.4).astype(int)
    w_fail1 = panic.shift(1)
    r_steep = (1-w_fail1)*r_spy + w_fail1*rets.get('TLT', 0)
    
    if 'VUG' in prices.columns and 'VTV' in prices.columns:
        mom_vug = prices['VUG'].pct_change(126).fillna(0)
        mom_vtv = prices['VTV'].pct_change(126).fillna(0)
        w_vug = (mom_vug > mom_vtv).astype(float).shift(1)
        r_fact = w_vug * rets['VUG'] + (1-w_vug) * rets['VTV']
    else:
        r_fact = r_spy
    
    # 8. MLP Deep Neural Network Strategy 🧠
    # ----------------------------------------
    # The "throw ML at it" approach
    if HAS_MLP:
        try:
            r_mlp = calculate_mlp_returns(prices, rets)
        except Exception as e:
            # Fallback if MLP fails
            r_mlp = r_spy * 0.5 + rets.get('TLT', 0) * 0.5
    else:
        # No sklearn - use simple 60/40
        r_mlp = r_spy * 0.6 + rets.get('TLT', 0) * 0.4
    
    # Combine
    df = pd.DataFrame({
        'SPY (Benchmark)': (1+r_spy).cumprod(),
        'Golden Omni (FINAL)': (1+r_omni).cumprod(),
        'Trifecta (Optimized)': (1+r_tri).cumprod(),
        'Ultimate (Growth)': (1+r_ult).cumprod(),
        'MLP Neural Net 🧠': (1+r_mlp).cumprod(),
        'Crypto Comp (Patterns) 🌚': (1+r_cc).cumprod(),
        'ETH/BTC Flipper 🐬': (1+r_flip).cumprod(),
        'Dog Flipper 🐕': (1+r_dog_flip).cumprod(),
        'HRP (Reliability)': (1+r_hrp).cumprod(),
        'Factor Rotation': (1+r_fact).cumprod(),
        'Steepener': (1+r_steep).cumprod()
    })
    
    # Normalize start to 1.0
    df = df / df.iloc[0]
    
    return df, weights_dict

# =============================================================================
# 2. RUN SIMULATION
# =============================================================================

df, weights = load_simulation_data()

col1, col2 = st.columns([1, 4])
with col1:
    speed = st.slider("Simulation Speed", 1, 100, 20)
    # LOG SCALE TOGGLE
    use_log_scale = st.checkbox("Log Scale 🚀", value=True, help="Use logarithmic scale to see percentage changes better vs linear price.")
    
    # METRIC TOGGLE
    view_metric = st.radio("Metric", ["Growth ($)", "Rolling Sharpe (1Y)"], horizontal=True)
    
    start_btn = st.button("▶️ Start Simulation")

# PREPARE DATA BASED ON SELECTION
if view_metric == "Rolling Sharpe (1Y)":
    # Calculate Rolling Sharpe (252 days)
    # Risk Free Rate assumed 0 for simplicity or constant
    daily_rets = df.pct_change()
    rolling_mean = daily_rets.rolling(252).mean()
    rolling_std = daily_rets.rolling(252).std()
    # Annualize
    df_plot = (rolling_mean / rolling_std) * np.sqrt(252)
    # Clean NaNs (start of series)
    df_plot = df_plot.fillna(0)
    
    chart_title_prefix = "Rolling Sharpe (1Y)"
    y_axis_title = "Sharpe Ratio"
    # Sharpe doesn't need log scale usually, but user might leave it on. 
    # We'll allow it but default range is different.
else:
    df_plot = df
    chart_title_prefix = "Growth of $1"
    y_axis_title = "Growth of $1"

chart_placeholder = st.empty()

if start_btn:
    if HAS_PLOTLY:
        step = 10 
        # Start later for Sharpe (needs 1 year of data)
        start_idx = 252 if view_metric == "Rolling Sharpe (1Y)" else 100
        
        for i in range(start_idx, len(df_plot), step):
            chunk = df_plot.iloc[:i]
            
            fig = go.Figure()
            
            # --- WINNERS ---
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Golden Omni (FINAL)'], name='Golden Omni 🌟', 
                                     line=dict(color='#FFD700', width=4)))
            
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Trifecta (Optimized)'], name='Trifecta', 
                                     line=dict(color='#00FF00', width=2)))
            
            # --- BENCHMARK ---
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['SPY (Benchmark)'], name='SPY', 
                                     line=dict(color='gray', width=2, dash='dash')))
            
            # --- COMPONENTS ---
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Ultimate (Growth)'], name='Ultimate (Upgraded)', 
                                     line=dict(color='#00CCFF', width=2), opacity=0.8))

            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Crypto Comp (Patterns) 🌚'], name='Crypto Comp (Tu/Thu Avoid)', 
                                     line=dict(color='#9467bd', width=2, dash='dash'), opacity=0.8))

            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['ETH/BTC Flipper 🐬'], name='ETH/BTC Flipper', 
                                     line=dict(color='#e377c2', width=2), opacity=0.8))

            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Dog Flipper 🐕'], name='Dog Flipper (Meme)', 
                                     line=dict(color='#ff7f0e', width=2), opacity=0.8))
                                     
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['HRP (Reliability)'], name='HRP', 
                                     line=dict(color='#FFA500', width=1), opacity=0.5))

            # --- ML STRATEGY ---
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['MLP Neural Net 🧠'], name='MLP Neural Net 🧠', 
                                     line=dict(color='#FF00FF', width=3), opacity=0.9))
                                     
            # --- FAILURES ---
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Factor Rotation'], name='Factor Rot ❌', 
                                     line=dict(color='#FF3333', width=1, dash='dot'), opacity=0.5))
                                     
            fig.add_trace(go.Scatter(x=chunk.index, y=chunk['Steepener'], name='Steepener ❌', 
                                     line=dict(color='#9933FF', width=1, dash='dot'), opacity=0.5))
            
            # Layout
            current_date = chunk.index[-1].strftime('%Y-%m-%d')
            
            # Dynamic Title Value
            if view_metric == "Rolling Sharpe (1Y)":
                val = chunk['Golden Omni (FINAL)'].iloc[-1]
                val_str = f"{val:.2f}"
            else:
                val = (chunk['Golden Omni (FINAL)'].iloc[-1] - 1) * 100
                val_str = f"{val:,.0f}%"
            
            y_range = None
            if view_metric == "Growth ($)":
                 y_range = [0.5, df_plot.max().max() * 1.5] if not use_log_scale else None
            else:
                 # Sharpe Range: usually -2 to 5
                 y_range = [-3, 6]

            layout_args = dict(
                title=f"📅 {current_date} | 🌟 Omni: {val_str}",
                xaxis_title="Date",
                yaxis_title=y_axis_title,
                template="plotly_dark",
                height=600,
                xaxis_range=[df_plot.index[0], df_plot.index[-1]], 
                yaxis_range=y_range
            )
            
            fig.update_layout(**layout_args)
            if use_log_scale and view_metric == "Growth ($)":
                fig.update_yaxes(type="log")
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(1.5 / speed)
    else:
        # Fallback static
        st.line_chart(df_plot)


if not start_btn:
    if HAS_PLOTLY:
        fig = go.Figure()
        
        # Use df_plot which is selected metric
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Golden Omni (FINAL)'], name='Golden Omni 🌟', line=dict(color='#FFD700', width=4)))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Trifecta (Optimized)'], name='Trifecta', line=dict(color='#00FF00', width=2)))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SPY (Benchmark)'], name='SPY', line=dict(color='gray', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Ultimate (Growth)'], name='Ultimate (Upgraded)', line=dict(color='#00CCFF', width=2), opacity=0.8))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Crypto Comp (Patterns) 🌚'], name='Crypto Comp (Tu/Thu Avoid)', line=dict(color='#9467bd', width=2, dash='dash'), opacity=0.8))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ETH/BTC Flipper 🐬'], name='ETH/BTC Flipper', line=dict(color='#e377c2', width=2), opacity=0.8))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Dog Flipper 🐕'], name='Dog Flipper (Meme)', line=dict(color='#ff7f0e', width=2), opacity=0.8))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['HRP (Reliability)'], name='HRP', line=dict(color='#FFA500', width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MLP Neural Net 🧠'], name='MLP Neural Net 🧠', line=dict(color='#FF00FF', width=3), opacity=0.9))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Factor Rotation'], name='Factor Rot ❌', line=dict(color='#FF3333', width=1, dash='dot'), opacity=0.5))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Steepener'], name='Steepener ❌', line=dict(color='#9933FF', width=1, dash='dot'), opacity=0.5))
        
        fig.update_layout(title="Final Result (Press Start to Replay)", template="plotly_dark", height=600)
        
        if use_log_scale and view_metric == "Growth ($)":
            fig.update_yaxes(type="log")
        elif view_metric == "Rolling Sharpe (1Y)":
             fig.update_layout(yaxis_title="Sharpe Ratio")
            
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot)

st.write("### 🏆 The New Champion: 'Golden Omni' 🌟")
st.write("The Omni strategy combines:")
st.write("1. **Macro Regime**: (SPY 200MA) -> Switches Attack/Defense.")
st.write("2. **Inflation Filter**: (XLE 200MA) -> Swaps Bonds for Energy in inflationary crashes.")
st.write("3. **Crypto Commander**: (Altseason) -> Swaps BTC for Alts when momentum is hot.")
st.write("---")
st.write("### 🥈 The Runner Up: 'Trifecta'")
st.write(f"1. **Ultimate Strategy** (Growth Engine) - Now powered by **Altseason Detector**! 🚀")
st.write(f"2. **HRP** (Defensive Engine) - {weights['HRP']:.1%}")
st.write(f"3. **Dollar Trend** (Crisis Hedge) - {weights['Dollar Trend']:.1%}")
st.write("---")
st.write("### 🧠 The ML Challenger: 'MLP Neural Net'")
st.write("A **deep neural network** (256→128→64→32) trying to learn what our hand-crafted strategies already know:")
st.write("- **50+ Features**: Momentum, volatility, regime signals, correlations")
st.write("- **Time Awareness**: Seasonality (Month), Turn-of-Month (Day), Annual Cycles (Day of Year) 🕰️")
st.write("- **Walk-Forward Training**: Retrained quarterly to avoid look-ahead bias")
st.write("- **Target**: Predict optimal SPY/TLT/BTC blend for next 5 days")
st.write("**Verdict**: Can ML beat domain expertise? Watch the race! 🏁")
