"""
NUCLEAR STRESS TESTING
======================
Simulating extreme market conditions for the AlphaMax/ERP Portfolio.

Scenarios:
1. Transaction Costs Sweep (Impact of slippage/commissions)
2. Liquidity Crisis (Bid-Ask spread widening 10x)
3. Correlation Breakdown (Stocks/Bonds correlation -> 1.0)
4. Black Swan Event (-20% Gap Down simulation)

RUN: python test_nuclear_option.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Strategy Imports
from erp_regime_validation import erp_regime_strategy
from alpha_max_strategy import AlphaMaxStrategy

# Params
ALLOCATION = {'AlphaMax': 0.60, 'ERP': 0.40}
INITIAL_CAPITAL = 1_000_000

# =============================================================================
# DATA FETCH
# =============================================================================

print("="*70)
print("☢️  NUCLEAR STRESS TEST INITIALIZED")
print("="*70)

print("\nFetching OOS Data (2022-Present)...")
tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'XLB', 'XLI', 'JNK', 'IEF', 'UUP', '^TNX', '^VIX']
start_date = '2015-01-01' # Need history for training
data = yf.download(tickers, start=start_date, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix = prices['^VIX']
macro_cols = ['^TNX', 'UUP', 'IEF', 'JNK']
macro_data = prices[macro_cols].copy()
prices = prices.drop(['^VIX', '^TNX'], axis=1)

oos_start = '2022-01-01'
oos_idx = prices.index.get_loc(prices.loc[oos_start:].index[0])
oos_prices = prices.iloc[oos_idx:]
returns = prices.pct_change()

# =============================================================================
# BASELINE OOS PERFORMANCE
# =============================================================================

print("\nGenerating Baseline Returns...")

# ERP
erp_w = erp_regime_strategy(prices, vix) # This returns full history weights
# AlphaMax (Train once on pre-2022)
am = AlphaMaxStrategy()
am.train(prices.iloc[:oos_idx], macro_data.iloc[:oos_idx])

# Generate AM weights for full history (using fast path)
am_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
features, _, asset_map = am._create_features(prices, macro_data)

print(f"Predicting AlphaMax OOS ({len(prices)-oos_idx} days)...")
# OOS Loop for AM (Fast Mockup using trained models)
# Note: For real rigorous OOS we'd retrain, but here we test Robustness of the Signals
for i in range(oos_idx, len(prices)):
    dt = prices.index[i]
    sigs = {}
    row = features.iloc[i:i+1]
    for t, model in am.models.items():
        cols = asset_map[t]
        feat_vec = row[cols]
        if not feat_vec.isnull().values.any():
            sigs[t] = model.predict(feat_vec)[0]
    
    # Weight & Scale
    rw = pd.Series(sigs)
    rw[rw<0] = 0
    if rw.sum() > 0:
        w = rw / rw.sum()
        # Vol target
        recent_vol = returns.iloc[i-20:i].std() * np.sqrt(252)
        pvol = (w * recent_vol[w.index]).sum() if not recent_vol.empty else 0.01
        scale = min(0.20/pvol, 2.0) if pvol>0 else 1.0
        w *= scale
        for k, v in w.items():
            if k in am_w.columns: am_w.loc[dt, k] = v

# Combine Portfolio
combined_w = (am_w * ALLOCATION['AlphaMax']) + (erp_w * ALLOCATION['ERP'])

# OOS Slice
combined_w_oos = combined_w.loc[oos_start:]
returns_oos = returns.loc[oos_start:]

# Baseline Stats
baseline_ret = (combined_w_oos.shift(1) * returns_oos).sum(axis=1)
base_mean = baseline_ret.mean() * 252
base_vol = baseline_ret.std() * np.sqrt(252)
base_sharpe = base_mean / base_vol

print(f"Baseline OOS Sharpe: {base_sharpe:.2f}")

# =============================================================================
# SCENARIO 1: TRANSACTION COSTS SWEEP
# =============================================================================
print("\n" + "-"*60)
print("SCENARIO 1: TRANSACTION COST SWEEP")
print("-"*60)

# Calculate Turnover
# Turnover = sum(|w_t - w_{t-1} * (1+r)|) approximately sum(|diff|)
turnover = combined_w_oos.diff().abs().sum(axis=1).mean() * 252 # Annualized turnover sum
print(f"Est. Annual Turnover: {turnover:.1f}x ({(turnover/252)*100:.1f}% daily)")

print(f"{'Cost (bps)':<12} {'Sharpe':<10} {'Return':<10} {'Breakeven?'}")
for bps in [0, 5, 10, 20, 50, 100]:
    cost_daily = (combined_w_oos.diff().abs().sum(axis=1) * (bps/10000))
    net_ret = baseline_ret - cost_daily
    
    ann = net_ret.mean() * 252
    vol = net_ret.std() * np.sqrt(252)
    sh = ann / vol if vol > 0 else 0
    
    print(f"{bps:<12} {sh:<10.2f} {ann:<10.1%} {'✅' if sh > 1.0 else '⚠️' if sh > 0 else '❌'}")

# =============================================================================
# SCENARIO 2: LIQUIDITY CRISIS (Widen Spreads 10x)
# =============================================================================
print("\n" + "-"*60)
print("SCENARIO 2: LIQUIDITY CRISIS")
print("-"*60)
print("Simulating 10x spread widening during high volatility days (VIX > 30)")

# Logic: If VIX > 30, transaction costs jump to 50bps instead of 5bps
crisis_cost = pd.Series(0.0005, index=baseline_ret.index) # 5bps base
# Start VIX aligned keys
vix_oos = vix.loc[oos_start:]
# Align indices
common_idx = crisis_cost.index.intersection(vix_oos.index)
crisis_cost.loc[common_idx] = np.where(vix_oos.loc[common_idx] > 30, 0.0050, 0.0005) # 50bps if VIX>30

daily_turnover = combined_w_oos.diff().abs().sum(axis=1)
crisis_penalty = daily_turnover * crisis_cost

crisis_ret = baseline_ret - crisis_penalty
c_sharpe = (crisis_ret.mean() * 252) / (crisis_ret.std() * np.sqrt(252))
c_dd = (crisis_ret.cumsum() - crisis_ret.cumsum().cummax()).min()

print(f"Crisis Sharpe: {c_sharpe:.2f}")
print(f"Crisis Max DD: {c_dd*100:.1f}% (Approx from daily log sum)")

if c_sharpe > 0.8:
    print("✅ SURVIVED LIQUIDITY SHOCK")
else:
    print("❌ FAILED LIQUIDITY SHOCK")

# =============================================================================
# SCENARIO 3: CORRELATION BREAKDOWN
# =============================================================================
print("\n" + "-"*60)
print("SCENARIO 3: CORRELATION BREAKDOWN (Stock/Bond = 1.0)")
print("-"*60)

# Simulate returns where TLT moves EXACTLY like SPY (loss of hedge)
# We replace TLT returns with SPY returns * (TLT_vol / SPY_vol)
mod_returns = returns_oos.copy()
spy_vol = mod_returns['SPY'].std()
tlt_vol = mod_returns['TLT'].std()
# Force TLT to be correlated 1.0 with SPY
mod_returns['TLT'] = mod_returns['SPY'] * (tlt_vol / spy_vol)

# Recalculate portfolio return with BROKEN correlations
broken_ret = (combined_w_oos.shift(1) * mod_returns).sum(axis=1)
b_sharpe = (broken_ret.mean() * 252) / (broken_ret.std() * np.sqrt(252))

print(f"Correlation Breakdown Sharpe: {b_sharpe:.2f}")
print(f"Change: {b_sharpe - base_sharpe:.2f}")

if b_sharpe > 0.5:
    print("✅ SURVIVED HEDGE FAILURE (Strategy has alpha beyond curve)")
else:
    print("❌ FAILED HEDGE FAILURE (Relied purely on diversification)")

# =============================================================================
# SCENARIO 4: BLACK SWAN (-20% Gap Day)
# =============================================================================
print("\n" + "-"*60)
print("SCENARIO 4: BLACK SWAN SIMULATION")
print("-"*60)

# Inject a -20% day for SPY and XLE, -10% for JNK
# Assume Gold/TLT flat (liquidity trap) or up slightly? 
# In liquidity crisis, everything sells off. Let's say GLD/TLT -5%.

bs_ret = baseline_ret.copy()
# Insert dummy black swan at the end
shock_impact = (
    combined_w_oos.iloc[-1]['SPY'] * -0.20 +
    combined_w_oos.iloc[-1]['XLE'] * -0.20 +
    combined_w_oos.iloc[-1]['JNK'] * -0.10 +
    combined_w_oos.iloc[-1]['GLD'] * -0.05 +
    combined_w_oos.iloc[-1]['TLT'] * -0.05
) * 2.0 # Assume we were leveraged 2x for worst case? No, strategy targets vol.
        # Strategy leverage is embedded in weights.
        # Wait, strategy weights sum to leverage (e.g. 1.5).
        # combined_w_oos ALREADY includes leverage scaling.
        # So manual calc:

shock_day_ret = 0
w_last = combined_w_oos.iloc[-1]
shock_day_ret += w_last.get('SPY', 0) * -0.20
shock_day_ret += w_last.get('XLE', 0) * -0.20
shock_day_ret += w_last.get('XLB', 0) * -0.20
shock_day_ret += w_last.get('XLI', 0) * -0.20
shock_day_ret += w_last.get('JNK', 0) * -0.10
shock_day_ret += w_last.get('GLD', 0) * -0.05
shock_day_ret += w_last.get('TLT', 0) * -0.05

print(f"Estimated Portfolio Impact of -20% Market Crash: {shock_day_ret:.1%}")
print(f"Current Portfolio Leverage: {w_last.sum():.2f}x")

print("\n" + "="*70)
print("FINAL STRESS TEST VERDICT")
print("="*70)

failures = 0
if c_sharpe < 0.5: failures += 1
if b_sharpe < 0.0: failures += 1 # Loss of correlation kills it?
if pd.isna(shock_day_ret) or shock_day_ret < -0.30: failures += 1 # Blowup risk

if failures == 0:
    print("🏆 NUCLEAR PROOF. Strategy is robust to extreme shocks.")
elif failures == 1:
    print("⚠️  RESILIENT. Minor weakness detected in extreme scenarios.")
else:
    print("❌ FRAGILE. Significant risks in extreme scenarios.")
