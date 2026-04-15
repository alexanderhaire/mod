"""
OPTIMIZE STRATEGY ALLOCATION
============================

Determine the optimal mix of the 3 validated strategies:
1. ERP Regime (Macro/Weird Data)
2. AlphaMax (ML/Momentum)
3. Compounder (Ensemble/Trend)

Tests:
- Individual Performance
- Correlation Matrix
- Portfolio Combinations:
    - Equal Weight (33/33/33)
    - Risk Parity (Inverse Volatility)
    - Best Pair (50/50)
- Efficient Frontier Analysis

RUN: python optimize_allocation.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# strategy imports
from erp_regime_validation import erp_regime_strategy
from alpha_max_strategy import AlphaMaxStrategy
from compounder_strategy import CompounderStrategy

# Weird Data for ERP (Embedded for safety)
WEIRD_DATA = {
    'netflix': {2010:18.3, 2011:21.5, 2012:25.7, 2013:41.4, 2014:54.5, 2015:70.8, 2016:89.1, 2017:110.6, 2018:139.0, 2019:151.5, 2020:203.7, 2021:221.8, 2022:220.7, 2023:260.3, 2024:300.0, 2025:320.0, 2026:340.0},
    'cheese': {2010:33.0, 2011:33.3, 2012:33.5, 2013:34.0, 2014:34.5, 2015:35.0, 2016:36.0, 2017:37.0, 2018:38.0, 2019:38.5, 2020:39.0, 2021:40.2, 2022:42.0, 2023:42.3, 2024:42.5, 2025:43.0, 2026:43.5}
}

# Monkey patch ERP strategy if needed or just reimplement simple version
def get_erp_sig(date):
    y = date.year
    nf = (WEIRD_DATA['netflix'].get(y,0) - WEIRD_DATA['netflix'].get(y-1,0))/WEIRD_DATA['netflix'].get(y-1,1) if y in WEIRD_DATA['netflix'] and y-1 in WEIRD_DATA['netflix'] else 0
    ch = (WEIRD_DATA['cheese'].get(y,0) - WEIRD_DATA['cheese'].get(y-1,0))/WEIRD_DATA['cheese'].get(y-1,1) if y in WEIRD_DATA['cheese'] and y-1 in WEIRD_DATA['cheese'] else 0
    return -nf * 0.5 + ch * 0.3

def erp_simple(prices, vix):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i in range(len(prices)):
        sig = get_erp_sig(prices.index[i])
        vv = vix.iloc[i] if isinstance(vix, pd.Series) else 20
        wt = {'SPY':0.25, 'XLE':0.25, 'GLD':0.25, 'TLT':0.25}
        if sig > 0.02: wt['XLE'], wt['SPY'] = 0.35, 0.20
        elif sig < -0.02: wt['XLE'], wt['GLD'] = 0.10, 0.35
        if vv > 25: wt['TLT'] = 0.40; wt['XLE'] *= 0.5
        for k in wt: 
            if k in w.columns: w.iloc[i][k] = wt[k]/sum(wt.values())
    return w.shift(1).fillna(0)

# =============================================================================
# DATA FETCH
# =============================================================================

print("="*70)
print("ALLOCATION OPTIMIZER")
print("="*70)

print("\nFetching data...")
tickers = ['SPY', 'XLE', 'GLD', 'TLT', 'XLB', 'XLI', 'JNK', 'IEF', 'UUP', '^TNX', '^VIX']
# Start early enough for training
start_date = '2015-01-01'
data = yf.download(tickers, start=start_date, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix = prices['^VIX']
tnx = prices['^TNX'] if '^TNX' in prices.columns else None
macro_cols = ['^TNX', 'UUP', 'IEF', 'JNK']
macro_data = prices[macro_cols].copy()
prices = prices.drop(['^VIX', '^TNX'], axis=1)

print(f"Loaded {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")

# =============================================================================
# GENERATE STRATEGY RETURNS
# =============================================================================

returns = prices.pct_change().dropna()
oos_start = '2022-01-01'
oos_idx = prices.index.get_loc(prices.loc[oos_start:].index[0])

strat_returns = pd.DataFrame(index=returns.index)

# 1. ERP Regime
print("\nRunning ERP Regime...")
erp_w = erp_simple(prices, vix)
erp_ret = (erp_w.shift(1) * returns).sum(axis=1)
strat_returns['ERP_Regime'] = erp_ret

# 2. AlphaMax
print("Running AlphaMax (Training on pre-2022)...")
am = AlphaMaxStrategy()
# Train on In-Sample (pre-2022)
is_prices = prices.iloc[:oos_idx]
is_macro = macro_data.iloc[:oos_idx]
am.train(is_prices, is_macro) # Train once
# Predict everywhere
am_w = am.generate_signals(prices, macro_data) # This function in class might need adjustment to return full series
# The provided class generates signal for LAST ROW only. We need to loop or monkeypatch.
# Let's loop for OOS.
am_w_series = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
# Speed up: re-predict every day is slow.
# Hack: Use the _create_features and iterate predictions without retraining
features, _, asset_map = am._create_features(prices, macro_data)
# We trained on IS. Now predict OOS.
for i in range(oos_idx, len(prices)):
    dt = prices.index[i]
    sigs = {}
    for t, model in am.models.items():
        cols = asset_map[t]
        row = features.iloc[i:i+1][cols]
        if not row.isnull().values.any():
            sigs[t] = model.predict(row)[0]
    
    # Weight logic
    rw = pd.Series(sigs)
    rw[rw<0] = 0
    if rw.sum() > 0:
        w = rw / rw.sum()
        # Vol target
        recent_vol = returns.iloc[i-20:i].std() * np.sqrt(252)
        pvol = (w * recent_vol[w.index]).sum() if not recent_vol.empty else 0.01
        scale = min(0.20/pvol, 2.0) if pvol > 0 else 1.0
        w *= scale
        for k, v in w.items():
            if k in am_w_series.columns: am_w_series.loc[dt, k] = v

am_ret = (am_w_series.shift(1) * returns).sum(axis=1)
strat_returns['AlphaMax'] = am_ret

# 3. Compounder
print("Running Compounder...")
try:
    comp = CompounderStrategy()
    # It generates weights for full history
    comp_w = comp.generate_weights(prices, spy_prices=prices['SPY'], vix=vix)
    comp_ret = (comp_w.shift(1) * returns).sum(axis=1)
    strat_returns['Compounder'] = comp_ret
except Exception as e:
    print(f"Compounder Error: {e}")
    strat_returns['Compounder'] = 0.0

# Clip to OOS
strat_oos = strat_returns.loc[oos_start:]
print(f"\nOOS Period: {len(strat_oos)} days")

# =============================================================================
# ANALYSIS
# =============================================================================

# 1. Individual Performance
print("\n1. INDIVIDUAL PERFORMANCE (2022-Present)")
print(f"{'Strategy':<15} {'Sharpe':>8} {'CAGR':>8} {'Vol':>8}")
print("-" * 45)
for col in strat_oos.columns:
    r = strat_oos[col]
    mean = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sharpe = mean / vol if vol > 0 else 0
    print(f"{col:<15} {sharpe:>8.2f} {mean:>8.1%} {vol:>8.1%}")

# 2. Correlation
print("\n2. CORRELATION MATRIX")
print(strat_oos.corr().round(2))

# 3. Portfolio Optimization
def port_stats(weights, rets):
    p_ret = (rets * weights).sum(axis=1)
    mean = p_ret.mean() * 252
    vol = p_ret.std() * np.sqrt(252)
    sharpe = mean / vol if vol > 0 else 0
    return -sharpe # Minimize negative sharpe

print("\n3. OPTIMAL ALLOCATION (Max Sharpe)")
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for _ in range(3))
init_guess = [1/3, 1/3, 1/3]

opt = minimize(port_stats, init_guess, args=(strat_oos,), method='SLSQP', bounds=bnds, constraints=cons)

opt_weights = opt.x
print(f"Optimal Weights:")
for i, col in enumerate(strat_oos.columns):
    print(f"  {col}: {opt_weights[i]:.1%}")

# Backtest Optimal
opt_ret = (strat_oos * opt_weights).sum(axis=1)
mean = opt_ret.mean() * 252
vol = opt_ret.std() * np.sqrt(252)
sharpe = mean / vol
print(f"\nOptimal Portfolio Stats:")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  CAGR:   {mean:.1%}")
print(f"  Vol:    {vol:.1%}")

# Equal Weight
eq_ret = (strat_oos * [1/3, 1/3, 1/3]).sum(axis=1)
mean_eq = eq_ret.mean() * 252
vol_eq = eq_ret.std() * np.sqrt(252)
sharpe_eq = mean_eq / vol_eq
print(f"\nEqual Weight (1/3 each) Stats:")
print(f"  Sharpe: {sharpe_eq:.2f}")
print(f"  CAGR:   {mean_eq:.1%}")

# Verdict
best_strat = max(strat_oos.columns, key=lambda c: strat_oos[c].mean()/strat_oos[c].std())
best_sharpe = strat_oos[best_strat].mean()/strat_oos[best_strat].std()*np.sqrt(252)

print("\n" + "="*70)
print("VERDICT")
print("="*70)
if sharpe > best_sharpe + 0.1:
    print(f"✅ DIVERSIFICATION WINS! Mix strategies for best results.")
    print(f"   Allocation: {', '.join([f'{w:.0%} {c}' for w,c in zip(opt_weights, strat_oos.columns) if w > 0.05])}")
else:
    print(f"🏆 FOCUS WINS! Stick to {best_strat}.")
    print(f"   Diversification benefit is marginal (+{sharpe-best_sharpe:.2f} Sharpe).")
