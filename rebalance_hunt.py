"""
Rebalancing Alpha Hunt
======================

Optimizing the "Meta-Rebalance" logic between Ultimate Strategy and HRP.
Goal: Maximize Sharpe Ratio after Transaction Costs.

Variables:
- Frequency: Daily, Weekly, Monthly, Quarterly.
- Threshold: 0% (Time-only), 5%, 10%, 15%, 20% (Drift Bands).

Cost Assumption: 10bps (0.10%) per turnover.

RUN: python rebalance_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. GENERATE RETURNS (ENGINES)
# =============================================================================

def fetch_data():
    print("⚖️ Fetching Data for Rebalance Lab...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', # Trad
        'BTC-USD', 'ETH-USD', # Crypto
        '^VIX', '^VIX3M' # Signals
    ]
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    prices = prices.drop(columns=[c for c in ['^VIX', '^VIX3M'] if c in prices.columns])
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days")
    return prices, vix, vix3m

def get_hrp_proxy(prices):
    # Proxy HRP with Inverse Volatility (0.9 corr, faster)
    returns = prices.pct_change().dropna()
    weights = pd.DataFrame(index=returns.index, columns=prices.columns)
    
    lookback = 126
    
    # Monthly rebal weights
    m_dates = returns.resample('M').last().index
    
    for t in m_dates:
        hist = returns[returns.index <= t].tail(lookback)
        if len(hist) < 60: continue
        vol = hist.std()
        w = (1/vol) / (1/vol).sum()
        # Find closest date in index
        try:
            weights.loc[t] = w
        except:
            pass
            
    weights = weights.ffill().dropna()
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    return port_ret

def get_ultimate(prices, vix, vix3m):
    # Simplified Ultimate Logic
    vix_sig = pd.Series(0, index=prices.index)
    if vix is not None and vix3m is not None:
        ratio = (vix/vix3m).rolling(5).mean()
        vix_sig[ratio < 0.90] = 1 # Bullish
        vix_sig[ratio > 1.05] = -1 # Bearish
        
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(1, len(prices)):
        vs = vix_sig.iloc[i]
        # Trad
        if vs > 0: w_t = {'SPY': 0.45, 'TLT': 0.10}
        elif vs < 0: w_t = {'SPY': 0.15, 'TLT': 0.35}
        else: w_t = {'SPY': 0.30, 'TLT': 0.22}
        
        # Crypto
        if 'BTC-USD' in prices.columns:
            # Simple mom check
            btc = prices['BTC-USD'].iloc[i]
            prev = prices['BTC-USD'].iloc[i-1] if i > 0 else btc
            if btc > prev:
                 w_c = {'BTC-USD': 0.25}
            else:
                 w_c = {'BTC-USD': 0.10}
        
        for t, w in w_t.items(): 
             if t in prices.columns: weights.iloc[i][t] = w
        for t, w in w_c.items():
             if t in prices.columns: weights.iloc[i][t] = w
             
    port_ret = (weights.shift(1) * prices.pct_change()).sum(axis=1)
    return port_ret

# =============================================================================
# 2. REBALANCE SIMULATOR
# =============================================================================

def simulate_rebalancing(r_ult, r_hrp, freq_days, threshold_pct, cost_bps=0.0010):
    """
    Simulate portfolio of 50/50 Ultimate/HRP with specific rebalancing rules.
    """
    # Align
    common = r_ult.index.intersection(r_hrp.index)
    r_ult = r_ult.loc[common]
    r_hrp = r_hrp.loc[common]
    
    # Init Portfolio
    cash = 1.0
    # Initial alloc 50/50
    w_u = 0.5
    w_h = 0.5
    
    val_u = cash * w_u
    val_h = cash * w_h
    
    portfolio_values = []
    costs = []
    
    # Loop daily
    # Can't vectorize easily because threshold depends on path
    
    last_rebal = 0
    days_since_rebal = 0
    
    for i in range(len(common)):
        # 1. Update Values
        ret_u = r_ult.iloc[i]
        ret_h = r_hrp.iloc[i]
        
        val_u *= (1 + ret_u)
        val_h *= (1 + ret_h)
        total_val = val_u + val_h
        
        # 2. Check Drift
        curr_w_u = val_u / total_val
        drift = abs(curr_w_u - 0.50)
        
        cost = 0.0
        rebalance = False
        
        # Check Rules
        days_since_rebal += 1
        
        # Time Trigger
        if freq_days > 0 and days_since_rebal >= freq_days:
            # If threshold is 0, always rebal
            if threshold_pct == 0:
                rebalance = True
            # If threshold > 0, check drift too
            elif drift > threshold_pct:
                rebalance = True
        
        # Drift Trigger (Independent of time? Usually paired. Let's assume OR condition or AND?
        # Usually "Smart Rebal" checks every day (freq=1) but only acts if drift > thresh.
        # So we use freq=1 for pure threshold strategies.
        
        if rebalance:
            # Target 50/50
            target_u = total_val * 0.5
            target_h = total_val * 0.5
            
            # Turnover amount
            trade_amt = abs(target_u - val_u) # Same as target_h - val_h
            
            # Cost
            cost = trade_amt * cost_bps
            total_val -= cost
            
            # Reset
            val_u = total_val * 0.5
            val_h = total_val * 0.5
            
            days_since_rebal = 0
            
        portfolio_values.append(total_val)
        costs.append(cost)
        
    # Metrics
    port_curve = pd.Series(portfolio_values, index=common)
    returns = port_curve.pct_change().dropna()
    
    ann = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    total_cost = sum(costs)
    turnover_ratio = total_cost / cost_bps / len(common) * 252 # Annualized turnover approx
    
    return {
        'Sharpe': sharpe,
        'Ann': ann,
        'Vol': vol,
        'CostDrag': total_cost / port_curve.iloc[-1] # roughly
    }

# =============================================================================
# 3. OPTIMIZATION GRID
# =============================================================================

def run_grid_search(r_ult, r_hrp):
    print("\n🔬 REBALANCE OPTIMIZATION GRID (Cost 10bps)")
    print(f"{'Freq':<12} {'Thresh':<8} {'Sharpe':<8} {'Ann':<8} {'CostDrag'}")
    print("-" * 60)
    
    # Settings
    # Freq: 1 (Daily/Check), 5 (Weekly), 21 (Monthly), 63 (Quarterly)
    # Thresh: 0.00, 0.05, 0.10, 0.15, 0.20
    
    configs = [
        (1, 0.00, "Daily Fixed"),
        (5, 0.00, "Weekly Fixed"),
        (21, 0.00, "Monthly Fixed"),
        (63, 0.00, "Qtly Fixed"),
        (1, 0.05, "Smart 5%"),
        (1, 0.10, "Smart 10%"),
        (1, 0.15, "Smart 15%"),
        (1, 0.20, "Smart 20%"),
        (21, 0.05, "Monthly+5%"), # Check monthly, rebal only if 5% drift
    ]
    
    best_sharpe = 0
    best_cfg = ""
    
    for freq, thresh, label in configs:
        res = simulate_rebalancing(r_ult, r_hrp, freq, thresh)
        print(f"{label:<12} {thresh:>6.0%} {res['Sharpe']:>8.2f} {res['Ann']:>7.1%} {res['CostDrag']:>8.2%}")
        
        if res['Sharpe'] > best_sharpe:
            best_sharpe = res['Sharpe']
            best_cfg = label
            
    print("-" * 60)
    print(f"🏆 WINNER: {best_cfg} (Sharpe {best_sharpe:.2f})")

if __name__ == "__main__":
    prices, vix, vix3m = fetch_data()
    r_ult = get_ultimate(prices, vix, vix3m)
    r_hrp = get_hrp_proxy(prices)
    run_grid_search(r_ult, r_hrp)
