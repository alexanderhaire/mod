
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')

def optimize_portfolio(rets):
    """Find weights that maximize Sharpe Ratio."""
    def neg_sharpe(weights, rets):
        p_ret = (rets * weights).sum(axis=1)
        mean = p_ret.mean() * 252 # Annualized (approx, assumes daily-like freq)
        vol = p_ret.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        return -sharpe

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(rets.shape[1]))
    init_guess = [1.0 / rets.shape[1]] * rets.shape[1]
    
    if len(rets) < 5:
        return init_guess
        
    try:
        opt = minimize(neg_sharpe, init_guess, args=(rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return init_guess

def fetch_recent_data():
    print("Fetching last 20+ YEARS (MAX) of DAILY data...")
    tickers = ['SPY', 'TLT', 'GLD', 'BTC-USD', '^VIX', 'UUP']
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    
    # Handle MultiIndex
    try:
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        open_price = data['Open']
    except:
        adj_close = data['Close']
        open_price = data['Open']
    
    # Filter to last 20 years approx
    start_date = '2004-01-01'
    adj_close = adj_close[adj_close.index >= start_date]
    open_price = open_price[open_price.index >= start_date]
        
    return adj_close, open_price

def analyze():
    close, open_ = fetch_recent_data()
    
    # Data is already daily
    daily_close = close.ffill() # Don't dropna yet, we want to keep early years even if BTC missing
    daily_open = open_.ffill()
    
    # Align indices
    common = daily_close.index.intersection(daily_open.index)
    daily_close = daily_close.loc[common]
    daily_open = daily_open.loc[common]
    
    daily_rets = daily_close.pct_change().fillna(0)
    
    # --- 1. STRATEGY PERFORMANCE ---
    
    # Ultimate (Simulated)
    # Handle missing VIX or BTC
    if '^VIX' in daily_close.columns:
        vix = daily_close['^VIX'].fillna(20) # Default to 20 if missing
        vix_ma = vix.rolling(20).mean()
        bull = vix < vix_ma
        bear = ~bull
    else:
        bull = pd.Series(True, index=daily_close.index) # Default Bull
        bear = ~bull

    w_ult = pd.DataFrame(0.0, index=daily_rets.index, columns=['SPY', 'TLT', 'BTC-USD'])
    
    w_ult.loc[bull, 'SPY'] = 0.45; w_ult.loc[bull, 'TLT'] = 0.10
    w_ult.loc[bear, 'SPY'] = 0.15; w_ult.loc[bear, 'TLT'] = 0.35
    w_ult['BTC-USD'] = 0.20
    
    # If BTC missing (e.g. 2004-2010), allocation should ideally go to SPY/TLT or Cash
    # For simplicity, we keep the weight but return is 0, effective "Cash" drag
    # Or cleaner: Re-normalize if asset missing
    
    # Shift signals
    sig_shifted = w_ult.shift(1)
    
    r_ult = pd.Series(0.0, index=daily_rets.index)
    for col in ['SPY', 'TLT', 'BTC-USD']:
        if col in daily_rets.columns:
            r_ult += sig_shifted[col] * daily_rets[col]
    
    # HRP
    hrp_assets = ['SPY', 'TLT', 'GLD']
    avail_hrp = [c for c in hrp_assets if c in daily_rets.columns]
    
    # Calculating Vol needs data.
    # We will compute HRP only where we have data for at least SPY/TLT
    # If GLD missing (pre 2004), exclude it dynamically?
    # Simplified: HRP requires all assets to be present to work "correctly" logic wise
    # but we can try dropna just for checking avail
    
    # Rolling vol
    vol = daily_rets[avail_hrp].rolling(20).std()
    inv_vol = 1 / vol.replace(0, np.nan)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * daily_rets[avail_hrp]).sum(axis=1)
    
    # Dollar Trend
    if 'UUP' in daily_close.columns:
        uup = daily_close['UUP']
        # UUP inception 2007. Before that 0 returns (Cash)
        uup_trend = (uup > uup.rolling(20).mean()).astype(int)
        r_uup = uup_trend.shift(1) * daily_rets['UUP']
    else:
        r_uup = pd.Series(0, index=daily_rets.index)
    
    # SPY
    r_spy = daily_rets['SPY']
    
    # BTC
    r_btc = daily_rets.get('BTC-USD', pd.Series(0, index=daily_rets.index))
    
    # --- 2. DAY vs NIGHT (SPY) ---
    # Intraday: Open to Close
    r_day = (daily_close['SPY'] / daily_open['SPY']) - 1
    
    # Overnight: Close(T-1) to Open(T)
    r_night = (daily_open['SPY'] / daily_close['SPY'].shift(1)) - 1
    
    r_spy_day = r_day.fillna(0)
    r_spy_night = r_night.fillna(0)
    
    # Combine
    # We dropna ONLY at the end, so we might lose early days if SPY data bad
    candidates = pd.DataFrame({
        'Ultimate': r_ult,
        'HRP': r_hrp,
        'Dollar Trend': r_uup,
        'SPY (Hold)': r_spy,
        'BTC (Hold)': r_btc,
        'SPY Day': r_spy_day,
        'SPY Night': r_spy_night
    }).dropna()
    
    # --- 3. OPTIMIZATION ---
    print("\nOPTIMIZING FOR 20-YEAR REGIME (2004-2024)")
    print("-" * 50)
    
    # Annualization for Daily
    ANN_FACTOR = np.sqrt(252)
    
    def optimize_daily(rets):
        def neg_sharpe(weights, rets):
            p_ret = (rets * weights).sum(axis=1)
            if p_ret.std() == 0: return 0
            sharpe = (p_ret.mean() / p_ret.std()) * ANN_FACTOR
            return -sharpe

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(rets.shape[1]))
        init = [1.0/rets.shape[1]] * rets.shape[1]
        try:
            return minimize(neg_sharpe, init, args=(rets,), bounds=bnds, constraints=cons).x
        except:
            return init
    
    # A. Single Strategy Stats
    print(f"{'Strategy':<20} {'Sharpe':<10} {'Return':<10}")
    print("-" * 50)
    for col in candidates.columns:
        r = candidates[col]
        cum = (1 + r).prod() - 1
        sharpe = (r.mean() / r.std()) * ANN_FACTOR if r.std() > 0 else 0
        print(f"{col:<20} {sharpe:<10.2f} {cum:<10.1%}")
        
    # B. Best Combination
    w = optimize_daily(candidates)
    
    print("\n" + "="*50)
    print("🏆 WINNING 20-YEAR MIX (Highest Sharpe)")
    print("="*50)
    for i, col in enumerate(candidates.columns):
        if w[i] > 0.01:
            print(f"  {col:<20}: {w[i]:.1%}")
            
    r_opt = (candidates * w).sum(axis=1)
    cum = (1 + r_opt).prod() - 1
    sharpe = (r_opt.mean() / r_opt.std()) * ANN_FACTOR if r_opt.std() > 0 else 0
    
    print("-" * 50)
    print(f"  Combined Sharpe   : {sharpe:.2f}")
    print(f"  Combined Return   : {cum:.1%}")
    print("="*50)

if __name__ == "__main__":
    analyze()
