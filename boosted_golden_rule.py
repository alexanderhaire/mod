
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def fetch_data():
    print("Fetching 20 Years of Data for Boosted Analysis...")
    tickers = [
        'SPY', 'QQQ', 'TQQQ', 'UPRO', # Risk / Lev
        'TLT', 'TMF', 'GLD', 'UUP', # Safety
        'BTC-USD', # Crypto 
        '^VIX', '^VIX3M' # Signals
    ]
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    
    try:
        adj_close = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    except:
        adj_close = data['Close']
        
    # Filter 2004+
    return adj_close[adj_close.index >= '2004-01-01'].ffill()

def calc_strategy_returns(prices):
    rets = prices.pct_change().fillna(0)
    
    # 1. THE SUB-STRATEGIES
    # =====================
    
    # A. Ultimate Strategy (The Growth Engine)
    # Simple Logic: 60% SPY/TLT (VIX timed) + 40% BTC (if avail)
    # We simulate a simplified version vectorized
    vix = prices['^VIX'].fillna(20)
    vix_ma = vix.rolling(20).mean()
    bull_vix = vix < vix_ma
    
    w_ult = pd.DataFrame(0.0, index=rets.index, columns=['SPY', 'TLT', 'BTC-USD'])
    w_ult.loc[bull_vix, 'SPY'] = 0.45; w_ult.loc[bull_vix, 'TLT'] = 0.10
    w_ult.loc[~bull_vix, 'SPY'] = 0.15; w_ult.loc[~bull_vix, 'TLT'] = 0.35
    w_ult['BTC-USD'] = 0.20 # Constant crypto exposure in Ultimate
    
    # Handle missing BTC
    if 'BTC-USD' not in rets.columns: w_ult['BTC-USD'] = 0
    
    r_ultimate = (w_ult.shift(1) * rets[['SPY', 'TLT', 'BTC-USD']]).sum(axis=1)
    
    # A2. Levered Growth (TQQQ) - For pure bull runs
    r_tqqq = rets['TQQQ'].fillna(rets['QQQ'] * 3) # Syntehtic backfill with QQQ x3
    
    # B. HRP Strategy (The Safety Net)
    # Inverse Vol weighting of SPY, TLT, GLD
    hrp_assets = ['SPY', 'TLT', 'GLD']
    avail_hrp = [c for c in hrp_assets if c in rets.columns]
    vol = rets[avail_hrp].rolling(20).std()
    inv_vol = 1 / vol.replace(0, np.nan)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[avail_hrp]).sum(axis=1)
    
    # C. Dollar Trend (The Hedge)
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        r_uup = ((uup > uup.rolling(200).mean()).shift(1) * rets['UUP'])
    else:
        r_uup = pd.Series(0, index=rets.index)
        
    # 2. THE MASTER SWITCH (Golden Rule)
    # ==================================
    spy = prices['SPY']
    ma200 = spy.rolling(200).mean()
    
    # Fill NAs to avoid masking errors. 
    # Default to Bearish (False) if insufficient data for MA
    regime_bull = (spy > ma200).shift(1).fillna(False) 
    
    # Align Bull/Bear indices (prices vs returns)
    # Reindex `regime_bull` to match `rets` just in case
    regime_bull = regime_bull.reindex(rets.index).fillna(False)
    
    # 3. COMBINATIONS
    # ===============
    
    strategies = {}
    
    # Baseline: Simple Golden Rule (SPY in Bull / TLT in Bear)
    r_base = pd.Series(0.0, index=rets.index)
    r_base[regime_bull] = rets['SPY']
    r_base[~regime_bull] = rets['TLT']
    strategies['Original Golden Rule'] = r_base
    
    # Boost 1: Ultimate in Bull / HRP in Bear
    # "The Smart Switch"
    r_smart = pd.Series(0.0, index=rets.index)
    r_smart[regime_bull] = r_ultimate
    r_smart[~regime_bull] = r_hrp
    strategies['Boosted: Ultimate/HRP'] = r_smart
    
    # Boost 2: Levered in Bull / HRP in Bear
    # "The Aggressive Switch"
    r_agg = pd.Series(0.0, index=rets.index)
    r_agg[regime_bull] = r_tqqq # 3x Tech
    r_agg[~regime_bull] = r_hrp # Safety
    strategies['Boosted: TQQQ/HRP'] = r_agg
    
    # Boost 3: Ultimate in Bull / Dollar in Bear
    # "The Crisis Hedge"
    r_crisis = pd.Series(0.0, index=rets.index)
    r_crisis[regime_bull] = r_ultimate
    r_crisis[~regime_bull] = r_uup
    strategies['Boosted: Ultimate/USD'] = r_crisis
    
    # Comparison
    print("\nBOOSTED GOLDEN RULE ANALYSIS (2004-2024)")
    print("=" * 65)
    print(f"{'Strategy':<25} {'Sharpe':<8} {'Return':<10} {'MaxDD':<10}")
    print("-" * 65)
    
    results = []
    for name, r in strategies.items():
        cum = (1 + r).prod() - 1
        vol = r.std() * np.sqrt(252)
        sharpe = r.mean() * 252 / vol if vol > 0 else 0
        
        wealth = (1 + r).cumprod()
        dd = (wealth / wealth.cummax()) - 1
        mdd = dd.min()
        
        results.append((name, sharpe, cum, mdd))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    for name, s, c, d in results:
        print(f"{name:<25} {s:<8.2f} {c:<10.0%} {d:<10.1%}")
        
    print("-" * 65)
    print(f"🏆 BEST STRATEGY: {results[0][0]}")
    print("=" * 65)

if __name__ == "__main__":
    p = fetch_data()
    calc_strategy_returns(p)
