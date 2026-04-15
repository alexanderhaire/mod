
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_data():
    print("Fetching Data for Trifecta Portfolio...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'UUP', 'XLE',
        'TQQQ', 'TMF', # Lev
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        '^VIX'
    ]
    data = yf.download(tickers, period='max', interval='1d', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    return prices[prices.index >= '2005-01-01'].ffill()

def calc_sub_strategies(prices):
    rets = prices.pct_change().fillna(0)
    
    # 1. ULTIMATE (Aggressive)
    vix = prices['^VIX'].fillna(20)
    bull_vix = (vix < vix.rolling(20).mean()).shift(1).fillna(False)
    
    r_ult = pd.Series(0.0, index=rets.index)
    # Crypto part
    btc = rets.get('BTC-USD', 0)
    # Bull: 45 SPY / 40 BTC / 10 TLT
    # Bear: 15 SPY / 40 BTC / 35 TLT
    # (High Crypto Weight as per user pref)
    r_bull = 0.45*rets['SPY'] + 0.10*rets['TLT'] + 0.40*btc
    r_bear = 0.15*rets['SPY'] + 0.35*rets['TLT'] + 0.40*btc
    r_ult[bull_vix] = r_bull[bull_vix]
    r_ult[~bull_vix] = r_bear[~bull_vix]
    
    # 2. HRP (Safe / Defensive)
    hrp_assets = ['SPY', 'TLT', 'GLD']
    avail = [c for c in hrp_assets if c in rets.columns]
    vol = rets[avail].rolling(20).std()
    w = (1/vol).div((1/vol).sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w * rets[avail]).sum(axis=1)
    
    # 3. DOLLAR TREND / ALPHA (Crisis Hedge)
    # If Dollar Strong: Buy UUP. If Weak: Buy XLE (Inflation) or TQQQ?
    # Simple: Buy UUP if > 200MA. Else cash.
    uup = prices.get('UUP', pd.Series(0, index=prices.index))
    if 'UUP' in prices:
        uup_trend = (uup > uup.rolling(200).mean()).shift(1).fillna(False)
        r_dollar = pd.Series(0.0, index=rets.index)
        r_dollar[uup_trend] = rets['UUP'][uup_trend]
    else:
        r_dollar = pd.Series(0.0, index=rets.index)
        
    return pd.DataFrame({
        'Ultimate': r_ult, 
        'HRP': r_hrp, 
        'Dollar Hedge': r_dollar
    })

def analyze_portfolio(strats):
    print("\nSIMULTANEOUS TRIFECTA PORTFOLIO (Rebalanced)")
    print("=" * 65)
    
    # Equal Weight Portfolio (1/3 each)
    r_port = strats.mean(axis=1) # Daily rebalance approx
    
    # Correlation Matrix
    print("Correlation Matrix:")
    print(strats.corr())
    print("-" * 65)
    
    stats = {}
    items = list(strats.columns) + ['Trifecta Portfolio']
    for col in items:
        if col == 'Trifecta Portfolio': r = r_port
        else: r = strats[col]
        
        mach = (1+r).prod() - 1
        vol = r.std() * np.sqrt(252)
        sharpe = r.mean() * 252 / vol if vol > 0 else 0
        w = (1+r).cumprod()
        dd = (w/w.cummax()) - 1
        mdd = dd.min()
        stats[col] = (sharpe, mach, mdd)
        
    print(f"{'Strategy':<20} {'Sharpe':<8} {'Return':<10} {'MaxDD':<10}")
    print("-" * 65)
    for k, v in stats.items():
        print(f"{k:<20} {v[0]:<8.2f} {v[1]:<10.0%} {v[2]:<10.1%}")
    print("=" * 65)
    
    if stats['Trifecta Portfolio'][0] > 1.2:
        print("✅ SUCCESS: Diversification significantly boosted Sharpe.")
    else:
        print("❌ FAILURE: Correlation is too high, failed to boost Sharpe.")

if __name__ == "__main__":
    p = fetch_data()
    s = calc_sub_strategies(p)
    analyze_portfolio(s)
