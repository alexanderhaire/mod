
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_data():
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        'ADA-USD', 'XRP-USD', 'AVAX-USD', 'SHIB-USD', 
        'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD',
        'SPY'
    ]
    print(f"Fetching data for IS/OOS test...")
    data = yf.download(tickers, start='2020-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                prices = data.xs('Adj Close', axis=1, level=1)
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    return prices.ffill().dropna(subset=['BTC-USD', 'SPY'])

def run_strategy(prices):
    # Expanded Universe Logic
    univ = [c for c in prices.columns if c != 'SPY']
    # Ensure BTC is in there for benchmarking
    if 'BTC-USD' not in univ: univ.append('BTC-USD')
    
    alts = [c for c in univ if c != 'BTC-USD']
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    LOOKBACK = 14
    TOP_N = 3
    
    btc_mom = prices['BTC-USD'].pct_change(LOOKBACK)
    
    for i in range(LOOKBACK+1, len(prices)):
        current_moms = {}
        for a in alts:
            if prices[a].iloc[i] > 0 and prices[a].iloc[i-LOOKBACK] > 0:
                try:
                    m = prices[a].iloc[i] / prices[a].iloc[i-LOOKBACK] - 1
                    current_moms[a] = m
                except:
                    pass
        
        if not current_moms:
            weights.iloc[i]['BTC-USD'] = 1.0
            continue
            
        sorted_alts = sorted(current_moms.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [x for x in sorted_alts[:TOP_N]]
        
        avg_alt_mom = np.mean([x[1] for x in top_candidates])
        btc_m = btc_mom.iloc[i]
        
        if avg_alt_mom > btc_m:
            targets = [x[0] for x in top_candidates]
            for t in targets:
                weights.iloc[i][t] = 1.0 / len(targets)
        else:
            weights.iloc[i]['BTC-USD'] = 1.0
            
    return weights.shift(1).fillna(0)

def calculate_metrics(prices, weights, label="Test"):
    # Calculate returns
    warmup = 1
    rets = prices.pct_change()
    w = weights
    
    # Align
    idx = w.index.intersection(rets.index)
    w = w.loc[idx]
    rets = rets.loc[idx]
    
    port_ret = (w * rets).sum(axis=1)
    
    total_ret = (1 + port_ret).prod() - 1
    days = len(port_ret)
    if days < 20: return
    
    ann_ret = (1 + total_ret) ** (365/days) - 1
    vol = port_ret.std() * np.sqrt(365)
    sharpe = ann_ret / vol if vol > 0 else 0
    max_dd = ((1+port_ret).cumprod() / (1+port_ret).cumprod().cummax() - 1).min()
    
    print(f"\n[{label}] ({len(port_ret)} days)")
    print(f"CAGR:   {ann_ret:.1%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"MaxDD:  {max_dd:.1%}")
    print(f"Total:  {total_ret:.1%}")

if __name__ == "__main__":
    raw_prices = fetch_data()
    
    # Split Date
    SPLIT_DATE = '2023-01-01'
    
    print(f"Split Date: {SPLIT_DATE}")
    
    # In-Sample
    prices_is = raw_prices[raw_prices.index < SPLIT_DATE]
    if len(prices_is) > 100:
        w_is = run_strategy(prices_is)
        calculate_metrics(prices_is, w_is, "IN-SAMPLE: 2020-2022")
    else:
        print("Not enough IS data")

    # Out-of-Sample
    prices_oos = raw_prices[raw_prices.index >= SPLIT_DATE]
    if len(prices_oos) > 100:
        w_oos = run_strategy(prices_oos)
        calculate_metrics(prices_oos, w_oos, "OUT-OF-SAMPLE: 2023-Present")
    else:
        print("Not enough OOS data")
