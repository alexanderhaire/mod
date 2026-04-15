
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    print(f"Fetching data for: {tickers}")
    data = yf.download(tickers, start='2020-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.ffill().dropna()

def backtest_concentration(prices, top_n=1):
    # Momentum lookback
    LOOKBACK = 14
    
    # Alts
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    alts = [c for c in alts if c in prices.columns]
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Logic:
    # If Alt Momentum > BTC Momentum -> Go Long Top N Alts
    # Else -> Go Long BTC
    
    btc_mom = prices['BTC-USD'].pct_change(LOOKBACK)
    
    for i in range(LOOKBACK+1, len(prices)):
        # Calculate Alt Mom
        current_moms = {}
        for a in alts:
            try:
                # Simple return over lookback
                m = prices[a].iloc[i] / prices[a].iloc[i-LOOKBACK] - 1
                current_moms[a] = m
            except:
                pass
        
        # Avg Alt Mom (for the switch decision)
        if not current_moms:
            weights.iloc[i]['BTC-USD'] = 1.0
            continue
            
        avg_alt_mom = np.mean(list(current_moms.values()))
        btc_m = btc_mom.iloc[i]
        
        if avg_alt_mom > btc_m:
            # Altseason!
            # Sort by momentum
            sorted_alts = sorted(current_moms.items(), key=lambda x: x[1], reverse=True)
            
            # Pick Top N
            targets = [x[0] for x in sorted_alts[:top_n]]
            
            for t in targets:
                weights.iloc[i][t] = 1.0 / len(targets)
        else:
            # BTC Season
            weights.iloc[i]['BTC-USD'] = 1.0
            
    return weights.shift(1).fillna(0)

def compute_stats(prices, weights):
    warmup = 20
    rets = prices.pct_change().iloc[warmup:]
    w = weights.iloc[warmup:]
    
    port_ret = (w * rets).sum(axis=1)
    
    total_ret = (1 + port_ret).prod() - 1
    ann_ret = (1 + port_ret).prod() ** (365/len(port_ret)) - 1
    vol = port_ret.std() * np.sqrt(365)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    return ann_ret, sharpe, total_ret

if __name__ == "__main__":
    prices = fetch_data()
    
    print("\n--- RESULTS (2020-Now) ---")
    
    # 1. Basket (Top 3)
    w_basket = backtest_concentration(prices, top_n=3)
    r_basket, s_basket, tot_basket = compute_stats(prices, w_basket)
    print(f"BASKET (Top 3): CAGR {r_basket:.1%} | Sharpe {s_basket:.2f} | Total {tot_basket:.0%}")
    
    # 2. Sniper (Top 1)
    w_sniper = backtest_concentration(prices, top_n=1)
    r_sniper, s_sniper, tot_sniper = compute_stats(prices, w_sniper)
    print(f"SNIPER (Top 1): CAGR {r_sniper:.1%} | Sharpe {s_sniper:.2f} | Total {tot_sniper:.0%}")
    
    if r_sniper > r_basket:
        print("\n>>> WINNER: SNIPER (Concentration wins)")
    else:
        print("\n>>> WINNER: BASKET (Diversification wins)")
