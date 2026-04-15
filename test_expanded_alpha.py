
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data():
    # The Expanded Universe
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        'ADA-USD', 'XRP-USD', 'AVAX-USD', 'SHIB-USD', 
        'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD'
    ]
    print(f"Fetching data for: {tickers}")
    data = yf.download(tickers, start='2020-01-01', progress=False)
    # yfinance formatting handling
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    return prices.ffill().dropna()

def backtest_basket(prices, universe_cols, top_n=3):
    # Momentum lookback
    LOOKBACK = 14
    
    # Filter universe
    avail_cols = [c for c in universe_cols if c in prices.columns]
    if 'BTC-USD' not in avail_cols:
        avail_cols.append('BTC-USD')
        
    alts = [c for c in avail_cols if c != 'BTC-USD']
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Logic:
    # If Avg Alt Momentum (of Top N) > BTC Momentum -> Go Long Top N Alts
    # Else -> Go Long BTC
    
    btc_mom = prices['BTC-USD'].pct_change(LOOKBACK)
    
    for i in range(LOOKBACK+1, len(prices)):
        # Calculate moms
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
            
        # Sort ALL alts by momentum
        sorted_alts = sorted(current_moms.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [x for x in sorted_alts[:top_n]]
        
        # Avg Momentum of the candidates vs BTC
        avg_alt_mom = np.mean([x[1] for x in top_candidates])
        btc_m = btc_mom.iloc[i]
        
        if avg_alt_mom > btc_m:
            # Altseason: Buy Top N
            targets = [x[0] for x in top_candidates]
            for t in targets:
                weights.iloc[i][t] = 1.0 / len(targets)
        else:
            # BTC Season
            weights.iloc[i]['BTC-USD'] = 1.0
            
    return weights.shift(1).fillna(0)

def compute_stats(prices, weights):
    warmup = 30
    rets = prices.pct_change().iloc[warmup:]
    w = weights.iloc[warmup:]
    
    # Align
    common = w.columns.intersection(rets.columns)
    port_ret = (w[common] * rets[common]).sum(axis=1)
    
    # Cumulative stats
    total_ret = (1 + port_ret).prod() - 1
    days = len(port_ret)
    if days == 0: return 0, 0, 0, 0, 0
    
    ann_ret = (1 + total_ret) ** (365/days) - 1
    vol = port_ret.std() * np.sqrt(365)
    sharpe = ann_ret / vol if vol > 0 else 0
    
    # Calculate Alpha vs SPY
    if 'SPY' in prices.columns:
        spy_ret = prices['SPY'].pct_change().iloc[warmup:]
        # Align dates
        common_idx = port_ret.index.intersection(spy_ret.index)
        y = port_ret.loc[common_idx]
        x = spy_ret.loc[common_idx]
        
        if len(y) > 30:
            import statsmodels.api as sm
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            alpha = model.params.iloc[0] * 365 # Annualized
            beta = model.params.iloc[1]
        else:
            alpha = 0
            beta = 0
    else:
        alpha = 0
        beta = 0
    
    return ann_ret, sharpe, total_ret, alpha, beta

if __name__ == "__main__":
    # Fetch Data more robustly
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 
        'ADA-USD', 'XRP-USD', 'AVAX-USD', 'SHIB-USD', 
        'DOT-USD', 'MATIC-USD', 'LINK-USD', 'UNI-USD', 'LTC-USD',
        'SPY' # Benchmark
    ]
    data = yf.download(tickers, start='2020-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to get Adj Close first
            if 'Adj Close' in data.columns.levels[0]:
                prices = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                # Fallback: flatten and grep
                prices = data.xs('Adj Close', axis=1, level=1)
        except:
             # Last resort: just take Close
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # Clean
    prices = prices.ffill().dropna(subset=['BTC-USD', 'SPY']) # Require at least BTC and SPY
    
    print("\n--- RESULTS (2020-Now) ---")
    
    # 1. Standard Trio
    print("Testing Standard Trio: [ETH, SOL, DOGE]")
    univ_std = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD']
    w_std = backtest_basket(prices, univ_std, top_n=3)
    r_std, s_std, tot_std, a_std, b_std = compute_stats(prices, w_std)
    
    # 2. Expanded Universe
    print("Testing Expanded Universe: [ETH, SOL, DOGE, ADA, XRP, AVAX, SHIB, DOT...]")
    # Exclude SPY from the crypto universe
    univ_exp = [c for c in prices.columns if c != 'SPY']
    w_exp = backtest_basket(prices, univ_exp, top_n=3)
    r_exp, s_exp, tot_exp, a_exp, b_exp = compute_stats(prices, w_exp)
    
    print("-" * 80)
    print(f"{'Universe':<20} {'CAGR':>10} {'Sharpe':>8} {'Alpha':>8} {'Beta':>8} {'Total':>10}")
    print("-" * 80)
    print(f"{'Standard Trio':<20} {r_std:>9.1%} {s_std:>8.2f} {a_std:>8.1%} {b_std:>8.2f} {tot_std:>9.0%}")
    print(f"{'Expanded Univ':<20} {r_exp:>9.1%} {s_exp:>8.2f} {a_exp:>8.1%} {b_exp:>8.2f} {tot_exp:>9.0%}")
    print("-" * 80)
