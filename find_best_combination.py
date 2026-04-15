
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
        mean = p_ret.mean() * 252
        vol = p_ret.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        return -sharpe

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(rets.shape[1]))
    init_guess = [1.0 / rets.shape[1]] * rets.shape[1]
    
    if len(rets) < 20:
        return init_guess
        
    try:
        opt = minimize(neg_sharpe, init_guess, args=(rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return init_guess

def calc_stats(r):
    cum = (1 + r).prod() - 1
    vol = r.std() * np.sqrt(252)
    sharpe = r.mean() * 252 / vol if vol > 0 else 0
    return cum, sharpe

def run_battle_royale():
    print("=" * 60)
    print("   GLOBAL PORTFOLIO BATTLE ROYALE")
    print("   Testing ALL combinations to find Maximum Sharpe")
    print("=" * 60)
    
    print("1. Fetching Data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', 
        'VUG', 'VTV', 'RSP',
        '^FVX', '^TYX', '^VIX', 
        'BTC-USD'
    ]
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    prices = prices.ffill().dropna()
    rets = prices.pct_change().fillna(0)
    
    # --- STRATEGY GENERATION ---
    print("2. Generating Strategies...")
    
    # A. Ultimate (Original)
    vix = prices['^VIX']
    vix_ma = vix.rolling(20).mean()
    signal = pd.Series(0, index=prices.index)
    signal[vix < vix_ma] = 1 
    signal[vix > vix_ma] = -1 
    
    w_ult = pd.DataFrame(0.0, index=prices.index, columns=['SPY', 'TLT', 'BTC-USD'])
    mask_bull = (signal > 0)
    mask_bear = (signal < 0)
    w_ult.loc[mask_bull, 'SPY'] = 0.45; w_ult.loc[mask_bull, 'TLT'] = 0.10
    w_ult.loc[mask_bear, 'SPY'] = 0.15; w_ult.loc[mask_bear, 'TLT'] = 0.35
    w_ult.loc[(~mask_bull) & (~mask_bear), 'SPY'] = 0.30; w_ult.loc[(~mask_bull) & (~mask_bear), 'TLT'] = 0.22
    w_ult['BTC-USD'] = 0.20
    
    sp = rets['SPY'] if 'SPY' in rets.columns else 0
    tl = rets['TLT'] if 'TLT' in rets.columns else 0
    bt = rets['BTC-USD'] if 'BTC-USD' in rets.columns else 0
    
    r_ult = (w_ult['SPY'].shift(1)*sp + w_ult['TLT'].shift(1)*tl + w_ult['BTC-USD'].shift(1)*bt)
    
    # B. HRP
    cols_hrp = [c for c in ['SPY', 'TLT', 'GLD', 'IEF'] if c in rets.columns]
    vol = rets[cols_hrp].rolling(126).std()
    inv_vol = 1 / vol.replace(0, 0.01)
    w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
    r_hrp = (w_hrp * rets[cols_hrp]).sum(axis=1)
    
    # C. Dollar Trend
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        uup_fast = uup.rolling(50).mean()
        uup_slow = uup.rolling(200).mean()
        uup_sig = (uup_fast > uup_slow).astype(float)
        r_uup = (uup_sig.shift(1) * rets['UUP'])
    else:
        r_uup = pd.Series(0, index=prices.index)
        
    # D. SPY Hold
    r_spy = rets['SPY']
    
    # E. 60/40
    r_6040 = 0.6 * rets['SPY'] + 0.4 * rets['TLT']
    
    strategies = {
        'Ultimate': r_ult,
        'HRP': r_hrp,
        'Dollar Trend': r_uup,
        'SPY': r_spy,
        '60/40': r_6040
    }
    
    strat_df = pd.DataFrame(strategies).dropna()
    
    # --- BATTLE ROYALE ---
    print("3. Testing Combinations...")
    print("-" * 60)
    
    results = []
    
    all_names = list(strategies.keys())
    
    # Test all subsets of size 1 to N
    for r in range(1, len(all_names) + 1):
        for combo in combinations(all_names, r):
            combo_rets = strat_df[list(combo)]
            
            # Optimize weights for this combo
            if len(combo) > 1:
                w = optimize_portfolio(combo_rets)
                combo_r = (combo_rets * w).sum(axis=1)
                weights_str = ", ".join([f"{n}: {W:.0%}" for n, W in zip(combo, w)])
            else:
                combo_r = combo_rets.iloc[:, 0]
                weights_str = "100%"
                
            cum, sharpe = calc_stats(combo_r)
            
            results.append({
                'Name': " + ".join(combo),
                'Sharpe': sharpe,
                'Return': cum,
                'Weights': weights_str
            })
            
    # Sort
    results.sort(key=lambda x: x['Sharpe'], reverse=True)
    
    # Limit output
    top_n = 10
    print(f"\nTOP {top_n} COMBINATIONS (Sorted by Sharpe)")
    print(f"{'Rank':<4} {'Sharpe':<6} {'Return':<8} {'Strategy Combo'}")
    print("-" * 60)
    
    for i, res in enumerate(results[:top_n]):
        print(f"{i+1:<4} {res['Sharpe']:<6.2f} {res['Return']:<8.1%} {res['Name']}")
        
    print("\n" + "=" * 60)
    print("WINNER DETAILS:")
    print(f"Strategy: {results[0]['Name']}")
    print(f"Sharpe:   {results[0]['Sharpe']:.3f} 🏆")
    print(f"Weights:  {results[0]['Weights']}")
    print("=" * 60)

if __name__ == "__main__":
    run_battle_royale()
