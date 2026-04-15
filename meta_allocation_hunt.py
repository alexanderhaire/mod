"""
Meta-Allocation Alpha Hunt
==========================

Combining the two validated Alpha Sources:
1. Ultimate Strategy (Timing Alpha) - High Vol, High Return.
2. HRP Strategy (Allocation Alpha) - Low Vol, Steady Return.

Goal: Optimize the mix. 
Hypothesis: Low Correlation + Risk Parity = High Sharpe.

RUN: python meta_allocation_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA FETCHING (SUPERSET)
# =============================================================================

def fetch_meta_data():
    print("🌍 Fetching Superset Data (Traditional + Crypto + VIX)...")
    
    # HRP Universe
    hrp_tickers = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'FXI', # Equities
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', # Bonds
        'GLD', 'USO', 'DBA', 'VNQ', # Real Assets
        'BTC-USD' # Crypto
    ]
    
    # Ultimate Universe (Adds specific Alts and VIX)
    ultimate_tickers = [
        'ETH-USD', 'SOL-USD', 'ADA-USD', 'AVAX-USD', 'DOGE-USD',
        '^VIX', '^VIX3M'
    ]
    
    tickers = list(set(hrp_tickers + ultimate_tickers))
    
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    # Extract VIX separately
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
    vix3m = prices['^VIX3M'].copy() if '^VIX3M' in prices.columns else None
    
    # Drop VIX from main price df
    cols_to_drop = [c for c in ['^VIX', '^VIX3M'] if c in prices.columns]
    prices = prices.drop(columns=cols_to_drop)
    
    # Cleaning
    prices = prices.ffill()
    
    # Align to SPY trading days (Primary Market)
    if 'SPY' in prices.columns:
        spy_days = prices['SPY'].dropna().index
        prices = prices.reindex(spy_days)
    
    prices = prices.ffill().dropna()
    print(f"   Data: {len(prices)} days | {len(prices.columns)} Assets")
    
    if vix is not None and len(vix) > 0:
        vix = vix.reindex(prices.index).ffill()
    if vix3m is not None and len(vix3m) > 0:
        vix3m = vix3m.reindex(prices.index).ffill()
        
    return prices, vix, vix3m

# =============================================================================
# 2. STRATEGY ENGINE 1: HRP (Allocation Alpha)
# =============================================================================

def run_hrp_engine(prices):
    print("   🛡️ Running HRP Engine...")
    # Subset to HRP Universe
    hrp_cols = [c for c in prices.columns if c in [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'FXI',
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
        'GLD', 'USO', 'DBA', 'VNQ', 'BTC-USD'
    ]]
    p_hrp = prices[hrp_cols]
    returns = p_hrp.pct_change().dropna()
    
    monthly_dates = returns.resample('M').last().index
    weights = pd.DataFrame(index=monthly_dates, columns=p_hrp.columns)
    
    lookback = 126 # 6 months
    
    for t in monthly_dates:
        hist_ret = returns[returns.index <= t].tail(lookback)
        if len(hist_ret) < 60: continue
        
        try:
            # HRP Logic Inline
            corr = hist_ret.corr().fillna(0)
            cov = hist_ret.cov().fillna(0)
            dist = np.sqrt((1 - corr) / 2).fillna(0)
            dist_vals = dist.values
            np.fill_diagonal(dist_vals, 0)
            dist_vals = squareform((dist_vals + dist_vals.T)/2)
            link = linkage(dist_vals, method='single')
            
            # Quasi Diag Sort
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0])
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            sort_ix = sort_ix.tolist()
            
            # Rec Bisection
            w_series = pd.Series(1.0, index=sort_ix)
            c_items = [sort_ix]
            while len(c_items) > 0:
                c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                for i in range(0, len(c_items), 2):
                    c0 = c_items[i]
                    c1 = c_items[i+1]
                    # Var
                    def get_var(idx_list):
                        sub_cov = cov.iloc[idx_list, idx_list]
                        iv = 1.0 / np.diag(sub_cov)
                        w_iv = iv / iv.sum()
                        return np.dot(np.dot(w_iv.T, sub_cov), w_iv)
                    v0 = get_var(c0)
                    v1 = get_var(c1)
                    alpha = 1 - v0 / (v0 + v1)
                    w_series[c0] *= alpha
                    w_series[c1] *= 1 - alpha

            final_w = pd.Series(w_series.values, index=p_hrp.columns[sort_ix]).sort_index()
            weights.loc[t] = final_w
            
        except:
             weights.loc[t] = 1.0/len(p_hrp.columns)
             
    weights = weights.reindex(returns.index).ffill().dropna()
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    return port_ret

# =============================================================================
# 3. STRATEGY ENGINE 2: ULTIMATE (Timing Alpha)
# =============================================================================

def run_ultimate_engine(prices, vix, vix3m):
    print("   🎯 Running Ultimate Strategy Engine...")
    
    # Signals
    # VIX Term Structure
    if vix is None or vix3m is None:
        vix_sig = pd.Series(0, index=prices.index)
    else:
        ratio = (vix / vix3m).rolling(5).mean()
        vix_sig = pd.Series(0, index=prices.index)
        vix_sig[ratio < 0.90] = 1 # Bullish
        vix_sig[ratio > 1.05] = -1 # Bearish
    
    # Crypto Signals
    alts = [c for c in prices.columns if c.endswith('-USD') and c != 'BTC-USD']
    if 'BTC-USD' in prices.columns:
        btc = prices['BTC-USD']
        btc_mom = btc.pct_change(14)
        
        alt_sig = pd.Series(0, index=prices.index)
        
        for i in range(20, len(prices)):
            # Altseason logic simplified for speed
            if i % 1 == 0: # Check daily
                btc_m = btc_mom.iloc[i]
                # Alt mom
                alt_moms = []
                for a in alts:
                     if a in prices.columns:
                         r = prices[a].iloc[i]/prices[a].iloc[i-14]-1
                         alt_moms.append(r)
                if len(alt_moms)>0 and np.mean(alt_moms) > btc_m * 1.2:
                    alt_sig.iloc[i] = 1
                else:
                    alt_sig.iloc[i] = -1
    else:
        alt_sig = pd.Series(0, index=prices.index)
        
    # Allocation Logic
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(1, len(prices)):
        # Trad Sleeve (60%)
        vs = vix_sig.iloc[i]
        if vs > 0:
             w_t = {'SPY': 0.45, 'TLT': 0.10, 'GLD': 0.05}
        elif vs < 0:
             w_t = {'SPY': 0.15, 'TLT': 0.35, 'GLD': 0.10}
        else:
             w_t = {'SPY': 0.30, 'TLT': 0.22, 'GLD': 0.08}
             
        # Crypto Sleeve (40%)
        asig = alt_sig.iloc[i]
        if asig > 0: # Altseason
             w_c = {'alts': 0.30, 'BTC-USD': 0.10}
        else:
             w_c = {'BTC-USD': 0.20} # Neutral/Bearish (Simpler for simulation)
             
        # Assign
        for t, w in w_t.items():
            if t in prices.columns: weights.iloc[i][t] = w
            
        for t, w in w_c.items():
            if t == 'alts':
                for a in alts: 
                    if a in prices.columns: weights.iloc[i][a] = w / len(alts)
            elif t in prices.columns:
                weights.iloc[i][t] = w
                
    port_ret = (weights.shift(1) * prices.pct_change()).sum(axis=1)
    return port_ret

# =============================================================================
# 4. META ALLOCATION
# =============================================================================

def meta_optimize(r_hrp, r_ult, r_spy):
    print("\n⚔️ META-ALLOCATION LAB")
    
    # Align
    common = r_hrp.index.intersection(r_ult.index).intersection(r_spy.index)
    r_hrp = r_hrp.loc[common]
    r_ult = r_ult.loc[common]
    r_spy = r_spy.loc[common]
    
    # 1. Correlation and Stats
    corr = r_hrp.corr(r_ult)
    print(f"   Correlation (HRP vs Ultimate): {corr:.2f}")
    if corr < 0.4:
        print("   ✅ LOW CORRELATION! Excellent diversification benefit.")
    
    vol_hrp = r_hrp.std() * np.sqrt(252)
    vol_ult = r_ult.std() * np.sqrt(252)
    print(f"   Vol HRP: {vol_hrp:.1%} | Vol Ultimate: {vol_ult:.1%}")
    
    # 2. Strategies
    
    # A. Equal Weight
    r_ew = 0.5 * r_hrp + 0.5 * r_ult
    
    # B. Risk Parity (Simple Inverse Vol)
    w_h = (1/vol_hrp) / ((1/vol_hrp) + (1/vol_ult))
    w_u = 1 - w_h
    r_rp = w_h * r_hrp + w_u * r_ult
    
    # C. Leveraged RP (Target 15% Vol)
    # Vol of RP portfolio
    vol_rp = r_rp.std() * np.sqrt(252)
    leverage = 0.15 / vol_rp
    r_lev = r_rp * leverage
    cost_of_lev = 0.05 / 252 # Assume 5% cost of borrowing
    r_lev = r_lev - (leverage - 1) * cost_of_lev
    
    # Metrics
    def get_stats(r, name):
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (1+r).cumprod().iloc[-1] - 1
        print(f"   {name:<25} | Ann {ann:.1%} | Vol {vol:.1%} | Sharpe {sharpe:.2f}")
        return sharpe
    
    print("-" * 75)
    get_stats(r_spy, "SPY (Benchmark)")
    get_stats(r_ult, "Ultimate (Timing)")
    get_stats(r_hrp, "HRP (Allocation)")
    print("-" * 75)
    get_stats(r_ew, "Meta: 50/50 Split")
    print(f"   Meta: Risk Parity Split   | Weights: {w_h:.0%} HRP / {w_u:.0%} Ult")
    s_rp = get_stats(r_rp, "Meta: Risk Parity")
    s_lev = get_stats(r_lev, "Meta: Lev RP (15% Vol)")
    
    print("-" * 75)
    if s_lev > 1.5:
        print("🚀 HOLY GRAIL STATUS: Sharpe > 1.5 achieved via Diversification.")
    elif s_rp > 1.2:
        print("✅ SYNERGY CONFIRMED: Combination beats individual components.")
    else:
        print("⚠️ NO SYNERGY: Diversification failed to improve risk-adjusted return.")

if __name__ == "__main__":
    prices, vix, vix3m = fetch_meta_data()
    r_hrp = run_hrp_engine(prices)
    r_ult = run_ultimate_engine(prices, vix, vix3m)
    
    # Get SPY for bench
    if 'SPY' in prices.columns:
        r_spy = prices['SPY'].pct_change().dropna()
    else:
        r_spy = r_hrp # fallback
        
    meta_optimize(r_hrp, r_ult, r_spy)
