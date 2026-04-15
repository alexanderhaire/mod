"""
Math Alpha Hunt (Hierarchical Risk Parity)
===========================================

Testing "Allocation Alpha". Can HRP outperform standard portfolios by 
clustering correlated assets and diversifying across risk sources?

Universe: 16 Global Assets (Equities, Bonds, Commods, Real Estate, Crypto).
Strategy: Rebalance Monthly using HRP weights derived from trailing 252d covariance.

RUN: python math_alpha_hunt.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. HRP ALGORITHM (Lopez de Prado)
# =============================================================================

def get_cov_matrix(returns):
    return returns.cov()

def get_corr_matrix(returns):
    return returns.corr()

def get_quasi_diag_indices(link):
    # Realign the covariance matrix rows/cols based on clustering hierarchy
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])  # re-index
    return sort_ix.tolist()

def get_cluster_var(cov, c_items):
    # Variance of a cluster
    cov_slice = cov.iloc[c_items, c_items]
    w = get_ivar_weights(cov_slice).values.reshape(-1, 1) # Inverse Variance weights within cluster
    c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return c_var

def get_ivar_weights(cov):
    # Inverse Variance weights (Risk Parity naive)
    iv = 1.0 / np.diag(cov)
    w = iv / iv.sum()
    return pd.Series(w, index=cov.index)

def get_rec_bipart(cov, sort_ix):
    # Recursive Bisection
    # Allocate weight 1.0 to top cluster. Then split in half.
    # Allocation to Left = Var(Right) / (Var(Left) + Var(Right)). 
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i] # Cluster 1
            c_items1 = c_items[i + 1] # Cluster 2
            
            # Map indices back to original col names if needed, but here sort_ix are integer locs based on cov
            # We need to map integer locs to cov labels
            # cov is dataframe. sort_ix are integers indexing cov.index/cols
            
            # Get variances
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
            
    return w

def run_hrp(returns):
    """
    Core HRP execution.
    """
    corr = returns.corr().fillna(0)
    cov = returns.cov().fillna(0)
    
    # 1. Dist Matrix
    dist = np.sqrt((1 - corr) / 2)
    dist = dist.fillna(0)
    
    # 2. Clustering (Linkage)
    try:
        # squareform checks for symmetry. dist matrix from corr is symmetric.
        # But pandas fillna might break perfect symmetry floating point?
        # Ensure symmetry
        dist_vals = dist.values
        np.fill_diagonal(dist_vals, 0)
        dist_vals = (dist_vals + dist_vals.T) / 2
        
        link = linkage(squareform(dist_vals), method='single')
        
    except Exception as e:
        # Fallback if clustering fails (e.g. singular matrix)
        # Return Equal Weight
        return pd.Series(1.0/len(returns.columns), index=returns.columns)
        
    # 3. Sort Indices
    sort_ix = get_quasi_diag_indices(link)
    
    # Reorder cov
    sort_ix_labels = returns.columns[sort_ix]
    cov_sorted = cov.loc[sort_ix_labels, sort_ix_labels]
    
    # 4. Allocation (Recursive Bisection)
    # We pass range(len) as indices for recursive function, which map to cov_sorted
    weights_nums = get_rec_bipart(cov_sorted, list(range(len(cov_sorted))))
    
    # 5. Map back to labels
    weights = pd.Series(weights_nums.values, index=sort_ix_labels)
    
    return weights.sort_index()

# =============================================================================
# 2. DATA
# =============================================================================

def fetch_multi_asset_data():
    print("🌍 Fetching Global Multi-Asset Universe...")
    tickers = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'FXI', # Equities
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG', # Bonds
        'GLD', 'USO', 'DBA', 'VNQ', # Real Assets
        'BTC-USD' # Crypto
    ]
    
    data = yf.download(tickers, start='2018-01-01', progress=False)
    
    # Clean Prices
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    # FIX: Align to SPY trading days to avoid Weekend artifacts affecting Stocks
    # If we have BTC (365 days) and SPY (252 days), combining them makes SPY have NaNs on weekends.
    # If we then dropCols with thresh=0.9, SPY gets dropped!
    # Soludion: Reindex to SPY valid days.
    
    if 'SPY' in prices.columns:
        spy_days = prices['SPY'].dropna().index
        prices = prices.ffill() # Fill crypto gaps if any
        prices = prices.reindex(spy_days) # Keep only trading days
    
    # Now drop columns that are truly broken
    prices = prices.dropna(axis=1, thresh=len(prices)*0.9)
    # Fill remaining small gaps
    prices = prices.ffill().dropna()
    
    print(f"   Assets: {list(prices.columns)}")
    print(f"   History: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices

# =============================================================================
# 3. BACKTEST (Monthly Rebal)
# =============================================================================

def backtest_strategies(prices):
    print("⚖️ Running Backtest (Monthly Rebalancing)...")
    
    returns = prices.pct_change().dropna()
    
    # Monthly Rebalance Dates
    monthly_dates = returns.resample('M').last().index
    
    # Weights DataFrames
    w_hrp = pd.DataFrame(index=monthly_dates, columns=prices.columns)
    w_rp = pd.DataFrame(index=monthly_dates, columns=prices.columns)
    w_ew = pd.DataFrame(index=monthly_dates, columns=prices.columns)
    
    # Lookback window for Covariance
    lookback = 252 // 2 # 6 months is responsive, 1 year standard. Let's use 6m.
    
    print(f"   Rebalancing {len(monthly_dates)} times...")
    
    for t in monthly_dates:
        # Get history up to t (exclusive of t return? No, t is decision date)
        # Use data leading up to t
        hist_ret = returns[returns.index <= t].tail(lookback)
        
        if len(hist_ret) < 60: continue # Need minimum data
        
        # 1. HRP
        try:
             w_hrp.loc[t] = run_hrp(hist_ret)
        except:
             w_hrp.loc[t] = 1.0/len(prices.columns) # Fallback
             
        # 2. Risk Parity (Inverse Vol)
        vol = hist_ret.std()
        inv_vol = 1.0 / vol
        w_rp.loc[t] = inv_vol / inv_vol.sum()
        
        # 3. Equal Weight
        w_ew.loc[t] = 1.0 / len(prices.columns)
        
    # Forward Fill Weights to daily (hold constant)
    w_hrp = w_hrp.reindex(returns.index).ffill().dropna()
    w_rp = w_rp.reindex(returns.index).ffill().dropna()
    w_ew = w_ew.reindex(returns.index).ffill().dropna()
    
    # Calculate Portfolio Returns
    # Return at T is Weight(T-1) * Ret(T)
    # Align
    def calc_port_ret(w, r):
        common = w.index.intersection(r.index)
        return (w.loc[common].shift(1) * r.loc[common]).sum(axis=1)
        
    p_hrp = calc_port_ret(w_hrp, returns)
    p_rp = calc_port_ret(w_rp, returns)
    p_ew = calc_port_ret(w_ew, returns)
    
    # Benchmark: 60/40 (SPY/TLT)
    if 'SPY' in prices.columns and 'TLT' in prices.columns:
        p_6040 = returns['SPY'] * 0.6 + returns['TLT'] * 0.4
    else:
        p_6040 = returns.mean(axis=1) # Fallback to EW?
        
    # Metrics
    def get_stats(r):
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
        dd = (r.cumsum() - r.cumsum().cummax()).min() # Simple log approx or standard
        cum = (1+r).cumprod().iloc[-1] - 1
        vol = r.std() * np.sqrt(252)
        return sharpe, cum, vol
        
    s_hrp, c_hrp, v_hrp = get_stats(p_hrp)
    s_rp, c_rp, v_rp = get_stats(p_rp)
    s_ew, c_ew, v_ew = get_stats(p_ew)
    s_6040, c_6040, v_6040 = get_stats(p_6040)
    
    print("\n📊 BATTLE ROYALE RESULTS (2018-2026):")
    print(f"{'Strategy':<20} {'Sharpe':<8} {'Return':<8} {'Vol':<8}")
    print("-" * 50)
    print(f"{'HRP (ML Math)':<20} {s_hrp:<8.2f} {c_hrp:<8.2%} {v_hrp:<8.2%}")
    print(f"{'Risk Parity':<20} {s_rp:<8.2f} {c_rp:<8.2%} {v_rp:<8.2%}")
    print(f"{'Equal Weight':<20} {s_ew:<8.2f} {c_ew:<8.2%} {v_ew:<8.2%}")
    print(f"{'60/40 Benchmark':<20} {s_6040:<8.2f} {c_6040:<8.2%} {v_6040:<8.2%}")
    
    if s_hrp > s_6040 + 0.1:
        print("\n✅ WINNER: HRP creates real math alpha!")
    else:
        print("\n❌ NO EDGE: Math didn't beat simple exposure.")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("📐 MATH ALPHA HUNT: HIERARCHICAL RISK PARITY")
    print("="*60)
    
    prices = fetch_multi_asset_data()
    backtest_strategies(prices)
    
    print("\n" + "=" * 60)
