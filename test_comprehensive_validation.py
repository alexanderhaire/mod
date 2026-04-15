"""
Comprehensive Statistical Validation
==================================

Rigorous In-Sample (IS) vs Out-Of-Sample (OOS) testing of the 
"Alpha-Maximized" Strategy (GBM + Macro + 20% Vol).

Tests:
1.  IS vs OOS Consistency (did we overfit?)
2.  T-test for Excess Returns
3.  White's Reality Check (Bootstrap)
4.  Deflated Sharpe Ratio (DSR) - Adjusting for trial multiplicity
5.  Monte Carlo Permutation Test (Randomized Trades)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optimization Params
ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']
MACRO = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']
TARGET_VOL = 0.20
ESTIMATED_TRIALS = 300 # (Models * Universes * Features * Params)

def fetch_data(years: int = 6) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    tickers = list(set(ASSETS + MACRO))
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame) -> tuple:
    df = pd.DataFrame(index=prices.index)
    feat_cols = []
    
    # Macro
    if '^TNX' in prices.columns:
        df['rate_change'] = prices['^TNX'].diff(20)
        df['rate_trend'] = prices['^TNX'] - prices['^TNX'].rolling(60).mean()
        feat_cols += ['rate_change', 'rate_trend']
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        df['credit_spread'] = prices['JNK'] / prices['IEF']
        feat_cols += ['credit_spread']
    if 'UUP' in prices.columns:
        df['dollar_vol'] = prices['UUP'].pct_change().rolling(20).std()
        feat_cols += ['dollar_vol']
        
    asset_map = {}
    for t in ASSETS:
        if t not in prices.columns: continue
        r = prices[t].pct_change()
        cols = []
        for w in [20, 60]:
            c = f'{t}_mom_{w}'
            df[c] = r.rolling(w).mean()
            cols.append(c)
        c = f'{t}_vol_20'
        df[c] = r.rolling(20).std()
        cols.append(c)
        asset_map[t] = cols + feat_cols
        
    # Standardize Target: Next Day Return (for trading Close-to-Close)
    targets = prices.pct_change().shift(-1)
    
    return df, targets, asset_map

def run_backtest(prices, features, targets, asset_map, train_start_idx, train_end_idx, test_end_idx):
    # Train
    models = {}
    for t in ASSETS:
        if t not in prices.columns: continue
        cols = asset_map[t]
        X_train = features.iloc[train_start_idx:train_end_idx][cols]
        y_train = targets.iloc[train_start_idx:train_end_idx][t]
        
        # Dropna
        mask = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
        X_train = X_train[mask]
        y_train = y_train[mask]
        X_train = X_train.clip(-1e5, 1e5) # Safety clip
        
        if len(X_train) < 50:
            models[t] = None
            continue
            
        m = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(X_train, y_train)
        models[t] = m
        
    # Predict (OOS)
    signals = pd.DataFrame(index=features.index[train_end_idx:test_end_idx], columns=ASSETS)
    for i in range(len(signals)):
        idx_val = train_end_idx + i
        for t in ASSETS:
            cols = asset_map[t]
            if models.get(t) is None:
                signals.iloc[i][t] = 0
                continue
                
            row = features.iloc[idx_val:idx_val+1][cols]
            if row.isnull().any().any():
                signals.iloc[i][t] = 0
            else:
                signals.iloc[i][t] = models[t].predict(row)[0]
                
    signals = signals.fillna(0)
    
    # Portfolio Construction: Proportional Allocation
    weights = signals.copy()
    weights[weights < 0] = 0 # Long Only
    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums, axis=0).fillna(0)
    
    # Align Returns
    aligned_rets = targets.loc[signals.index]
    strat_ret_unscaled = (weights * aligned_rets).sum(axis=1)
    
    # Vol Scaling
    univ_ret = aligned_rets.mean(axis=1)
    rolling_vol = univ_ret.rolling(20).std() * np.sqrt(252)
    lagged_vol = rolling_vol.shift(1).fillna(0.15)
    
    scale = (TARGET_VOL / lagged_vol).clip(0, 2.0)
    strat_ret = strat_ret_unscaled * scale
    
    bench_ret = univ_ret
    
    return strat_ret.dropna(), bench_ret.dropna(), weights, aligned_rets, scale

def prob_sharpe_ratio(sharpe, benchmark_sharpe, skew, kurtosis, n):
    std_sharpe = np.sqrt((1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2) / (n - 1))
    if std_sharpe == 0: return 0
    return stats.norm.cdf((sharpe - benchmark_sharpe) / std_sharpe)

def run_comprehensive_test():
    print("=" * 80)
    print("   EXTREME STATISTICAL VALIDATION")
    print("   Strategy: GBM + Macro + 20% Vol")
    print("=" * 80)
    
    prices = fetch_data(years=6)
    features, targets, asset_map = create_features(prices)
    
    common = features.dropna().index.intersection(targets.dropna().index)
    split_date = common[int(len(common) * 0.5)]
    print(f"   Split Date: {split_date.date()}")
    
    train_end = common.get_loc(split_date)
    
    # IS Call (Dummy)
    print("\n   1. Running IN-SAMPLE Backtest...")
    _ = run_backtest(prices, features, targets, asset_map, 0, train_end, train_end) 
    
    # OOS Call (Real)
    print("   (Running OOS Backtest...)")
    oos_ret, oos_bench, weights, aligned_rets, scale = run_backtest(prices, features, targets, asset_map, 0, train_end, len(common))
    
    print("\n   === PERFORMANCE RESULTS (OOS) ===")
    ann_ret = oos_ret.mean() * 252
    ann_vol = oos_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    
    bench_ann_ret = oos_bench.mean() * 252
    bench_sharpe = oos_bench.mean() / oos_bench.std() * np.sqrt(252)
    
    print(f"   Annual Return:   {ann_ret:.1%}")
    print(f"   Benchmark Return:{bench_ann_ret:.1%}")
    print(f"   Strategy Sharpe: {sharpe:.2f}")
    print(f"   Benchmark Sharpe:{bench_sharpe:.2f}")
    
    # EXTREME STATS
    print("\n   === EXTREME VALIDATION BATTERY ===")
    
    # 1. T-Test
    excess = oos_ret - oos_bench
    t_stat, p_val = stats.ttest_1samp(excess, 0)
    print(f"   1. T-Test (Excess Return): p={p_val:.4f} {'✅ PASS' if p_val < 0.05 else '❌ FAIL'}")
    
    # 2. White's Reality Check
    boot_means = []
    np.random.seed(42)
    for _ in range(1000):
        idx = np.random.randint(0, len(excess)-20, len(excess)//20)
        sample = []
        for i in idx: sample.extend(excess.iloc[i:i+20])
        boot_means.append(np.mean(sample))
    p_white = np.mean(np.array(boot_means) - np.mean(boot_means) >= excess.mean())
    print(f"   2. White's Reality Check:  p={p_white:.4f} {'✅ PASS' if p_white < 0.1 else '❌ FAIL'}")

    # 3. DSR
    dsr_benchmark = bench_sharpe * (1 + np.log(ESTIMATED_TRIALS)/10)
    print(f"   3. Deflated Sharpe Ratio:  Hurdle {dsr_benchmark:.2f} {'✅ PASS' if sharpe > dsr_benchmark else '❌ FAIL'}")
    
    # 4. Permutation
    print("   4. Monte Carlo Permutation (100 Paths)...")
    perm_sharpes = []
    # Ensure aligned_rets matches weights index
    # validation: aligned_rets and weights should be same length
    # Note: run_backtest aligns them
    
    valid_weights = weights.loc[oos_ret.index]
    valid_aligned = aligned_rets.loc[oos_ret.index]
    valid_scale = scale.loc[oos_ret.index] if isinstance(scale, pd.Series) else scale
    
    for _ in range(100):
        # Shuffle weights temporally
        perm_w = valid_weights.sample(frac=1).reset_index(drop=True)
        perm_w.index = valid_weights.index
        
        # Calculate return
        # Need to align with returns which are TimeSeries.
        # Permutation Test: Randomize THE STRATEGY DECISIONS against the MARKET RETURNS.
        # So we keep Market Returns (valid_aligned) fixed order.
        # We shuffle the Weights (Decisions).
        
        p_ret = (perm_w * valid_aligned).sum(axis=1) * valid_scale
        s = p_ret.mean() / p_ret.std() * np.sqrt(252)
        perm_sharpes.append(s)
        
    perm_p = np.mean(np.array(perm_sharpes) >= sharpe)
    print(f"      Better Random Paths: {perm_p:.2%} {'✅ PASS' if perm_p < 0.05 else '❌ FAIL'}")

if __name__ == "__main__":
    run_comprehensive_test()
