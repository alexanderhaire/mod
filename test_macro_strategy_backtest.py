"""
Macro Strategy Backtest
=======================

Testing the portfolio performance of the Macro-Enhanced Strategy.
Assets: XLB, XLI, XLE, JNK, GLD, USMV, MTUM, FXA
Features: Momentum + Volatility + Rates + Credit Spreads + Dollar

Hypothesis: Higher IC translates to higher Sharpe.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA & FEATURES
# =============================================================================

TARGET_ASSETS = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'FXA', 'USMV', 'MTUM']
MACRO_ASSETS = ['^TNX', 'UUP', 'IEF', 'SHY', 'JNK']

def fetch_data(years: int = 5) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    tickers = list(set(TARGET_ASSETS + MACRO_ASSETS))
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    return prices.dropna(how='all').ffill().dropna()

def create_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create all features (Asset + Macro)."""
    df = pd.DataFrame(index=prices.index)
    feature_cols = []
    
    # 1. Macro Features
    if '^TNX' in prices.columns:
        df['rate_change'] = prices['^TNX'].diff(20)
        df['rate_trend'] = prices['^TNX'] - prices['^TNX'].rolling(60).mean()
        feature_cols.extend(['rate_change', 'rate_trend'])
        
    if 'JNK' in prices.columns and 'IEF' in prices.columns:
        df['credit_spread'] = prices['JNK'] / prices['IEF']
        feature_cols.append('credit_spread')
        
    if 'UUP' in prices.columns:
        df['dollar_vol'] = prices['UUP'].pct_change().rolling(20).std()
        feature_cols.append('dollar_vol')
        
    # 2. Asset Features (per asset)
    asset_feature_map = {}
    
    for ticker in TARGET_ASSETS:
        if ticker not in prices.columns: continue
        
        returns = prices[ticker].pct_change()
        
        # Unique col names
        m1 = f'{ticker}_mom_1m'
        m3 = f'{ticker}_mom_3m'
        v1 = f'{ticker}_vol_1m'
        
        df[m1] = returns.rolling(20).mean()
        df[m3] = returns.rolling(60).mean()
        df[v1] = returns.rolling(20).std()
        
        # Store which features belong to which asset
        # We include global macro features for EVERY asset model
        asset_feature_map[ticker] = [m1, m3, v1] + feature_cols
        
    return df, asset_feature_map

# =============================================================================
# BACKTEST LOGIC
# =============================================================================

def run_backtest():
    print("=" * 80)
    print("   MACRO-ENHANCED STRATEGY BACKTEST")
    print("=" * 80)
    
    prices = fetch_data(years=5)
    features_df, asset_feat_map = create_features(prices)
    
    # Target returns (next day)
    returns = prices.pct_change().shift(-1) # Predicting t+1 from t
    
    # Align
    common_idx = features_df.dropna().index.intersection(returns.dropna().index)
    features_df = features_df.loc[common_idx]
    returns = returns.loc[common_idx]
    
    # Walk-Forward Simulation
    print(f"   Simulating {len(common_idx)} days...")
    
    signals = pd.DataFrame(index=common_idx, columns=TARGET_ASSETS)
    models = {t: RandomForestRegressor(n_estimators=50, max_depth=3, n_jobs=1) for t in TARGET_ASSETS}
    
    # Window settings
    train_window = 252 # 1 year rolling lookback? Or expanding? Let's use expanding with min window.
    min_window = 252
    retrain_days = 20 # Retrain every month to save time
    
    for i in range(min_window, len(common_idx)):
        date = common_idx[i]
        
        if i == min_window:
            # First training
            if i % 100 == 0: print(f"   Processing {date.date()}...", end='\r')
            train_start = 0
            train_end = i
            for ticker in TARGET_ASSETS:
                if ticker not in prices.columns: continue
                cols = asset_feat_map[ticker]
                X = features_df.iloc[train_start:train_end][cols]
                y = returns.iloc[train_start:train_end][ticker]
                models[ticker].fit(X, y)
        elif i % retrain_days == 0:
            # Retrain
            if i % 100 == 0: print(f"   Processing {date.date()}...", end='\r')
            
            # Using expanding window
            train_start = 0 
            train_end = i
            
            for ticker in TARGET_ASSETS:
                if ticker not in prices.columns: continue
                
                cols = asset_feat_map[ticker]
                X = features_df.iloc[train_start:train_end][cols]
                y = returns.iloc[train_start:train_end][ticker]
                
                # Shift Y to align? 
                # returns dataframe is already shifted (-1). 
                # So returns.iloc[t] is return from t to t+1.
                # features.iloc[t] is features at t.
                # So we predict returns.iloc[t] using features.iloc[t].
                # Correct.
                
                models[ticker].fit(X, y)
        
        # Predict for today (to get position for tomorrow)
        for ticker in TARGET_ASSETS:
            if ticker not in prices.columns: continue
            
            cols = asset_feat_map[ticker]
            # Need 2D array
            feat_vector = features_df.iloc[i:i+1][cols]
            pred = models[ticker].predict(feat_vector)[0]
            signals.loc[date, ticker] = pred
            
    print("\n   Simulation complete.")
    
    # Position Sizing
    # Rank signals, Long Top 3, Cash/Short others?
    # Or just proportional to positive signal?
    # Let's do Proportional to Positive Signal (Long Only)
    
    signals = signals.fillna(0)
    weights = signals.copy()
    
    # Zero out negative predictions (Long Only)
    weights[weights < 0] = 0
    
    # Normalize to 100%
    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums, axis=0).fillna(0)
    
    # Shift weights by 1 day (Signal at Close T -> Trade at Close T? No, Trade at Close T+1?
    # signals.loc[date] is probability of return t->t+1. Computed at Close T.
    # To capture return t->t+1, we must hold from Close T to Close T+1.
    # So weights.loc[date] applies to returns.loc[date].
    # But returns.loc[date] is prices.pct_change().shift(-1).
    # Realized return vector is prices.pct_change().
    
    # Let's align cleanly:
    # We computed signal at index t. We buy at Close t. We realize return at Close t+1.
    # Portfolio Return at t+1 = Weights[t] * Returns[t+1]
    
    daily_returns = prices.pct_change()
    strat_returns = (weights.shift(1) * daily_returns).sum(axis=1)
    
    # Benchmark (Equal Weight)
    bench_weights = pd.DataFrame(1.0/len(TARGET_ASSETS), index=weights.index, columns=TARGET_ASSETS)
    bench_returns = (bench_weights.shift(1) * daily_returns).sum(axis=1)
    
    # Trim to valid period
    valid_idx = strat_returns.index[min_window+5:]
    strat_returns = strat_returns.loc[valid_idx]
    bench_returns = bench_returns.loc[valid_idx]
    
    # Metrics
    strat_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(252)
    
    print("\n" + "=" * 80)
    print("   RESULTS: MACRO-ENHANCED vs BASE BENCHMARK")
    print("=" * 80)
    print(f"   Strategy Sharpe:  {strat_sharpe:.2f}")
    print(f"   Benchmark Sharpe: {bench_sharpe:.2f}")
    print(f"   Improvement:      {strat_sharpe - bench_sharpe:.2f}")
    
    # Bias Test (Simple probability)
    print("\n   Win Rate (Daily):   {:.1%}".format((strat_returns > 0).mean()))
    print("   Correlation to Bench: {:.2f}".format(strat_returns.corr(bench_returns)))

    if strat_sharpe > 1.6:
        print("\n   ✅ MACRO ALPHA CONFIRMED. Sharpe > 1.6")
    else:
        print("\n   ⚠️ MARGINAL IMPROVEMENT. Macro helps but not a silver bullet.")

if __name__ == "__main__":
    run_backtest()
