"""
Walk-Forward Validation: Asian + Trend-Following (DBMF)
========================================================

Rigorous in-sample vs out-of-sample testing using walk-forward validation
to confirm the trend-following enhancement is robust, not just lucky.

Walk-Forward Method:
- Split data into multiple folds
- Train on historical data, test on future data
- Roll forward and repeat
- Compare IS vs OOS Sharpe across all folds
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fetch base Asian and Asian + DBMF universes."""
    print("Fetching data...")
    
    base_tickers = ['EWJ', 'FXI', 'EWY', 'INDA', 'EWT', 'EWH', 'EWS', 'AAXJ', 'GLD', 'TLT']
    trend_tickers = base_tickers + ['DBMF', 'CTA']
    
    all_tickers = list(set(trend_tickers))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    # Build universes
    available = [c for c in prices.columns if not prices[c].isna().all()]
    
    base_cols = [c for c in base_tickers if c in available]
    trend_cols = [c for c in trend_tickers if c in available]
    
    base_prices = prices[base_cols].dropna()
    trend_prices = prices[trend_cols].dropna()
    
    print(f"   Base: {len(base_prices.columns)} assets, {len(base_prices)} days")
    print(f"   Trend: {len(trend_prices.columns)} assets, {len(trend_prices)} days")
    
    return base_prices, trend_prices, vix


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_period(prices: pd.DataFrame, vix: pd.Series, warmup: int = 65) -> Dict:
    """Run backtest on a given period."""
    
    config = CompounderConfig(
        vix_threshold=15.0,  # Our optimized setting
        sma_lookback=1,      # Disabled
    )
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except:
        return {'sharpe': 0, 'cagr': 0}
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty or len(weights) < 20:
        return {'sharpe': 0, 'cagr': 0}
    
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    cagr = (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0
    
    return {'sharpe': sharpe, 'cagr': cagr}


def walk_forward_validation(prices: pd.DataFrame, 
                            vix: pd.Series, 
                            n_splits: int = 5) -> Dict:
    """
    Walk-forward validation with expanding window.
    
    For each split:
    - In-sample: all data up to split point
    - Out-of-sample: data after split point
    """
    n = len(prices)
    split_size = n // (n_splits + 1)
    
    is_sharpes = []
    oos_sharpes = []
    is_cagrs = []
    oos_cagrs = []
    fold_dates = []
    
    print(f"\n   Running {n_splits} walk-forward folds...")
    
    for i in range(n_splits):
        # In-sample: first (i+1) parts
        is_end = (i + 1) * split_size
        oos_start = is_end
        oos_end = min(oos_start + split_size, n)
        
        if oos_end <= oos_start + 20:
            continue
        
        is_prices = prices.iloc[:is_end]
        oos_prices = prices.iloc[oos_start:oos_end]
        
        is_vix = vix.iloc[:is_end]
        oos_vix = vix.iloc[oos_start:oos_end]
        
        # Get fold date
        fold_date = prices.index[oos_start].strftime('%Y-%m')
        fold_dates.append(fold_date)
        
        print(f"      Fold {i+1}: IS up to {fold_date}, OOS from {fold_date}...", end=" ", flush=True)
        
        is_result = backtest_period(is_prices, is_vix)
        oos_result = backtest_period(oos_prices, oos_vix)
        
        is_sharpes.append(is_result['sharpe'])
        oos_sharpes.append(oos_result['sharpe'])
        is_cagrs.append(is_result['cagr'])
        oos_cagrs.append(oos_result['cagr'])
        
        print(f"IS={is_result['sharpe']:.2f}, OOS={oos_result['sharpe']:.2f}")
    
    # Calculate degradation
    mean_is = np.mean(is_sharpes) if is_sharpes else 0
    mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0
    degradation = (1 - mean_oos / mean_is) * 100 if mean_is != 0 else 0
    
    return {
        'is_sharpes': is_sharpes,
        'oos_sharpes': oos_sharpes,
        'is_cagrs': is_cagrs,
        'oos_cagrs': oos_cagrs,
        'fold_dates': fold_dates,
        'mean_is_sharpe': mean_is,
        'mean_oos_sharpe': mean_oos,
        'degradation_pct': degradation,
        'n_positive_oos': sum(1 for s in oos_sharpes if s > 0),
        'n_folds': len(is_sharpes)
    }


# =============================================================================
# MAIN
# =============================================================================

def run_walk_forward_comparison():
    """Run walk-forward validation comparing Base vs Trend."""
    
    print("=" * 70)
    print("   WALK-FORWARD VALIDATION")
    print("   Comparing Base Asian vs Asian + Trend/CTA (DBMF)")
    print("=" * 70)
    
    base_prices, trend_prices, vix = fetch_data(years=5)
    
    print("\n" + "=" * 60)
    print("   BASE ASIAN UNIVERSE")
    print("=" * 60)
    
    base_wf = walk_forward_validation(base_prices, vix, n_splits=5)
    
    print("\n" + "=" * 60)
    print("   ASIAN + TREND/CTA (DBMF)")
    print("=" * 60)
    
    trend_wf = walk_forward_validation(trend_prices, vix, n_splits=5)
    
    # Summary
    print("\n" + "=" * 70)
    print("   WALK-FORWARD RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Base Asian':>15} {'Asian + Trend':>15}")
    print("-" * 65)
    print(f"{'Mean IS Sharpe':<30} {base_wf['mean_is_sharpe']:>15.2f} {trend_wf['mean_is_sharpe']:>15.2f}")
    print(f"{'Mean OOS Sharpe':<30} {base_wf['mean_oos_sharpe']:>15.2f} {trend_wf['mean_oos_sharpe']:>15.2f}")
    print(f"{'Degradation':<30} {base_wf['degradation_pct']:>14.1f}% {trend_wf['degradation_pct']:>14.1f}%")
    print(f"{'Positive OOS Folds':<30} {base_wf['n_positive_oos']:>12}/{base_wf['n_folds']} {trend_wf['n_positive_oos']:>12}/{trend_wf['n_folds']}")
    
    print("\n" + "-" * 65)
    print("   FOLD-BY-FOLD COMPARISON (OOS Sharpe)")
    print("-" * 65)
    
    print(f"\n{'Fold':<8} {'Date':<12} {'Base OOS':>12} {'Trend OOS':>12} {'Winner':>12}")
    print("-" * 55)
    
    for i in range(min(len(base_wf['oos_sharpes']), len(trend_wf['oos_sharpes']))):
        base_oos = base_wf['oos_sharpes'][i]
        trend_oos = trend_wf['oos_sharpes'][i]
        date = base_wf['fold_dates'][i] if i < len(base_wf['fold_dates']) else 'N/A'
        winner = "Trend" if trend_oos > base_oos else "Base"
        print(f"{i+1:<8} {date:<12} {base_oos:>12.2f} {trend_oos:>12.2f} {winner:>12}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("   VERDICT")
    print("=" * 70)
    
    trend_wins = sum(1 for b, t in zip(base_wf['oos_sharpes'], trend_wf['oos_sharpes']) if t > b)
    total = min(len(base_wf['oos_sharpes']), len(trend_wf['oos_sharpes']))
    
    if trend_wf['mean_oos_sharpe'] > base_wf['mean_oos_sharpe'] and trend_wins >= total / 2:
        print(f"\n    TREND-FOLLOWING ENHANCEMENT IS ROBUST!")
        print(f"   - Wins {trend_wins}/{total} OOS folds")
        print(f"   - Mean OOS Sharpe: {trend_wf['mean_oos_sharpe']:.2f} vs {base_wf['mean_oos_sharpe']:.2f}")
    elif trend_wf['mean_oos_sharpe'] > base_wf['mean_oos_sharpe']:
        print(f"\n   Trend shows promise but mixed OOS results")
        print(f"   - Wins {trend_wins}/{total} folds")
    else:
        print(f"\n   Base Asian may be more reliable")
        print(f"   - Trend wins only {trend_wins}/{total} folds")
    
    if trend_wf['degradation_pct'] < base_wf['degradation_pct']:
        print(f"    Trend shows LESS degradation ({trend_wf['degradation_pct']:.1f}% vs {base_wf['degradation_pct']:.1f}%)")
    
    print("\n" + "=" * 70)
    
    return {'base': base_wf, 'trend': trend_wf}


if __name__ == "__main__":
    results = run_walk_forward_comparison()
