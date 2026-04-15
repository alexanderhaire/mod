"""
Defensive Strategy Enhancement Test - REAL DATA
================================================

Tests various defensive strategies to improve performance during high VIX:

1. LONG VOLATILITY / CONVEXITY
   - VIXY: ProShares VIX Short-Term Futures (long vol)
   - DBMF: iMGP DBi Managed Futures Strategy (trend-following/CTA)
   - TAIL: Cambria Tail Risk ETF (put spreads / crash protection)

2. QUALITY / DEFENSIVE FACTORS
   - QUAL: iShares MSCI USA Quality Factor
   - SPLV: Invesco S&P 500 Low Volatility
   - USMV: iShares MSCI USA Min Vol Factor
   - VIG: Vanguard Dividend Appreciation

3. SHORT-DURATION / CASH-LIKE
   - SHY: iShares 1-3 Year Treasury
   - BIL: SPDR Bloomberg 1-3 Month T-Bill

All using REAL Yahoo Finance data. No simulations.
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
# DATA FETCHING - REAL DATA ONLY
# =============================================================================

def fetch_all_universes(years: int = 5) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Fetch multiple universe configurations for comparison.
    Uses shorter period (5 years) because some defensive ETFs are newer.
    """
    print("=" * 60)
    print("   FETCHING REAL DATA FROM YAHOO FINANCE")
    print("=" * 60)
    
    # Base Asian universe
    asian_tickers = ['EWJ', 'FXI', 'EWY', 'INDA', 'EWT', 'EWH', 'EWS', 'AAXJ', 'GLD', 'TLT']
    
    # Defensive additions
    long_vol_tickers = ['VIXY', 'TAIL']  # Note: DBMF may have limited history
    trend_tickers = ['DBMF', 'CTA']  # Managed futures
    quality_tickers = ['QUAL', 'SPLV', 'USMV', 'VIG']
    short_duration_tickers = ['SHY', 'BIL']
    
    all_tickers = list(set(asian_tickers + long_vol_tickers + trend_tickers + 
                           quality_tickers + short_duration_tickers))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"\n   Downloading {len(all_tickers)} tickers...")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    
    # Download all data
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    # VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    # Clean
    prices = prices.dropna(how='all').ffill()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    # Report available tickers
    available = [c for c in prices.columns if not prices[c].isna().all()]
    missing = [t for t in all_tickers if t not in available]
    
    print(f"\n   Available: {len(available)} tickers")
    if missing:
        print(f"   Missing/No data: {missing}")
    
    # Build universe configurations
    universes = {}
    
    # 1. Base Asian only
    base_cols = [c for c in asian_tickers if c in available]
    if len(base_cols) >= 5:
        universes['Base Asian'] = prices[base_cols].dropna()
    
    # 2. Asian + Long Vol
    longvol_cols = base_cols + [c for c in long_vol_tickers if c in available]
    if len(longvol_cols) > len(base_cols):
        universes['Asian + Long Vol'] = prices[longvol_cols].dropna()
    
    # 3. Asian + Trend Following
    trend_cols = base_cols + [c for c in trend_tickers if c in available]
    if len(trend_cols) > len(base_cols):
        universes['Asian + Trend/CTA'] = prices[trend_cols].dropna()
    
    # 4. Asian + Quality/Defensive Factors
    quality_cols = base_cols + [c for c in quality_tickers if c in available]
    if len(quality_cols) > len(base_cols):
        universes['Asian + Quality'] = prices[quality_cols].dropna()
    
    # 5. Asian + Short Duration Bonds
    short_dur_cols = base_cols + [c for c in short_duration_tickers if c in available]
    if len(short_dur_cols) > len(base_cols):
        universes['Asian + Short Duration'] = prices[short_dur_cols].dropna()
    
    # 6. Asian + Low Vol factors only
    lowvol_cols = base_cols + [c for c in ['SPLV', 'USMV'] if c in available]
    if len(lowvol_cols) > len(base_cols):
        universes['Asian + Low Vol Factors'] = prices[lowvol_cols].dropna()
    
    # 7. Full defensive suite
    all_defensive = base_cols + [c for c in (quality_tickers + short_duration_tickers) if c in available]
    if len(all_defensive) > len(base_cols):
        universes['Asian + All Defensive'] = prices[all_defensive].dropna()
    
    # 8. Crash protection focus
    crash_cols = base_cols + [c for c in (long_vol_tickers + short_duration_tickers) if c in available]
    if len(crash_cols) > len(base_cols):
        universes['Asian + Crash Protection'] = prices[crash_cols].dropna()
    
    print(f"\n   Built {len(universes)} universe configurations:")
    for name, df in universes.items():
        print(f"      {name}: {len(df.columns)} assets, {len(df)} days")
    
    return universes, vix


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_universe(prices: pd.DataFrame, 
                      vix: pd.Series,
                      warmup: int = 65,
                      vix_threshold: float = 15.0) -> Dict:
    """
    Backtest using our optimized configuration (VIX 15, no SMA).
    """
    # Use optimized config
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=1,  # Effectively disabled
    )
    
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except Exception as e:
        print(f"      Error: {e}")
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
    # Normalize and smooth
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    # Portfolio returns
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # High VIX performance
    vix_aligned = vix.reindex(net_returns.index).ffill()
    if isinstance(vix_aligned, pd.DataFrame):
        vix_aligned = vix_aligned.iloc[:, 0]
    high_vix_mask = vix_aligned > 25
    high_vix_returns = net_returns[high_vix_mask.values]
    high_vix_sharpe = high_vix_returns.mean() / high_vix_returns.std() * np.sqrt(252) if len(high_vix_returns) > 20 and high_vix_returns.std() > 0 else None
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': ((equity - equity.cummax()) / equity.cummax()).min(),
        'volatility': net_returns.std() * np.sqrt(252),
        'high_vix_sharpe': high_vix_sharpe,
        'high_vix_days': high_vix_mask.sum(),
        'daily_returns': net_returns
    }


def analyze_high_vix_performance(results: Dict[str, Dict], vix: pd.Series):
    """Analyze which configurations do best during high VIX."""
    print("\n" + "=" * 60)
    print("   HIGH VIX PERFORMANCE ANALYSIS (VIX > 25)")
    print("=" * 60)
    
    print(f"\n{'Universe':<30} {'Days':>8} {'HV Sharpe':>12} {'HV Return':>12}")
    print("-" * 65)
    
    for name, result in results.items():
        hv_sharpe = result.get('high_vix_sharpe')
        hv_days = result.get('high_vix_days', 0)
        
        if hv_sharpe is not None:
            # Calculate high VIX return
            daily_ret = result.get('daily_returns')
            if daily_ret is not None:
                vix_aligned = vix.reindex(daily_ret.index).ffill()
                hv_returns = daily_ret[vix_aligned > 25]
                hv_annual = hv_returns.mean() * 252 if len(hv_returns) > 0 else 0
                print(f"{name:<30} {hv_days:>8} {hv_sharpe:>12.2f} {hv_annual:>11.1%}")
            else:
                print(f"{name:<30} {hv_days:>8} {hv_sharpe:>12.2f} {'N/A':>12}")
        else:
            print(f"{name:<30} {hv_days:>8} {'N/A':>12}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_comprehensive_test():
    """Test all defensive strategy configurations."""
    
    print("=" * 70)
    print("   DEFENSIVE STRATEGY ENHANCEMENT TEST")
    print("   Using REAL Yahoo Finance Data")
    print("=" * 70)
    
    # Fetch all universes
    universes, vix = fetch_all_universes(years=5)
    
    if not universes:
        print("\n ERROR: No valid universes could be built")
        return None
    
    # OOS split (use last 30%)
    results = {}
    
    print("\n" + "=" * 60)
    print("   BACKTESTING UNIVERSES (OOS)")
    print("=" * 60)
    
    print(f"\n{'Universe':<30} {'Assets':>8} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-" * 75)
    
    for name, prices in universes.items():
        # OOS split
        split_point = int(len(prices) * 0.7)
        oos_prices = prices.iloc[split_point:]
        oos_vix = vix.reindex(oos_prices.index).ffill()
        
        print(f"   Training {name}...", end=" ", flush=True)
        
        result = backtest_universe(oos_prices, oos_vix, vix_threshold=15.0)
        results[name] = result
        
        print("Done")
        print(f"{name:<30} {len(oos_prices.columns):>8} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_dd']:>9.1%}")
    
    # Rank by Sharpe
    print("\n" + "=" * 60)
    print("   RESULTS RANKED BY SHARPE")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Universe':<35} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 65)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<6} {name:<35} {result['sharpe']:>10.2f} {result['max_dd']:>9.1%}{marker}")
    
    # High VIX analysis
    analyze_high_vix_performance(results, vix)
    
    # Compare best vs base
    base_result = results.get('Base Asian', list(results.values())[0])
    best_name, best_result = sorted_results[0]
    
    print("\n" + "=" * 60)
    print("   IMPROVEMENT SUMMARY")
    print("=" * 60)
    
    base_sharpe = base_result['sharpe']
    best_sharpe = best_result['sharpe']
    improvement = best_sharpe - base_sharpe
    
    print(f"\n   Base Asian Sharpe:    {base_sharpe:.2f}")
    print(f"   Best Config:          {best_name}")
    print(f"   Best Sharpe:          {best_sharpe:.2f}")
    print(f"   Improvement:          {improvement:+.2f}")
    
    if improvement > 0.1:
        print(f"\n    SIGNIFICANT IMPROVEMENT with {best_name}!")
    elif improvement > 0:
        print(f"\n    Marginal improvement with {best_name}")
    else:
        print(f"\n   Base Asian universe remains optimal")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_test()
