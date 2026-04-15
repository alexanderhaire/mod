"""
Beat the Champion: Asian ETFs + VIX 25 (Sharpe 1.56)
=====================================================

Testing MANY variations to find something that beats the current best.

Categories to test:
1. VIX threshold variations (10, 12, 15, 18, 20, 22, 25, 30)
2. Universe variations (add DBMF trend, add US tech, sector rotation)
3. SMA lookback variations (100, 150, 200, 250)
4. Position cap variations (10%, 15%, 20%, 25%, 30%)
5. Leverage overlay variations
6. Combined best of each
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# UNIVERSES TO TEST
# =============================================================================

# Current champion
ASIAN_BASE = ['EWJ', 'FXI', 'EWY', 'INDA', 'EWT', 'EWH', 'EWS', 'AAXJ', 'GLD', 'TLT']

# Variations
ASIAN_EXTENDED = ASIAN_BASE + ['DBMF']  # Add trend-following
ASIAN_US_TECH = ASIAN_BASE + ['AAPL', 'MSFT', 'NVDA']  # Add US mega-caps
ASIAN_EMERGING = ['FXI', 'INDA', 'EWT', 'EWS', 'AAXJ', 'VWO', 'GLD', 'TLT']  # EM focus
ASIAN_CHINA_HEAVY = ['FXI', 'KWEB', 'MCHI', 'EWH', 'GLD', 'TLT']  # China focus
ASIAN_JAPAN_KOREA = ['EWJ', 'EWY', 'AAXJ', 'GLD', 'TLT']  # JP+KR focus
ASIAN_MINIMUM = ['AAXJ', 'EWJ', 'FXI', 'GLD', 'TLT']  # Minimal set


# =============================================================================
# FUNCTIONS
# =============================================================================

def fetch_universe(tickers: List[str], years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch data for a universe."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill()
    available = [c for c in prices.columns if not prices[c].isna().all()]
    prices = prices[available].dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def backtest_config(prices: pd.DataFrame, vix: pd.Series, 
                    vix_threshold: float = 25.0,
                    sma_lookback: int = 200,
                    max_position: float = 0.20,
                    warmup: int = 65) -> Dict:
    """Run backtest with specific configuration."""
    
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=sma_lookback,
        max_position_pct=max_position
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
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': ((equity - equity.cummax()) / equity.cummax()).min(),
    }


def run_exhaustive_test():
    """Run exhaustive search for best configuration."""
    
    print("=" * 80)
    print("   EXHAUSTIVE SEARCH: BEAT THE CHAMPION (Asian ETFs VIX 25 = 1.56)")
    print("=" * 80)
    
    # Fetch all possible tickers
    print("\nFetching all possible tickers...")
    all_tickers = list(set(ASIAN_BASE + ASIAN_EXTENDED + ASIAN_US_TECH + 
                           ASIAN_EMERGING + ASIAN_CHINA_HEAVY + ASIAN_JAPAN_KOREA))
    
    all_prices, vix = fetch_universe(all_tickers, years=5)
    available_tickers = list(all_prices.columns)
    
    print(f"   Available: {len(available_tickers)}")
    
    # OOS split
    split_point = int(len(all_prices) * 0.7)
    oos_prices = all_prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    # Define test matrix
    universes = {
        'Asian Base': [t for t in ASIAN_BASE if t in available_tickers],
        'Asian + DBMF': [t for t in ASIAN_EXTENDED if t in available_tickers],
        'Asian + US Tech': [t for t in ASIAN_US_TECH if t in available_tickers],
        'EM Focus': [t for t in ASIAN_EMERGING if t in available_tickers],
        'China Heavy': [t for t in ASIAN_CHINA_HEAVY if t in available_tickers],
        'Japan+Korea': [t for t in ASIAN_JAPAN_KOREA if t in available_tickers],
        'Minimal Set': [t for t in ASIAN_MINIMUM if t in available_tickers],
    }
    
    vix_thresholds = [15, 18, 20, 22, 25, 28, 30]
    sma_lookbacks = [100, 150, 200]
    position_caps = [0.15, 0.20, 0.25, 0.30]
    
    # Calculate total tests
    total_tests = len(universes) * len(vix_thresholds) * len(sma_lookbacks) * len(position_caps)
    print(f"\n   Running {total_tests} configuration tests...")
    
    champion_sharpe = 1.56
    results = []
    test_num = 0
    
    for u_name, tickers in universes.items():
        if len(tickers) < 4:
            continue
            
        u_prices = oos_prices[tickers].dropna()
        
        for vix_t in vix_thresholds:
            for sma_l in sma_lookbacks:
                for pos_cap in position_caps:
                    test_num += 1
                    if test_num % 20 == 0:
                        print(f"   Progress: {test_num}/{total_tests}...", end="\r", flush=True)
                    
                    result = backtest_config(u_prices, oos_vix, 
                                             vix_threshold=vix_t,
                                             sma_lookback=sma_l,
                                             max_position=pos_cap)
                    
                    results.append({
                        'universe': u_name,
                        'vix': vix_t,
                        'sma': sma_l,
                        'pos_cap': pos_cap,
                        'sharpe': result['sharpe'],
                        'cagr': result.get('cagr', 0),
                        'max_dd': result.get('max_dd', 0),
                        'beats_champion': result['sharpe'] > champion_sharpe
                    })
    
    print(f"   Progress: {total_tests}/{total_tests} - Complete!")
    
    # Sort by Sharpe
    results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    # Results
    print("\n" + "=" * 80)
    print("   TOP 20 CONFIGURATIONS BY SHARPE")
    print("=" * 80)
    
    print(f"\n{'Rank':<6} {'Universe':<18} {'VIX':>6} {'SMA':>6} {'Cap':>6} {'Sharpe':>10} {'CAGR':>8} {'Beats?':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results[:20], 1):
        beats = "YES!" if r['beats_champion'] else "no"
        marker = " <--" if r['beats_champion'] else ""
        print(f"{i:<6} {r['universe']:<18} {r['vix']:>6} {r['sma']:>6} {r['pos_cap']:>5.0%} {r['sharpe']:>10.2f} {r['cagr']:>7.1%} {beats:>8}{marker}")
    
    # Count configs that beat champion
    beats_count = sum(1 for r in results if r['beats_champion'])
    
    print("\n" + "=" * 80)
    print("   SUMMARY")
    print("=" * 80)
    
    print(f"\n   Champion: Asian ETFs + VIX 25 = Sharpe 1.56")
    print(f"   Total configurations tested: {len(results)}")
    print(f"   Configurations that BEAT champion: {beats_count}")
    
    if beats_count > 0:
        print(f"\n    FOUND {beats_count} CONFIGURATIONS THAT BEAT THE CHAMPION!")
        print(f"\n   Best configuration:")
        best = results[0]
        print(f"      Universe:      {best['universe']}")
        print(f"      VIX threshold: {best['vix']}")
        print(f"      SMA lookback:  {best['sma']}")
        print(f"      Position cap:  {best['pos_cap']:.0%}")
        print(f"      Sharpe:        {best['sharpe']:.2f}")
        print(f"      CAGR:          {best['cagr']:.1%}")
    else:
        print(f"\n   No configuration beats the champion.")
        print(f"   Asian ETFs + VIX 25 remains THE BEST!")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_exhaustive_test()
