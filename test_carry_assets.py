"""
Carry Trade Direct Inclusion Test
=================================

Previous test showed carry signal overlay had no effect.
This test directly includes carry-sensitive assets in the tradeable universe:

- FXY (inverse = short JPY = carry long)
- CEW (EM currency fund)
- EMHY or similar (EM high yield)

The ML model should naturally pick up carry momentum if it's a valid signal.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy


def fetch_extended_universe(years: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch extended universe including carry-sensitive assets.
    """
    print("Fetching extended universe with carry assets...")
    
    # Extended universe: Asian + Carry + Defensive
    tickers = {
        # Core Asian
        'EWJ': 'Japan',
        'FXI': 'China',
        'EWY': 'South Korea',
        'INDA': 'India',
        'EWT': 'Taiwan',
        'EWH': 'Hong Kong',
        'EWS': 'Singapore',
        'AAXJ': 'Asia ex-Japan',
        # Carry-sensitive
        'FXY': 'JPY Currency',    # Short this = carry long
        'CEW': 'EM Currencies',   # Long = carry long
        'BWX': 'Intl Bonds',      # Yield exposure
        # Defensive
        'GLD': 'Gold',
        'TLT': 'Long Treasuries',
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    prices.columns = [tickers.get(c, c) for c in prices.columns]
    
    # VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    print(f"   Loaded {len(prices)} days for {len(prices.columns)} assets (includes carry)")
    
    return prices, vix


def backtest_strategy(prices: pd.DataFrame, 
                      vix: pd.Series,
                      warmup: int = 65) -> Dict:
    """Run compounder strategy backtest."""
    
    strategy = CompounderStrategy()
    weights = strategy.generate_weights(prices, vix=vix)
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
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
        'daily_returns': net_returns,
        'weights': weights
    }


def run_comparison():
    """Compare original vs carry-extended universe."""
    
    print("=" * 70)
    print("   CARRY ASSET INCLUSION COMPARISON")
    print("=" * 70)
    
    # Original universe
    print("\n--- Testing Original Universe ---")
    orig_tickers = {
        'EWJ': 'Japan', 'FXI': 'China', 'EWY': 'South Korea', 'INDA': 'India',
        'EWT': 'Taiwan', 'EWH': 'Hong Kong', 'EWS': 'Singapore', 
        'AAXJ': 'Asia ex-Japan', 'GLD': 'Gold', 'TLT': 'Long Treasuries'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10 * 365)
    
    orig_data = yf.download(list(orig_tickers.keys()), start=start_date, end=end_date, progress=False)
    orig_prices = orig_data['Adj Close'] if 'Adj Close' in orig_data.columns else orig_data['Close']
    orig_prices.columns = [orig_tickers.get(c, c) for c in orig_prices.columns]
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    orig_prices = orig_prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(orig_prices.index).ffill().fillna(15)
    
    # OOS split
    split_point = int(len(orig_prices) * 0.7)
    oos_orig = orig_prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    print(f"   Training on {len(oos_orig.columns)} assets...")
    orig_result = backtest_strategy(oos_orig, oos_vix)
    
    # Extended universe
    print("\n--- Testing Extended Universe (with Carry) ---")
    ext_prices, ext_vix = fetch_extended_universe(years=10)
    
    # Same OOS period
    ext_oos = ext_prices.iloc[split_point:len(ext_prices)]
    ext_oos_vix = ext_vix.iloc[split_point:len(ext_vix)]
    
    print(f"   Training on {len(ext_oos.columns)} assets...")
    ext_result = backtest_strategy(ext_oos, ext_oos_vix)
    
    # Results
    print("\n" + "=" * 70)
    print("   COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Universe':<30} {'Assets':<10} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-" * 75)
    print(f"{'Original (10 assets)':<30} {len(oos_orig.columns):<10} {orig_result['sharpe']:>10.2f} {orig_result['cagr']:>9.1%} {orig_result['max_dd']:>9.1%}")
    print(f"{'Extended (+Carry, 13 assets)':<30} {len(ext_oos.columns):<10} {ext_result['sharpe']:>10.2f} {ext_result['cagr']:>9.1%} {ext_result['max_dd']:>9.1%}")
    
    improvement = ext_result['sharpe'] - orig_result['sharpe']
    
    print("\n" + "=" * 70)
    
    if improvement > 0.05:
        print(f"    ADDING CARRY ASSETS IMPROVES SHARPE BY {improvement:+.2f}")
    elif improvement < -0.05:
        print(f"    CARRY ASSETS HURT PERFORMANCE BY {improvement:.2f}")
    else:
        print(f"   Minimal difference ({improvement:+.2f})")
    
    # Show allocation to carry assets in extended portfolio
    if 'weights' in ext_result:
        weights = ext_result['weights']
        carry_cols = [c for c in weights.columns if c in ['JPY Currency', 'EM Currencies', 'Intl Bonds']]
        if carry_cols:
            avg_carry_weight = weights[carry_cols].abs().mean().sum()
            print(f"\n   Average allocation to carry assets: {avg_carry_weight:.1%}")
    
    print("\n" + "=" * 70)
    
    return {'original': orig_result, 'extended': ext_result}


if __name__ == "__main__":
    results = run_comparison()
