"""
High VIX Shorting Strategy Experiment
======================================

Tests whether shorting during high VIX periods (instead of going to cash)
improves the Asian market Compounder strategy.

Variants tested:
1. Original: BEAR = 0% (cash)
2. Short BEAR: BEAR = -100% (full short)
3. Inverse BEAR: BEAR = -50% (partial short)
4. Dynamic: Scale short size by VIX level
5. Inverse Momentum: Short worst performers during high VIX
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import (
    CompounderConfig,
    CompounderStrategy,
    RegimeDetector,
    EnsembleSignalGenerator,
    PositionSizer
)


# =============================================================================
# MODIFIED REGIME DETECTOR WITH SHORTING
# =============================================================================

class ShortingRegimeDetector(RegimeDetector):
    """Modified regime detector that enables shorting in BEAR regimes."""
    
    def __init__(self, vix_threshold: float = 25.0, sma_lookback: int = 200,
                 bear_exposure: float = -1.0,  # Negative = short
                 caution_exposure: float = 0.5):
        super().__init__(vix_threshold, sma_lookback)
        self.bear_exposure = bear_exposure
        self.caution_exposure = caution_exposure
    
    def get_exposure_multiplier(self, regime: str) -> float:
        """Get exposure multiplier - can be negative for shorting."""
        if regime == 'BULL':
            return 1.0
        elif regime == 'CAUTION':
            return self.caution_exposure
        else:  # BEAR
            return self.bear_exposure  # Can be negative!


class ShortingCompounderStrategy(CompounderStrategy):
    """Modified strategy that can short during bear regimes."""
    
    def __init__(self, config: CompounderConfig = None, 
                 bear_exposure: float = -1.0,
                 caution_exposure: float = 0.5):
        super().__init__(config)
        # Replace regime detector with shorting version
        self.regime_detector = ShortingRegimeDetector(
            vix_threshold=self.config.vix_threshold,
            sma_lookback=self.config.sma_lookback,
            bear_exposure=bear_exposure,
            caution_exposure=caution_exposure
        )


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_asian_data(years: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch real Asian market data from Yahoo Finance."""
    print("Fetching ASIAN MARKET data...")
    
    tickers = {
        'EWJ': 'Japan',
        'FXI': 'China Large-Cap',
        'EWY': 'South Korea',
        'INDA': 'India',
        'EWT': 'Taiwan',
        'EWH': 'Hong Kong',
        'EWS': 'Singapore',
        'AAXJ': 'Asia ex-Japan',
        'GLD': 'Gold',
        'TLT': 'Long Treasuries'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
        
        prices.columns = [tickers.get(c, c) for c in prices.columns]
        
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if 'Close' in vix_data.columns:
            vix = vix_data['Close']
        else:
            vix = vix_data.iloc[:, 0] if len(vix_data.columns) > 0 else pd.Series()
        
        prices = prices.dropna(how='all').ffill().dropna()
        vix = vix.reindex(prices.index).ffill().fillna(15)
        
        print(f"   Loaded {len(prices)} days for {len(prices.columns)} assets")
        return prices, vix
        
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return pd.DataFrame(), pd.Series()


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_strategy(prices: pd.DataFrame, 
                      strategy,
                      vix: pd.Series = None,
                      transaction_cost: float = 0.001,
                      warmup: int = 65) -> Dict:
    """Run backtest for a given strategy."""
    
    # Generate weights
    weights = strategy.generate_weights(prices, vix=vix)
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return {'daily_returns': pd.Series(), 'sharpe': 0, 'equity_curve': pd.Series()}
    
    # Normalize weights
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    
    # Smooth positions
    smoothed = normalized.ewm(span=5).mean()
    
    # Portfolio returns
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * transaction_cost
    
    # Metrics
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    return {
        'daily_returns': net_returns,
        'sharpe': sharpe,
        'equity_curve': equity,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'volatility': net_returns.std() * np.sqrt(252),
        'max_drawdown': ((equity - equity.cummax()) / equity.cummax()).min()
    }


def analyze_regime_returns(returns: pd.Series, vix: pd.Series) -> Dict:
    """Analyze returns by regime."""
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    common_idx = returns.index.intersection(vix.index)
    returns = returns.loc[common_idx]
    vix_aligned = vix.loc[common_idx]
    
    low_vol = vix_aligned < 15
    normal_vol = (vix_aligned >= 15) & (vix_aligned < 25)
    high_vol = vix_aligned >= 25
    
    results = {}
    for regime, mask in [('Low VIX', low_vol), ('Normal VIX', normal_vol), ('High VIX', high_vol)]:
        regime_returns = returns[mask.values]
        if len(regime_returns) > 5:
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
            results[regime] = {
                'days': len(regime_returns),
                'sharpe': sharpe,
                'return': regime_returns.mean() * 252
            }
    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_shorting_experiment():
    """Test various shorting strategies during high VIX."""
    
    print("=" * 70)
    print("   HIGH VIX SHORTING EXPERIMENT - ASIAN MARKETS")
    print("=" * 70)
    
    # Fetch data
    prices, vix = fetch_asian_data(years=10)
    
    if prices.empty:
        print("Failed to load data")
        return
    
    # Split into OOS only (we care about real performance)
    split_point = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    print(f"\n   Testing on OOS period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    print(f"   ({len(oos_prices)} days)")
    
    # Define variants to test
    variants = {
        'Original (Cash in BEAR)': {'bear': 0.0, 'caution': 0.5},
        'Full Short BEAR': {'bear': -1.0, 'caution': 0.5},
        'Half Short BEAR': {'bear': -0.5, 'caution': 0.5},
        'Quarter Short BEAR': {'bear': -0.25, 'caution': 0.5},
        'Full Short, No Caution': {'bear': -1.0, 'caution': 1.0},
        'Short BEAR + Short CAUTION': {'bear': -1.0, 'caution': -0.5},
        'Aggressive Caution Only': {'bear': 0.0, 'caution': 0.25},
    }
    
    print("\n" + "=" * 70)
    print("   TESTING REGIME EXPOSURE VARIANTS")
    print("=" * 70)
    
    results = {}
    
    print(f"\n{'Variant':<30} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'Vol':>10}")
    print("-" * 75)
    
    for name, params in variants.items():
        print(f"   Training {name}...", end=" ", flush=True)
        
        strategy = ShortingCompounderStrategy(
            bear_exposure=params['bear'],
            caution_exposure=params['caution']
        )
        
        result = backtest_strategy(oos_prices, strategy, oos_vix)
        results[name] = result
        
        print("Done")
        print(f"{name:<30} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_drawdown']:>9.1%} {result['volatility']:>9.1%}")
    
    # Find the best
    print("\n" + "=" * 70)
    print("   RESULTS RANKED BY SHARPE RATIO")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Variant':<35} {'Sharpe':>10} {'CAGR':>10}")
    print("-" * 65)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<6} {name:<35} {result['sharpe']:>10.2f} {result['cagr']:>9.1%}{marker}")
    
    # Compare regime performance for best vs original
    print("\n" + "=" * 70)
    print("   REGIME BREAKDOWN: ORIGINAL vs BEST")
    print("=" * 70)
    
    original = results['Original (Cash in BEAR)']
    best_name = sorted_results[0][0]
    best = sorted_results[0][1]
    
    orig_regimes = analyze_regime_returns(original['daily_returns'], oos_vix)
    best_regimes = analyze_regime_returns(best['daily_returns'], oos_vix)
    
    print(f"\n   Original (Cash in BEAR):")
    for regime, data in orig_regimes.items():
        print(f"      {regime}: Sharpe={data['sharpe']:.2f}, Return={data['return']:.1%} ({data['days']} days)")
    
    print(f"\n   {best_name}:")
    for regime, data in best_regimes.items():
        print(f"      {regime}: Sharpe={data['sharpe']:.2f}, Return={data['return']:.1%} ({data['days']} days)")
    
    # Improvement summary
    print("\n" + "=" * 70)
    print("   IMPROVEMENT SUMMARY")
    print("=" * 70)
    
    sharpe_improvement = best['sharpe'] - original['sharpe']
    cagr_improvement = best['cagr'] - original['cagr']
    
    print(f"\n   Best Variant: {best_name}")
    print(f"   Sharpe Improvement: {sharpe_improvement:+.2f} ({original['sharpe']:.2f} -> {best['sharpe']:.2f})")
    print(f"   CAGR Improvement:   {cagr_improvement:+.1%} ({original['cagr']:.1%} -> {best['cagr']:.1%})")
    
    if sharpe_improvement > 0:
        print(f"\n    SHORTING DURING HIGH VIX IMPROVES THE STRATEGY!")
    else:
        print(f"\n    Original cash approach remains best")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_shorting_experiment()
