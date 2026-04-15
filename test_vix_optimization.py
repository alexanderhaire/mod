"""
VIX Threshold Optimization
==========================

The previous test showed that high VIX periods still have massive losses (-144%)
even with the "cash" approach. This suggests:
1. The VIX threshold (25) may not be triggering early enough
2. The 200-day SMA check may be interfering

Let's test different VIX thresholds and see if we can improve protection.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import (
    CompounderConfig,
    CompounderStrategy,
    RegimeDetector,
)


class TunableRegimeDetector(RegimeDetector):
    """Regime detector with tunable VIX threshold."""
    
    def __init__(self, vix_threshold: float = 25.0, sma_lookback: int = 200,
                 bear_exposure: float = 0.0,
                 caution_exposure: float = 0.5,
                 use_sma: bool = True):
        super().__init__(vix_threshold, sma_lookback)
        self.bear_exposure = bear_exposure
        self.caution_exposure = caution_exposure
        self.use_sma = use_sma
    
    def get_regime(self, spy_prices: pd.Series, vix: pd.Series = None) -> str:
        """Determine regime - optionally ignore SMA."""
        if not self.use_sma:
            # Only use VIX for regime detection
            if vix is not None and not vix.empty:
                if isinstance(vix, pd.DataFrame):
                    current_vix = vix.iloc[-1].iloc[0]
                else:
                    current_vix = vix.iloc[-1]
                
                if current_vix > self.vix_threshold:
                    return 'BEAR'
                elif current_vix > self.vix_threshold * 0.7:  # Caution zone
                    return 'CAUTION'
            return 'BULL'
        else:
            return super().get_regime(spy_prices, vix)
    
    def get_exposure_multiplier(self, regime: str) -> float:
        if regime == 'BULL':
            return 1.0
        elif regime == 'CAUTION':
            return self.caution_exposure
        else:
            return self.bear_exposure


class TunableCompounderStrategy(CompounderStrategy):
    """Strategy with tunable VIX parameters."""
    
    def __init__(self, config: CompounderConfig = None,
                 vix_threshold: float = 25.0,
                 bear_exposure: float = 0.0,
                 caution_exposure: float = 0.5,
                 use_sma: bool = True):
        super().__init__(config)
        self.regime_detector = TunableRegimeDetector(
            vix_threshold=vix_threshold,
            sma_lookback=self.config.sma_lookback,
            bear_exposure=bear_exposure,
            caution_exposure=caution_exposure,
            use_sma=use_sma
        )


def fetch_asian_data(years: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch Asian market data."""
    print("Fetching data...")
    
    tickers = {
        'EWJ': 'Japan', 'FXI': 'China Large-Cap', 'EWY': 'South Korea',
        'INDA': 'India', 'EWT': 'Taiwan', 'EWH': 'Hong Kong',
        'EWS': 'Singapore', 'AAXJ': 'Asia ex-Japan', 'GLD': 'Gold', 'TLT': 'Long Treasuries'
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
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def backtest_strategy(prices: pd.DataFrame, strategy, vix: pd.Series, warmup: int = 65) -> Dict:
    """Run backtest."""
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
        'daily_returns': net_returns
    }


def run_optimization():
    """Test different VIX thresholds and configurations."""
    
    print("=" * 70)
    print("   VIX THRESHOLD OPTIMIZATION - ASIAN MARKETS")
    print("=" * 70)
    
    prices, vix = fetch_asian_data(years=10)
    
    split_point = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    print(f"\n   OOS Period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    
    # Test different configurations
    configs = []
    
    # VIX thresholds (lower = more conservative)
    for vix_thresh in [15, 18, 20, 22, 25, 30]:
        for use_sma in [True, False]:
            for bear_exp in [0.0, -0.5, -1.0]:
                configs.append({
                    'vix_threshold': vix_thresh,
                    'use_sma': use_sma,
                    'bear_exposure': bear_exp,
                    'caution_exposure': 0.5
                })
    
    print(f"\n   Testing {len(configs)} configurations...")
    
    results = []
    
    for i, cfg in enumerate(configs):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(configs)}...", end="\r", flush=True)
        
        strategy = TunableCompounderStrategy(
            vix_threshold=cfg['vix_threshold'],
            bear_exposure=cfg['bear_exposure'],
            caution_exposure=cfg['caution_exposure'],
            use_sma=cfg['use_sma']
        )
        
        result = backtest_strategy(oos_prices, strategy, oos_vix)
        
        results.append({
            **cfg,
            'sharpe': result['sharpe'],
            'cagr': result['cagr'],
            'max_dd': result['max_dd']
        })
    
    print(f"   Progress: {len(configs)}/{len(configs)} - Done!")
    
    # Sort by Sharpe
    results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print("\n" + "=" * 70)
    print("   TOP 10 CONFIGURATIONS BY SHARPE")
    print("=" * 70)
    
    print(f"\n{'Rank':<5} {'VIX Thresh':>10} {'Use SMA':>8} {'Bear Exp':>10} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10], 1):
        sma_str = "Yes" if r['use_sma'] else "No"
        bear_str = f"{r['bear_exposure']:.1f}"
        print(f"{i:<5} {r['vix_threshold']:>10} {sma_str:>8} {bear_str:>10} {r['sharpe']:>8.2f} {r['cagr']:>7.1%} {r['max_dd']:>7.1%}")
    
    print("\n" + "=" * 70)
    print("   BOTTOM 5 CONFIGURATIONS (avoid these)")
    print("=" * 70)
    
    print(f"\n{'Rank':<5} {'VIX Thresh':>10} {'Use SMA':>8} {'Bear Exp':>10} {'Sharpe':>8} {'CAGR':>8}")
    print("-" * 60)
    
    for i, r in enumerate(results[-5:], len(results)-4):
        sma_str = "Yes" if r['use_sma'] else "No"
        bear_str = f"{r['bear_exposure']:.1f}"
        print(f"{i:<5} {r['vix_threshold']:>10} {sma_str:>8} {bear_str:>10} {r['sharpe']:>8.2f} {r['cagr']:>7.1%}")
    
    # Best configuration
    best = results[0]
    print("\n" + "=" * 70)
    print("   OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"\n   VIX Threshold:  {best['vix_threshold']}")
    print(f"   Use SMA:        {'Yes' if best['use_sma'] else 'No'}")
    print(f"   Bear Exposure:  {best['bear_exposure']}")
    print(f"   Sharpe:         {best['sharpe']:.2f}")
    print(f"   CAGR:           {best['cagr']:.1%}")
    print(f"   Max Drawdown:   {best['max_dd']:.1%}")
    
    # Compare to baseline
    baseline = next(r for r in results if r['vix_threshold'] == 25 and r['use_sma'] and r['bear_exposure'] == 0.0)
    
    print(f"\n   vs Baseline (VIX 25, SMA, Cash):")
    print(f"   Sharpe improvement: {best['sharpe'] - baseline['sharpe']:+.2f}")
    print(f"   CAGR improvement:   {best['cagr'] - baseline['cagr']:+.1%}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_optimization()
