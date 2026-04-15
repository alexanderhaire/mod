"""
All-Weather Strategy Backtesting
=================================

Comprehensive backtesting script that evaluates all strategy variants
and generates performance metrics matching Table 1 from the specification.

Usage:
    python all_weather_backtest.py

Output:
    - Console: Performance table for all 7+ strategy variants
    - all_weather_results.png: Equity curves chart
    - all_weather_metrics.csv: Raw performance data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from all_weather_strategy import (
    strategy_baseline_ems,
    strategy_ems_with_trend,
    strategy_ems_with_risk_parity,
    strategy_ems_with_value_momentum,
    strategy_ems_trend_risk_parity,
    strategy_stat_arb_only,
    strategy_vrp_only,
    strategy_all_weather,
    AllWeatherEnsemble,
    AllWeatherConfig
)

# Import data fetching utility
try:
    from external_data import fetch_market_data_pool
    HAS_EXTERNAL_DATA = True
except ImportError:
    HAS_EXTERNAL_DATA = False


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class AllWeatherBacktester:
    """
    Comprehensive backtester for All-Weather strategy variants.
    
    Features:
    - Transaction cost modeling
    - Position smoothing to reduce turnover
    - Full metrics calculation (Sharpe, CAGR, Vol, MaxDD, etc.)
    - Equity curve generation
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 transaction_cost_pct: float = 0.001,
                 smoothing_factor: float = 0.3):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.smoothing_factor = smoothing_factor
    
    def run(self, 
            prices: pd.DataFrame, 
            signal_func: Callable,
            vix: Optional[pd.Series] = None,
            warmup: int = 65) -> Dict[str, Any]:
        """
        Run backtest for a given signal function.
        
        Args:
            prices: Price DataFrame (dates x assets)
            signal_func: Function that generates signals from prices
            vix: Optional VIX series for VRP strategies
            warmup: Number of initial days to skip
            
        Returns:
            Dictionary with performance metrics and equity curve
        """
        # Generate signals
        try:
            # Check if function accepts vix parameter
            if 'vix' in signal_func.__code__.co_varnames:
                signals = signal_func(prices, vix)
            else:
                signals = signal_func(prices)
        except Exception as e:
            print(f"  Signal generation error: {e}")
            signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Skip warmup period
        signals = signals.iloc[warmup:].copy()
        returns = returns.iloc[warmup:].copy()
        
        if signals.empty:
            return self._empty_result()
        
        # Normalize signals to sum of abs = 1 (fully invested)
        abs_sum = signals.abs().sum(axis=1).replace(0, 1)
        normalized_signals = signals.div(abs_sum, axis=0)
        
        # Apply position smoothing (EMA)
        smoothed_signals = normalized_signals.copy() * 0
        current_weights = {col: 0.0 for col in prices.columns}
        
        for i in range(len(normalized_signals)):
            for col in prices.columns:
                target = normalized_signals[col].iloc[i]
                current = current_weights[col]
                new_weight = self.smoothing_factor * target + (1 - self.smoothing_factor) * current
                smoothed_signals[col].iloc[i] = new_weight
                current_weights[col] = new_weight
        
        # Calculate portfolio returns
        portfolio_returns = (smoothed_signals.shift(1) * returns).sum(axis=1)
        
        # Calculate turnover and transaction costs
        turnover = smoothed_signals.diff().abs().sum(axis=1)
        costs = turnover * self.transaction_cost_pct
        net_returns = portfolio_returns - costs
        
        # Calculate metrics
        return self._calculate_metrics(net_returns, smoothed_signals)
    
    def _calculate_metrics(self, 
                           net_returns: pd.Series,
                           signals: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        
        # Handle empty or constant returns
        if net_returns.empty or net_returns.std() == 0:
            return self._empty_result()
        
        # Cumulative returns
        cumulative = (1 + net_returns).cumprod()
        
        # Total return
        total_return = cumulative.iloc[-1] - 1
        
        # CAGR
        n_years = len(net_returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 and total_return > -1 else 0
        
        # Volatility (annualized)
        daily_vol = net_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (net_returns.mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
        
        # Maximum Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win Rate (on days with position)
        active_days = (signals.abs().sum(axis=1) > 0.01).shift(1).fillna(False)
        active_returns = net_returns[active_days]
        win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0
        
        # Average Turnover
        avg_turnover = signals.diff().abs().sum(axis=1).mean()
        
        # Sortino Ratio (using downside deviation)
        downside_returns = net_returns[net_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (net_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'volatility': annual_vol,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_turnover': avg_turnover,
            'sortino': sortino,
            'calmar': calmar,
            'equity_curve': cumulative * self.initial_capital
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for failed backtests."""
        return {
            'total_return': 0,
            'cagr': 0,
            'sharpe': 0,
            'volatility': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'avg_turnover': 0,
            'sortino': 0,
            'calmar': 0,
            'equity_curve': pd.Series()
        }


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(assets: List[str], timeframe: str = "10y") -> pd.DataFrame:
    """
    Fetch market data for backtesting.
    
    Falls back to synthetic data if API fails.
    """
    if HAS_EXTERNAL_DATA:
        try:
            print(f"📊 Fetching {timeframe} of market data for {len(assets)} assets...")
            prices = fetch_market_data_pool(assets, timeframe=timeframe)
            if prices is not None and not prices.empty and len(prices) > 252:
                print(f"✅ Loaded {len(prices)} days of real market data")
                return prices
        except Exception as e:
            print(f"⚠️ API error: {e}")
    
    # Fallback to synthetic data with realistic market regimes
    print("⚠️ Using synthetic data with market regime simulation...")
    return generate_synthetic_data(n_days=2520, n_assets=len(assets))


def generate_synthetic_data(n_days: int = 2520, n_assets: int = 10) -> pd.DataFrame:
    """
    Generate synthetic price data with realistic market regimes.
    
    Simulates:
    - Bull markets (positive drift)
    - Bear markets (negative drift)
    - High volatility periods
    - Correlation structure
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2016-01-01', periods=n_days, freq='B')
    assets = [f'Asset_{i}' for i in range(n_assets)]
    
    # Create prices with regime changes
    prices = np.zeros((n_days, n_assets))
    prices[0] = 100
    
    for t in range(1, n_days):
        # Define market regime based on time
        year_progress = (t % 252) / 252
        cycle_position = t / n_days
        
        # Multi-regime simulation
        if t < n_days * 0.15:
            drift = 0.0006  # Bull
            vol = 0.012
        elif t < n_days * 0.25:
            drift = -0.0008  # Bear (2018 correction style)
            vol = 0.018
        elif t < n_days * 0.40:
            drift = 0.0007  # Bull
            vol = 0.011
        elif t < n_days * 0.45:
            drift = -0.0025  # Sharp bear (COVID style crash)
            vol = 0.035
        elif t < n_days * 0.55:
            drift = 0.0015  # Strong recovery
            vol = 0.020
        elif t < n_days * 0.75:
            drift = 0.0005  # Bull
            vol = 0.013
        elif t < n_days * 0.85:
            drift = -0.0006  # Bear (2022 style)
            vol = 0.016
        else:
            drift = 0.0004  # Recovery
            vol = 0.012
        
        # Generate correlated returns
        market_factor = np.random.randn() * vol
        
        for j in range(n_assets):
            # Each asset has different beta and idiosyncratic risk
            beta = 0.5 + 0.7 * (j / n_assets)
            idio = np.random.randn() * vol * 0.5
            
            ret = drift * beta + market_factor * beta + idio
            prices[t, j] = prices[t-1, j] * (1 + ret)
    
    return pd.DataFrame(prices, index=dates, columns=assets)


def generate_synthetic_vix(prices: pd.DataFrame) -> pd.Series:
    """
    Generate synthetic VIX based on realized volatility.
    
    VIX typically trades at a premium to realized vol.
    """
    returns = prices.mean(axis=1).pct_change()
    realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100
    
    # VIX = realized vol * 1.2 + noise
    vix = realized_vol * 1.2 + np.random.randn(len(prices)) * 2
    vix = vix.clip(10, 80)  # Realistic VIX range
    
    return vix.fillna(15)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_comprehensive_backtest():
    """
    Run comprehensive backtest across all strategy variants.
    
    Generates:
    - Performance table (Table 1 replication)
    - Equity curves chart
    - Correlation analysis
    """
    print("=" * 70)
    print("   ALL-WEATHER STRATEGY COMPREHENSIVE BACKTEST")
    print("   Testing & Validating Table 1 Performance Claims")
    print("=" * 70)
    
    # Define assets
    assets = [
        "S&P 500", "Nasdaq 100", "Russell 2000", "Dow Jones",
        "XLF (Financials)", "XLK (Technology)", "XLE (Energy)",
        "XLV (Health Care)", "GLD (Gold ETF)", "TLT (20Y+ Treasury)"
    ]
    
    # Fetch data
    prices = fetch_data(assets, timeframe="10y")
    vix = generate_synthetic_vix(prices)
    
    print(f"\n📅 Backtest Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"📊 Assets: {len(prices.columns)}")
    print(f"📈 Total Days: {len(prices)}")
    
    # Define strategies to test
    strategies = {
        "1. Baseline EMS (Lasso+Ridge+EN)": strategy_baseline_ems,
        "2. + Trend Following Filter": strategy_ems_with_trend,
        "3. + Risk Parity Weighting": strategy_ems_with_risk_parity,
        "4. + Value & Momentum Combo": strategy_ems_with_value_momentum,
        "5. + Trend + Risk Parity": strategy_ems_trend_risk_parity,
        "6. Stat Arb Overlay (standalone)": strategy_stat_arb_only,
        "7. VRP Overlay (standalone)": lambda p: strategy_vrp_only(p, vix),
        "8. FULL All-Weather Ensemble": lambda p: strategy_all_weather(p, vix),
    }
    
    # Run backtests
    print("\n🚀 Running backtests...")
    print("-" * 70)
    
    backtester = AllWeatherBacktester(
        initial_capital=10000.0,
        transaction_cost_pct=0.001,
        smoothing_factor=0.3
    )
    
    results = {}
    for name, func in strategies.items():
        print(f"   Testing {name}...", end=" ", flush=True)
        try:
            result = backtester.run(prices, func, vix=vix)
            results[name] = result
            if result['sharpe'] != 0:
                print(f"Sharpe: {result['sharpe']:.2f}")
            else:
                print("⚠️ No trades")
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[name] = backtester._empty_result()
    
    # Display results table
    print("\n" + "=" * 70)
    print("   TABLE 1: PERFORMANCE OF STRATEGY VARIANTS")
    print("=" * 70)
    
    print(f"\n{'Strategy Variant':<40} {'Sharpe':>8} {'CAGR':>10} {'Vol':>8} {'MaxDD':>10}")
    print("-" * 80)
    
    # Get baseline for comparison
    baseline_sharpe = results.get("1. Baseline EMS (Lasso+Ridge+EN)", {}).get('sharpe', 0)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v['sharpe'] != 0 or v.get('equity_curve') is not None],
        key=lambda x: x[1]['sharpe'],
        reverse=True
    )
    
    for name, r in sorted_results:
        sharpe = r['sharpe']
        cagr = r['cagr']
        vol = r['volatility']
        max_dd = r['max_drawdown']
        
        # Mark improvements over baseline
        improvement = sharpe > baseline_sharpe and "Baseline" not in name
        marker = "⭐" if improvement else "  "
        
        print(f"{marker}{name:<38} {sharpe:>8.2f} {cagr:>9.1%} {vol:>7.1%} {max_dd:>9.1%}")
    
    # Analysis summary
    print("\n" + "=" * 70)
    print("   ANALYSIS SUMMARY")
    print("=" * 70)
    
    if sorted_results:
        best_name, best = sorted_results[0]
        
        print(f"\n🏆 BEST PERFORMING: {best_name}")
        print(f"   Sharpe:       {best['sharpe']:.2f}")
        print(f"   CAGR:         {best['cagr']:.1%}")
        print(f"   Max Drawdown: {best['max_drawdown']:.1%}")
        print(f"   Sortino:      {best['sortino']:.2f}")
        
        if baseline_sharpe != 0:
            improvement_pct = ((best['sharpe'] / baseline_sharpe) - 1) * 100
            print(f"\n📈 Sharpe Improvement vs Baseline: {improvement_pct:+.1f}%")
        
        # Trend filter specific analysis
        trend_result = results.get("2. + Trend Following Filter", {})
        if trend_result.get('sharpe', 0) != 0:
            print(f"\n📊 Trend Filter Impact:")
            print(f"   Sharpe: {baseline_sharpe:.2f} → {trend_result['sharpe']:.2f}")
            print(f"   MaxDD:  {results['1. Baseline EMS (Lasso+Ridge+EN)'].get('max_drawdown', 0):.1%} → {trend_result['max_drawdown']:.1%}")
            
            dd_improvement = (1 - trend_result['max_drawdown'] / results['1. Baseline EMS (Lasso+Ridge+EN)'].get('max_drawdown', -0.01)) * 100
            print(f"   Drawdown Reduction: {dd_improvement:.0f}%")
    
    # Correlation analysis
    print("\n" + "-" * 70)
    print("   STRATEGY CORRELATION ANALYSIS")
    print("-" * 70)
    
    # Extract equity curves for correlation
    curves = {}
    for name, r in results.items():
        if isinstance(r.get('equity_curve'), pd.Series) and len(r['equity_curve']) > 100:
            curves[name[:25]] = r['equity_curve']
    
    if len(curves) >= 3:
        curves_df = pd.DataFrame(curves)
        returns_df = curves_df.pct_change().dropna()
        corr = returns_df.corr()
        
        print("\n(Lower correlation = better diversification)")
        print()
        
        # Show correlations with baseline
        baseline_col = "1. Baseline EMS (Lasso+Ri"
        if baseline_col in corr.columns:
            for col in sorted(corr.columns):
                if col != baseline_col:
                    c = corr.loc[baseline_col, col]
                    div_quality = "🟢 Good" if c < 0.5 else "🟡 Moderate" if c < 0.75 else "🔴 High"
                    print(f"   {col:<25}: {c:.2f} {div_quality}")
    
    # Generate charts
    print("\n" + "-" * 70)
    print("   GENERATING OUTPUT FILES")
    print("-" * 70)
    
    generate_equity_chart(results)
    generate_metrics_csv(results)
    
    print("\n" + "=" * 70)
    print("   BACKTEST COMPLETE")
    print("=" * 70)
    
    return results


def generate_equity_chart(results: Dict[str, Dict]):
    """Generate and save equity curves chart."""
    try:
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, (name, r) in enumerate(results.items()):
            if isinstance(r.get('equity_curve'), pd.Series) and len(r['equity_curve']) > 0:
                curve = r['equity_curve']
                # Shorten name for legend
                short_name = name.split('.')[1].strip()[:30] if '.' in name else name[:30]
                
                # Make baseline dashed, others solid
                ls = '--' if 'Baseline' in name else '-'
                lw = 2 if 'All-Weather' in name or 'Baseline' in name else 1.5
                
                plt.plot(curve.index, curve.values, label=short_name, 
                        color=colors[i], linestyle=ls, linewidth=lw)
        
        plt.title('All-Weather Strategy Equity Curves', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('all_weather_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Saved: all_weather_results.png")
        
    except Exception as e:
        print(f"⚠️ Chart generation failed: {e}")


def generate_metrics_csv(results: Dict[str, Dict]):
    """Generate and save metrics to CSV."""
    try:
        rows = []
        for name, r in results.items():
            rows.append({
                'Strategy': name,
                'Sharpe': r['sharpe'],
                'CAGR': r['cagr'],
                'Volatility': r['volatility'],
                'Max_Drawdown': r['max_drawdown'],
                'Win_Rate': r['win_rate'],
                'Sortino': r['sortino'],
                'Calmar': r['calmar'],
                'Avg_Turnover': r['avg_turnover']
            })
        
        df = pd.DataFrame(rows)
        df.to_csv('all_weather_metrics.csv', index=False)
        
        print("✅ Saved: all_weather_metrics.csv")
        
    except Exception as e:
        print(f"⚠️ CSV generation failed: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_comprehensive_backtest()
