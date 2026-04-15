"""
Strategy Combination Analysis

Tests combining the existing Lasso Momentum strategy with proven complementary strategies
to see if we can achieve higher Sharpe ratios.

Strategies tested:
1. Lasso Momentum (Baseline) - Your current strategy
2. Time-Series Momentum (Trend Following) - Classic CTA approach
3. Mean Reversion - Trade against short-term extremes
4. Risk Parity Weighting - Equal risk contribution
5. Lasso + Mean Reversion Ensemble - Combine signals
6. Lasso + Trend Following Ensemble - Combine signals
"""
import pandas as pd
import numpy as np
from typing import Dict, Callable, List
import warnings
warnings.filterwarnings('ignore')

from external_data import fetch_market_data_pool

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

def strategy_lasso_momentum(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Simplified Lasso-style momentum strategy.
    Uses multiple return horizons + volatility to generate signals.
    """
    returns = prices.pct_change()
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float).fillna(0)
    
    for col in prices.columns:
        # Multi-horizon momentum features
        r1 = prices[col].pct_change(1).fillna(0)
        r5 = prices[col].pct_change(5).fillna(0)
        r20 = prices[col].pct_change(20).fillna(0)
        
        # Volatility (inverse weight)
        vol = returns[col].rolling(20).std().fillna(0.01)
        inv_vol = 1 / (vol + 0.01)
        
        # Combined momentum signal (weighted by inverse vol)
        raw_signal = (0.3 * r1 + 0.3 * r5 + 0.4 * r20) * inv_vol
        
        # Normalize - use expanding mean after warmup
        norm_factor = raw_signal.abs().rolling(60, min_periods=20).mean().fillna(0.01) + 0.001
        signals[col] = (raw_signal / norm_factor).fillna(0)
    
    # Cap signals
    signals = signals.clip(-2, 2)
    return signals


def strategy_trend_following(prices: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    """
    Classic time-series momentum / trend following.
    Long when price > moving average, Short when below.
    Uses dual moving average crossover.
    """
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float).fillna(0)
    
    for col in prices.columns:
        ma_fast = prices[col].rolling(fast).mean()
        ma_slow = prices[col].rolling(slow).mean()
        
        # Trend signal: +1 when fast > slow, -1 when fast < slow
        raw_signal = np.sign(ma_fast - ma_slow)
        
        # Strength: how far is price from slow MA (normalized)
        strength = (prices[col] - ma_slow) / (ma_slow * 0.01 + 0.001)
        strength = strength.clip(-2, 2)
        
        signals[col] = raw_signal * (0.5 + 0.5 * strength.abs().clip(0, 1))
    
    return signals


def strategy_mean_reversion(prices: pd.DataFrame, lookback: int = 20, threshold: float = 1.5) -> pd.DataFrame:
    """
    Mean reversion strategy.
    Buy when price is oversold (below lower band), Sell when overbought.
    """
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float).fillna(0)
    
    for col in prices.columns:
        ma = prices[col].rolling(lookback).mean()
        std = prices[col].rolling(lookback).std()
        
        # Z-score
        z = (prices[col] - ma) / (std + 0.001)
        
        # Mean reversion: buy oversold, sell overbought (opposite of momentum)
        signals[col] = -z.clip(-3, 3) / 3  # Normalize to [-1, 1]
    
    return signals


def strategy_risk_parity(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Risk parity weighting - equal volatility contribution.
    Always long, but sized by inverse volatility.
    """
    returns = prices.pct_change()
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float).fillna(0)
    
    for i in range(lookback, len(prices)):
        window_returns = returns.iloc[i-lookback:i]
        vols = window_returns.std()
        
        # Inverse volatility weights
        inv_vols = 1 / (vols + 0.001)
        weights = inv_vols / inv_vols.sum()
        
        signals.iloc[i] = weights.values
    
    return signals


def ensemble_lasso_meanrev(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble: 70% Lasso Momentum + 30% Mean Reversion.
    Momentum for trends, Mean reversion for extremes.
    """
    lasso = strategy_lasso_momentum(prices)
    mr = strategy_mean_reversion(prices)
    
    return 0.7 * lasso + 0.3 * mr


def ensemble_lasso_trend(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble: 60% Lasso Momentum + 40% Trend Following.
    Double momentum exposure for stronger trends.
    """
    lasso = strategy_lasso_momentum(prices)
    trend = strategy_trend_following(prices)
    
    return 0.6 * lasso + 0.4 * trend


def ensemble_all_weather(prices: pd.DataFrame) -> pd.DataFrame:
    """
    All-Weather Ensemble: Combine all strategies.
    40% Lasso + 30% Trend + 20% Mean Reversion + 10% Risk Parity
    """
    lasso = strategy_lasso_momentum(prices)
    trend = strategy_trend_following(prices)
    mr = strategy_mean_reversion(prices)
    rp = strategy_risk_parity(prices)
    
    return 0.40 * lasso + 0.30 * trend + 0.20 * mr + 0.10 * rp


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest_strategy(
    prices: pd.DataFrame, 
    signal_func: Callable,
    transaction_cost: float = 0.001,
    smoothing: float = 0.3
) -> Dict:
    """
    Backtest a strategy with transaction costs and position smoothing.
    """
    returns = prices.pct_change()
    signals = signal_func(prices)
    
    # Skip warmup period
    warmup = 65
    signals = signals.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    # Normalize signals to sum of abs = 1 (fully invested)
    abs_sum = signals.abs().sum(axis=1)
    abs_sum = abs_sum.replace(0, 1)  # Avoid division by zero
    normalized_signals = signals.div(abs_sum, axis=0)
    
    # Apply position smoothing (EMA)
    smoothed_signals = normalized_signals.copy() * 0
    current_weights = {col: 0.0 for col in prices.columns}
    
    for i in range(len(normalized_signals)):
        for col in prices.columns:
            target = normalized_signals[col].iloc[i]
            current = current_weights[col]
            new_weight = smoothing * target + (1 - smoothing) * current
            smoothed_signals[col].iloc[i] = new_weight
            current_weights[col] = new_weight
    
    # Calculate portfolio returns
    portfolio_returns = (smoothed_signals.shift(1) * returns).sum(axis=1)
    
    # Calculate turnover and transaction costs
    turnover = smoothed_signals.diff().abs().sum(axis=1)
    costs = turnover * transaction_cost
    net_returns = portfolio_returns - costs
    
    # Performance metrics
    total_return = (1 + net_returns).prod() - 1
    n_years = len(net_returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    daily_vol = net_returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    
    sharpe = (net_returns.mean() / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
    
    # Drawdown
    cumulative = (1 + net_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate (on days with position)
    active_days = (smoothed_signals.abs().sum(axis=1) > 0.01).shift(1).fillna(False)
    active_returns = net_returns[active_days]
    win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'volatility': annual_vol,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'avg_turnover': turnover.mean(),
        'equity_curve': cumulative
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("   STRATEGY COMBINATION ANALYSIS")
    print("   Finding the best ensemble to maximize Sharpe Ratio")
    print("=" * 70)
    
    # Fetch market data
    print("\n📊 Fetching 2 years of market data...")
    assets = [
        "S&P 500", "Nasdaq 100", "Russell 2000", "Dow Jones",
        "XLF (Financials)", "XLK (Technology)", "XLE (Energy)", 
        "XLV (Health Care)", "GLD (Gold ETF)", "TLT (20Y+ Treasury)"
    ]
    
    try:
        prices = fetch_market_data_pool(assets, timeframe="2y")
        if prices is None or prices.empty:
            raise ValueError("Empty data")
        print(f"✅ Loaded {len(prices)} days for {len(prices.columns)} assets")
    except Exception as e:
        print(f"⚠️ Using synthetic data: {e}")
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='B')
        assets = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLV', 'GLD', 'TLT']
        # Create correlated returns
        returns = np.random.randn(500, len(assets)) * 0.012 + 0.0003
        # Add some correlation structure
        market_factor = np.random.randn(500) * 0.008
        for i in range(len(assets)):
            beta = 0.3 + 0.7 * np.random.rand()
            returns[:, i] += market_factor * beta
        prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), 
                             index=dates, columns=assets)
    
    # Define strategies to test
    strategies = {
        "1. Lasso Momentum (Baseline)": strategy_lasso_momentum,
        "2. Trend Following": strategy_trend_following,
        "3. Mean Reversion": strategy_mean_reversion,
        "4. Risk Parity": strategy_risk_parity,
        "5. Lasso + Mean Reversion": ensemble_lasso_meanrev,
        "6. Lasso + Trend Following": ensemble_lasso_trend,
        "7. All-Weather Ensemble": ensemble_all_weather,
    }
    
    # Run backtests
    print("\n🚀 Running backtests...")
    print("-" * 70)
    
    results = {}
    for name, func in strategies.items():
        print(f"   Testing {name}...", end=" ", flush=True)
        try:
            result = backtest_strategy(prices, func)
            results[name] = result
            print(f"Sharpe: {result['sharpe']:.2f}")
        except Exception as e:
            print(f"Failed: {e}")
            results[name] = None
    
    # Display results
    print("\n" + "=" * 70)
    print("   RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<35} {'Sharpe':>8} {'CAGR':>10} {'Vol':>8} {'MaxDD':>10} {'WinRate':>8}")
    print("-" * 85)
    
    baseline_sharpe = results["1. Lasso Momentum (Baseline)"]['sharpe'] if results.get("1. Lasso Momentum (Baseline)") else 0
    
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1]['sharpe'],
        reverse=True
    )
    
    for name, r in sorted_results:
        improvement = ((r['sharpe'] / baseline_sharpe - 1) * 100) if baseline_sharpe > 0 else 0
        marker = "⭐" if r['sharpe'] > baseline_sharpe and "Baseline" not in name else "  "
        
        print(f"{marker}{name:<33} {r['sharpe']:>8.2f} {r['cagr']:>9.1%} {r['volatility']:>7.1%} {r['max_drawdown']:>9.1%} {r['win_rate']:>7.1%}")
    
    # Find best combination
    best_name, best_result = sorted_results[0]
    
    print("\n" + "=" * 70)
    print("   RECOMMENDATION")
    print("=" * 70)
    
    if best_result['sharpe'] > baseline_sharpe:
        improvement = (best_result['sharpe'] / baseline_sharpe - 1) * 100
        print(f"\n🏆 BEST STRATEGY: {best_name}")
        print(f"   Sharpe Improvement: +{improvement:.1f}% over baseline")
        print(f"   Sharpe: {best_result['sharpe']:.2f} vs {baseline_sharpe:.2f} baseline")
        print(f"   CAGR: {best_result['cagr']:.1%}")
        print(f"   Max Drawdown: {best_result['max_drawdown']:.1%}")
    else:
        print(f"\n✅ Your current Lasso Momentum strategy is already optimal!")
        print(f"   Sharpe: {baseline_sharpe:.2f}")
    
    # Correlation analysis
    print("\n" + "-" * 70)
    print("   STRATEGY CORRELATION ANALYSIS")
    print("-" * 70)
    
    if len(sorted_results) >= 3:
        # Get equity curves
        curves = {name: r['equity_curve'] for name, r in sorted_results[:5] if r is not None}
        
        if curves:
            curves_df = pd.DataFrame(curves)
            returns_df = curves_df.pct_change().dropna()
            corr = returns_df.corr()
            
            print("\nStrategy Return Correlations:")
            print("(Lower correlation = better diversification)")
            print()
            
            # Print correlation of each strategy with Lasso baseline
            baseline_col = "1. Lasso Momentum (Baseline)"
            if baseline_col in corr.columns:
                for col in corr.columns:
                    if col != baseline_col:
                        c = corr.loc[baseline_col, col]
                        diversification = "🟢 Good" if c < 0.7 else "🟡 Moderate" if c < 0.85 else "🔴 High"
                        print(f"   {col[:35]:<35}: {c:.2f} {diversification}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
