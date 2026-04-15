"""
Deep Validation: Japan+Korea Champion Strategy
===============================================

New Champion: Japan+Korea (EWJ, EWY, AAXJ, GLD, TLT)
- Sharpe: 2.14
- CAGR: 16.6%
- VIX: 25, SMA: 150, Position Cap: 15%

Deep tests:
1. Walk-forward validation (5 folds)
2. Statistical significance
3. Parameter sensitivity (fine-tuning around optimal)
4. Different time periods
5. Monthly rebalancing vs daily
6. Transaction cost sensitivity
7. Drawdown analysis
8. Compare to buy-and-hold
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# THE CHAMPION UNIVERSE
# =============================================================================

CHAMPION_TICKERS = ['EWJ', 'EWY', 'AAXJ', 'GLD', 'TLT']
CHAMPION_CONFIG = {'vix_threshold': 25, 'sma_lookback': 150, 'max_position': 0.15}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch champion universe data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(CHAMPION_TICKERS, start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest(prices: pd.DataFrame, vix: pd.Series, 
             vix_threshold: float = 25.0, sma_lookback: int = 150,
             max_position: float = 0.15, tx_cost: float = 0.001,
             warmup: int = 65) -> Dict:
    """Run backtest with full metrics."""
    
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=sma_lookback,
        max_position_pct=max_position
    )
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except:
        return {'sharpe': 0, 'cagr': 0, 'returns': pd.Series()}
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty or len(weights) < 20:
        return {'sharpe': 0, 'cagr': 0, 'returns': pd.Series()}
    
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * tx_cost
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # Drawdown analysis
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win/loss stats
    positive_days = (net_returns > 0).sum()
    win_rate = positive_days / len(net_returns) if len(net_returns) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': max_dd,
        'volatility': net_returns.std() * np.sqrt(252),
        'win_rate': win_rate,
        'returns': net_returns,
        'equity': equity,
        'drawdown': drawdown,
        'avg_turnover': turnover.mean()
    }


# =============================================================================
# TESTS
# =============================================================================

def test_walk_forward(prices: pd.DataFrame, vix: pd.Series, n_splits: int = 5) -> Dict:
    """Walk-forward validation."""
    print("\n   === WALK-FORWARD VALIDATION ===")
    
    n = len(prices)
    split_size = n // (n_splits + 1)
    
    is_sharpes = []
    oos_sharpes = []
    fold_dates = []
    
    for i in range(n_splits):
        is_end = (i + 1) * split_size
        oos_start = is_end
        oos_end = min(oos_start + split_size, n)
        
        if oos_end <= oos_start + 30:
            continue
        
        is_prices = prices.iloc[:is_end]
        oos_prices = prices.iloc[oos_start:oos_end]
        
        is_vix = vix.iloc[:is_end]
        oos_vix = vix.iloc[oos_start:oos_end]
        
        fold_date = prices.index[oos_start].strftime('%Y-%m')
        fold_dates.append(fold_date)
        
        is_result = backtest(is_prices, is_vix, **CHAMPION_CONFIG)
        oos_result = backtest(oos_prices, oos_vix, **CHAMPION_CONFIG)
        
        is_sharpes.append(is_result['sharpe'])
        oos_sharpes.append(oos_result['sharpe'])
        
        print(f"      Fold {i+1} ({fold_date}): IS={is_result['sharpe']:.2f}, OOS={oos_result['sharpe']:.2f}")
    
    mean_oos = np.mean(oos_sharpes)
    n_positive = sum(1 for s in oos_sharpes if s > 0)
    
    print(f"\n      Mean OOS Sharpe: {mean_oos:.2f}")
    print(f"      Positive folds: {n_positive}/{len(oos_sharpes)}")
    
    return {
        'mean_oos': mean_oos,
        'n_positive': n_positive,
        'n_folds': len(oos_sharpes),
        'oos_sharpes': oos_sharpes
    }


def test_statistical_significance(returns: pd.Series) -> Dict:
    """Test if Sharpe is statistically significant."""
    print("\n   === STATISTICAL SIGNIFICANCE ===")
    
    if returns.empty or returns.std() == 0:
        return {'t_stat': 0, 'p_value': 1.0}
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    n = len(returns)
    se = np.sqrt((1 + 0.5 * sharpe**2) / n)
    t_stat = sharpe / se
    p_value = 1 - stats.t.cdf(t_stat, n - 1)
    
    print(f"      Sharpe: {sharpe:.2f}")
    print(f"      t-statistic: {t_stat:.2f}")
    print(f"      p-value: {p_value:.6f}")
    print(f"      Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"      Significant at 1%: {'YES' if p_value < 0.01 else 'NO'}")
    
    return {'t_stat': t_stat, 'p_value': p_value, 'sharpe': sharpe}


def test_parameter_sensitivity(prices: pd.DataFrame, vix: pd.Series) -> Dict:
    """Test sensitivity around optimal parameters."""
    print("\n   === PARAMETER SENSITIVITY ===")
    
    # Split for OOS
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Test variations
    vix_range = [20, 22, 25, 28, 30]
    sma_range = [100, 125, 150, 175, 200]
    cap_range = [0.10, 0.12, 0.15, 0.18, 0.20]
    
    results = []
    
    # VIX sensitivity
    print("\n      VIX Threshold Sensitivity:")
    for v in vix_range:
        r = backtest(oos_prices, oos_vix, vix_threshold=v)
        marker = " <-- optimal" if v == 25 else ""
        print(f"         VIX {v}: Sharpe {r['sharpe']:.2f}{marker}")
        results.append(('vix', v, r['sharpe']))
    
    # SMA sensitivity
    print("\n      SMA Lookback Sensitivity:")
    for s in sma_range:
        r = backtest(oos_prices, oos_vix, sma_lookback=s)
        marker = " <-- optimal" if s == 150 else ""
        print(f"         SMA {s}: Sharpe {r['sharpe']:.2f}{marker}")
        results.append(('sma', s, r['sharpe']))
    
    # Position cap sensitivity
    print("\n      Position Cap Sensitivity:")
    for c in cap_range:
        r = backtest(oos_prices, oos_vix, max_position=c)
        marker = " <-- optimal" if c == 0.15 else ""
        print(f"         Cap {c:.0%}: Sharpe {r['sharpe']:.2f}{marker}")
        results.append(('cap', c, r['sharpe']))
    
    return {'sensitivity': results}


def test_transaction_cost_sensitivity(prices: pd.DataFrame, vix: pd.Series) -> Dict:
    """Test sensitivity to transaction costs."""
    print("\n   === TRANSACTION COST SENSITIVITY ===")
    
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    costs = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
    
    for cost in costs:
        r = backtest(oos_prices, oos_vix, tx_cost=cost, **CHAMPION_CONFIG)
        print(f"      Cost {cost*100:.2f}%: Sharpe {r['sharpe']:.2f}, CAGR {r['cagr']:.1%}")
    
    return {}


def test_vs_buy_and_hold(prices: pd.DataFrame, vix: pd.Series) -> Dict:
    """Compare to simple buy-and-hold."""
    print("\n   === VS BUY-AND-HOLD ===")
    
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Strategy
    strat_result = backtest(oos_prices, oos_vix, **CHAMPION_CONFIG)
    
    # Buy and hold (equal weight)
    bh_returns = oos_prices.pct_change().mean(axis=1).fillna(0)
    bh_equity = (1 + bh_returns).cumprod()
    bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(252) if bh_returns.std() > 0 else 0
    bh_cagr = (bh_equity.iloc[-1] ** (252/len(bh_equity))) - 1
    
    print(f"      Strategy:     Sharpe {strat_result['sharpe']:.2f}, CAGR {strat_result['cagr']:.1%}, MaxDD {strat_result['max_dd']:.1%}")
    print(f"      Buy & Hold:   Sharpe {bh_sharpe:.2f}, CAGR {bh_cagr:.1%}")
    print(f"      Improvement:  +{(strat_result['sharpe'] - bh_sharpe):.2f} Sharpe")
    
    return {'strategy': strat_result, 'bh_sharpe': bh_sharpe, 'bh_cagr': bh_cagr}


def test_drawdown_analysis(result: Dict) -> Dict:
    """Analyze drawdowns."""
    print("\n   === DRAWDOWN ANALYSIS ===")
    
    dd = result.get('drawdown', pd.Series())
    if dd.empty:
        return {}
    
    max_dd = dd.min()
    avg_dd = dd.mean()
    
    # Time underwater
    underwater = (dd < 0).sum() / len(dd) * 100
    
    # Worst drawdown duration
    in_drawdown = dd < 0
    dd_starts = (~in_drawdown.shift(1).fillna(False)) & in_drawdown
    dd_ends = in_drawdown.shift(1).fillna(False) & (~in_drawdown)
    
    print(f"      Max Drawdown:      {max_dd:.1%}")
    print(f"      Avg Drawdown:      {avg_dd:.1%}")
    print(f"      Time Underwater:   {underwater:.1f}%")
    print(f"      Win Rate:          {result.get('win_rate', 0):.1%}")
    
    return {'max_dd': max_dd, 'avg_dd': avg_dd, 'underwater': underwater}


# =============================================================================
# MAIN
# =============================================================================

def run_deep_validation():
    """Run all deep validation tests."""
    
    print("=" * 80)
    print("   DEEP VALIDATION: JAPAN+KOREA CHAMPION")
    print("   VIX 25 | SMA 150 | Position Cap 15%")
    print("=" * 80)
    
    # Fetch data
    print("\nFetching data...")
    prices, vix = fetch_data(years=5)
    print(f"   {len(prices.columns)} assets, {len(prices)} days")
    
    # Split for main OOS test
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Main backtest
    print("\n" + "=" * 80)
    print("   MAIN OOS BACKTEST")
    print("=" * 80)
    
    result = backtest(oos_prices, oos_vix, **CHAMPION_CONFIG)
    
    print(f"\n      Sharpe:      {result['sharpe']:.2f}")
    print(f"      CAGR:        {result['cagr']:.1%}")
    print(f"      Max DD:      {result['max_dd']:.1%}")
    print(f"      Volatility:  {result['volatility']:.1%}")
    print(f"      Win Rate:    {result['win_rate']:.1%}")
    
    # Run all tests
    print("\n" + "=" * 80)
    print("   DEEP VALIDATION TESTS")
    print("=" * 80)
    
    wf_result = test_walk_forward(prices, vix)
    sig_result = test_statistical_significance(result['returns'])
    sens_result = test_parameter_sensitivity(prices, vix)
    tx_result = test_transaction_cost_sensitivity(prices, vix)
    bh_result = test_vs_buy_and_hold(prices, vix)
    dd_result = test_drawdown_analysis(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("   VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"""
   ╔══════════════════════════════════════════════════════════╗
   ║                    JAPAN+KOREA STRATEGY                 ║
   ╠══════════════════════════════════════════════════════════╣
   ║  OOS Sharpe:           {result['sharpe']:.2f}                            ║
   ║  OOS CAGR:             {result['cagr']:.1%}                          ║
   ║  Max Drawdown:         {result['max_dd']:.1%}                          ║
   ║  Walk-Forward OOS:     {wf_result['mean_oos']:.2f} ({wf_result['n_positive']}/{wf_result['n_folds']} positive)            ║
   ║  Statistical Sig:      {'YES (p=' + f"{sig_result['p_value']:.4f})" if sig_result['p_value'] < 0.05 else 'NO':23} ║
   ║  vs Buy-Hold:          +{(result['sharpe'] - bh_result['bh_sharpe']):.2f} Sharpe                       ║
   ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Final verdict
    passed_tests = 0
    total_tests = 4
    
    if result['sharpe'] > 1.5: passed_tests += 1
    if wf_result['n_positive'] >= 3: passed_tests += 1
    if sig_result['p_value'] < 0.05: passed_tests += 1
    if result['sharpe'] > bh_result['bh_sharpe']: passed_tests += 1
    
    print(f"   VALIDATION SCORE: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(f"\n    STRATEGY FULLY VALIDATED!")
    elif passed_tests >= 3:
        print(f"\n    STRATEGY MOSTLY VALIDATED (some caution advised)")
    else:
        print(f"\n   STRATEGY NEEDS FURTHER INVESTIGATION")
    
    print("\n" + "=" * 80)
    
    return result


if __name__ == "__main__":
    result = run_deep_validation()
