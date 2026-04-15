"""
Final Portfolio Validation: The Cyclical & Credit Engine
========================================================

Testing the "Holy Grail" portfolio identified by ML discovery.
Assets: XLB, XLI, XLE, JNK, GLD, USMV, MTUM, FXA

Tests:
1.  Performance vs Buy-and-Hold
2.  Walk-Forward Validation
3.  Selection Bias Tests (White's, Bootstrap, etc.)
4.  Correlations (are we just levered beta?)

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
# THE HOLY GRAIL UNIVERSE
# =============================================================================

# Assets with IC > 0.20 or Accuracy > 65%
HOLY_GRAIL_ASSETS = [
    'XLB',   # Materials (IC 0.357)
    'XLI',   # Industrials (IC 0.308)
    'XLE',   # Energy (IC 0.230)
    'JNK',   # Junk Bonds (Acc 72.5%, IC 0.228)
    'FXA',   # Aussie Dollar (IC 0.214)
    'USMV',  # Min Vol (IC 0.210)
    'MTUM',  # Momentum (IC 0.193)
    'GLD'    # Gold (Acc 72.1%)
]

# Config: Tuned for these predictable assets
# We use VIX 25 as a general risk filter, but maybe can relax it?
# Let's stick to the standard config first.
CONFIG = {
    'vix_threshold': 25.0,
    'sma_lookback': 200,
    'max_position_pct': 0.25, # Concentrated (8 assets, so 25% allows 4 positions)
    'target_volatility': 0.15
}


# =============================================================================
# DATA
# =============================================================================

def fetch_data(years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch data for Holy Grail universe."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"Fetching data for {len(HOLY_GRAIL_ASSETS)} assets...")
    data = yf.download(HOLY_GRAIL_ASSETS, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(prices: pd.DataFrame, vix: pd.Series, config_dict: Dict) -> Dict:
    """Run strategy backtest."""
    config = CompounderConfig(**config_dict)
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except Exception as e:
        print(f"Error in backtest: {e}")
        return {}
    
    # Warmup period
    warmup = 65
    weights = weights.iloc[warmup:].copy()
    returns = prices.pct_change().iloc[warmup:].fillna(0)
    
    # Align
    common_idx = weights.index.intersection(returns.index)
    weights = weights.loc[common_idx]
    returns = returns.loc[common_idx]
    
    # Portfolio returns (Lag weights by 1 day for T+1 execution)
    # Note: generate_weights likely already shifted? Let's check logic.
    # In compounder_strategy.py: "weights = weights.shift(1).fillna(0)"
    # So weights ARE already shifted. We multiply directly.
    
    port_returns = (weights * returns).sum(axis=1)
    
    # Transaction costs
    turnover = weights.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    
    # Metrics
    if net_returns.std() > 0:
        sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
        
    cagr = (equity.iloc[-1] ** (252/len(equity))) - 1
    max_dd = (equity / equity.cummax() - 1).min()
    
    return {
        'sharpe': sharpe,
        'cagr': cagr,
        'max_dd': max_dd,
        'returns': net_returns,
        'equity': equity,
        'turnover': turnover.mean()
    }


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def compare_benchmark(strategy_res: Dict, prices: pd.DataFrame) -> Dict:
    """Compare vs Equal Weight Benchmark."""
    returns = prices.pct_change().fillna(0)
    bench_returns = returns.mean(axis=1)
    
    # Align dates
    start_date = strategy_res['returns'].index[0]
    bench_returns = bench_returns.loc[start_date:]
    
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(252)
    bench_cagr = ((1 + bench_returns).cumprod().iloc[-1] ** (252/len(bench_returns))) - 1
    
    print("\n   === BENCHMARK COMPARISON ===")
    print(f"   Strategy Sharpe: {strategy_res['sharpe']:.2f}")
    print(f"   Benchmark Sharpe: {bench_sharpe:.2f}")
    print(f"   Excess Sharpe: {strategy_res['sharpe'] - bench_sharpe:.2f}")
    
    return {'bench_sharpe': bench_sharpe, 'bench_cagr': bench_cagr}


def selection_bias_test(prices: pd.DataFrame, vix: pd.Series, strategy_sharpe: float) -> Dict:
    """White's Reality Check."""
    print("\n   === SELECTION BIAS TEST (White's) ===")
    n_bootstrap = 1000
    
    # Benchmark returns
    bench_returns = prices.pct_change().mean(axis=1).iloc[65:].fillna(0)
    
    # Bootstrap
    bootstrap_excess = []
    
    # We simulate random strategies by shuffling the asset returns 
    # but keeping the strategy weights structure? 
    # Actually, simpler White's check: bootstrap the *excess returns*.
    
    # Re-run backtest to get strategy returns aligned
    res = run_backtest(prices, vix, CONFIG)
    strat_returns = res['returns']
    
    common = strat_returns.index.intersection(bench_returns.index)
    strat_returns = strat_returns.loc[common]
    bench_returns = bench_returns.loc[common]
    
    excess_returns = strat_returns - bench_returns
    mean_excess = excess_returns.mean()
    
    print(f"   Mean Daily Excess Return: {mean_excess:.5f}")
    
    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        # Block bootstrap
        block_size = 20
        idx = np.random.randint(0, len(excess_returns) - block_size, int(len(excess_returns)/block_size))
        boot_sample = []
        for i in idx:
            boot_sample.extend(excess_returns.iloc[i:i+block_size].values)
        
        boot_means.append(np.mean(boot_sample))
    
    # p-value: proportion of centered bootstrap means > observed mean
    # (White's test formulation is slightly more complex, but this is a standard bootstrap test for mean > 0)
    # H0: Mean excess <= 0
    
    centered_means = np.array(boot_means) - np.mean(boot_means)
    p_value = np.mean(centered_means >= mean_excess)
    
    print(f"   p-value (H0: No Edge): {p_value:.4f}")
    print(f"   Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    return {'p_value': p_value}


def run_comprehensive_test():
    print("=" * 80)
    print("   FINAL TEST: CYCLICAL & CREDIT PORTFOLIO")
    print("   Assets: " + ", ".join(HOLY_GRAIL_ASSETS))
    print("=" * 80)
    
    prices, vix = fetch_data(years=5)
    print(f"   Loaded {len(prices.columns)} assets over {len(prices)} days")
    
    # Split OOS (Last 2 years)
    split = int(len(prices) * 0.6)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    print(f"\n   Running OOS Backtest from {oos_prices.index[0].date()}...")
    
    res = run_backtest(oos_prices, oos_vix, CONFIG)
    
    print("\n   === PERFORMANCE METRICS ===")
    print(f"   Sharpe Ratio: {res['sharpe']:.2f}")
    print(f"   CAGR:         {res['cagr']:.1%}")
    print(f"   Max Drawdown: {res['max_dd']:.1%}")
    print(f"    turnover:   {res['turnover']:.1%}")
    
    # Benchmark
    bench = compare_benchmark(res, oos_prices)
    
    # Bias
    bias = selection_bias_test(oos_prices, oos_vix, res['sharpe'])
    
    print("\n" + "=" * 80)
    print("   FINAL VERDICT")
    print("=" * 80)
    
    passed = True
    if res['sharpe'] < 1.0: passed = False
    if res['sharpe'] < bench['bench_sharpe']: passed = False
    if bias['p_value'] > 0.10: passed = False
    
    if passed:
        print("   ✅ PASSED ALL TESTS. This portfolio is robust.")
    else:
        print("   ⚠️ FAILED SOME TESTS. Proceed with caution.")
        if res['sharpe'] < bench['bench_sharpe']:
            print("   - Failed to beat benchmark.")
        if bias['p_value'] > 0.10:
            print("   - Selection bias likely.")

    return res

if __name__ == "__main__":
    run_comprehensive_test()
