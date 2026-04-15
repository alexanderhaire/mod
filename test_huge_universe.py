"""
Massive Universe Test
=====================

Testing a combined universe of:
1. US Sector ETFs
2. Asian Country ETFs
3. Mega-cap Tech Stocks (US & Asian)
4. Defensive Assets (Gold, Bonds, Trend)

Hypothesis: Does massive diversification combined with the compounder logic provide a true edge?
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
# MASSIVE UNIVERSE DEFINITION
# =============================================================================

# 1. US Broad & Sectors
US_ETFS = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLU', 'XLY']

# 2. Asian / Global
GLOBAL_ETFS = ['EWJ', 'FXI', 'EWY', 'INDA', 'EWT', 'EWH', 'EWS', 'AAXJ', 'EEM', 'EFA']

# 3. Individual Stocks (High Quality / Tech)
STOCKS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSM', 'BABA', 'SONY', 'JPM', 'JNJ', 'V']

# 4. Defensives
DEFENSIVES = ['GLD', 'TLT', 'IEF', 'SHy', 'DBMF'] 

FULL_UNIVERSE = list(set(US_ETFS + GLOBAL_ETFS + STOCKS + DEFENSIVES))


# =============================================================================
# FUNCTIONS
# =============================================================================

def fetch_data(tickers: List[str], years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch data for massive universe."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"Fetching data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    # Drop columns with too much missing data
    prices = prices.dropna(axis=1, thresh=int(len(prices)*0.9))
    prices = prices.ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def run_backtest(prices: pd.DataFrame, vix: pd.Series, 
                 vix_threshold: float = 25.0, sma_lookback: int = 200, 
                 max_position: float = 0.10) -> Dict:
    """Run backtest. Note lower max_position due to larger universe."""
    
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=sma_lookback,
        max_position_pct=max_position
    )
    strategy = CompounderStrategy(config)
    
    print("Generating weights (this may take a moment due to universe size)...")
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except Exception as e:
        print(f"Error generating weights: {e}")
        return {}
    
    returns = prices.pct_change().fillna(0)
    warmup = 65
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return {}
    
    # Normalize weights
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    # Estimate transaction costs (higher for stock turnover)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # Benchmark (Equal Weight of Universe)
    bench_returns = returns.mean(axis=1)
    bench_sharpe = bench_returns.mean() / bench_returns.std() * np.sqrt(252) if bench_returns.std() > 0 else 0
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': ((equity - equity.cummax()) / equity.cummax()).min(),
        'bench_sharpe': bench_sharpe,
        'volatility': net_returns.std() * np.sqrt(252),
        'n_assets': len(prices.columns)
    }


def run_massive_test():
    print("=" * 80)
    print("   MASSIVE UNIVERSE TEST")
    print("   Combining US, Asian, Stocks, and Defensives")
    print("=" * 80)
    
    prices, vix = fetch_data(FULL_UNIVERSE, years=5)
    print(f"   Successfully loaded {len(prices.columns)} assets over {len(prices)} days")
    
    # Run OOS test (last 30%)
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    # Test a few configurations
    configs = [
        # (Name, VIX, SMA, Cap)
        ("Standard (VIX 25, 10% Cap)", 25.0, 200, 0.10),
        ("Defensive (VIX 20, 5% Cap)", 20.0, 200, 0.05),
        ("Aggressive (VIX 30, 20% Cap)", 30.0, 150, 0.20),
    ]
    
    results = []
    
    print("\n   RUNNING BACKTESTS (OOS)...")
    print(f"   {'Configuration':<30} {'Sharpe':>8} {'Bench':>8} {'Edge':>8} {'CAGR':>8} {'MaxDD':>8}")
    print("   " + "-" * 75)
    
    for name, v, s, c in configs:
        res = run_backtest(oos_prices, oos_vix, vix_threshold=v, sma_lookback=s, max_position=c)
        edge = res['sharpe'] - res['bench_sharpe']
        print(f"   {name:<30} {res['sharpe']:>8.2f} {res['bench_sharpe']:>8.2f} {edge:>8.2f} {res['cagr']:>8.1%} {res['max_dd']:>8.1%}")
        results.append((name, res))
        
    print("\n" + "=" * 80)
    
    # Best result analysis
    best_res = max(results, key=lambda x: x[1]['sharpe'])
    print(f"   BEST CONFIG: {best_res[0]}")
    print(f"   Sharpe: {best_res[1]['sharpe']:.2f} vs Benchmark {best_res[1]['bench_sharpe']:.2f}")
    
    if best_res[1]['sharpe'] > best_res[1]['bench_sharpe'] + 0.2:
        print("\n   ✅ SIGNIFICANT OUTPERFORMANCE found with massive universe!")
        print("   The strategy creates value by selecting winners from the large pool.")
    else:
        print("\n   ⚠️ PERFORMANCE IS CLOSE TO BENCHMARK.")
        print("   Even with a massive universe, simply holding everything (Index Fund style)")
        print("   performs similarly to the active strategy.")
        
    return results

if __name__ == "__main__":
    run_massive_test()
