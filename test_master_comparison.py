"""
Master Comparison: All Strategies vs Original PDF
==================================================

The PDF strategy (EMS Compounder) achieved:
- OOS Sharpe: 1.53 (Walk-forward 2 windows)
- CAGR: 67.7% 
- Universe: US stocks (SPY, QQQ, sector ETFs)
- Key config: 20% max position, VIX > 25 OR SPY < 200MA = risk-off

This script compares all our experiments against that baseline.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# UNIVERSES TO COMPARE
# =============================================================================

# Original PDF universe (US ETFs)
ORIGINAL_PDF = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'GLD': 'Gold',
    'TLT': 'Long Treasuries',
}

# Asian ETFs
ASIAN_ETFS = {
    'EWJ': 'Japan', 'FXI': 'China', 'EWY': 'South Korea',
    'INDA': 'India', 'EWT': 'Taiwan', 'EWH': 'Hong Kong',
    'EWS': 'Singapore', 'AAXJ': 'Asia ex-Japan',
    'GLD': 'Gold', 'TLT': 'Long Treasuries'
}

# Best Mix (from our experiments)
BEST_MIX = {
    'TSM': 'TSMC', 'BABA': 'Alibaba', 'SONY': 'Sony', 'SE': 'Sea',
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 'GOOGL': 'Google',
    'GLD': 'Gold', 'TLT': 'Long Treasuries'
}

# Tech Focus (winner from stock test)
TECH_FOCUS = {
    'TSM': 'TSMC', 'BABA': 'Alibaba', 'SONY': 'Sony', 'SE': 'Sea',
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'NVIDIA', 
    'GOOGL': 'Google', 'AMZN': 'Amazon', 'META': 'Meta',
    'GLD': 'Gold', 'TLT': 'Long Treasuries'
}


# =============================================================================
# BACKTESTING
# =============================================================================

def fetch_and_backtest(tickers: Dict, name: str, years: int = 5, vix_threshold: float = 25.0) -> Dict:
    """Fetch data and run backtest."""
    print(f"\n--- {name} ---")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    available = [c for c in prices.columns if not prices[c].isna().all()]
    prices = prices[available].dropna()
    
    print(f"   {len(prices.columns)} assets, {len(prices)} days")
    
    # OOS split
    split_point = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    # Backtest with specified VIX threshold
    config = CompounderConfig(vix_threshold=vix_threshold, sma_lookback=200)
    strategy = CompounderStrategy(config)
    
    print(f"   Training (VIX threshold: {vix_threshold})...", end=" ", flush=True)
    
    try:
        weights = strategy.generate_weights(oos_prices, vix=oos_vix)
    except Exception as e:
        print(f"Error: {e}")
        return {'sharpe': 0, 'cagr': 0}
    
    returns = oos_prices.pct_change().fillna(0)
    warmup = 65
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty:
        return {'sharpe': 0, 'cagr': 0}
    
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    cagr = (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
    
    # Get top holdings
    avg_weights = smoothed.abs().mean()
    top_3 = avg_weights.nlargest(3).index.tolist()
    
    print("Done")
    
    return {
        'sharpe': sharpe,
        'cagr': cagr,
        'max_dd': max_dd,
        'top_holdings': top_3,
        'n_assets': len(oos_prices.columns),
        'oos_days': len(net_returns)
    }


def run_master_comparison():
    """Compare all universes against the original PDF."""
    
    print("=" * 70)
    print("   MASTER COMPARISON: ALL STRATEGIES vs ORIGINAL PDF")
    print("=" * 70)
    print("\n   PDF Target: Sharpe 1.53 OOS (Walk-forward)")
    print("                CAGR 67.7%, Max DD ~-30%")
    
    results = {}
    
    # 1. Original PDF universe (VIX 25)
    results['PDF Original (VIX 25)'] = fetch_and_backtest(ORIGINAL_PDF, 'PDF Original (VIX 25)', vix_threshold=25.0)
    
    # 2. Original PDF universe (VIX 15 - our optimized)
    results['PDF + VIX 15'] = fetch_and_backtest(ORIGINAL_PDF, 'PDF + VIX 15 (Optimized)', vix_threshold=15.0)
    
    # 3. Asian ETFs (VIX 25)
    results['Asian ETFs (VIX 25)'] = fetch_and_backtest(ASIAN_ETFS, 'Asian ETFs (VIX 25)', vix_threshold=25.0)
    
    # 4. Asian ETFs (VIX 15)
    results['Asian ETFs (VIX 15)'] = fetch_and_backtest(ASIAN_ETFS, 'Asian ETFs (VIX 15)', vix_threshold=15.0)
    
    # 5. Best Mix stocks
    results['Best Mix Stocks'] = fetch_and_backtest(BEST_MIX, 'Best Mix Stocks', vix_threshold=15.0)
    
    # 6. Tech Focus
    results['Tech Focus'] = fetch_and_backtest(TECH_FOCUS, 'Tech Focus', vix_threshold=15.0)
    
    # Results table
    print("\n" + "=" * 70)
    print("   RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<30} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10} {'Top Holdings':<25}")
    print("-" * 90)
    
    baseline_sharpe = 1.53  # PDF target
    
    for name, result in results.items():
        top = ', '.join(result.get('top_holdings', [])[:2])
        vs_pdf = result['sharpe'] / baseline_sharpe * 100 if baseline_sharpe > 0 else 0
        print(f"{name:<30} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_dd']:>9.1%}   {top:<25} ({vs_pdf:.0f}% of PDF)")
    
    # Rank
    print("\n" + "=" * 70)
    print("   RANKED BY SHARPE")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Strategy':<35} {'Sharpe':>10} {'vs PDF Target':<15}")
    print("-" * 70)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        vs_pdf = result['sharpe'] - baseline_sharpe
        marker = " <-- BEST" if i == 1 else ""
        status = "ABOVE" if vs_pdf > 0 else "BELOW"
        print(f"{i:<6} {name:<35} {result['sharpe']:>10.2f}   {vs_pdf:+.2f} ({status}){marker}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY")
    print("=" * 70)
    
    best_name, best_result = sorted_results[0]
    
    print(f"\n   PDF Target Sharpe:    1.53")
    print(f"   Best Strategy:        {best_name}")
    print(f"   Best Sharpe:          {best_result['sharpe']:.2f}")
    print(f"   Gap vs Target:        {best_result['sharpe'] - 1.53:+.2f}")
    
    if best_result['sharpe'] >= 1.0:
        print(f"\n    Strategy achieves Sharpe >= 1.0 - INVESTABLE")
    else:
        print(f"\n   Strategy below Sharpe 1.0 - needs improvement")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_master_comparison()
