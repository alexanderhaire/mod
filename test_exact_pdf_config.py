"""
Exact PDF Configuration Test
=============================

PDF explicitly states: "reduce exposure when VIX > 25 OR SPY < 200MA"
(See lines 144, 147, 177 of pdf_content.txt)

This test compares both VIX thresholds to verify which is better.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


def fetch_pdf_universe(years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch the EXACT PDF universe."""
    
    # PDF recommends: "Liquid compounders (tech, healthcare, leaders) plus defensive bonds"
    tickers = {
        'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
        'XLK': 'Technology', 'XLV': 'Healthcare', 'XLF': 'Financials', 'XLE': 'Energy',
        'GLD': 'Gold', 'TLT': 'Long Treasuries',
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def fetch_asian_universe(years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch Asian ETF universe."""
    
    tickers = {
        'EWJ': 'Japan', 'FXI': 'China', 'EWY': 'South Korea',
        'INDA': 'India', 'EWT': 'Taiwan', 'EWH': 'Hong Kong',
        'EWS': 'Singapore', 'AAXJ': 'Asia ex-Japan',
        'GLD': 'Gold', 'TLT': 'Long Treasuries'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


def backtest(prices: pd.DataFrame, vix: pd.Series, vix_threshold: float, warmup: int = 65) -> Dict:
    """Run backtest with specific VIX threshold."""
    
    config = CompounderConfig(
        vix_threshold=vix_threshold,
        sma_lookback=200  # SPY 200-day MA as per PDF
    )
    strategy = CompounderStrategy(config)
    
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
    }


def run_test():
    """Compare PDF and Asian with both VIX thresholds."""
    
    print("=" * 80)
    print("   EXACT PDF CONFIG TEST: VIX 25 vs VIX 15")
    print("   PDF states: 'reduce exposure when VIX > 25 OR SPY < 200MA'")
    print("=" * 80)
    
    # Fetch data
    print("\nFetching data...")
    pdf_prices, pdf_vix = fetch_pdf_universe(years=5)
    asian_prices, asian_vix = fetch_asian_universe(years=5)
    
    print(f"   PDF: {len(pdf_prices.columns)} assets")
    print(f"   Asian: {len(asian_prices.columns)} assets")
    
    # OOS split
    pdf_split = int(len(pdf_prices) * 0.7)
    asian_split = int(len(asian_prices) * 0.7)
    
    pdf_oos = pdf_prices.iloc[pdf_split:]
    pdf_oos_vix = pdf_vix.iloc[pdf_split:]
    
    asian_oos = asian_prices.iloc[asian_split:]
    asian_oos_vix = asian_vix.iloc[asian_split:]
    
    # Test all combinations
    configs = [
        ('PDF Universe + VIX 25 (EXACT PDF)', pdf_oos, pdf_oos_vix, 25.0),
        ('PDF Universe + VIX 15', pdf_oos, pdf_oos_vix, 15.0),
        ('Asian ETFs + VIX 25', asian_oos, asian_oos_vix, 25.0),
        ('Asian ETFs + VIX 15', asian_oos, asian_oos_vix, 15.0),
    ]
    
    results = []
    
    print("\n" + "=" * 80)
    print("   OUT-OF-SAMPLE RESULTS (30% holdout)")
    print("=" * 80)
    
    print(f"\n{'Strategy':<35} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-" * 70)
    
    for name, prices, vix, vix_thresh in configs:
        print(f"   Training {name}...", end=" ", flush=True)
        result = backtest(prices, vix, vix_thresh)
        results.append((name, result))
        print("Done")
        print(f"{name:<35} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_dd']:>9.1%}")
    
    # Rank
    print("\n" + "=" * 80)
    print("   RANKED BY OOS SHARPE")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Strategy':<40} {'Sharpe':>10}")
    print("-" * 60)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<6} {name:<40} {result['sharpe']:>10.2f}{marker}")
    
    # Summary
    print("\n" + "=" * 80)
    print("   KEY FINDINGS")
    print("=" * 80)
    
    pdf_25 = next(r for n, r in results if 'PDF' in n and '25' in n)
    pdf_15 = next(r for n, r in results if 'PDF' in n and '15' in n)
    asian_25 = next(r for n, r in results if 'Asian' in n and '25' in n)
    asian_15 = next(r for n, r in results if 'Asian' in n and '15' in n)
    
    print(f"""
   PDF Universe (US ETFs):
      VIX 25 (exact PDF): Sharpe {pdf_25['sharpe']:.2f}, CAGR {pdf_25['cagr']:.1%}
      VIX 15 (modified):  Sharpe {pdf_15['sharpe']:.2f}, CAGR {pdf_15['cagr']:.1%}
      
   Asian ETFs:
      VIX 25:             Sharpe {asian_25['sharpe']:.2f}, CAGR {asian_25['cagr']:.1%}
      VIX 15:             Sharpe {asian_15['sharpe']:.2f}, CAGR {asian_15['cagr']:.1%}
      
   WINNER: {sorted_results[0][0]} (Sharpe {sorted_results[0][1]['sharpe']:.2f})
    """)
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_test()
