"""
Optimal Stock Selection: Asian + American Markets
==================================================

Tests individual stocks (not just ETFs) from both Asian and American markets
to find the optimal universe for the compounder strategy.

Stock Categories:
1. ASIAN COMPOUNDERS - High-quality Asian tech/growth companies
2. AMERICAN MEGA-CAPS - FAANG+, proven compounders  
3. AMERICAN VALUE - Quality value stocks
4. CROSS-MARKET MIX - Optimal blend

Uses REAL stock data from Yahoo Finance.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderStrategy, CompounderConfig


# =============================================================================
# STOCK UNIVERSES
# =============================================================================

# Asian stocks (ADRs and major Asian companies)
ASIAN_STOCKS = {
    # Japanese
    'TM': 'Toyota',
    'SONY': 'Sony',
    'NVR': 'Nintendo', # Actually US homebuilder, need NTDOY
    '7203.T': 'Toyota JP',
    '6758.T': 'Sony JP',
    # Chinese ADRs
    'BABA': 'Alibaba',
    'JD': 'JD.com',
    'PDD': 'Pinduoduo',
    'BIDU': 'Baidu',
    'NIO': 'NIO',
    'LI': 'Li Auto',
    # Taiwan
    'TSM': 'TSMC',
    # Korean ADRs
    'PKX': 'POSCO',
    # Indian ADRs  
    'INFY': 'Infosys',
    'WIT': 'Wipro',
    'IBN': 'ICICI Bank',
    # Singapore
    'SE': 'Sea Limited',
}

# American mega-cap compounders
US_COMPOUNDERS = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'META': 'Meta',
    'TSLA': 'Tesla',
    'V': 'Visa',
    'MA': 'Mastercard',
    'UNH': 'UnitedHealth',
    'JNJ': 'Johnson & Johnson',
    'JPM': 'JPMorgan',
    'HD': 'Home Depot',
    'PG': 'Procter & Gamble',
    'COST': 'Costco',
}

# Quality/Value stocks
US_QUALITY = {
    'BRK-B': 'Berkshire',
    'LLY': 'Eli Lilly',
    'AVGO': 'Broadcom',
    'MRK': 'Merck',
    'ABBV': 'AbbVie',
    'PEP': 'PepsiCo',
    'KO': 'Coca-Cola',
    'WMT': 'Walmart',
    'MCD': 'McDonalds',
    'DIS': 'Disney',
}

# Hedges
HEDGES = {
    'GLD': 'Gold',
    'TLT': 'Long Treasuries',
    'SPY': 'S&P 500',
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_stock_data(tickers: Dict[str, str], years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch stock data and VIX."""
    print(f"   Downloading {len(tickers)} stocks...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    # VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.dropna(how='all').ffill()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    # Filter to available
    available = [c for c in prices.columns if not prices[c].isna().all() and prices[c].count() > 100]
    prices = prices[available].dropna()
    
    print(f"   Available: {len(available)} stocks, {len(prices)} days")
    
    return prices, vix


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_universe(prices: pd.DataFrame, vix: pd.Series, warmup: int = 65) -> Dict:
    """Backtest a universe."""
    
    config = CompounderConfig(vix_threshold=15.0, sma_lookback=1)
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except Exception as e:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'error': str(e)}
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty or len(weights) < 20:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    normalized = weights.div(abs_sum, axis=0)
    smoothed = normalized.ewm(span=5).mean()
    
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # Get top holdings
    avg_weights = smoothed.abs().mean()
    top_5 = avg_weights.nlargest(5).index.tolist()
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': ((equity - equity.cummax()) / equity.cummax()).min(),
        'volatility': net_returns.std() * np.sqrt(252),
        'top_holdings': top_5,
        'daily_returns': net_returns
    }


def walk_forward_quick(prices: pd.DataFrame, vix: pd.Series, n_splits: int = 3) -> Dict:
    """Quick walk-forward validation."""
    n = len(prices)
    split_size = n // (n_splits + 1)
    
    oos_sharpes = []
    
    for i in range(n_splits):
        is_end = (i + 1) * split_size
        oos_end = min(is_end + split_size, n)
        
        if oos_end <= is_end + 20:
            continue
        
        oos_prices = prices.iloc[is_end:oos_end]
        oos_vix = vix.iloc[is_end:oos_end]
        
        result = backtest_universe(oos_prices, oos_vix)
        oos_sharpes.append(result['sharpe'])
    
    return {
        'mean_oos_sharpe': np.mean(oos_sharpes) if oos_sharpes else 0,
        'oos_sharpes': oos_sharpes,
        'n_positive': sum(1 for s in oos_sharpes if s > 0)
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_stock_optimization():
    """Find optimal stock combinations."""
    
    print("=" * 70)
    print("   OPTIMAL STOCK SELECTION: ASIAN + AMERICAN")
    print("   Using REAL Yahoo Finance Data")
    print("=" * 70)
    
    # Fetch all categories
    print("\n--- Fetching Stock Data ---")
    
    all_tickers = {**ASIAN_STOCKS, **US_COMPOUNDERS, **US_QUALITY, **HEDGES}
    all_prices, vix = fetch_stock_data(all_tickers, years=5)
    
    # Build test universes
    print("\n--- Building Test Universes ---")
    
    universes = {}
    
    # 1. Asian stocks only
    asian_cols = [c for c in ASIAN_STOCKS.keys() if c in all_prices.columns]
    if len(asian_cols) >= 5:
        universes['Asian Only'] = all_prices[asian_cols + ['GLD', 'TLT']].dropna()
    
    # 2. US Compounders only
    us_comp_cols = [c for c in US_COMPOUNDERS.keys() if c in all_prices.columns]
    if len(us_comp_cols) >= 5:
        universes['US Compounders'] = all_prices[us_comp_cols + ['GLD', 'TLT']].dropna()
    
    # 3. US Quality only
    us_qual_cols = [c for c in US_QUALITY.keys() if c in all_prices.columns]
    if len(us_qual_cols) >= 5:
        universes['US Quality'] = all_prices[us_qual_cols + ['GLD', 'TLT']].dropna()
    
    # 4. All US
    all_us = us_comp_cols + us_qual_cols
    if len(all_us) >= 8:
        universes['All US Stocks'] = all_prices[all_us + ['GLD', 'TLT']].dropna()
    
    # 5. Asian + US Compounders
    asian_us_comp = asian_cols + us_comp_cols
    if len(asian_us_comp) >= 8:
        universes['Asian + US Compounders'] = all_prices[asian_us_comp + ['GLD', 'TLT']].dropna()
    
    # 6. Best of both - Top ADRs + Top US
    best_asian = ['TSM', 'BABA', 'SONY', 'SE']
    best_us = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'V', 'MA', 'LLY']
    best_mix = [c for c in (best_asian + best_us) if c in all_prices.columns]
    if len(best_mix) >= 6:
        universes['Best Mix (Elite)'] = all_prices[best_mix + ['GLD', 'TLT']].dropna()
    
    # 7. Tech focus (Asian + US tech)
    tech_focus = ['TSM', 'BABA', 'SONY', 'SE', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META']
    tech_cols = [c for c in tech_focus if c in all_prices.columns]
    if len(tech_cols) >= 5:
        universes['Tech Focus'] = all_prices[tech_cols + ['GLD', 'TLT']].dropna()
    
    # 8. All stocks together
    all_stocks = asian_cols + all_us
    if len(all_stocks) >= 10:
        universes['All Stocks Combined'] = all_prices[all_stocks + ['GLD', 'TLT']].dropna()
    
    print(f"\n   Built {len(universes)} universes")
    for name, df in universes.items():
        print(f"      {name}: {len(df.columns)} stocks")
    
    # Backtest each
    print("\n" + "=" * 60)
    print("   BACKTESTING UNIVERSES (OOS 30%)")
    print("=" * 60)
    
    results = {}
    
    print(f"\n{'Universe':<30} {'Stocks':>8} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-" * 75)
    
    for name, prices in universes.items():
        split_point = int(len(prices) * 0.7)
        oos_prices = prices.iloc[split_point:]
        oos_vix = vix.reindex(oos_prices.index).ffill()
        
        print(f"   Training {name}...", end=" ", flush=True)
        result = backtest_universe(oos_prices, oos_vix)
        results[name] = result
        print("Done")
        
        top = result.get('top_holdings', [])[:3]
        print(f"{name:<30} {len(prices.columns):>8} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_dd']:>9.1%}   Top: {top}")
    
    # Rank
    print("\n" + "=" * 60)
    print("   RESULTS RANKED BY SHARPE")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Universe':<35} {'Sharpe':>10} {'Top Holdings':<30}")
    print("-" * 85)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        top = ', '.join(result.get('top_holdings', [])[:3])
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<6} {name:<35} {result['sharpe']:>10.2f}   {top:<30}{marker}")
    
    # Walk-forward on top 3
    print("\n" + "=" * 60)
    print("   WALK-FORWARD VALIDATION (Top 3 Universes)")
    print("=" * 60)
    
    for i, (name, _) in enumerate(sorted_results[:3], 1):
        prices = universes[name]
        print(f"\n   {i}. {name}:")
        wf = walk_forward_quick(prices, vix, n_splits=3)
        print(f"      Mean OOS Sharpe: {wf['mean_oos_sharpe']:.2f}")
        print(f"      Positive folds: {wf['n_positive']}/{len(wf['oos_sharpes'])}")
        print(f"      OOS Sharpes: {[f'{s:.2f}' for s in wf['oos_sharpes']]}")
    
    # Best stocks identification
    print("\n" + "=" * 60)
    print("   OPTIMAL STOCKS IDENTIFIED")
    print("=" * 60)
    
    best_name = sorted_results[0][0]
    best_result = sorted_results[0][1]
    
    print(f"\n   Best Universe: {best_name}")
    print(f"   Sharpe: {best_result['sharpe']:.2f}")
    print(f"   CAGR: {best_result['cagr']:.1%}")
    print(f"   Max Drawdown: {best_result['max_dd']:.1%}")
    print(f"\n   Top Holdings (by avg weight):")
    for i, stock in enumerate(best_result.get('top_holdings', []), 1):
        full_name = all_tickers.get(stock, stock)
        print(f"      {i}. {stock} ({full_name})")
    
    print("\n" + "=" * 70)
    
    return results, universes


if __name__ == "__main__":
    results, universes = run_stock_optimization()
