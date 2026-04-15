"""
Chemical Alpha Hunt (LME Expanded)
==================================

Searching for Alpha in the derivatives of the user's specific chemical portfolio using Liquid Proxies.
Now includes London Stock Exchange (LSE) proxies for LME exposure.

Portfolio Mapping:
- Copper -> CPER (ETF), COPX (Miners), FCX (Freeport)
- Nickel -> VALE (Major Producer)
- Cobalt / Lithium -> LIT (ETF), SQM (Major Producer)
- Molybdenum -> REMX (Rare Earths/Strategic Metals)
- Broad Base -> DBB (Base Metals Fund)

London (LME Proxies):
- Glencore (GLEN.L) -> The King of Commodity Trading
- Rio Tinto (RIO.L) -> Diversified Major
- Antofagasta (ANTO.L) -> Copper Specialist
- Anglo American (AAL.L) -> Diversified / PGM

Benchmarks:
- US Market: SPY
- UK Market: ^FTSE (for London stocks)
- Broad Commodities: DBC

RUN: python chemical_alpha_hunt_lme.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PROXIES = {
    # US / Global
    'Copper (ETF)': 'CPER',
    'Copper (Miners)': 'COPX',
    'Freeport (Copper/Gold)': 'FCX',
    'Nickel/Iron (Vale)': 'VALE',
    'Lithium/Cobalt (ETF)': 'LIT',
    'Lithium/Iodine (SQM)': 'SQM',
    'Strategic Metals (Moly)': 'REMX',
    'Base Metals (Fund)': 'DBB',
    'Teck Resources (Zinc)': 'TECK',
    
    # London (LME Proxies)
    'Glencore (LME King)': 'GLEN.L',
    'Rio Tinto (Diversified)': 'RIO.L',
    'Antofagasta (Copper)': 'ANTO.L',
    'Anglo American (PGM/Div)': 'AAL.L'
}

BENCHMARKS = ['SPY', 'DBC', '^FTSE']

# =============================================================================
# 2. DATA ENGINE
# =============================================================================

def fetch_chemical_data():
    print("🧪 Fetching Chemical & LME Proxy Data...")
    
    tickers = list(PROXIES.values()) + BENCHMARKS
    data = yf.download(tickers, start='2015-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna()
    print(f"   Data Range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Tickers Found: {len(prices.columns)}")
    
    return prices

# =============================================================================
# 3. ALPHA CALCULATOR
# =============================================================================

def calculate_alpha_metrics(prices):
    print("\n🔬 Calculating Alpha & Volatility Metrics...")
    print("-" * 115)
    print(f"{'Asset':<25} {'Ticker':<8} {'Region':<6} {'Vol':<8} {'Beta':<6} {'Alpha':<8} {'Sharpe':<6} {'Benchmark'}")
    print("-" * 115)
    
    results = []
    
    for name, ticker in PROXIES.items():
        if ticker not in prices.columns:
            continue
            
        # Determine Benchmark
        if ticker.endswith('.L'):
            bench_ticker = '^FTSE'
            region = 'UK'
        else:
            bench_ticker = 'SPY'
            region = 'US'
            
        if bench_ticker not in prices.columns:
            continue

        # Returns
        asset_ret = prices[ticker].pct_change().dropna()
        bench_ret = prices[bench_ticker].pct_change().dropna()
        
        # Align dates
        common_idx = asset_ret.index.intersection(bench_ret.index)
        y = asset_ret.loc[common_idx]
        x = bench_ret.loc[common_idx]
        
        # 1. Volatility
        vol = y.std() * np.sqrt(252)
        
        # 2. Beta & Alpha (CAPM)
        if len(y) > 60:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            beta = slope
            # Annualize Alpha (approx)
            alpha_ann = intercept * 252 
        else:
            beta, alpha_ann = np.nan, np.nan
            
        # 3. Sharpe
        sharpe = (y.mean() * 252) / (vol) if vol > 0 else 0
        
        print(f"{name:<25} {ticker:<8} {region:<6} {vol:<8.1%} {beta:<6.2f} {alpha_ann:<8.1%} {sharpe:<6.2f} {bench_ticker}")
        
        results.append({
            'Name': name,
            'Ticker': ticker,
            'Region': region,
            'Vol': vol,
            'Alpha': alpha_ann,
            'Sharpe': sharpe
        })
        
    print("-" * 115)
    return pd.DataFrame(results)

# =============================================================================
# 4. TREND DETECTOR
# =============================================================================

def detect_trends(prices):
    print("\n📈 Current Trend Status (Daily)")
    print("-" * 65)
    print(f"{'Asset':<25} {'Price':<10} {'SMA200':<10} {'Status'}")
    print("-" * 65)
    
    for name, ticker in PROXIES.items():
        if ticker not in prices.columns:
            continue
            
        p = prices[ticker]
        current_price = p.iloc[-1]
        sma_200 = p.rolling(200).mean().iloc[-1]
        
        status = "BULLISH 🟢" if current_price > sma_200 else "BEARISH 🔴"
        dist = (current_price / sma_200) - 1
        
        print(f"{name:<25} {current_price:<10.2f} {sma_200:<10.2f} {status} ({dist:+.1%})")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    prices = fetch_chemical_data()
    metrics = calculate_alpha_metrics(prices)
    detect_trends(prices)
    
    # Recommendation
    print("\n💡 OBSERVATIONS:")
    if not metrics.empty:
        best_alpha = metrics.sort_values('Alpha', ascending=False).iloc[0]
        best_sharpe = metrics.sort_values('Sharpe', ascending=False).iloc[0]
        
        print(f"   Highest Alpha Generator: {best_alpha['Name']} ({best_alpha['Alpha']:.1%} annual alpha)")
        print(f"   Best Risk-Adjusted:      {best_sharpe['Name']} (Sharpe {best_sharpe['Sharpe']:.2f})")
