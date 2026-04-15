"""
Chemical Alpha Hunt
===================

Searching for Alpha in the derivatives of the user's specific chemical portfolio using Liquid Proxies.

Portfolio Mapping:
- Copper -> CPER (ETF), COPX (Miners), FCX (Freeport)
- Nickel -> VALE (Major Producer)
- Cobalt / Lithium -> LIT (ETF), SQM (Major Producer)
- Molybdenum -> REMX (Rare Earths/Strategic Metals)
- Broad Base -> DBB (Base Metals Fund)

Benchmarks:
- Market: SPY
- Broad Commodities: DBC

RUN: python chemical_alpha_hunt.py
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
    'Copper (ETF)': 'CPER',
    'Copper (Miners)': 'COPX',
    'Freeport (Copper/Gold)': 'FCX',
    'Nickel/Iron (Vale)': 'VALE',
    'Lithium/Cobalt (ETF)': 'LIT',
    'Lithium/Iodine (SQM)': 'SQM',
    'Strategic Metals (Moly)': 'REMX',
    'Base Metals (Fund)': 'DBB',
    'Teck Resources (Zinc/Copper)': 'TECK'
}

BENCHMARKS = ['SPY', 'DBC']

# =============================================================================
# 2. DATA ENGINE
# =============================================================================

def fetch_chemical_data():
    print("🧪 Fetching Chemical Proxy Data...")
    
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
    print("-" * 100)
    print(f"{'Asset':<30} {'Ticker':<8} {'Vol (Ann)':<12} {'Beta (SPY)':<10} {'Alpha (vs SPY)':<15} {'Sharpe':<8}")
    print("-" * 100)
    
    spy_ret = prices['SPY'].pct_change().dropna()
    results = []
    
    for name, ticker in PROXIES.items():
        if ticker not in prices.columns:
            continue
            
        # Returns
        asset_ret = prices[ticker].pct_change().dropna()
        
        # Align dates
        common_idx = asset_ret.index.intersection(spy_ret.index)
        y = asset_ret.loc[common_idx]
        x = spy_ret.loc[common_idx]
        
        # 1. Volatility
        vol = y.std() * np.sqrt(252)
        
        # 2. Beta & Alpha (CAPM)
        # y = alpha + beta * x
        if len(y) > 60:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            beta = slope
            # Annualize Alpha (approx)
            alpha_ann = intercept * 252 
        else:
            beta, alpha_ann = np.nan, np.nan
            
        # 3. Sharpe
        sharpe = (y.mean() * 252) / (vol) if vol > 0 else 0
        
        print(f"{name:<30} {ticker:<8} {vol:<12.1%} {beta:<10.2f} {alpha_ann:<15.1%} {sharpe:<8.2f}")
        
        results.append({
            'Name': name,
            'Ticker': ticker,
            'Vol': vol,
            'Alpha': alpha_ann,
            'Sharpe': sharpe
        })
        
    print("-" * 100)
    return pd.DataFrame(results)

# =============================================================================
# 4. TREND DETECTOR
# =============================================================================

def detect_trends(prices):
    print("\n📈 Current Trend Status (Daily)")
    print("-" * 60)
    print(f"{'Asset':<30} {'Price':<10} {'SMA200':<10} {'Status'}")
    print("-" * 60)
    
    for name, ticker in PROXIES.items():
        if ticker not in prices.columns:
            continue
            
        p = prices[ticker]
        current_price = p.iloc[-1]
        sma_200 = p.rolling(200).mean().iloc[-1]
        
        status = "BULLISH 🟢" if current_price > sma_200 else "BEARISH 🔴"
        dist = (current_price / sma_200) - 1
        
        print(f"{name:<30} {current_price:<10.2f} {sma_200:<10.2f} {status} ({dist:+.1%})")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    prices = fetch_chemical_data()
    metrics = calculate_alpha_metrics(prices)
    detect_trends(prices)
    
    # Recommendation
    print("\n💡 OBSERVATIONS:")
    best_alpha = metrics.sort_values('Alpha', ascending=False).iloc[0]
    best_sharpe = metrics.sort_values('Sharpe', ascending=False).iloc[0]
    
    print(f"   Highest Alpha Generator: {best_alpha['Name']} ({best_alpha['Alpha']:.1%} annual alpha)")
    print(f"   Best Risk-Adjusted:      {best_sharpe['Name']} (Sharpe {best_sharpe['Sharpe']:.2f})")
    print(f"   Most Volatile:           {metrics.sort_values('Vol', ascending=False).iloc[0]['Name']} ({metrics.sort_values('Vol', ascending=False).iloc[0]['Vol']:.1%})")
