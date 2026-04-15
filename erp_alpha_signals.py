"""
ERP Alpha Signal Generator
===========================

Uses the findings from correlation_report.txt (118 correlations!) to create
tradeable signals. Since we know that items like NO3FE correlate strongly
with cheese_consumption (r=0.96), we can:

1. Track changes in the "weird data" factors
2. Generate signals for related assets (XLB, MOO, CF, etc.)

Key from your correlation_report.txt:
- cheese_consumption: predicts 17 chemicals
- netflix_subscribers: predicts 16 chemicals  
- coffee_price: predicts 15 chemicals
- starbucks_stores: predicts 14 chemicals
- butter_consumption: predicts 12 chemicals

Trading Hypothesis:
- If weird data predicts YOUR chemical costs
- And YOUR chemical costs are tied to agricultural demand
- Then weird data may predict agricultural commodities

Usage:
    python erp_alpha_signals.py
"""

import datetime
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# WEIRD DATA SIGNALS
# From your chemical_correlation_scanner.py
# =============================================================================

WEIRD_DATA = {
    # USDA: Butter consumption per capita (lbs/year)
    "butter_consumption": {
        2010: 4.9, 2011: 5.0, 2012: 5.2, 2013: 5.3, 2014: 5.5,
        2015: 5.6, 2016: 5.7, 2017: 5.8, 2018: 5.9, 2019: 6.0,
        2020: 6.1, 2021: 6.2, 2022: 6.3, 2023: 6.5, 2024: 6.8
    },

    # USDA: Cheese consumption per capita (lbs/year)
    "cheese_consumption": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5
    },

    # Company reports: Netflix subscribers (millions, global)
    "netflix_subscribers": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0
    },

    # BLS: Average coffee price per pound (USD)
    "coffee_price": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32
    },

    # Company reports: Starbucks US store count (thousands)
    "starbucks_stores": {
        2010: 10.6, 2011: 10.8, 2012: 11.2, 2013: 11.6, 2014: 12.0,
        2015: 12.5, 2016: 13.0, 2017: 13.5, 2018: 14.2, 2019: 15.0,
        2020: 15.3, 2021: 15.7, 2022: 15.9, 2023: 16.4, 2024: 16.9
    },

    # NPS: National Park visits (millions)
    "national_park_visits": {
        2010: 281.3, 2011: 278.7, 2012: 282.8, 2013: 273.6, 2014: 292.8,
        2015: 307.2, 2016: 331.0, 2017: 330.9, 2018: 318.2, 2019: 327.5,
        2020: 237.0, 2021: 297.1, 2022: 312.0, 2023: 325.0, 2024: 331.9
    },
}

# Target assets to trade based on weird data correlations
TRADE_TARGETS = {
    "XLB": "SPDR Materials ETF",
    "MOO": "VanEck Agriculture ETF", 
    "XLE": "Energy Select ETF",
    "CF": "CF Industries (Nitrogen)",
    "MOS": "Mosaic (Fertilizer)",
    "NTR": "Nutrien (Fertilizer)",
    "DBA": "Invesco DB Agriculture",
}


def fetch_annual_returns(tickers: list, start_year: int = 2010) -> pd.DataFrame:
    """Fetch annual returns for given tickers."""
    if not HAS_YFINANCE:
        return pd.DataFrame()
    
    print(f"Fetching {len(tickers)} tickers...")
    data = yf.download(tickers, start=f"{start_year}-01-01", progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    # Annual returns (year-end to year-end)
    annual = prices.resample('YE').last()
    returns = annual.pct_change().dropna()
    returns.index = returns.index.year
    
    return returns


def calculate_weird_data_vs_markets():
    """
    Main analysis: correlate weird data directly with market returns.
    This tests if the same factors predicting your chemical costs
    also predict market movements.
    """
    print("=" * 70)
    print("  WEIRD DATA → MARKET CORRELATIONS")
    print("=" * 70)
    
    # Fetch market data
    tickers = list(TRADE_TARGETS.keys())
    returns = fetch_annual_returns(tickers, 2010)
    
    if returns.empty:
        print("Could not fetch market data")
        return []
    
    print(f"\nMarket data: {len(returns)} years")
    
    results = []
    
    # Test each weird factor against each market
    for weird_name, weird_values in WEIRD_DATA.items():
        for ticker in returns.columns:
            # Align data
            common_years = []
            for year in weird_values:
                if year in returns.index:
                    common_years.append(year)
            
            if len(common_years) < 8:
                continue
            
            x = np.array([weird_values[y] for y in common_years])
            y = returns.loc[common_years, ticker].values
            
            # Test lag=0 (concurrent) and lag=1 (predictive)
            for lag in [0, 1]:
                if lag > 0:
                    # Weird data predicts NEXT year's return
                    x_lagged = x[:-lag]
                    y_lagged = y[lag:]
                else:
                    x_lagged = x
                    y_lagged = y
                
                if len(x_lagged) < 6:
                    continue
                
                r, p = stats.pearsonr(x_lagged, y_lagged)
                
                results.append({
                    "weird_factor": weird_name,
                    "ticker": ticker,
                    "lag": lag,
                    "correlation": r,
                    "p_value": p,
                    "n_years": len(x_lagged),
                })
    
    # Filter and sort
    significant = [r for r in results if abs(r["correlation"]) > 0.4]
    significant.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return significant


def generate_signal_report():
    """Generate full analysis report."""
    print("\n" + "=" * 70)
    print("  ERP ALPHA SIGNAL ANALYSIS")
    print("  From Your Correlation Findings")
    print("=" * 70)
    
    results = calculate_weird_data_vs_markets()
    
    if not results:
        print("\nNo significant correlations found.")
        return
    
    print(f"\nFound {len(results)} significant correlations (|r| > 0.4):\n")
    
    # Print strongest
    print("-" * 60)
    print("TOP 15 WEIRD DATA → MARKET CORRELATIONS")
    print("-" * 60)
    
    for i, r in enumerate(results[:15], 1):
        direction = "↑" if r["correlation"] > 0 else "↓"
        lag_str = "(PREDICTIVE!)" if r["lag"] > 0 else ""
        ticker_name = TRADE_TARGETS.get(r["ticker"], r["ticker"])
        
        print(f"{i:2}. {r['weird_factor']:20} → {r['ticker']:5} | r={r['correlation']:+.3f} {direction} | p={r['p_value']:.3f} {lag_str}")
        print(f"    Trade: {ticker_name}")
    
    # Focus on predictive signals (lag > 0)
    predictive = [r for r in results if r["lag"] > 0]
    if predictive:
        print("\n" + "=" * 60)
        print("🔮 PREDICTIVE SIGNALS (Weird Data Leads Market by 1 Year)")
        print("=" * 60)
        
        for r in predictive[:10]:
            direction = "UP" if r["correlation"] > 0 else "DOWN"
            print(f"\n{r['weird_factor']} → {r['ticker']}")
            print(f"  Correlation: {r['correlation']:+.3f}")
            print(f"  Signal: When {r['weird_factor']} rises, {r['ticker']} tends to go {direction} next year")
    
    # Summary by factor
    print("\n" + "=" * 60)
    print("SUMMARY: Most Predictive Weird Factors")
    print("=" * 60)
    
    factor_counts = {}
    for r in results:
        factor_counts[r["weird_factor"]] = factor_counts.get(r["weird_factor"], 0) + 1
    
    for factor, count in sorted(factor_counts.items(), key=lambda x: -x[1])[:5]:
        tickers = [r["ticker"] for r in results if r["weird_factor"] == factor]
        print(f"\n{factor}: predicts {count} assets")
        print(f"  Assets: {', '.join(set(tickers))}")
    
    # Trading implications
    print("\n" + "=" * 60)
    print("TRADING IMPLICATIONS")
    print("=" * 60)
    print("""
Based on correlations, potential signals:

1. DAIRY CONSUMPTION SIGNALS (butter, cheese):
   - Rising consumption → Materials/Agriculture may rise
   - These correlate with your NO3FE, CHEGLUCO costs
   - Trade: Long XLB, MOO when dairy consumption rising

2. STREAMING/TECH SIGNALS (Netflix):
   - Growing subscribers → Economic strength signal
   - Correlates with your chemical demand
   - Trade: Materials tend to follow consumer trends

3. COFFEE PRICES:
   - Coffee price spikes → May signal inflation
   - Your chemical costs follow similar patterns
   - Trade: Watch coffee as early commodity indicator

⚠️  CAUTION: These are spurious correlations until proven otherwise.
    Backtest rigorously before trading!
""")
    
    return results


if __name__ == "__main__":
    results = generate_signal_report()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv("erp_alpha_signals.csv", index=False)
        print("\n📊 Results saved to erp_alpha_signals.csv")
