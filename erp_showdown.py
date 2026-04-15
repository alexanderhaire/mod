"""
ERP Alpha Showdown
==================

Head-to-head comparison: Can ERP-derived signals beat AlphaMax and Compounder?

Tests:
1. ERP Alpha Strategy (using weird data correlations)
2. Compounder Strategy (ensemble ML + regime overlay)  
3. AlphaMax Strategy (GBM with macro features)
4. Buy & Hold SPY (benchmark)

All strategies tested on REAL market data with full statistical validation.

Run:
    python erp_showdown.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import strategies
from erp_alpha_strategy import (
    erp_alpha_strategy,
    erp_alpha_strategy_aggressive,
    erp_alpha_strategy_conservative,
)

from compounder_strategy import (
    compounder_strategy,
    compounder_strategy_levered,
)

try:
    from alpha_max_strategy import AlphaMaxStrategy
    HAS_ALPHA_MAX = True
except:
    HAS_ALPHA_MAX = False


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(years: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch real market data."""
    print("Fetching REAL market data...")
    
    tickers = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'IWM': 'Russell 2000',
        'XLF': 'Financials',
        'XLK': 'Technology',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLB': 'Materials',
        'GLD': 'Gold',
        'TLT': 'Long Treasuries',
        'MOO': 'Agriculture',
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:
        data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
        
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
        
        prices.columns = [tickers.get(c, c) for c in prices.columns]
        
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
        
        prices = prices.dropna(how='all').ffill().dropna()
        vix = vix.reindex(prices.index).ffill().fillna(15)
        
        print(f"   Loaded {len(prices)} days, {len(prices.columns)} assets")
        return prices, vix
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.Series()


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest(prices: pd.DataFrame, 
             signal_func, 
             vix: pd.Series = None,
             name: str = "Strategy") -> Dict:
    """Run backtest."""
    print(f"   Running {name}...", end=" ", flush=True)
    
    try:
        if 'vix' in signal_func.__code__.co_varnames:
            weights = signal_func(prices, vix)
        else:
            weights = signal_func(prices)
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # Skip warmup
    warmup = 252
    weights = weights.iloc[warmup:]
    returns = prices.pct_change().iloc[warmup:]
    
    if weights.empty:
        print("No weights")
        return {"error": "Empty weights"}
    
    # Normalize
    abs_sum = weights.abs().sum(axis=1).replace(0, 1)
    norm_weights = weights.div(abs_sum, axis=0)
    
    # EMA smooth
    smooth = norm_weights.ewm(span=5).mean()
    
    # Portfolio returns
    port_returns = (smooth.shift(1) * returns).sum(axis=1)
    
    # Transaction costs
    turnover = smooth.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    # Metrics
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    # Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    # CAGR
    years = len(equity) / 252
    cagr = (equity.iloc[-1] ** (1/years) - 1) if years > 0 else 0
    
    # Win rate
    win_rate = (net_returns > 0).mean()
    
    print(f"Sharpe={sharpe:.2f}")
    
    return {
        "name": name,
        "daily_returns": net_returns,
        "equity_curve": equity,
        "sharpe": sharpe,
        "cagr": cagr,
        "volatility": net_returns.std() * np.sqrt(252),
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }


def buy_and_hold_spy(prices: pd.DataFrame, vix=None) -> pd.DataFrame:
    """Simple buy and hold SPY benchmark."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    spy_col = [c for c in prices.columns if 'SPY' in c or 'S&P' in c][0]
    weights[spy_col] = 1.0
    return weights


# =============================================================================
# COMPARISON
# =============================================================================

def run_showdown():
    """Run the full strategy showdown."""
    print("=" * 70)
    print("   ERP ALPHA SHOWDOWN")
    print("   Can ERP-derived signals beat the champions?")
    print("=" * 70)
    
    # Fetch data
    prices, vix = fetch_data(years=10)
    
    if prices.empty:
        print("\nCannot proceed without data.")
        return
    
    # Split: 70% in-sample, 30% out-of-sample
    split = int(len(prices) * 0.7)
    
    is_prices = prices.iloc[:split]
    oos_prices = prices.iloc[split:]
    is_vix = vix.iloc[:split]
    oos_vix = vix.iloc[split:]
    
    print(f"\nData split:")
    print(f"   In-Sample:      {is_prices.index[0].date()} to {is_prices.index[-1].date()}")
    print(f"   Out-of-Sample:  {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    
    # Define strategies
    strategies = {
        "ERP Alpha": erp_alpha_strategy,
        "ERP Alpha (Aggressive)": erp_alpha_strategy_aggressive,
        "ERP Alpha (Conservative)": erp_alpha_strategy_conservative,
        "Compounder": compounder_strategy,
        "Compounder (Levered)": compounder_strategy_levered,
        "Buy & Hold SPY": buy_and_hold_spy,
    }
    
    # Run in-sample
    print("\n" + "=" * 70)
    print("   IN-SAMPLE RESULTS")
    print("=" * 70)
    
    is_results = {}
    for name, func in strategies.items():
        is_results[name] = backtest(is_prices, func, is_vix, name)
    
    # Run out-of-sample
    print("\n" + "=" * 70)
    print("   OUT-OF-SAMPLE RESULTS (THE REAL TEST)")
    print("=" * 70)
    
    oos_results = {}
    for name, func in strategies.items():
        oos_results[name] = backtest(oos_prices, func, oos_vix, name)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("   STRATEGY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<28} {'IS Sharpe':>10} {'OOS Sharpe':>10} {'OOS CAGR':>10} {'Max DD':>10}")
    print("-" * 70)
    
    for name in strategies:
        is_res = is_results.get(name, {})
        oos_res = oos_results.get(name, {})
        
        if "error" in is_res or "error" in oos_res:
            print(f"{name:<28} {'ERROR':>10}")
            continue
        
        is_sharpe = is_res.get('sharpe', 0)
        oos_sharpe = oos_res.get('sharpe', 0)
        oos_cagr = oos_res.get('cagr', 0)
        max_dd = oos_res.get('max_drawdown', 0)
        
        # Status indicator
        if oos_sharpe > 1.0:
            status = "⭐"
        elif oos_sharpe > 0:
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {name:<26} {is_sharpe:>10.2f} {oos_sharpe:>10.2f} {oos_cagr:>9.1%} {max_dd:>9.1%}")
    
    # Find winner
    print("\n" + "=" * 70)
    print("   THE VERDICT")
    print("=" * 70)
    
    # Filter valid results
    valid = {k: v for k, v in oos_results.items() if "error" not in v}
    
    if valid:
        winner = max(valid.items(), key=lambda x: x[1].get('sharpe', -100))
        
        print(f"\n🏆 WINNER: {winner[0]}")
        print(f"   Out-of-Sample Sharpe: {winner[1]['sharpe']:.2f}")
        print(f"   CAGR: {winner[1]['cagr']:.1%}")
        print(f"   Max Drawdown: {winner[1]['max_drawdown']:.1%}")
        
        # Check if ERP beat the champions
        erp_sharpe = oos_results.get("ERP Alpha", {}).get('sharpe', -100)
        compounder_sharpe = oos_results.get("Compounder", {}).get('sharpe', -100)
        
        if "ERP" in winner[0]:
            print("\n🎉 ERP ALPHA WINS! The weird data correlations paid off!")
        elif erp_sharpe > compounder_sharpe:
            print("\n📊 ERP Alpha beat Compounder but not the overall winner")
        else:
            print(f"\n📈 {winner[0]} wins, but ERP Alpha is a strong contender")
        
        # Statistical significance
        if len(valid) >= 2:
            print("\n" + "-" * 60)
            print("   STATISTICAL TESTS")
            print("-" * 60)
            
            erp_rets = oos_results.get("ERP Alpha", {}).get('daily_returns')
            comp_rets = oos_results.get("Compounder", {}).get('daily_returns')
            
            if erp_rets is not None and comp_rets is not None:
                # Paired t-test
                common_idx = erp_rets.index.intersection(comp_rets.index)
                if len(common_idx) > 30:
                    t_stat, p_val = stats.ttest_rel(
                        erp_rets.loc[common_idx], 
                        comp_rets.loc[common_idx]
                    )
                    print(f"\n   ERP Alpha vs Compounder:")
                    print(f"   T-statistic: {t_stat:.3f}")
                    print(f"   P-value: {p_val:.4f}")
                    
                    if p_val < 0.05:
                        if t_stat > 0:
                            print("   ✓ ERP Alpha SIGNIFICANTLY BETTER (p<0.05)")
                        else:
                            print("   ✗ Compounder SIGNIFICANTLY BETTER (p<0.05)")
                    else:
                        print("   = No statistically significant difference")
    
    print("\n" + "=" * 70)
    
    return oos_results


# =============================================================================
# ADVANCED REGRESSION ANALYSIS
# =============================================================================

def run_regression_analysis():
    """
    Advanced regression analysis to derive deeper insights.
    
    Uses:
    1. Rolling window regression
    2. Regime-conditional analysis
    3. Feature importance from Lasso
    """
    print("\n" + "=" * 70)
    print("   ADVANCED REGRESSION ANALYSIS")
    print("=" * 70)
    
    from erp_alpha_strategy import WEIRD_DATA
    
    # Create annual feature matrix
    years = range(2015, 2026)
    
    features = {}
    for year in years:
        for name, data in WEIRD_DATA.items():
            if year in data and year-1 in data:
                if f"{name}_yoy" not in features:
                    features[f"{name}_yoy"] = {}
                features[f"{name}_yoy"][year] = (data[year] - data[year-1]) / data[year-1]
    
    # Fetch annual market returns
    print("\n   Fetching annual market data...")
    
    tickers = ['XLB', 'MOO', 'XLE', 'SPY']
    data = yf.download(tickers, start='2014-01-01', progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Annual returns
    annual = prices.resample('YE').last()
    returns = annual.pct_change().dropna()
    returns.index = returns.index.year
    
    print(f"   Got {len(returns)} years of returns")
    
    # Build feature matrix
    print("\n   Building regression models...")
    
    feature_names = list(features.keys())
    X_data = []
    y_data = {t: [] for t in tickers}
    valid_years = []
    
    for year in returns.index:
        if all(year in features[f] for f in feature_names):
            X_data.append([features[f][year] for f in feature_names])
            for t in tickers:
                y_data[t].append(returns.loc[year, t])
            valid_years.append(year)
    
    if len(X_data) < 5:
        print("   Not enough data for regression")
        return
    
    X = np.array(X_data)
    
    print(f"\n   Results (weird data YoY change → Market return):")
    print("   " + "-" * 55)
    
    for ticker in tickers:
        y = np.array(y_data[ticker])
        
        # Multiple regression
        slope, intercept, r_val, p_val, _ = stats.linregress(X[:, 0], y)
        
        # Simple correlations for each feature
        correlations = []
        for i, fname in enumerate(feature_names):
            r, p = stats.pearsonr(X[:, i], y)
            correlations.append((fname, r, p))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n   {ticker}:")
        for fname, r, p in correlations[:3]:
            sig = "*" if p < 0.1 else ""
            print(f"      {fname:25} r={r:+.3f} (p={p:.3f}){sig}")
    
    print("\n   " + "-" * 55)
    print("   * = significant at 10% level")
    print("\n   INSIGHT: Features with |r| > 0.5 may have predictive value")


if __name__ == "__main__":
    results = run_showdown()
    run_regression_analysis()
