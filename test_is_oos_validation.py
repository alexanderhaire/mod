"""
In-Sample vs Out-of-Sample Validation
======================================

Rigorous comparison of:
1. PDF Original Strategy (US ETFs)
2. Asian ETFs (VIX 25)

Tests:
- In-Sample (training) performance
- Out-of-Sample (unseen) performance
- Walk-Forward validation (5 folds)
- IS vs OOS degradation
- Statistical significance
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
# UNIVERSES
# =============================================================================

PDF_UNIVERSE = {
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'IWM': 'Russell 2000',
    'XLF': 'Financials', 'XLK': 'Technology', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'GLD': 'Gold', 'TLT': 'Long Treasuries',
}

ASIAN_UNIVERSE = {
    'EWJ': 'Japan', 'FXI': 'China', 'EWY': 'South Korea',
    'INDA': 'India', 'EWT': 'Taiwan', 'EWH': 'Hong Kong',
    'EWS': 'Singapore', 'AAXJ': 'Asia ex-Japan',
    'GLD': 'Gold', 'TLT': 'Long Treasuries'
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_data(tickers: Dict, years: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    """Fetch price data and VIX."""
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
    available = [c for c in prices.columns if not prices[c].isna().all()]
    prices = prices[available].dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    return prices, vix


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest(prices: pd.DataFrame, vix: pd.Series, vix_threshold: float = 25.0, warmup: int = 65) -> Dict:
    """Run backtest and return metrics."""
    
    config = CompounderConfig(vix_threshold=vix_threshold, sma_lookback=200)
    strategy = CompounderStrategy(config)
    
    try:
        weights = strategy.generate_weights(prices, vix=vix)
    except:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'returns': pd.Series()}
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    if weights.empty or len(weights) < 20:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'returns': pd.Series()}
    
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
        'volatility': net_returns.std() * np.sqrt(252),
        'returns': net_returns,
        'equity': equity
    }


def walk_forward_validation(prices: pd.DataFrame, vix: pd.Series, n_splits: int = 5, vix_threshold: float = 25.0) -> Dict:
    """Walk-forward validation with expanding window."""
    
    n = len(prices)
    split_size = n // (n_splits + 1)
    
    is_sharpes = []
    oos_sharpes = []
    is_cagrs = []
    oos_cagrs = []
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
        
        is_result = backtest(is_prices, is_vix, vix_threshold)
        oos_result = backtest(oos_prices, oos_vix, vix_threshold)
        
        is_sharpes.append(is_result['sharpe'])
        oos_sharpes.append(oos_result['sharpe'])
        is_cagrs.append(is_result['cagr'])
        oos_cagrs.append(oos_result['cagr'])
    
    mean_is = np.mean(is_sharpes) if is_sharpes else 0
    mean_oos = np.mean(oos_sharpes) if oos_sharpes else 0
    degradation = (1 - mean_oos / mean_is) * 100 if mean_is != 0 else 0
    
    return {
        'is_sharpes': is_sharpes,
        'oos_sharpes': oos_sharpes,
        'is_cagrs': is_cagrs,
        'oos_cagrs': oos_cagrs,
        'fold_dates': fold_dates,
        'mean_is_sharpe': mean_is,
        'mean_oos_sharpe': mean_oos,
        'degradation_pct': degradation,
        'n_positive_oos': sum(1 for s in oos_sharpes if s > 0),
        'n_folds': len(is_sharpes)
    }


def sharpe_significance_test(returns: pd.Series, null_sharpe: float = 0) -> Tuple[float, float]:
    """Test if Sharpe is statistically significant."""
    if returns.empty or returns.std() == 0:
        return 0, 1.0
    
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    n = len(returns)
    se = np.sqrt((1 + 0.5 * sharpe**2) / n)
    t_stat = (sharpe - null_sharpe) / se
    p_value = 1 - stats.t.cdf(t_stat, n - 1)
    
    return t_stat, p_value


# =============================================================================
# MAIN
# =============================================================================

def run_is_oos_comparison():
    """Run comprehensive IS vs OOS comparison."""
    
    print("=" * 80)
    print("   IN-SAMPLE vs OUT-OF-SAMPLE VALIDATION")
    print("   PDF Strategy vs Asian ETFs (VIX 25)")
    print("=" * 80)
    
    # Fetch data
    print("\n--- Fetching Data ---")
    pdf_prices, pdf_vix = fetch_data(PDF_UNIVERSE, years=5)
    asian_prices, asian_vix = fetch_data(ASIAN_UNIVERSE, years=5)
    
    print(f"   PDF Universe: {len(pdf_prices.columns)} assets, {len(pdf_prices)} days")
    print(f"   Asian Universe: {len(asian_prices.columns)} assets, {len(asian_prices)} days")
    
    # Split points
    pdf_split = int(len(pdf_prices) * 0.7)
    asian_split = int(len(asian_prices) * 0.7)
    
    # =========================================================================
    # 1. IN-SAMPLE vs OUT-OF-SAMPLE (Single Split)
    # =========================================================================
    print("\n" + "=" * 80)
    print("   TEST 1: IN-SAMPLE vs OUT-OF-SAMPLE (70%/30% Split)")
    print("=" * 80)
    
    strategies = {}
    
    # PDF Strategy
    print("\n   PDF Original Strategy (VIX 25):")
    print("   Training (In-Sample)...", end=" ", flush=True)
    pdf_is_result = backtest(pdf_prices.iloc[:pdf_split], pdf_vix.iloc[:pdf_split], vix_threshold=25.0)
    print("Done")
    print("   Testing (Out-of-Sample)...", end=" ", flush=True)
    pdf_oos_result = backtest(pdf_prices.iloc[pdf_split:], pdf_vix.iloc[pdf_split:], vix_threshold=25.0)
    print("Done")
    
    strategies['PDF (VIX 25)'] = {
        'is': pdf_is_result,
        'oos': pdf_oos_result
    }
    
    # Asian ETFs
    print("\n   Asian ETFs Strategy (VIX 25):")
    print("   Training (In-Sample)...", end=" ", flush=True)
    asian_is_result = backtest(asian_prices.iloc[:asian_split], asian_vix.iloc[:asian_split], vix_threshold=25.0)
    print("Done")
    print("   Testing (Out-of-Sample)...", end=" ", flush=True)
    asian_oos_result = backtest(asian_prices.iloc[asian_split:], asian_vix.iloc[asian_split:], vix_threshold=25.0)
    print("Done")
    
    strategies['Asian (VIX 25)'] = {
        'is': asian_is_result,
        'oos': asian_oos_result
    }
    
    # Results table
    print("\n   " + "-" * 75)
    print(f"   {'Strategy':<20} {'IS Sharpe':>12} {'OOS Sharpe':>12} {'Degradation':>12} {'OOS CAGR':>10}")
    print("   " + "-" * 75)
    
    for name, data in strategies.items():
        is_sharpe = data['is']['sharpe']
        oos_sharpe = data['oos']['sharpe']
        degradation = (1 - oos_sharpe / is_sharpe) * 100 if is_sharpe != 0 else 0
        print(f"   {name:<20} {is_sharpe:>12.2f} {oos_sharpe:>12.2f} {degradation:>11.1f}% {data['oos']['cagr']:>9.1%}")
    
    # =========================================================================
    # 2. WALK-FORWARD VALIDATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("   TEST 2: WALK-FORWARD VALIDATION (5 Folds)")
    print("=" * 80)
    
    print("\n   PDF Strategy Walk-Forward...")
    pdf_wf = walk_forward_validation(pdf_prices, pdf_vix, n_splits=5, vix_threshold=25.0)
    
    print("   Asian ETFs Walk-Forward...")
    asian_wf = walk_forward_validation(asian_prices, asian_vix, n_splits=5, vix_threshold=25.0)
    
    print("\n   " + "-" * 75)
    print(f"   {'Strategy':<20} {'Mean IS':>10} {'Mean OOS':>10} {'Degrad.':>10} {'Positive':>10}")
    print("   " + "-" * 75)
    print(f"   {'PDF (VIX 25)':<20} {pdf_wf['mean_is_sharpe']:>10.2f} {pdf_wf['mean_oos_sharpe']:>10.2f} {pdf_wf['degradation_pct']:>9.1f}% {pdf_wf['n_positive_oos']}/{pdf_wf['n_folds']:>8}")
    print(f"   {'Asian (VIX 25)':<20} {asian_wf['mean_is_sharpe']:>10.2f} {asian_wf['mean_oos_sharpe']:>10.2f} {asian_wf['degradation_pct']:>9.1f}% {asian_wf['n_positive_oos']}/{asian_wf['n_folds']:>8}")
    
    # Fold details
    print("\n   Fold-by-Fold OOS Sharpe:")
    print(f"   {'Fold':<8} {'Date':<12} {'PDF OOS':>12} {'Asian OOS':>12} {'Winner':>12}")
    print("   " + "-" * 60)
    
    for i in range(min(len(pdf_wf['oos_sharpes']), len(asian_wf['oos_sharpes']))):
        pdf_oos = pdf_wf['oos_sharpes'][i]
        asian_oos = asian_wf['oos_sharpes'][i]
        date = pdf_wf['fold_dates'][i] if i < len(pdf_wf['fold_dates']) else 'N/A'
        winner = "ASIAN" if asian_oos > pdf_oos else "PDF"
        print(f"   {i+1:<8} {date:<12} {pdf_oos:>12.2f} {asian_oos:>12.2f} {winner:>12}")
    
    # Count wins
    asian_wins = sum(1 for p, a in zip(pdf_wf['oos_sharpes'], asian_wf['oos_sharpes']) if a > p)
    total_folds = min(len(pdf_wf['oos_sharpes']), len(asian_wf['oos_sharpes']))
    
    print(f"\n   Asian wins: {asian_wins}/{total_folds} folds")
    
    # =========================================================================
    # 3. STATISTICAL SIGNIFICANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("   TEST 3: STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    
    # Test OOS Sharpes
    pdf_t, pdf_p = sharpe_significance_test(pdf_oos_result['returns'])
    asian_t, asian_p = sharpe_significance_test(asian_oos_result['returns'])
    
    print(f"\n   {'Strategy':<20} {'OOS Sharpe':>12} {'t-statistic':>12} {'p-value':>12} {'Significant?':>15}")
    print("   " + "-" * 75)
    print(f"   {'PDF (VIX 25)':<20} {pdf_oos_result['sharpe']:>12.2f} {pdf_t:>12.2f} {pdf_p:>12.4f} {'YES' if pdf_p < 0.05 else 'NO':>15}")
    print(f"   {'Asian (VIX 25)':<20} {asian_oos_result['sharpe']:>12.2f} {asian_t:>12.2f} {asian_p:>12.4f} {'YES' if asian_p < 0.05 else 'NO':>15}")
    
    # =========================================================================
    # FINAL VERDICT
    # =========================================================================
    print("\n" + "=" * 80)
    print("   FINAL VERDICT")
    print("=" * 80)
    
    print(f"""
   METRIC                          PDF (VIX 25)    Asian (VIX 25)     WINNER
   -----------------------------------------------------------------------
   OOS Sharpe (30% holdout):       {pdf_oos_result['sharpe']:>10.2f}      {asian_oos_result['sharpe']:>10.2f}         {'ASIAN' if asian_oos_result['sharpe'] > pdf_oos_result['sharpe'] else 'PDF'}
   OOS CAGR:                       {pdf_oos_result['cagr']:>9.1%}       {asian_oos_result['cagr']:>9.1%}          {'ASIAN' if asian_oos_result['cagr'] > pdf_oos_result['cagr'] else 'PDF'}
   OOS Max Drawdown:               {pdf_oos_result['max_dd']:>9.1%}       {asian_oos_result['max_dd']:>9.1%}          {'ASIAN' if asian_oos_result['max_dd'] > pdf_oos_result['max_dd'] else 'PDF'}
   Walk-Forward Mean OOS:          {pdf_wf['mean_oos_sharpe']:>10.2f}      {asian_wf['mean_oos_sharpe']:>10.2f}         {'ASIAN' if asian_wf['mean_oos_sharpe'] > pdf_wf['mean_oos_sharpe'] else 'PDF'}
   Walk-Forward Wins:              {total_folds - asian_wins}/{total_folds}             {asian_wins}/{total_folds}              {'ASIAN' if asian_wins > total_folds/2 else 'PDF'}
   Statistically Significant:      {'YES' if pdf_p < 0.05 else 'NO':>10}      {'YES' if asian_p < 0.05 else 'NO':>10}
   """)
    
    overall_winner = "ASIAN ETFs (VIX 25)" if asian_oos_result['sharpe'] > pdf_oos_result['sharpe'] else "PDF Original"
    
    print(f"   OVERALL WINNER: {overall_winner}")
    print("\n" + "=" * 80)
    
    return {
        'pdf_is': pdf_is_result,
        'pdf_oos': pdf_oos_result,
        'pdf_wf': pdf_wf,
        'asian_is': asian_is_result,
        'asian_oos': asian_oos_result,
        'asian_wf': asian_wf
    }


if __name__ == "__main__":
    results = run_is_oos_comparison()
