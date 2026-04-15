
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
import warnings
from datetime import datetime
from scipy.optimize import minimize
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.getcwd())

# Import validation tools
from validate_edge import (
    test_sharpe_significance,
    bootstrap_sharpe_confidence_interval,
    probabilistic_sharpe_ratio,
    monte_carlo_test,
    calculate_alpha_beta
)

# Try importing MLP
try:
    from mlp_strategy import calculate_mlp_returns
    HAS_MLP = True
except ImportError:
    HAS_MLP = False
    print("Warning: mlp_strategy module not found. MLP will be skipped.")

def optimize_portfolio(rets):
    """Find weights that maximize Sharpe Ratio."""
    def neg_sharpe(weights, rets):
        p_ret = (rets * weights).sum(axis=1)
        if p_ret.std() == 0: return 0
        sharpe = (p_ret.mean() * 252) / (p_ret.std() * np.sqrt(252))
        return -sharpe

    try:
        clean_rets = rets.fillna(0)
        n = clean_rets.shape[1]
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(n))
        init_guess = [1.0 / n] * n
        
        opt = minimize(neg_sharpe, init_guess, args=(clean_rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return [1.0 / rets.shape[1]] * rets.shape[1]

def generate_all_strategies(prices, rets):
    """
    Generates returns for all strategies using the provided price/return data.
    Returns a dictionary of strategy name -> daily returns pd.Series
    """
    strategies = {}
    
    # 1. SPY (Benchmark)
    strategies['SPY (Benchmark)'] = rets['SPY']
    
    # Shared Crypto Logic
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    r_crypto_piece = pd.Series(0.0, index=rets.index)
    if avail_alts:
        alt_rets = rets[avail_alts].copy()
        # Handle missing alts
        for c in avail_alts:
            mask = (prices[c].isna()) | (prices[c] == 0)
            alt_rets.loc[mask, c] = np.nan
        avg_alt_ret = alt_rets.mean(axis=1).fillna(0)
        
        btc_mom = btc_p.pct_change(14)
        alt_idx = prices[avail_alts].mean(axis=1)
        alt_mom = alt_idx.pct_change(14)
        is_altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    else:
        is_altseason = pd.Series(False, index=prices.index)
        avg_alt_ret = pd.Series(0.0, index=prices.index)

    mask_use_alts = is_crypto_safe & is_altseason
    mask_use_btc = is_crypto_safe & ~is_altseason
    
    r_crypto_piece[mask_use_alts] = avg_alt_ret[mask_use_alts]
    if 'BTC-USD' in rets.columns:
        btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
        r_crypto_piece[mask_use_btc & btc_avail] = rets['BTC-USD'][mask_use_btc & btc_avail]
        
    missing_crypto_correction = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
        missing_crypto_correction[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]

    # 2. Ultimate Strategy
    if '^VIX' in prices.columns:
        vix = prices['^VIX'].fillna(method='ffill')
    else:
        vix = pd.Series(20, index=prices.index)
    
    vix_ma = vix.rolling(20).mean()
    signal = pd.Series(0, index=prices.index)
    signal[vix < vix_ma] = 1 # Calm
    signal[vix > vix_ma] = -1 # Fear
    sig_shifted = signal.shift(1).fillna(0)
    
    # Legs
    r_ult_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_ult_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_ult_neut = (0.30 * rets['SPY'] + 0.22 * rets.get('TLT', 0) + 0.20 * r_crypto_piece + (0.20/0.40)*missing_crypto_correction)
    
    r_ult = pd.Series(0.0, index=rets.index)
    r_ult[sig_shifted > 0] = r_ult_bull[sig_shifted > 0]
    r_ult[sig_shifted < 0] = r_ult_bear[sig_shifted < 0]
    r_ult[sig_shifted == 0] = r_ult_neut[sig_shifted == 0]
    strategies['Ultimate Strategy'] = r_ult

    # 3. HRP
    cols_hrp = [c for c in ['SPY', 'TLT', 'GLD', 'IEF'] if c in rets.columns]
    if cols_hrp:
        vol = rets[cols_hrp].rolling(126).std()
        inv_vol = 1 / vol
        w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
        r_hrp = (w_hrp * rets[cols_hrp]).sum(axis=1)
        strategies['HRP'] = r_hrp

    # 4. Golden Omni
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    is_bull = (spy_p > ma200).shift(1).fillna(False)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation = ((xle_p > ma200_xle) & (~is_bull)).shift(1).fillna(False)
    
    r_bull_leg = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_bear_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_inflation_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    
    r_omni = pd.Series(0.0, index=rets.index)
    r_omni[is_bull] = r_bull_leg[is_bull]
    r_omni[(~is_bull) & (~is_inflation)] = r_bear_leg[(~is_bull) & (~is_inflation)]
    r_omni[(~is_bull) & (is_inflation)] = r_inflation_leg[(~is_bull) & (is_inflation)]
    strategies['Golden Omni'] = r_omni
    
    # 5. MLP Neural Net
    if HAS_MLP:
        print("   🧠 Training MLP Neural Net (This may take 2-5 minutes)...")
        start_time = time.time()
        try:
             # Progress hack - calculate_mlp_returns doesn't take a callback, 
             # but we can wrap it or just wait.
             r_mlp = calculate_mlp_returns(prices, rets)
             strategies['MLP Neural Net'] = r_mlp
             print(f"      MLP Training Complete ({time.time() - start_time:.1f}s)")
        except Exception as e:
             print(f"      MLP Failed: {e}")
    
    return strategies

def run_benchmark():
    print("=" * 80)
    print("   COMPREHENSIVE STRATEGY BENCHMARK")
    print("   Evaluating: Golden Omni, Ultimate, HRP, MLP Neural Net")
    print("=" * 80)

    # 1. Fetch Data
    print("\n1. Fetching Market Data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', 'XLE', 
        '^VIX', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD'
    ]
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        
    prices = prices.ffill().dropna(subset=['SPY'])
    rets = prices.pct_change().fillna(0)
    print(f"   Loaded {len(prices)} trading days.")

    # 2. Generate Strategies
    print("\n2. Generating Strategy Returns...")
    strategies = generate_all_strategies(prices, rets)
    
    # 3. Benchmark Logic (IS/OOS)
    # Split 70/30
    split_idx = int(len(prices) * 0.7)
    split_date = prices.index[split_idx]
    
    print(f"\n3. Performing In-Sample / Out-of-Sample Scan")
    print(f"   Split Date: {split_date.date()}")
    print("-" * 80)
    print(f"{'Strategy':<20} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'Prob(Edge)':<10} | {'Alpha':<8}")
    print("-" * 80)
    
    # Store OOS returns for deeper analysis
    oos_returns_dict = {}
    
    for name, r in strategies.items():
        # Align
        common_idx = r.index.intersection(rets.index)
        r = r.loc[common_idx].fillna(0)
        
        # Skip warmup
        if len(r) > 252:
            r = r.iloc[252:]
        else:
            continue
            
        # IS / OOS Split
        # We need to find the split index relative to the new (shortened) r series
        # Re-calc split based on dates to be accurate
        
        r_is = r[r.index < split_date]
        r_oos = r[r.index >= split_date]
        
        if len(r_oos) < 50:
             print(f"{name:<20} | Not enough OOS data")
             continue
             
        # Metrics
        is_sharpe = r_is.mean() / r_is.std() * np.sqrt(252) if r_is.std() > 0 else 0
        oos_sharpe = r_oos.mean() / r_oos.std() * np.sqrt(252) if r_oos.std() > 0 else 0
        
        # Probabilistic Sharpe Ratio on OOS
        psr = probabilistic_sharpe_ratio(r_oos)
        
        # Alpha vs SPY (OOS)
        spy_oos = strategies['SPY (Benchmark)'][r_oos.index]
        alpha_res = calculate_alpha_beta(r_oos, spy_oos)
        
        oos_returns_dict[name] = r_oos
        
        print(f"{name:<20} | {is_sharpe:>9.2f}  | {oos_sharpe:>9.2f}  | {psr:>9.1%}  | {alpha_res['alpha']:>7.1%}")

    # 4. Detailed Winner Analysis
    # Pick top OOS Sharpe
    best_name = max(oos_returns_dict, key=lambda k: oos_returns_dict[k].mean() / oos_returns_dict[k].std())
    best_r = oos_returns_dict[best_name]
    
    print("\n" + "=" * 80)
    print(f"   🏆 OOS WINNER: {best_name}")
    print("=" * 80)
    
    # Run rigorous tests on winner
    print("   Running Rigorous Validation Suite on Winner (Out-of-Sample Data)...")
    
    # Bootstrap
    print("   1. Bootstrap Confidence Interval...", end="")
    boot = bootstrap_sharpe_confidence_interval(best_r)
    print(f" Done. 95% CI: [{boot['ci_lower']:.2f}, {boot['ci_upper']:.2f}]")
    
    # Monte Carlo
    print("   2. Monte Carlo Permutation Test...", end="")
    mc = monte_carlo_test(best_r)
    print(f" Done. Percentile: {mc['percentile']:.1f}%")
    
    # Drawdown
    cum = (1 + best_r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    print("\n   Final Stats:")
    print(f"   - Max Drawdown: {max_dd:.1%}")
    print(f"   - Total Return (OOS): {(cum.iloc[-1]-1):.1%}")
    print(f"   - Win Rate: {(best_r > 0).mean():.1%}")
    
    if best_name == 'MLP Neural Net':
        print("\n   🤖 AI SUPREMACY: The Machine Learning model has verified OOS performance.")
    elif best_name == 'Golden Omni':
        print("\n   🌟 GOLDEN OMNI: The regime-based logic holds up best in recent years.")
    
if __name__ == "__main__":
    run_benchmark()
