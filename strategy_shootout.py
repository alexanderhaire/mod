"""
Strategy Shootout: The Final Verdict
====================================

Rigorously compares all "Alpha Hunt" strategies using advanced statistics.
Replicates logic from `pages/live_simulation.py` to ensure 1:1 fidelity.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import warnings
import sys
import os

# Add current directory to path to find modules
sys.path.append(os.getcwd())

try:
    from mlp_strategy import calculate_mlp_returns
except ImportError:
    # Handle if mlp_strategy is not found
    print("Warning: mlp_strategy module not found. MLP will be skipped.")
    def calculate_mlp_returns(p, r): return r['SPY']

from validate_edge import (
    test_sharpe_significance,
    bootstrap_sharpe_confidence_interval,
    probabilistic_sharpe_ratio,
    monte_carlo_test,
    calculate_alpha_beta
)

warnings.filterwarnings('ignore')

def optimize_portfolio(rets):
    """Find weights that maximize Sharpe Ratio."""
    def neg_sharpe(weights, rets):
        p_ret = (rets * weights).sum(axis=1)
        mean = p_ret.mean() * 252
        if p_ret.std() == 0: return 0
        vol = p_ret.std() * np.sqrt(252)
        sharpe = mean / vol if vol > 0 else 0
        return -sharpe

    try:
        clean_rets = rets.fillna(0)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for _ in range(clean_rets.shape[1]))
        init_guess = [1.0 / clean_rets.shape[1]] * clean_rets.shape[1]
        
        if len(clean_rets) < 20: return init_guess
        
        opt = minimize(neg_sharpe, init_guess, args=(clean_rets,), method='SLSQP', bounds=bnds, constraints=cons)
        return opt.x
    except:
        return [1.0 / rets.shape[1]] * rets.shape[1]

def generate_strategy_returns():
    """
    Replicates logic from live_simulation.py to generate DAILY RETURNS for all strategies.
    Returns: Dict[str, pd.Series]
    """
    print("🚀 Fetching market data...")
    tickers = [
        'SPY', 'TLT', 'GLD', 'IEF', 'QQQ', 'UUP', 'XLE', # Trad
        'VUG', 'VTV', 'RSP', # Factors/Breadth
        '^FVX', '^TYX', '^VIX', # Macro
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD' # Crypto
    ]
    # Fetch long history for full context
    data = yf.download(tickers, start='2005-01-01', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
             prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        except:
             prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # Forward fill and drop rows where SPY is missing
    prices = prices.ffill()
    prices = prices.dropna(subset=['SPY'])
    
    rets = prices.pct_change().fillna(0)
    
    # 1. SPY
    r_spy = rets['SPY']
    
    # 2. Crypto Components
    btc_p = prices.get('BTC-USD', pd.Series(np.nan, index=prices.index))
    btc_vol = btc_p.pct_change().rolling(30).std() * np.sqrt(365) * 100
    is_crypto_safe = (btc_vol < 100).shift(1).fillna(True)
    
    alts = ['ETH-USD', 'SOL-USD', 'DOGE-USD']
    avail_alts = [c for c in alts if c in prices.columns]
    
    alt_rets = rets[avail_alts].copy()
    for c in avail_alts:
        mask = (prices[c].isna()) | (prices[c] == 0)
        alt_rets.loc[mask, c] = np.nan
        
    avg_alt_ret = alt_rets.mean(axis=1).fillna(0)
    
    r_crypto_piece = pd.Series(0.0, index=rets.index)
    btc_mom = btc_p.pct_change(14)
    
    if avail_alts:
        alt_idx = prices[avail_alts].mean(axis=1)
        alt_mom = alt_idx.pct_change(14)
        is_altseason = (alt_mom > btc_mom).shift(1).fillna(False)
    else:
        is_altseason = pd.Series(False, index=prices.index)
        
    mask_use_alts = is_crypto_safe & is_altseason
    mask_use_btc = is_crypto_safe & ~is_altseason
    
    r_crypto_piece[mask_use_alts] = avg_alt_ret[mask_use_alts]
    
    btc_avail = prices['BTC-USD'].notna() & (prices['BTC-USD'] > 0)
    if 'BTC-USD' in rets.columns:
        r_crypto_piece[mask_use_btc] = rets['BTC-USD'][mask_use_btc]
    
    missing_crypto_correction = pd.Series(0.0, index=rets.index)
    if 'BTC-USD' in prices.columns:
        missing_crypto_correction[~btc_avail] = 0.40 * rets['SPY'][~btc_avail]
        
    # 3. Ultimate Strategy
    if '^VIX' not in prices.columns:
        vix = pd.Series(20, index=prices.index)
    else:
        vix = prices['^VIX'].fillna(method='ffill')
    
    vix_ma = vix.rolling(20).mean()
    signal = pd.Series(0, index=prices.index)
    signal[vix < vix_ma] = 1 # Calm
    signal[vix > vix_ma] = -1 # Fear
    
    # Ensure all rets components align
    r_ult_bull = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_ult_bear = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
    r_ult_neut = (0.30 * rets['SPY'] + 0.22 * rets.get('TLT', 0) + 0.20 * r_crypto_piece + (0.20/0.40)*missing_crypto_correction)
    
    sig_shifted = signal.shift(1).fillna(0)
    r_ult = pd.Series(0.0, index=rets.index)
    r_ult[sig_shifted > 0] = r_ult_bull[sig_shifted > 0]
    r_ult[sig_shifted < 0] = r_ult_bear[sig_shifted < 0]
    r_ult[sig_shifted == 0] = r_ult_neut[sig_shifted == 0]
    
    # 4. HRP
    cols_hrp = [c for c in ['SPY', 'TLT', 'GLD', 'IEF'] if c in rets.columns]
    if cols_hrp:
        vol = rets[cols_hrp].rolling(126).std()
        inv_vol = 1 / vol
        w_hrp = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1).fillna(0)
        r_hrp = (w_hrp * rets[cols_hrp]).sum(axis=1)
    else:
        r_hrp = r_spy
    
    # 5. Dollar Trend
    if 'UUP' in prices.columns:
        uup = prices['UUP']
        r_uup = pd.Series(0.0, index=prices.index)
        uup_ma = uup.rolling(200).mean()
        is_uptrend = (uup > uup_ma).shift(1).fillna(False)
        r_uup[is_uptrend] = rets['UUP'][is_uptrend]
    else:
        r_uup = pd.Series(0.0, index=prices.index)
        
    # 6. Golden Omni
    spy_p = prices['SPY']
    ma200 = spy_p.rolling(200).mean()
    
    # Raw Daily Signals
    is_bull_daily = (spy_p > ma200)
    
    xle_p = prices.get('XLE', spy_p)
    ma200_xle = xle_p.rolling(200).mean()
    is_inflation_daily = (xle_p > ma200_xle) & (~is_bull_daily)
    
    # WEEKLY REBALANCING (Friday Close)
    # We resample the signal to Friday, then forward fill it for the next week.
    bull_weekly = is_bull_daily.resample('W-FRI').last()
    inf_weekly = is_inflation_daily.resample('W-FRI').last()
    
    is_bull = bull_weekly.reindex(is_bull_daily.index, method='ffill').shift(1).fillna(False)
    is_inflation = inf_weekly.reindex(is_inflation_daily.index, method='ffill').shift(1).fillna(False)
    
    try:
        r_bull_leg = (0.45 * rets['SPY'] + 0.10 * rets.get('TLT', 0) + 0.05 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
        r_bear_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('TLT', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
        r_inflation_leg = (0.15 * rets['SPY'] + 0.35 * rets.get('XLE', 0) + 0.10 * rets.get('GLD', 0) + 0.40 * r_crypto_piece + missing_crypto_correction)
        
        r_omni = pd.Series(0.0, index=rets.index)
        r_omni[is_bull] = r_bull_leg[is_bull]
        r_omni[(~is_bull) & (~is_inflation)] = r_bear_leg[(~is_bull) & (~is_inflation)]
        r_omni[(~is_bull) & (is_inflation)] = r_inflation_leg[(~is_bull) & (is_inflation)]
    except Exception as e:
        print(f"Error calculating Omni: {e}")
        r_omni = r_spy
    
    # 7. Crypto Comp
    r_cc = pd.Series(0.0, index=rets.index)
    dow = r_cc.index.dayofweek
    mask_thurs = (dow == 3)
    mask_prime = (dow.isin([0, 2]))
    mask_other = (~mask_thurs) & (~mask_prime)
    
    r_cc[mask_thurs] = 0.0
    r_cc[mask_prime] = r_crypto_piece[mask_prime]
    if 'BTC-USD' in rets.columns:
        r_cc[mask_other] = 0.7 * rets['BTC-USD'][mask_other]
        
    # 8. ETH/BTC Flipper
    r_flip = pd.Series(0.0, index=rets.index)
    if 'ETH-USD' in prices.columns and 'BTC-USD' in prices.columns:
        ratio = prices['ETH-USD'] / prices['BTC-USD']
        z_score = (ratio - ratio.rolling(90).mean()) / ratio.rolling(90).std()
        
        state = pd.Series(np.nan, index=rets.index)
        state[z_score < -1.5] = 1 # Cheap ETH
        state[z_score > 1.5] = -1 # Expensive ETH
        state = state.ffill().shift(1).fillna(-1)
        
        r_flip[state == 1] = rets['ETH-USD'][state == 1]
        r_flip[state == -1] = rets['BTC-USD'][state == -1]
    elif 'BTC-USD' in rets.columns:
        r_flip[:] = rets['BTC-USD']
        
    # 9. Dog Flipper
    r_dog_flip = pd.Series(0.0, index=rets.index)
    if 'DOGE-USD' in prices.columns and 'BTC-USD' in prices.columns:
        dog_ratio = prices['DOGE-USD'] / prices['BTC-USD']
        # Fix infs
        dog_ratio = dog_ratio.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        
        dr_z = (dog_ratio - dog_ratio.rolling(90).mean()) / dog_ratio.rolling(90).std()
        
        d_state = pd.Series(np.nan, index=rets.index)
        d_state[dr_z < -1.5] = 1
        d_state[dr_z > 1.5] = -1
        d_state = d_state.ffill().shift(1).fillna(-1)
        
        doge_ok = prices['DOGE-USD'].notna() & (prices['DOGE-USD'] > 0)
        
        # Handle cases where DOGE is not yet available
        mask_doge = (d_state == 1) & doge_ok
        mask_btc = (d_state == -1) | ((d_state == 1) & (~doge_ok))
        
        if mask_doge.any():
            r_dog_flip[mask_doge] = rets['DOGE-USD'][mask_doge]
        if mask_btc.any():
            r_dog_flip[mask_btc] = rets['BTC-USD'][mask_btc]

    elif 'BTC-USD' in rets.columns:
        r_dog_flip[:] = rets['BTC-USD']
        
    # 10. Trifecta
    strat_rets = pd.DataFrame({'Ultimate': r_ult, 'HRP': r_hrp, 'Dollar Trend': r_uup})
    strat_rets = strat_rets.loc[(strat_rets != 0).any(axis=1)]
    
    r_tri = r_ult # fallback
    if not strat_rets.empty:
        try:
           opt_w = optimize_portfolio(strat_rets)
           r_tri = (opt_w[0] * r_ult + opt_w[1] * r_hrp + opt_w[2] * r_uup)
        except:
           pass
        
    # 11. MLP Neural Net
    print("🧠 Running MLP Neural Net (this takes a moment)...")
    try:
        r_mlp = calculate_mlp_returns(prices, rets)
    except Exception as e:
        print(f"MLP Error: {e}")
        r_mlp = r_spy # Fallback
        
    return {
        'SPY (Benchmark)': r_spy,
        'Golden Omni': r_omni,
        'Trifecta': r_tri,
        'Ultimate': r_ult,
        'MLP Neural Net': r_mlp,
        'Crypto Comp': r_cc,
        'ETH/BTC Flipper': r_flip,
        'Dog Flipper': r_dog_flip,
        'HRP': r_hrp,
        'Dollar Trend': r_uup
    }

def run_shootout():
    strategies = generate_strategy_returns()
    
    print("\n" + "=" * 80)
    print(f"   🥊 STRATEGY SHOOTOUT: STATISTICAL RANKING")
    print("=" * 80)
    
    # Dataframe to store metrics
    metrics = []
    
    # Need SPY for Alpha calc - ensure alignment
    spy_full = strategies['SPY (Benchmark)']
    
    for name, r in strategies.items():
        # Clean series, drop leading zeros/NaNs
        # BUT we must align with SPY for Alpha calc, so let's keep index intersection
        common_idx = r.index.intersection(spy_full.index)
        r = r.loc[common_idx].fillna(0)
        spy_curr = spy_full.loc[common_idx].fillna(0)
        
        # Trim warmup (first 252 days usually volatile/init)
        if len(r) > 252:
            r = r.iloc[252:]
            spy_curr = spy_curr.iloc[252:]
        else:
            continue
            
        # Analyze using validate_edge logic
        if r.std() == 0: continue
        
        sharpe = r.mean() / r.std() * np.sqrt(252)
        psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0)
        
        # Bootstrap
        boot = bootstrap_sharpe_confidence_interval(r, n_bootstrap=2000)
        
        # Drawdown
        cum = (1 + r).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        
        # Annual Return (CAGR)
        total_ret = cum.iloc[-1]
        years = len(cum) / 252
        cagr = total_ret**(1/years) - 1
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Alpha
        alpha_res = calculate_alpha_beta(r, spy_curr)
        
        metrics.append({
            'Strategy': name,
            'Sharpe': sharpe,
            'PSR': psr,
            'MaxDD': max_dd,
            'Calmar': calmar,
            'Alpha': alpha_res['alpha'],
            'Alpha Sig': alpha_res['alpha_significant'],
            'Lower CI': boot['ci_lower'],
            'Upper CI': boot['ci_upper']
        })
        
    # Convert to DF
    df_metrics = pd.DataFrame(metrics)
    
    # Sort by Sharpe
    df_metrics = df_metrics.sort_values('Sharpe', ascending=False).reset_index(drop=True)
    
    # Print Table
    print(f"\n{'Rank':<4} {'Strategy':<20} {'Sharpe':>8} {'PSR':>8} {'MaxDD':>10} {'Calmar':>8} {'Alpha':>8}")
    print("-" * 80)
    
    for i, row in df_metrics.iterrows():
        name = row['Strategy']
        marker = "👑" if i == 0 else ""
        if row['Strategy'] == 'SPY (Benchmark)': marker = "🛡️"
        
        print(f"{i+1:<4} {marker}{name:<19} {row['Sharpe']:>8.2f} {row['PSR']:>7.0%} {row['MaxDD']:>9.1%} {row['Calmar']:>8.2f} {row['Alpha']:>7.1%}")
        
    # Detailed Verdict
    if not df_metrics.empty:
        winner = df_metrics.iloc[0]
        print("\n" + "=" * 80)
        print(f"   🏆 WINNER: {winner['Strategy']}")
        print("=" * 80)
        print(f"   Detailed Stats for {winner['Strategy']}:")
        print(f"   - Sharpe Ratio: {winner['Sharpe']:.2f} (Outstanding > 1.0, Excellent > 2.0)")
        print(f"   - Confidence (PSR): {winner['PSR']:.1%} (Probability that Sharpe > 0)")
        print(f"   - Confidence Interval (95%): [{winner['Lower CI']:.2f}, {winner['Upper CI']:.2f}]")
        print(f"   - Alpha vs SPY: {winner['Alpha']:.1%} ({'Significant' if winner['Alpha Sig'] else 'Not Significant'})")
        print(f"   - Risk: Max Drawdown of {winner['MaxDD']:.1%} (Calmar: {winner['Calmar']:.2f})")
        
        if winner['Strategy'] == 'MLP Neural Net':
            print("\n   🤖 AI VERDICT: The Machine Learning approach has validated its edge.")
            print("   The combination of regime detection and seasonal features provides superior risk-adjusted returns.")
        elif winner['Strategy'] == 'Golden Omni':
            print("\n   🌟 VERDICT: The 'Golden Omni' remains the king.")
            print("   Human intuition on inflation and crypto cycles still beats the black box.")
    else:
        print("No metrics generated.")
    
if __name__ == "__main__":
    run_shootout()
