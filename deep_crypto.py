"""
Deep Crypto Pattern Mining
============================

Digging deeper into crypto patterns:
1. Day-of-Week Exploitation (that Wed=+142%, Thu=-62% finding)
2. Weekend vs Weekday patterns
3. Altcoin Momentum (DOGE, SOL, etc.)
4. Hour-of-day proxies (via open/close gaps)
5. BTC Dominance cycles
6. Alt-season detection

RUN: python deep_crypto.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA
# =============================================================================

def fetch_crypto_data():
    print("📊 Fetching extended crypto data...")
    
    # Major cryptos available on yfinance
    cryptos = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD',
        'XRP-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD',
        'SHIB-USD', 'LTC-USD', 'UNI-USD', 'ATOM-USD',
    ]
    
    # Also get SPY for correlation
    tickers = cryptos + ['SPY', 'TLT', 'GLD']
    
    data = yf.download(tickers, start='2020-01-01', progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    prices = prices.ffill()
    
    # Keep only columns with enough data
    valid_cols = prices.columns[prices.count() > 500]
    prices = prices[valid_cols].dropna()
    
    print(f"   Data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Assets: {list(prices.columns)}")
    
    return prices

# =============================================================================
# DAY OF WEEK STRATEGIES
# =============================================================================

def day_of_week_crypto_strategy(prices, crypto='BTC-USD'):
    """
    Exploit day-of-week patterns in crypto.
    Long on historically good days, flat on bad days.
    """
    if crypto not in prices.columns:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        dow = prices.index[i].dayofweek
        
        # Based on our analysis: Wed=best, Mon=2nd, Thu=worst
        if dow in [0, 2]:  # Monday, Wednesday - best days
            weights.iloc[i][crypto] = 1.0
        elif dow == 3:  # Thursday - worst day
            weights.iloc[i][crypto] = 0.0  # Avoid
        else:
            weights.iloc[i][crypto] = 0.5  # Reduced exposure
    
    return weights.shift(1).fillna(0)

def avoid_thursday_strategy(prices, crypto='BTC-USD'):
    """Simply avoid Thursday, full exposure other days."""
    if crypto not in prices.columns:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        dow = prices.index[i].dayofweek
        
        if dow == 3:  # Thursday
            weights.iloc[i][crypto] = 0.0
        else:
            weights.iloc[i][crypto] = 1.0
    
    return weights.shift(1).fillna(0)

def weekend_crypto_strategy(prices, crypto='BTC-USD'):
    """
    Weekend vs Weekday patterns.
    Crypto trades 24/7 - weekends often behave differently.
    """
    if crypto not in prices.columns:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(60, len(prices)):
        dow = prices.index[i].dayofweek
        
        if dow >= 5:  # Weekend (5=Sat, 6=Sun)
            weights.iloc[i][crypto] = 0.8  # Slightly reduced
        else:  # Weekday
            weights.iloc[i][crypto] = 1.0
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ALTCOIN STRATEGIES
# =============================================================================

def btc_dominance_signal(prices):
    """
    BTC Dominance proxy: BTC vs total crypto cap (approximated by BTC vs ETH+alts).
    High BTC dominance = risk-off in crypto
    Low BTC dominance = altseason
    """
    if 'BTC-USD' not in prices.columns or 'ETH-USD' not in prices.columns:
        return None
    
    # Use BTC/ETH ratio as dominance proxy
    btc = prices['BTC-USD']
    eth = prices['ETH-USD']
    
    ratio = btc / eth
    ratio_ma = ratio.rolling(30).mean()
    
    signal = pd.Series(0.0, index=prices.index)
    
    for i in range(30, len(prices)):
        if ratio.iloc[i] > ratio_ma.iloc[i] * 1.1:
            signal.iloc[i] = -1  # BTC dominance rising = risk-off
        elif ratio.iloc[i] < ratio_ma.iloc[i] * 0.9:
            signal.iloc[i] = 1   # ETH outperforming = altseason
    
    return signal

def altcoin_momentum_strategy(prices, lookback=30, top_n=3):
    """
    Altcoin momentum: buy the top performing alts.
    """
    alts = [c for c in prices.columns if c.endswith('-USD') and c not in ['BTC-USD', 'ETH-USD']]
    
    if len(alts) < 3:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(lookback + 10, len(prices)):
        # Calculate momentum for all alts
        mom = {}
        for alt in alts:
            if prices[alt].iloc[i-lookback:i].count() > lookback * 0.8:
                try:
                    ret = prices[alt].iloc[i] / prices[alt].iloc[i-lookback] - 1
                    mom[alt] = ret
                except:
                    pass
        
        if len(mom) >= top_n:
            # Pick top N
            sorted_mom = sorted(mom.items(), key=lambda x: x[1], reverse=True)
            top_alts = [x[0] for x in sorted_mom[:top_n]]
            
            for alt in top_alts:
                weights.iloc[i][alt] = 1.0 / top_n
    
    return weights.shift(1).fillna(0)

def btc_eth_rotation(prices, lookback=20):
    """
    Simple BTC/ETH rotation based on momentum.
    """
    if 'BTC-USD' not in prices.columns or 'ETH-USD' not in prices.columns:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    eth_mom = prices['ETH-USD'].pct_change(lookback)
    
    for i in range(lookback + 10, len(prices)):
        if btc_mom.iloc[i] > eth_mom.iloc[i]:
            weights.iloc[i]['BTC-USD'] = 1.0
        else:
            weights.iloc[i]['ETH-USD'] = 1.0
    
    return weights.shift(1).fillna(0)

def altseason_detector_strategy(prices, lookback=14):
    """
    Altseason strategy:
    - When alts are outperforming BTC, load up on alts
    - When BTC is outperforming, stick with BTC
    """
    alts = [c for c in prices.columns if c.endswith('-USD') and c != 'BTC-USD']
    
    if 'BTC-USD' not in prices.columns or len(alts) < 3:
        return None
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    btc_mom = prices['BTC-USD'].pct_change(lookback)
    
    for i in range(lookback + 10, len(prices)):
        # Calculate average alt momentum
        alt_moms = []
        for alt in alts:
            try:
                ret = prices[alt].iloc[i] / prices[alt].iloc[i-lookback] - 1
                if not np.isnan(ret):
                    alt_moms.append(ret)
            except:
                pass
        
        avg_alt_mom = np.mean(alt_moms) if len(alt_moms) > 0 else 0
        
        if avg_alt_mom > btc_mom.iloc[i] * 1.2:
            # Altseason: spread across top alts
            sorted_alts = sorted(zip(alts, [prices[a].iloc[i]/prices[a].iloc[i-lookback]-1 for a in alts if prices[a].iloc[i-lookback:i].count() > lookback*0.8]), 
                                key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
            top_3 = [x[0] for x in sorted_alts[:3] if not np.isnan(x[1])]
            for alt in top_3:
                weights.iloc[i][alt] = 1.0 / len(top_3)
        else:
            # BTC dominance: stick with BTC
            weights.iloc[i]['BTC-USD'] = 1.0
    
    return weights.shift(1).fillna(0)

# =============================================================================
# COMBINED STRATEGIES
# =============================================================================

def combined_crypto_strategy(prices):
    """
    Combine: Day-of-week + Altseason + Momentum
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    alts = [c for c in prices.columns if c.endswith('-USD') and c != 'BTC-USD']
    
    btc_mom = prices['BTC-USD'].pct_change(14) if 'BTC-USD' in prices.columns else None
    
    for i in range(60, len(prices)):
        dow = prices.index[i].dayofweek
        
        # Day-of-week filter
        if dow == 3:  # Thursday = worst day
            # Stay in stablecoins (cash proxy - just don't allocate)
            continue
        
        # Altseason detection
        alt_moms = []
        for alt in alts:
            try:
                ret = prices[alt].iloc[i] / prices[alt].iloc[i-14] - 1
                if not np.isnan(ret):
                    alt_moms.append((alt, ret))
            except:
                pass
        
        avg_alt_mom = np.mean([x[1] for x in alt_moms]) if len(alt_moms) > 0 else 0
        btc_m = btc_mom.iloc[i] if btc_mom is not None and i < len(btc_mom) else 0
        
        if dow in [0, 2]:  # Best days - full exposure
            if avg_alt_mom > btc_m * 1.1:
                # Altseason
                sorted_alts = sorted(alt_moms, key=lambda x: x[1], reverse=True)[:3]
                for alt, _ in sorted_alts:
                    weights.iloc[i][alt] = 1.0 / len(sorted_alts)
            else:
                weights.iloc[i]['BTC-USD'] = 1.0
        else:
            # Other days - reduced exposure
            weights.iloc[i]['BTC-USD'] = 0.7
    
    return weights.shift(1).fillna(0)

# =============================================================================
# ANALYSIS
# =============================================================================

def compute_returns(prices, weights, warmup=60):
    returns = prices.pct_change()
    weights = weights.iloc[warmup:]
    returns = returns.iloc[warmup:]
    
    common = weights.columns.intersection(returns.columns)
    abs_sum = weights[common].abs().sum(axis=1).replace(0, 1)
    
    port_ret = (weights[common].shift(1) * returns[common]).sum(axis=1)
    
    # Handle days with no position (keep as 0 return)
    return port_ret.dropna()

def compute_metrics(returns):
    if len(returns) < 20:
        return None
    
    # Filter out zero returns (no position days)
    active_returns = returns[returns != 0]
    if len(active_returns) < 20 or active_returns.std() == 0:
        return None
    
    sharpe = active_returns.mean() / active_returns.std() * np.sqrt(252)
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
    
    cum = (1 + returns).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    
    return {'sharpe': sharpe, 'cagr': cagr * 100, 'max_dd': max_dd * 100, 'active_days': len(active_returns)}

def compute_active_stats(r_c, r_b):
    common_idx = r_b.index.intersection(r_c.index)
    active = r_c.loc[common_idx] - r_b.loc[common_idx]
    active = active.dropna()
    
    if len(active) < 20 or active.std() == 0:
        return None
    
    ir = active.mean() / active.std() * np.sqrt(252)
    t_stat = active.mean() / (active.std() / np.sqrt(len(active)))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(active)-1))
    
    return {'ir': ir, 't_stat': t_stat, 'p_val': p_val}

# =============================================================================
# DIRECT ANALYSIS
# =============================================================================

def analyze_all_cryptos_dow(prices):
    """Analyze day-of-week patterns for all cryptos."""
    cryptos = [c for c in prices.columns if c.endswith('-USD')]
    
    results = {}
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for crypto in cryptos:
        ret = prices[crypto].pct_change().dropna()
        
        dow_data = {}
        for dow in range(7):
            d_ret = ret[ret.index.dayofweek == dow]
            if len(d_ret) > 50:
                dow_data[dow_names[dow]] = {
                    'mean': d_ret.mean() * 365 * 100,
                    'sharpe': d_ret.mean() / d_ret.std() * np.sqrt(365) if d_ret.std() > 0 else 0
                }
        
        if len(dow_data) == 7:
            results[crypto] = dow_data
    
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("   DEEP CRYPTO PATTERN MINING")
    print("   Searching for exploitable patterns")
    print("=" * 80)
    
    prices = fetch_crypto_data()
    
    # Day-of-week analysis for all cryptos
    print("\n" + "=" * 80)
    print("   DAY-OF-WEEK PATTERNS ACROSS CRYPTOS")
    print("=" * 80)
    
    dow_results = analyze_all_cryptos_dow(prices)
    
    print(f"\n   {'Crypto':<12} {'Best Day':>10} {'Return':>10} {'Worst Day':>10} {'Return':>10}")
    print("   " + "-" * 60)
    
    for crypto, dow_data in list(dow_results.items())[:8]:
        sorted_days = sorted(dow_data.items(), key=lambda x: x[1]['mean'], reverse=True)
        best = sorted_days[0]
        worst = sorted_days[-1]
        print(f"   {crypto:<12} {best[0]:>10} {best[1]['mean']:>9.1f}% {worst[0]:>10} {worst[1]['mean']:>9.1f}%")
    
    # Build strategies
    print("\n📈 Building strategies...")
    
    strategies = {
        'BTC Buy-Hold': pd.DataFrame({c: 0.0 for c in prices.columns}, index=prices.index).assign(**{'BTC-USD': 1.0}).shift(1).fillna(0) if 'BTC-USD' in prices.columns else None,
        'BTC Avoid Thursday': avoid_thursday_strategy(prices),
        'BTC DOW Optimized': day_of_week_crypto_strategy(prices),
        'BTC/ETH Rotation': btc_eth_rotation(prices),
        'Altcoin Momentum': altcoin_momentum_strategy(prices),
        'Altseason Detector': altseason_detector_strategy(prices),
        'Combined Crypto': combined_crypto_strategy(prices),
    }
    
    strategies = {k: v for k, v in strategies.items() if v is not None}
    
    # Test
    print("\n" + "=" * 80)
    print("   STRATEGY COMPARISON")
    print("=" * 80)
    
    results = {}
    
    print(f"\n   {'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("   " + "-" * 60)
    
    for name, weights in strategies.items():
        returns = compute_returns(prices, weights)
        metrics = compute_metrics(returns)
        results[name] = {'metrics': metrics, 'returns': returns}
        
        if metrics:
            print(f"   {name:<25} {metrics['sharpe']:>10.2f} {metrics['cagr']:>9.1f}% {metrics['max_dd']:>9.1f}%")
    
    # Statistical significance vs buy-hold
    print("\n" + "=" * 80)
    print("   SIGNIFICANCE vs BTC BUY-HOLD")
    print("=" * 80)
    
    print(f"\n   {'Strategy':<25} {'IR':>8} {'t-stat':>8} {'p-val':>8} {'Sig':>5}")
    print("   " + "-" * 60)
    
    base_ret = results['BTC Buy-Hold']['returns'] if 'BTC Buy-Hold' in results else None
    
    best_pval = 1.0
    best_strat = None
    
    if base_ret is not None:
        for name in strategies.keys():
            if name == 'BTC Buy-Hold':
                continue
            
            strat_ret = results[name]['returns']
            stats_result = compute_active_stats(strat_ret, base_ret)
            
            if stats_result:
                sig = "**" if stats_result['p_val'] < 0.05 else "*" if stats_result['p_val'] < 0.10 else ""
                print(f"   {name:<25} {stats_result['ir']:>8.2f} {stats_result['t_stat']:>8.2f} {stats_result['p_val']:>8.3f} {sig}")
                
                if stats_result['p_val'] < best_pval:
                    best_pval = stats_result['p_val']
                    best_strat = name
    
    print("\n   ** p < 0.05, * p < 0.10")
    
    # Verdict
    print("\n" + "=" * 80)
    print("   VERDICT")
    print("=" * 80)
    
    if best_pval < 0.05:
        print(f"\n   ✅ SIGNIFICANT: {best_strat} (p={best_pval:.3f})")
        print("   We found a real crypto edge! 🚀💰")
    elif best_pval < 0.10:
        print(f"\n   ⚠️  MARGINAL: {best_strat} (p={best_pval:.3f})")
        print("   Crypto patterns show promise! 🪙")
    else:
        print(f"\n   ❌ NO SIGNIFICANT EDGE vs BTC buy-hold")
        print(f"   Best: {best_strat} (p={best_pval:.3f})")
        print("   BTC buy-hold is hard to beat in crypto too! 📉")
    
    print("\n" + "=" * 80)
