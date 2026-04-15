"""
ERP REGIME ENHANCED - MAXIMUM ALPHA EXTRACTION
===============================================

Building on the validated ERP Regime strategy with additional alpha sources:

1. ERP SIGNALS (Base) - Netflix/Cheese correlations
2. MOMENTUM OVERLAY - Add momentum confirmation
3. VOLATILITY TARGETING - Scale exposure to target vol
4. SECTOR ROTATION - Add more tradeable assets
5. CONVICTION SCALING - Increase position on signal convergence
6. VIX HEDGING - Reduce exposure in high vol, increase in low vol

RUN: python erp_regime_enhanced.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("   💰 ERP REGIME ENHANCED - MAXIMUM ALPHA")
print("   Adding Momentum, Vol Targeting, Sector Rotation")
print("=" * 80)

# =============================================================================
# DATA
# =============================================================================

print("\n📊 Fetching data...")

# Expanded universe
tickers = ['SPY', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLB', 'XLI', 'XLV', 'XLU',
           'GLD', 'TLT', 'IEF', 'JNK', 'DBC', 'VNQ', 'EEM', 'IWM']

end = datetime.now()
start = end - timedelta(days=365*12)

data = yf.download(tickers + ['^VIX'], start=start, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix = prices['^VIX'].copy() if '^VIX' in prices.columns else None
if vix is not None:
    prices = prices.drop('^VIX', axis=1)

print(f"   Loaded {len(prices)} days, {len(prices.columns)} assets")

# =============================================================================
# ENHANCED ERP STRATEGY
# =============================================================================

WEIRD_DATA = {
    "netflix": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0,
        2025: 320.0, 2026: 340.0,
    },
    "cheese": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
        2025: 43.0, 2026: 43.5,
    },
    "coffee": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
        2025: 6.50, 2026: 6.70,
    },
}


def get_erp_signals(date):
    """Get all ERP-derived signals."""
    year = date.year
    signals = {}
    
    for name, data in WEIRD_DATA.items():
        if year in data and year-1 in data:
            signals[f"{name}_yoy"] = (data[year] - data[year-1]) / data[year-1]
    
    # Combined signal for XLE
    netflix = signals.get("netflix_yoy", 0)
    cheese = signals.get("cheese_yoy", 0)
    coffee = signals.get("coffee_yoy", 0)
    
    signals["xle_signal"] = -netflix * 0.5 + cheese * 0.3 + coffee * 0.2
    signals["spy_signal"] = -cheese * 0.3  # Inflation headwind
    
    return signals


def erp_regime_enhanced(prices, vix, config=None):
    """
    Enhanced ERP Regime with multiple alpha sources.
    """
    config = config or {
        "target_vol": 0.15,
        "max_position": 0.35,
        "momentum_weight": 0.3,
        "erp_weight": 0.4,
        "regime_weight": 0.3,
        "conviction_boost": 1.5,
        "vix_low": 15,
        "vix_high": 25,
        "vix_extreme": 35,
    }
    
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change()
    
    # Asset groups for rotation
    risk_on = ['SPY', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLB', 'XLI', 'IWM', 'EEM']
    risk_off = ['GLD', 'TLT', 'IEF', 'XLU']
    
    warmup = 252
    
    for i in range(warmup, len(prices)):
        date = prices.index[i]
        
        # 1. ERP SIGNALS
        erp = get_erp_signals(date)
        xle_signal = erp["xle_signal"]
        spy_signal = erp["spy_signal"]
        
        # 2. MOMENTUM (60-day, vol-adjusted)
        mom_signals = {}
        for asset in prices.columns:
            r = returns[asset].values[:i]
            if len(r) > 60:
                mom = np.mean(r[-60:])
                vol = np.std(r[-20:])
                mom_signals[asset] = mom / vol if vol > 0 else 0
            else:
                mom_signals[asset] = 0
        
        # 3. VIX REGIME
        current_vix = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            current_vix = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        # Regime multiplier
        if current_vix < config["vix_low"]:
            regime_mult = 1.3  # Low vol - more aggressive
            risk_tilt = 0.8   # Tilt to risk-on
        elif current_vix < config["vix_high"]:
            regime_mult = 1.0  # Normal
            risk_tilt = 0.5
        elif current_vix < config["vix_extreme"]:
            regime_mult = 0.7  # High vol - reduce exposure
            risk_tilt = 0.3
        else:
            regime_mult = 0.4  # Extreme vol - defensive
            risk_tilt = 0.1
        
        # 4. BUILD ASSET SCORES
        scores = {}
        
        for asset in prices.columns:
            base_score = 0
            
            # Momentum component
            mom = mom_signals.get(asset, 0)
            base_score += mom * config["momentum_weight"] * 10  # Scale up
            
            # ERP component
            if asset == 'XLE':
                base_score += xle_signal * config["erp_weight"] * 5
            elif asset == 'SPY':
                base_score += spy_signal * config["erp_weight"] * 3
            elif asset in ['XLB', 'XLI']:  # Cyclicals follow XLE signal
                base_score += xle_signal * config["erp_weight"] * 2
            
            # Risk-on/off tilt
            if asset in risk_on:
                base_score *= (1 + risk_tilt * 0.5)
            elif asset in risk_off:
                base_score *= (1 + (1 - risk_tilt) * 0.5)
            
            scores[asset] = base_score
        
        # 5. CONVICTION BOOST
        # When ERP and momentum agree, boost positions
        if xle_signal > 0.02 and mom_signals.get('XLE', 0) > 0:
            scores['XLE'] *= config["conviction_boost"]
        if xle_signal < -0.02 and mom_signals.get('XLE', 0) < 0:
            scores['GLD'] *= config["conviction_boost"]
            scores['TLT'] *= config["conviction_boost"]
        
        # 6. NORMALIZE AND APPLY VOL TARGET
        score_series = pd.Series(scores)
        
        # Long only
        score_series[score_series < 0] = 0
        
        if score_series.sum() > 0:
            raw_weights = score_series / score_series.sum()
        else:
            # Default to equal weight risk-off
            raw_weights = pd.Series(0.0, index=prices.columns)
            for a in risk_off:
                if a in raw_weights.index:
                    raw_weights[a] = 1.0 / len(risk_off)
        
        # Cap positions
        raw_weights = raw_weights.clip(0, config["max_position"])
        
        # Renormalize
        if raw_weights.sum() > 0:
            raw_weights = raw_weights / raw_weights.sum()
        
        # Vol targeting
        recent_vols = returns.iloc[i-20:i].std() * np.sqrt(252)
        if not recent_vols.empty:
            port_vol = (raw_weights * recent_vols[raw_weights.index]).sum()
            if port_vol > 0:
                vol_scale = config["target_vol"] / port_vol
                vol_scale = np.clip(vol_scale, 0.5, 2.0)  # Cap leverage
            else:
                vol_scale = 1.0
        else:
            vol_scale = 1.0
        
        # Apply regime and vol scaling
        final_weights = raw_weights * vol_scale * regime_mult
        
        # Store
        for asset in final_weights.index:
            weights.loc[date, asset] = final_weights[asset]
    
    return weights.shift(1).fillna(0)


def erp_original(prices, vix):
    """Original ERP Regime for comparison."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    assets = [a for a in ['SPY', 'XLE', 'GLD', 'TLT'] if a in prices.columns]
    
    for i in range(252, len(prices)):
        date = prices.index[i]
        erp = get_erp_signals(date)
        sig = erp["xle_signal"]
        
        v = 20
        if vix is not None and i < len(vix):
            v_val = vix.iloc[i]
            v = float(v_val) if not isinstance(v_val, pd.Series) else float(v_val.iloc[0])
        
        w = {a: 0.25 for a in assets}
        if 'XLE' in w:
            if sig > 0.02: w['XLE'], w['SPY'] = 0.35, 0.20
            elif sig < -0.02: w['XLE'], w['GLD'] = 0.10, 0.35
        if v > 25 and 'TLT' in w:
            w['TLT'] = 0.40
            if 'XLE' in w: w['XLE'] *= 0.5
        
        total = sum(w.values())
        for a in w: 
            if a in weights.columns:
                weights.iloc[i][a] = w[a] / total
    
    return weights.shift(1).fillna(0)


# =============================================================================
# BACKTEST COMPARISON
# =============================================================================

print("\n🔬 Running comparison...")

# OOS split
split = int(len(prices) * 0.7)
oos_prices = prices.iloc[split:]
oos_vix = vix.iloc[split:] if vix is not None else None

print(f"   OOS: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")

strategies = {
    "ERP Original": erp_original,
    "ERP Enhanced": erp_regime_enhanced,
}

# Different configs for enhanced
configs = {
    "Enhanced (Base)": {},
    "Enhanced (Aggressive)": {"target_vol": 0.20, "conviction_boost": 2.0, "max_position": 0.40},
    "Enhanced (Conservative)": {"target_vol": 0.12, "conviction_boost": 1.2, "max_position": 0.25},
}

results = []

# Test original
print("\n   Testing ERP Original...", end=" ", flush=True)
try:
    w = erp_original(oos_prices, oos_vix)
    ret = oos_prices.pct_change()
    warmup = 300
    w = w.iloc[warmup:]
    ret = ret.iloc[warmup:]
    common = w.columns.intersection(ret.columns)
    abs_sum = w[common].abs().sum(axis=1).replace(0, 1)
    norm = w[common].div(abs_sum, axis=0)
    port_ret = (norm.shift(1) * ret[common]).sum(axis=1)
    
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    equity = (1 + port_ret).cumprod()
    cagr = equity.iloc[-1] ** (252/len(equity)) - 1
    max_dd = (equity / equity.cummax() - 1).min()
    
    results.append({"name": "ERP Original", "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd})
    print(f"Sharpe={sharpe:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Test enhanced variants
for config_name, config in configs.items():
    print(f"   Testing {config_name}...", end=" ", flush=True)
    try:
        w = erp_regime_enhanced(oos_prices, oos_vix, config)
        ret = oos_prices.pct_change()
        warmup = 300
        w = w.iloc[warmup:]
        ret = ret.iloc[warmup:]
        common = w.columns.intersection(ret.columns)
        abs_sum = w[common].abs().sum(axis=1).replace(0, 1)
        norm = w[common].div(abs_sum, axis=0)
        port_ret = (norm.shift(1) * ret[common]).sum(axis=1)
        
        sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
        equity = (1 + port_ret).cumprod()
        cagr = equity.iloc[-1] ** (252/len(equity)) - 1
        max_dd = (equity / equity.cummax() - 1).min()
        
        results.append({"name": config_name, "sharpe": sharpe, "cagr": cagr, "max_dd": max_dd})
        print(f"Sharpe={sharpe:.2f}")
    except Exception as e:
        print(f"Error: {e}")

# SPY benchmark
print("   Testing SPY...", end=" ", flush=True)
spy_ret = oos_prices['SPY'].pct_change().iloc[300:]
spy_sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252)
spy_eq = (1 + spy_ret).cumprod()
spy_cagr = spy_eq.iloc[-1] ** (252/len(spy_eq)) - 1
spy_dd = (spy_eq / spy_eq.cummax() - 1).min()
results.append({"name": "SPY Buy&Hold", "sharpe": spy_sharpe, "cagr": spy_cagr, "max_dd": spy_dd})
print(f"Sharpe={spy_sharpe:.2f}")

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("   📊 FINAL COMPARISON")
print("=" * 70)

results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Rank':<6} {'Strategy':<25} {'Sharpe':>8} {'CAGR':>8} {'Max DD':>8}")
print("-" * 60)

for i, r in enumerate(results, 1):
    medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
    print(f"{medal} {i:<4} {r['name']:<25} {r['sharpe']:>8.2f} {r['cagr']:>7.1%} {r['max_dd']:>7.1%}")

# Alpha improvement
if len(results) >= 2:
    original = next((r for r in results if r['name'] == 'ERP Original'), None)
    best_enhanced = next((r for r in results if 'Enhanced' in r['name']), None)
    
    if original and best_enhanced:
        sharpe_improvement = best_enhanced['sharpe'] - original['sharpe']
        cagr_improvement = best_enhanced['cagr'] - original['cagr']
        
        print(f"\n   📈 Alpha Improvement:")
        print(f"   Sharpe: +{sharpe_improvement:.2f} ({sharpe_improvement/original['sharpe']*100:.1f}% better)")
        print(f"   CAGR: +{cagr_improvement:.1%}")

print("\n" + "=" * 70)
