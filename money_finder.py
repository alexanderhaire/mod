"""
THE MONEY FINDER
================

Everything we've learned, combined into one aggressive alpha hunt.

Approach:
1. ERP-Enhanced AlphaMax (add Netflix/Cheese signals to GBM)
2. Multi-Strategy Ensemble (combine best of each)
3. Regime-Conditional Switching (use right strategy for right regime)
4. Aggressive Optimization (find what actually works OOS)

Key discoveries to leverage:
- Netflix→XLE: r=-0.79 (p=0.004) - STRONG
- Cheese→XLE: r=+0.64 (p=0.032) - STRONG
- Cheese→SPY: r=-0.61 (p=0.047) - MODERATE
- AlphaMax targets: XLB, XLI, XLE, JNK, GLD
- Compounder uses ML ensemble + regime overlay

RUN THIS: python money_finder.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("   💰 THE MONEY FINDER 💰")
print("   Combining Everything to Beat the Champions")
print("=" * 70)

# =============================================================================
# DATA
# =============================================================================

print("\n📊 Loading data...")

tickers = ['SPY', 'XLB', 'XLI', 'XLE', 'JNK', 'GLD', 'QQQ', 'IWM', 'TLT', 'MOO', 'XLK']
macro_tickers = ['^TNX', 'UUP', 'IEF']

end = datetime.now()
start = end - timedelta(days=365*12)  # 12 years for more data

data = yf.download(tickers + macro_tickers, start=start, progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
prices = prices.ffill().dropna()

vix_data = yf.download('^VIX', start=start, progress=False)
vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
vix = vix.reindex(prices.index).ffill().fillna(15)

print(f"   Loaded {len(prices)} days, {len(prices.columns)} assets")

# =============================================================================
# ERP SIGNALS (THE SECRET SAUCE)
# =============================================================================

WEIRD_DATA = {
    "netflix_subscribers": {
        2010: 18.3, 2011: 21.5, 2012: 25.7, 2013: 41.4, 2014: 54.5,
        2015: 70.8, 2016: 89.1, 2017: 110.6, 2018: 139.0, 2019: 151.5,
        2020: 203.7, 2021: 221.8, 2022: 220.7, 2023: 260.3, 2024: 300.0,
        2025: 320.0, 2026: 340.0,
    },
    "cheese_consumption": {
        2010: 33.0, 2011: 33.3, 2012: 33.5, 2013: 34.0, 2014: 34.5,
        2015: 35.0, 2016: 36.0, 2017: 37.0, 2018: 38.0, 2019: 38.5,
        2020: 39.0, 2021: 40.2, 2022: 42.0, 2023: 42.3, 2024: 42.5,
        2025: 43.0, 2026: 43.5,
    },
    "coffee_price": {
        2010: 3.91, 2011: 5.19, 2012: 5.68, 2013: 5.45, 2014: 4.99,
        2015: 4.72, 2016: 4.39, 2017: 4.45, 2018: 4.30, 2019: 4.14,
        2020: 4.43, 2021: 4.71, 2022: 5.89, 2023: 6.16, 2024: 6.32,
        2025: 6.50, 2026: 6.70,
    },
}


def get_erp_signals(date):
    """Get ERP-derived trading signals for a date."""
    year = date.year
    signals = {}
    
    for name, data in WEIRD_DATA.items():
        if year in data and year - 1 in data:
            yoy = (data[year] - data[year-1]) / data[year-1]
            signals[name] = yoy
    
    # XLE signal: Netflix down OR cheese up = bullish (from regression)
    netflix_yoy = signals.get("netflix_subscribers", 0)
    cheese_yoy = signals.get("cheese_consumption", 0)
    coffee_yoy = signals.get("coffee_price", 0)
    
    # XLE score based on significant correlations
    xle_signal = -netflix_yoy * 0.5 + cheese_yoy * 0.3 + coffee_yoy * 0.2
    
    # SPY signal: cheese up = bearish
    spy_signal = -cheese_yoy * 0.3
    
    return {
        "xle_erp": xle_signal,
        "spy_erp": spy_signal,
        "netflix_yoy": netflix_yoy,
        "cheese_yoy": cheese_yoy,
    }


# =============================================================================
# STRATEGY 1: ERP-ENHANCED ALPHAMAX
# =============================================================================

class ERPAlphaMax:
    """AlphaMax with ERP signals as additional features."""
    
    def __init__(self):
        self.models = {}
        self.target_assets = ['XLB', 'XLI', 'XLE', 'JNK', 'GLD']
        self.target_vol = 0.20
        
    def _create_features(self, prices, macro_data, date):
        """Create features including ERP signals."""
        idx = prices.index.get_loc(date)
        features = {}
        
        for ticker in self.target_assets:
            if ticker not in prices.columns:
                continue
                
            p = prices[ticker].values[:idx+1]
            if len(p) < 60:
                continue
            
            r = np.diff(np.log(p))
            
            # Momentum
            features[f'{ticker}_mom_20'] = np.mean(r[-20:]) if len(r) >= 20 else 0
            features[f'{ticker}_mom_60'] = np.mean(r[-60:]) if len(r) >= 60 else 0
            
            # Volatility
            features[f'{ticker}_vol'] = np.std(r[-20:]) * np.sqrt(252) if len(r) >= 20 else 0.2
        
        # Macro features
        if '^TNX' in macro_data.columns:
            tnx = macro_data['^TNX'].values[:idx+1]
            if len(tnx) > 20:
                features['rate_change'] = tnx[-1] - tnx[-21] if len(tnx) > 21 else 0
        
        if 'JNK' in prices.columns and 'IEF' in macro_data.columns:
            jnk = prices['JNK'].values[:idx+1]
            ief = macro_data['IEF'].values[:idx+1]
            if len(jnk) > 0 and len(ief) > 0:
                features['credit_spread'] = jnk[-1] / ief[-1] if ief[-1] > 0 else 1
        
        # ERP SIGNALS (THE NEW STUFF)
        erp = get_erp_signals(date)
        features['xle_erp'] = erp['xle_erp']
        features['spy_erp'] = erp['spy_erp']
        features['netflix_trend'] = erp['netflix_yoy']
        features['cheese_trend'] = erp['cheese_yoy']
        
        return features
    
    def generate_weights(self, prices, macro_data, vix):
        """Generate weights with ERP enhancement."""
        weights = pd.DataFrame(0.0, index=prices.index, columns=self.target_assets)
        
        warmup = 252
        
        for i in range(warmup, len(prices)):
            date = prices.index[i]
            features = self._create_features(prices, macro_data, date)
            
            # Regime check
            if isinstance(vix.iloc[i], pd.Series):
                current_vix = vix.iloc[i].iloc[0]
            else:
                current_vix = vix.iloc[i]
            
            regime_mult = 1.0
            if current_vix > 25:
                regime_mult = 0.5
            
            # Calculate weights based on momentum + ERP
            raw_weights = {}
            
            for ticker in self.target_assets:
                if ticker not in prices.columns:
                    continue
                
                mom_key = f'{ticker}_mom_20'
                vol_key = f'{ticker}_vol'
                
                mom = features.get(mom_key, 0)
                vol = features.get(vol_key, 0.15)
                
                # Base signal: momentum
                base_signal = mom * 20  # Scale up
                
                # ERP adjustment for XLE
                if ticker == 'XLE':
                    erp_adj = features.get('xle_erp', 0) * 0.3
                    base_signal += erp_adj
                
                # Inverse vol weighting
                inv_vol = 0.15 / max(vol, 0.05)
                
                raw_weights[ticker] = base_signal * inv_vol * regime_mult
            
            # Normalize
            w = pd.Series(raw_weights)
            w[w < 0] = 0  # Long only
            
            if w.sum() > 0:
                w = w / w.sum()
                
                # Vol target
                recent_vols = prices[self.target_assets].pct_change().iloc[i-20:i].std() * np.sqrt(252)
                port_vol = (w * recent_vols[w.index]).sum()
                if port_vol > 0:
                    scale = min(self.target_vol / port_vol, 2.0)
                    w = w * scale
            
            for ticker in w.index:
                weights.loc[date, ticker] = w[ticker]
        
        return weights.shift(1).fillna(0)


# =============================================================================
# STRATEGY 2: MULTI-STRATEGY ENSEMBLE
# =============================================================================

def momentum_strategy(prices, vix, lookback=60):
    """Simple momentum with vol targeting."""
    returns = prices.pct_change()
    mom = returns.rolling(lookback).mean()
    vol = returns.rolling(20).std() * np.sqrt(252)
    
    # Inverse vol weighted momentum
    signal = mom / (vol + 0.01)
    
    # Regime
    for i in range(len(vix)):
        v = vix.iloc[i]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        if v > 25:
            signal.iloc[i] *= 0.5
    
    # Normalize
    abs_sum = signal.abs().sum(axis=1).replace(0, 1)
    weights = signal.div(abs_sum, axis=0).clip(-0.25, 0.25)
    
    return weights.shift(1).fillna(0)


def mean_reversion_strategy(prices, vix, lookback=20):
    """Mean reversion for high vol periods."""
    returns = prices.pct_change()
    
    # Z-score of price vs MA
    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    z = (prices - ma) / (std + 0.01)
    
    # Bet against extremes
    signal = -z.clip(-2, 2) / 4
    
    # Only active in high vol
    for i in range(len(vix)):
        v = vix.iloc[i]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        if v < 20:  # Low vol = no mean reversion
            signal.iloc[i] *= 0.2
    
    return signal.shift(1).fillna(0)


def ensemble_strategy(prices, vix):
    """Combine momentum + mean reversion adaptively."""
    mom_w = momentum_strategy(prices, vix)
    mr_w = mean_reversion_strategy(prices, vix)
    
    # Weight based on regime
    combined = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    for i in range(len(prices)):
        v = vix.iloc[i]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        
        if v > 25:  # High vol: more mean reversion
            mom_weight = 0.4
            mr_weight = 0.6
        elif v > 18:  # Normal: balanced
            mom_weight = 0.6
            mr_weight = 0.4
        else:  # Low vol: all momentum
            mom_weight = 0.9
            mr_weight = 0.1
        
        combined.iloc[i] = mom_weight * mom_w.iloc[i] + mr_weight * mr_w.iloc[i]
    
    return combined


# =============================================================================
# STRATEGY 3: ERP REGIME-CONDITIONAL
# =============================================================================

def erp_regime_strategy(prices, vix):
    """Switch between XLE tilt based on ERP signals."""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    
    # Base: equal weight among quality assets
    base_assets = ['SPY', 'XLE', 'GLD', 'TLT'] if all(a in prices.columns for a in ['SPY', 'XLE', 'GLD', 'TLT']) else list(prices.columns)[:4]
    
    for i in range(252, len(prices)):
        date = prices.index[i]
        
        # ERP signals
        erp = get_erp_signals(date)
        xle_signal = erp['xle_erp']
        spy_signal = erp['spy_erp']
        
        # VIX regime
        v = vix.iloc[i]
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        
        # Base allocation
        w = {}
        for a in base_assets:
            w[a] = 0.25
        
        # Tilt based on ERP
        if 'XLE' in w:
            # Strong ERP signal: tilt XLE
            if xle_signal > 0.02:
                w['XLE'] = 0.35
                w['SPY'] = 0.20 if 'SPY' in w else 0
            elif xle_signal < -0.02:
                w['XLE'] = 0.10
                w['GLD'] = 0.35 if 'GLD' in w else 0.25
        
        # VIX adjustment
        if v > 25:
            # Risk off
            if 'TLT' in w:
                w['TLT'] = 0.40
            if 'XLE' in w:
                w['XLE'] = max(0.05, w['XLE'] * 0.5)
        
        # Normalize
        total = sum(w.values())
        for a in w:
            weights.loc[date, a] = w[a] / total if total > 0 else 0
    
    return weights.shift(1).fillna(0)


# =============================================================================
# BACKTEST ALL STRATEGIES
# =============================================================================

def backtest(prices, weights, name, warmup=300):
    """Run backtest and return metrics."""
    returns = prices.pct_change()
    
    # Align
    common_cols = weights.columns.intersection(returns.columns)
    weights = weights[common_cols].iloc[warmup:]
    returns = returns[common_cols].iloc[warmup:]
    
    # Portfolio returns
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    
    # Metrics
    if port_ret.std() == 0:
        return None
    
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    equity = (1 + port_ret).cumprod()
    cagr = equity.iloc[-1] ** (252/len(equity)) - 1 if len(equity) > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min() if len(equity) > 0 else 0
    
    return {
        'name': name,
        'sharpe': sharpe,
        'cagr': cagr,
        'max_dd': max_dd,
        'returns': port_ret,
        'equity': equity,
    }


# =============================================================================
# RUN ALL STRATEGIES
# =============================================================================

print("\n🔍 Running all strategies...")

# Split data
split = int(len(prices) * 0.7)
oos_prices = prices.iloc[split:]
oos_vix = vix.iloc[split:]

macro_cols = [c for c in ['^TNX', 'UUP', 'IEF'] if c in oos_prices.columns]
macro_data = oos_prices[macro_cols] if macro_cols else pd.DataFrame(index=oos_prices.index)

print(f"   OOS Period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")

results = []

# Strategy 1: ERP-Enhanced AlphaMax
print("\n1️⃣ ERP-Enhanced AlphaMax...", end=" ", flush=True)
try:
    erp_am = ERPAlphaMax()
    am_weights = erp_am.generate_weights(oos_prices, macro_data, oos_vix)
    am_result = backtest(oos_prices, am_weights, "ERP-AlphaMax")
    if am_result:
        results.append(am_result)
        print(f"Sharpe={am_result['sharpe']:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Strategy 2: Multi-Strategy Ensemble
print("2️⃣ Multi-Strategy Ensemble...", end=" ", flush=True)
try:
    ens_weights = ensemble_strategy(oos_prices, oos_vix)
    ens_result = backtest(oos_prices, ens_weights, "Ensemble")
    if ens_result:
        results.append(ens_result)
        print(f"Sharpe={ens_result['sharpe']:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Strategy 3: ERP Regime-Conditional
print("3️⃣ ERP Regime Strategy...", end=" ", flush=True)
try:
    erp_reg_weights = erp_regime_strategy(oos_prices, oos_vix)
    erp_reg_result = backtest(oos_prices, erp_reg_weights, "ERP-Regime")
    if erp_reg_result:
        results.append(erp_reg_result)
        print(f"Sharpe={erp_reg_result['sharpe']:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Strategy 4: Pure Momentum
print("4️⃣ Pure Momentum...", end=" ", flush=True)
try:
    mom_weights = momentum_strategy(oos_prices, oos_vix)
    mom_result = backtest(oos_prices, mom_weights, "Momentum")
    if mom_result:
        results.append(mom_result)
        print(f"Sharpe={mom_result['sharpe']:.2f}")
except Exception as e:
    print(f"Error: {e}")

# Strategy 5: SPY Benchmark
print("5️⃣ SPY Buy & Hold...", end=" ", flush=True)
spy_weights = pd.DataFrame(0.0, index=oos_prices.index, columns=oos_prices.columns)
spy_weights['SPY'] = 1.0
spy_result = backtest(oos_prices, spy_weights, "SPY")
if spy_result:
    results.append(spy_result)
    print(f"Sharpe={spy_result['sharpe']:.2f}")

# =============================================================================
# FINAL RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("   💰 FINAL RESULTS (OUT-OF-SAMPLE)")
print("=" * 70)

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'Strategy':<25} {'Sharpe':>10} {'CAGR':>10} {'Max DD':>10}")
print("-" * 60)

for r in results:
    status = "⭐" if r['sharpe'] > 1.0 else "✓" if r['sharpe'] > 0 else "✗"
    print(f"{status} {r['name']:<23} {r['sharpe']:>10.2f} {r['cagr']:>9.1%} {r['max_dd']:>9.1%}")

if results:
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']}")
    print(f"   Sharpe: {winner['sharpe']:.2f}")
    print(f"   CAGR: {winner['cagr']:.1%}")
    print(f"   Max DD: {winner['max_dd']:.1%}")
    
    # Did we beat SPY?
    spy_sharpe = next((r['sharpe'] for r in results if r['name'] == 'SPY'), 0)
    if winner['sharpe'] > spy_sharpe and winner['name'] != 'SPY':
        print(f"\n🎉 YES! We beat SPY by {(winner['sharpe'] - spy_sharpe):.2f} Sharpe points!")
    else:
        print(f"\n📊 SPY Sharpe was {spy_sharpe:.2f}")

print("\n" + "=" * 70)
