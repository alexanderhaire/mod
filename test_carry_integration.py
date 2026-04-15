"""
Carry Trade Integration for Asian Strategy
===========================================

Explores whether carry trade signals can enhance the Asian market Compounder strategy.

Carry Trade Concept:
- Borrow in low-interest-rate currencies (JPY)
- Invest in high-interest-rate currencies/assets
- Profits from interest rate differential + potential currency appreciation

Key Insight: Carry trades tend to CRASH during high VIX (risk-off)
This aligns perfectly with our regime detection!

Available Proxies (via Yahoo Finance):
- FXY: Japanese Yen ETF (inverse = carry long)
- UUP: US Dollar ETF (carry funding alternative)
- CEW: WisdomTree Emerging Currency Strategy (EM carry)
- EMHY: iShares Emerging Markets High Yield Bond (yield proxy)

Integration Approaches:
1. Add carry momentum as alpha signal
2. Use carry spread as additional regime indicator
3. Directly add carry ETFs to portfolio
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from compounder_strategy import CompounderConfig, CompounderStrategy


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_carry_data(years: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fetch Asian market data plus carry trade proxies.
    
    Returns:
        prices: Asian market ETFs
        carry_data: Carry trade proxies
        vix: VIX series
    """
    print("Fetching data with carry proxies...")
    
    # Asian market tickers
    asian_tickers = {
        'EWJ': 'Japan',
        'FXI': 'China Large-Cap',
        'EWY': 'South Korea',
        'INDA': 'India',
        'EWT': 'Taiwan',
        'EWH': 'Hong Kong',
        'EWS': 'Singapore',
        'AAXJ': 'Asia ex-Japan',
        'GLD': 'Gold',
        'TLT': 'Long Treasuries'
    }
    
    # Carry trade proxies
    carry_tickers = {
        'FXY': 'JPY',           # Yen (inverse = carry long)
        'UUP': 'USD',           # Dollar (carry funding)
        'CEW': 'EM Currency',   # Emerging market currencies
        'BWX': 'IntlBond',      # International bonds (yield)
        'IGOV': 'GovtBond',     # Intl govt bonds
    }
    
    all_tickers = {**asian_tickers, **carry_tickers}
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Fetch prices
    data = yf.download(list(all_tickers.keys()), start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    
    prices.columns = [all_tickers.get(c, c) for c in prices.columns]
    
    # Split into Asian and carry
    asian_prices = prices[[asian_tickers[t] for t in asian_tickers if asian_tickers[t] in prices.columns]]
    carry_prices = prices[[carry_tickers[t] for t in carry_tickers if carry_tickers[t] in prices.columns]]
    
    # VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    # Clean
    asian_prices = asian_prices.dropna(how='all').ffill().dropna()
    carry_prices = carry_prices.reindex(asian_prices.index).ffill()
    vix = vix.reindex(asian_prices.index).ffill().fillna(15)
    
    print(f"   Loaded {len(asian_prices)} days: {len(asian_prices.columns)} Asian + {len(carry_prices.columns)} Carry assets")
    
    return asian_prices, carry_prices, vix


# =============================================================================
# CARRY SIGNAL GENERATORS
# =============================================================================

def compute_carry_momentum(carry_prices: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Compute carry momentum signal.
    
    Carry momentum = inverse of JPY momentum (when JPY weakens, carry profits)
    Combined with EM currency strength as confirmation.
    """
    signals = pd.DataFrame(index=carry_prices.index)
    
    # JPY weakness = carry strength (inverse returns)
    if 'JPY' in carry_prices.columns:
        jpy_ret = carry_prices['JPY'].pct_change(lookback)
        signals['jpy_carry'] = -jpy_ret  # Inverse: JPY down = carry up
    
    # EM currency strength = carry confirmation
    if 'EM Currency' in carry_prices.columns:
        em_ret = carry_prices['EM Currency'].pct_change(lookback)
        signals['em_carry'] = em_ret
    
    # Bond yield spread proxy (higher yield bonds vs lower)
    if 'IntlBond' in carry_prices.columns and 'GovtBond' in carry_prices.columns:
        bond_spread = carry_prices['IntlBond'] / carry_prices['GovtBond']
        signals['yield_proxy'] = bond_spread.pct_change(lookback)
    
    # Combine signals (simple average)
    carry_signal = signals.mean(axis=1).fillna(0)
    
    # Normalize to -1 to 1 range
    carry_signal = carry_signal / (carry_signal.rolling(60).std() + 0.01)
    carry_signal = carry_signal.clip(-2, 2) / 2
    
    return carry_signal


def compute_carry_regime(carry_prices: pd.DataFrame, vix: pd.Series) -> pd.Series:
    """
    Compute carry regime indicator.
    
    Carry works well in low vol, crashes in high vol.
    Returns: 1 = carry favorable, 0 = neutral, -1 = carry unfavorable
    """
    regime = pd.Series(0.0, index=carry_prices.index)
    
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    vix_aligned = vix.reindex(carry_prices.index).ffill()
    
    # Simple regime based on VIX
    regime[vix_aligned < 15] = 1.0    # Low vol = carry favorable
    regime[vix_aligned > 25] = -1.0   # High vol = carry unfavorable
    
    # JPY momentum confirmation
    if 'JPY' in carry_prices.columns:
        jpy_mom = carry_prices['JPY'].pct_change(20)
        # If JPY is strengthening rapidly, reduce carry confidence
        regime[jpy_mom > 0.02] *= 0.5
    
    return regime


# =============================================================================
# INTEGRATION STRATEGIES
# =============================================================================

class CarryEnhancedStrategy:
    """
    Compounder strategy enhanced with carry signals.
    
    Three integration approaches:
    1. Signal overlay: Boost/reduce positions based on carry momentum
    2. Regime overlay: Use carry regime to further filter trades
    3. Asset inclusion: Add carry-exposed assets to portfolio
    """
    
    def __init__(self, config: CompounderConfig = None,
                 carry_weight: float = 0.2,
                 use_carry_regime: bool = True,
                 include_carry_assets: bool = False):
        self.base_strategy = CompounderStrategy(config)
        self.carry_weight = carry_weight
        self.use_carry_regime = use_carry_regime
        self.include_carry_assets = include_carry_assets
    
    def generate_weights(self, 
                         asian_prices: pd.DataFrame,
                         carry_prices: pd.DataFrame,
                         vix: pd.Series) -> pd.DataFrame:
        """Generate enhanced portfolio weights."""
        
        # 1. Get base strategy weights
        base_weights = self.base_strategy.generate_weights(asian_prices, vix=vix)
        
        # 2. Compute carry signals
        carry_momentum = compute_carry_momentum(carry_prices)
        carry_regime = compute_carry_regime(carry_prices, vix)
        
        # 3. Apply carry signal overlay
        enhanced_weights = base_weights.copy()
        
        for date in enhanced_weights.index:
            if date not in carry_momentum.index:
                continue
            
            carry_sig = carry_momentum.loc[date]
            
            # Boost weights when carry is favorable
            carry_mult = 1.0 + (carry_sig * self.carry_weight)
            enhanced_weights.loc[date] *= carry_mult
            
            # Apply carry regime filter
            if self.use_carry_regime and date in carry_regime.index:
                creg = carry_regime.loc[date]
                if creg < 0:
                    enhanced_weights.loc[date] *= 0.5  # Reduce exposure
        
        # 4. Optionally add carry assets
        if self.include_carry_assets:
            # Add inverse JPY (carry long) as small position
            if 'JPY' in carry_prices.columns:
                for date in enhanced_weights.index:
                    if date in carry_regime.index and carry_regime.loc[date] > 0:
                        # Add 5% short JPY position during favorable carry regime
                        pass  # Would need to expand DataFrame structure
        
        # Re-normalize
        abs_sum = enhanced_weights.abs().sum(axis=1).replace(0, 1)
        enhanced_weights = enhanced_weights.div(abs_sum, axis=0)
        
        return enhanced_weights


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_strategy(prices: pd.DataFrame, 
                      weights: pd.DataFrame,
                      warmup: int = 65) -> Dict:
    """Run backtest with given weights."""
    
    returns = prices.pct_change().fillna(0)
    weights = weights.iloc[warmup:].copy()
    returns = returns.iloc[warmup:].copy()
    
    # Align
    common_idx = weights.index.intersection(returns.index)
    weights = weights.loc[common_idx]
    returns = returns.loc[common_idx]
    
    if weights.empty:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
    # Smooth
    smoothed = weights.ewm(span=5).mean()
    
    # Portfolio returns
    port_returns = (smoothed.shift(1) * returns).sum(axis=1)
    turnover = smoothed.diff().abs().sum(axis=1)
    net_returns = port_returns - turnover * 0.001
    
    equity = (1 + net_returns).cumprod()
    sharpe = net_returns.mean() / net_returns.std() * np.sqrt(252) if net_returns.std() > 0 else 0
    
    return {
        'sharpe': sharpe,
        'cagr': (equity.iloc[-1] ** (252/len(equity))) - 1 if len(equity) > 0 else 0,
        'max_dd': ((equity - equity.cummax()) / equity.cummax()).min(),
        'daily_returns': net_returns,
        'equity_curve': equity
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_carry_experiment():
    """Test carry trade integration approaches."""
    
    print("=" * 70)
    print("   CARRY TRADE INTEGRATION EXPERIMENT")
    print("=" * 70)
    
    # Fetch data
    asian_prices, carry_prices, vix = fetch_carry_data(years=10)
    
    # OOS split
    split_point = int(len(asian_prices) * 0.7)
    oos_asian = asian_prices.iloc[split_point:]
    oos_carry = carry_prices.iloc[split_point:]
    oos_vix = vix.iloc[split_point:]
    
    print(f"\n   OOS: {oos_asian.index[0].date()} to {oos_asian.index[-1].date()}")
    
    # Test configurations
    configs = [
        {'name': 'Baseline (No Carry)', 'carry_weight': 0.0, 'use_regime': False},
        {'name': 'Carry Momentum 10%', 'carry_weight': 0.1, 'use_regime': False},
        {'name': 'Carry Momentum 20%', 'carry_weight': 0.2, 'use_regime': False},
        {'name': 'Carry Momentum 30%', 'carry_weight': 0.3, 'use_regime': False},
        {'name': 'Carry + Regime 10%', 'carry_weight': 0.1, 'use_regime': True},
        {'name': 'Carry + Regime 20%', 'carry_weight': 0.2, 'use_regime': True},
        {'name': 'Carry + Regime 30%', 'carry_weight': 0.3, 'use_regime': True},
    ]
    
    print("\n" + "=" * 70)
    print("   TESTING CARRY INTEGRATION APPROACHES")
    print("=" * 70)
    
    results = []
    
    print(f"\n{'Config':<30} {'Sharpe':>10} {'CAGR':>10} {'MaxDD':>10}")
    print("-" * 65)
    
    for cfg in configs:
        print(f"   Training {cfg['name']}...", end=" ", flush=True)
        
        strategy = CarryEnhancedStrategy(
            carry_weight=cfg['carry_weight'],
            use_carry_regime=cfg['use_regime']
        )
        
        weights = strategy.generate_weights(oos_asian, oos_carry, oos_vix)
        result = backtest_strategy(oos_asian, weights)
        
        results.append({
            'name': cfg['name'],
            **result
        })
        
        print("Done")
        print(f"{cfg['name']:<30} {result['sharpe']:>10.2f} {result['cagr']:>9.1%} {result['max_dd']:>9.1%}")
    
    # Rank
    print("\n" + "=" * 70)
    print("   RESULTS RANKED BY SHARPE")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Config':<35} {'Sharpe':>10}")
    print("-" * 55)
    
    for i, r in enumerate(sorted_results, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:<6} {r['name']:<35} {r['sharpe']:>10.2f}{marker}")
    
    # Compare best carry vs baseline
    baseline = next(r for r in results if 'Baseline' in r['name'])
    best = sorted_results[0]
    
    print("\n" + "=" * 70)
    print("   CARRY TRADE IMPACT")
    print("=" * 70)
    
    improvement = best['sharpe'] - baseline['sharpe']
    
    print(f"\n   Baseline Sharpe:     {baseline['sharpe']:.2f}")
    print(f"   Best Config:         {best['name']}")
    print(f"   Best Sharpe:         {best['sharpe']:.2f}")
    print(f"   Improvement:         {improvement:+.2f}")
    
    if improvement > 0:
        print(f"\n    CARRY TRADING IMPROVES THE STRATEGY!")
    else:
        print(f"\n    Carry integration does not improve performance")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_carry_experiment()
