"""
ERP-Enhanced Compounder Strategy
=================================

The HYBRID approach: Use ERP-derived signals to ENHANCE the winning
Compounder strategy rather than replacing it.

Key insight: Compounder has Sharpe 1.37 OOS, ERP signals found significant
correlations. Combine them:

1. Base: Compounder's ML ensemble momentum signals
2. Overlay: ERP macro adjustment from weird data
3. Result: Potentially improved risk-adjusted returns

The ERP signals are used to:
- Tilt toward/away from XLE based on Netflix/Cheese signals
- Adjust overall exposure based on inflation proxy (cheese→SPY signal)
- Add conviction when ERP and momentum agree

Usage:
    from erp_enhanced_compounder import erp_enhanced_compounder
    weights = erp_enhanced_compounder(prices, vix)
"""

import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

# Import base Compounder
from compounder_strategy import (
    CompounderStrategy,
    CompounderConfig,
    EnsembleSignalGenerator,
)

# ERP signals from V2
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


@dataclass
class ERPEnhancedConfig(CompounderConfig):
    """Extended config with ERP settings."""
    erp_weight: float = 0.2  # Weight of ERP adjustment (0-1)
    energy_boost_factor: float = 1.5  # Extra factor for energy when signals align


class ERPEnhancedCompounder:
    """
    Compounder + ERP Signals = Enhanced Strategy
    """
    
    def __init__(self, config: ERPEnhancedConfig = None):
        self.config = config or ERPEnhancedConfig()
        self.base_compounder = CompounderStrategy(self.config)
    
    def get_erp_adjustments(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate ERP-based adjustments for today.
        
        Returns dict of asset adjustments (multiply base weight by this).
        """
        year = date.year
        adjustments = {}
        
        # Calculate YoY changes
        netflix_yoy = 0
        cheese_yoy = 0
        coffee_yoy = 0
        
        for name, data in WEIRD_DATA.items():
            if year in data and year - 1 in data:
                yoy = (data[year] - data[year - 1]) / data[year - 1]
                if name == "netflix_subscribers":
                    netflix_yoy = yoy
                elif name == "cheese_consumption":
                    cheese_yoy = yoy
                elif name == "coffee_price":
                    coffee_yoy = yoy
        
        # Energy adjustment: Netflix down OR cheese up = bullish
        # Correlation: Netflix→XLE = -0.79, Cheese→XLE = +0.64
        energy_signal = -netflix_yoy * 0.5 + cheese_yoy * 0.3 + coffee_yoy * 0.2
        
        # Normalize to [-1, 1]
        energy_signal = np.clip(energy_signal * 5, -1, 1)  # Amplify small changes
        
        # SPY adjustment: cheese up = bearish for SPY
        # Correlation: Cheese→SPY = -0.61
        spy_signal = -cheese_yoy
        spy_signal = np.clip(spy_signal * 3, -0.5, 0.5)
        
        # Apply adjustments
        # > 0 means boost the position, < 0 means reduce
        adjustments["Energy"] = 1 + energy_signal * self.config.erp_weight
        adjustments["XLE"] = 1 + energy_signal * self.config.erp_weight
        adjustments["S&P 500"] = 1 + spy_signal * self.config.erp_weight
        adjustments["SPY"] = 1 + spy_signal * self.config.erp_weight
        
        # When signals are very strong, boost energy even more
        if energy_signal > 0.5 and cheese_yoy > 0.02:
            adjustments["Energy"] *= self.config.energy_boost_factor
            adjustments["XLE"] *= self.config.energy_boost_factor
        
        return adjustments
    
    def generate_weights(self,
                          prices: pd.DataFrame,
                          vix: pd.Series = None) -> pd.DataFrame:
        """Generate weights using Compounder + ERP adjustments."""
        
        # Get base Compounder weights
        base_weights = self.base_compounder.generate_weights(prices, vix=vix)
        
        # Apply ERP adjustments day by day
        enhanced_weights = base_weights.copy()
        
        for i in range(len(prices)):
            date = prices.index[i]
            
            # Get ERP adjustments
            adjustments = self.get_erp_adjustments(date)
            
            # Apply to each asset
            for asset in prices.columns:
                asset_upper = asset.upper()
                
                # Find matching adjustment
                mult = 1.0
                for key, adj in adjustments.items():
                    if key.upper() in asset_upper or asset_upper in key.upper():
                        mult = adj
                        break
                
                # Apply
                enhanced_weights.loc[date, asset] *= mult
        
        # Re-normalize
        for date in enhanced_weights.index:
            row = enhanced_weights.loc[date]
            abs_sum = row.abs().sum()
            if abs_sum > 0:
                enhanced_weights.loc[date] = row / abs_sum
        
        # Re-cap
        enhanced_weights = enhanced_weights.clip(
            -self.config.max_position_pct,
            self.config.max_position_pct
        )
        
        return enhanced_weights


def erp_enhanced_compounder(prices: pd.DataFrame,
                             vix: pd.Series = None) -> pd.DataFrame:
    """Main entry point for ERP-Enhanced Compounder."""
    strategy = ERPEnhancedCompounder()
    return strategy.generate_weights(prices, vix)


def erp_enhanced_compounder_high(prices: pd.DataFrame,
                                   vix: pd.Series = None) -> pd.DataFrame:
    """Higher ERP weight version."""
    config = ERPEnhancedConfig(
        erp_weight=0.35,
        energy_boost_factor=2.0,
    )
    strategy = ERPEnhancedCompounder(config)
    return strategy.generate_weights(prices, vix)


# =============================================================================
# COMPARISON TEST
# =============================================================================

def compare_strategies():
    """Compare Enhanced vs Base Compounder."""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("=" * 70)
    print("   ERP-ENHANCED COMPOUNDER vs BASE COMPOUNDER")
    print("=" * 70)
    
    # Fetch data
    tickers = ['SPY', 'QQQ', 'XLE', 'XLF', 'XLK', 'XLB', 'GLD', 'TLT', 'MOO']
    end = datetime.now()
    start = end - timedelta(days=365*10)
    
    print("\nFetching data...")
    data = yf.download(tickers, start=start, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    vix_data = yf.download('^VIX', start=start, progress=False)
    vix = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
    
    prices = prices.ffill().dropna()
    vix = vix.reindex(prices.index).ffill().fillna(15)
    
    # Rename for friendly names in Compounder
    rename = {'SPY': 'S&P 500', 'XLE': 'Energy', 'XLK': 'Technology', 
              'XLB': 'Materials', 'XLF': 'Financials', 'MOO': 'Agriculture'}
    prices.columns = [rename.get(c, c) for c in prices.columns]
    
    # Split OOS
    split = int(len(prices) * 0.7)
    oos_prices = prices.iloc[split:]
    oos_vix = vix.iloc[split:]
    
    print(f"Out-of-sample period: {oos_prices.index[0].date()} to {oos_prices.index[-1].date()}")
    
    # Run strategies
    from compounder_strategy import compounder_strategy
    
    strategies = {
        "Base Compounder": lambda p, v: compounder_strategy(p, v),
        "ERP-Enhanced": erp_enhanced_compounder,
        "ERP-Enhanced (High)": erp_enhanced_compounder_high,
    }
    
    results = {}
    returns = oos_prices.pct_change()
    
    print("\n" + "-" * 60)
    
    for name, func in strategies.items():
        print(f"Running {name}...", end=" ", flush=True)
        
        try:
            weights = func(oos_prices, oos_vix)
            
            # Normalize and smooth
            abs_sum = weights.abs().sum(axis=1).replace(0, 1)
            norm = weights.div(abs_sum, axis=0)
            smooth = norm.ewm(span=5).mean()
            
            # Portfolio returns
            port_ret = (smooth.shift(1) * returns).sum(axis=1)
            
            # Skip warmup
            warmup = 65
            port_ret = port_ret.iloc[warmup:]
            
            # Metrics
            sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
            equity = (1 + port_ret).cumprod()
            cagr = equity.iloc[-1] ** (252/len(equity)) - 1
            max_dd = (equity / equity.cummax() - 1).min()
            
            results[name] = {
                'sharpe': sharpe,
                'cagr': cagr,
                'max_dd': max_dd,
            }
            
            print(f"Sharpe={sharpe:.2f}")
            
        except Exception as e:
            print(f"Error: {e}")
            results[name] = {'error': str(e)}
    
    # Comparison
    print("\n" + "=" * 60)
    print("   RESULTS")
    print("=" * 60)
    
    print(f"\n{'Strategy':<25} {'OOS Sharpe':>12} {'CAGR':>10} {'Max DD':>10}")
    print("-" * 60)
    
    for name, res in results.items():
        if 'error' not in res:
            status = "⭐" if res['sharpe'] > 1.0 else "✓" if res['sharpe'] > 0 else "✗"
            print(f"{status} {name:<23} {res['sharpe']:>12.2f} {res['cagr']:>9.1%} {res['max_dd']:>9.1%}")
    
    # Winner?
    valid = {k: v for k, v in results.items() if 'error' not in v}
    if valid:
        winner = max(valid.items(), key=lambda x: x[1]['sharpe'])
        
        print(f"\n🏆 WINNER: {winner[0]} (Sharpe={winner[1]['sharpe']:.2f})")
        
        if "Enhanced" in winner[0]:
            print("   🎉 ERP ENHANCEMENT IMPROVED THE STRATEGY!")
        else:
            print("   📊 Base Compounder remains the champion")
    
    return results


if __name__ == "__main__":
    compare_strategies()
