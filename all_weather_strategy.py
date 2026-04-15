"""
All-Weather Ensemble Momentum Strategy
========================================

Comprehensive implementation of 5 strategy enhancements to transform
the bull-market-dependent EMS into a robust all-weather portfolio.

Components:
1. Trend Following Filter - 12M momentum + 200DMA filter
2. Risk Parity Weighting - Inverse-volatility position sizing
3. Value + Momentum Combination - Composite factor scoring
4. Statistical Arbitrage Overlay - Market-neutral pairs trading
5. Volatility Risk Premium Overlay - Short volatility carry

Author: All-Weather Strategy Enhancement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

LOGGER = logging.getLogger(__name__)


# =============================================================================
# 1. TREND FOLLOWING FILTER
# =============================================================================

class TrendFollowingFilter:
    """
    Trend filter that only allows trades in the direction of the dominant trend.
    
    Uses:
    - 12-month (252-day) momentum: Only trade if trailing return agrees with signal
    - 200-day SMA: Long only above, neutral/short below
    
    This dramatically reduces drawdowns by avoiding contrarian bets during
    strong trends. Based on Faber's 10-month SMA research showing max DD
    reduction from ~60% to ~18%.
    """
    
    def __init__(self, momentum_lookback: int = 252, sma_lookback: int = 200):
        self.momentum_lookback = momentum_lookback
        self.sma_lookback = sma_lookback
    
    def get_trend_signal(self, prices: pd.Series) -> float:
        """
        Calculate trend signal for a single asset.
        
        Returns:
            +1.0: Strong uptrend (above 200DMA, positive 12M momentum)
            +0.5: Weak uptrend (above 200DMA only)
             0.0: Neutral/No trend
            -0.5: Weak downtrend (below 200DMA only)
            -1.0: Strong downtrend (below 200DMA, negative 12M momentum)
        """
        if len(prices) < max(self.momentum_lookback, self.sma_lookback):
            return 0.0
        
        current_price = prices.iloc[-1]
        sma_200 = prices.rolling(self.sma_lookback).mean().iloc[-1]
        
        # 12-month momentum (approximately 252 trading days)
        if len(prices) >= self.momentum_lookback:
            momentum_12m = (current_price / prices.iloc[-self.momentum_lookback]) - 1
        else:
            momentum_12m = 0.0
        
        # Combine signals
        above_sma = current_price > sma_200
        positive_momentum = momentum_12m > 0
        
        if above_sma and positive_momentum:
            return 1.0  # Strong uptrend
        elif above_sma:
            return 0.5  # Weak uptrend
        elif not above_sma and not positive_momentum:
            return -1.0  # Strong downtrend
        elif not above_sma:
            return -0.5  # Weak downtrend
        else:
            return 0.0
    
    def apply(self, prices: pd.DataFrame, raw_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trend filter to raw signals.
        
        Only allows:
        - Long positions when trend is positive (signal * trend > 0 for longs)
        - Short positions when trend is negative
        - Otherwise, reduces position to zero
        
        Args:
            prices: Price DataFrame (dates x assets)
            raw_signals: Raw signal DataFrame (same shape)
            
        Returns:
            Filtered signals DataFrame
        """
        filtered = raw_signals.copy()
        
        for col in prices.columns:
            for i in range(len(prices)):
                if i < self.sma_lookback:
                    continue
                    
                price_slice = prices[col].iloc[:i+1]
                trend = self.get_trend_signal(price_slice)
                raw_signal = raw_signals[col].iloc[i]
                
                # Only trade in direction of trend
                if trend > 0 and raw_signal > 0:
                    # Uptrend + Long signal: Allow
                    filtered[col].iloc[i] = raw_signal
                elif trend < 0 and raw_signal < 0:
                    # Downtrend + Short signal: Allow
                    filtered[col].iloc[i] = raw_signal
                else:
                    # Signal conflicts with trend: Reduce to zero
                    filtered[col].iloc[i] = 0.0
        
        return filtered


# =============================================================================
# 2. RISK PARITY WEIGHTING
# =============================================================================

class RiskParityWeighter:
    """
    Position sizing based on inverse volatility for equal risk contribution.
    
    Each asset's weight is proportional to 1/σ, ensuring that high-volatility
    assets get smaller allocations. This prevents concentrated risk in
    volatile assets and improves Sharpe ratio through better diversification.
    
    Empirically: Inverse-vol weighted portfolios achieve ~50% higher Sharpe
    than equal-weighted portfolios.
    """
    
    def __init__(self, vol_lookback: int = 60, target_vol: float = 0.15):
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol  # Annual target volatility
    
    def calculate_inverse_vol_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate inverse-volatility weights for all assets.
        
        Returns normalized weights summing to 1.0.
        """
        if len(returns) < self.vol_lookback:
            # Not enough data - return equal weights
            n = len(returns.columns)
            return pd.Series({c: 1.0/n for c in returns.columns})
        
        # Calculate rolling volatility (annualized)
        recent_returns = returns.iloc[-self.vol_lookback:]
        vols = recent_returns.std() * np.sqrt(252)
        
        # Inverse volatility weights
        inv_vols = 1.0 / (vols + 0.001)  # Add small constant to avoid division by zero
        weights = inv_vols / inv_vols.sum()
        
        return weights
    
    def apply(self, returns: pd.DataFrame, raw_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply risk parity weighting to signals.
        
        Scales each asset's signal by its inverse-volatility weight.
        """
        weighted = raw_signals.copy()
        
        for i in range(len(raw_signals)):
            if i < self.vol_lookback:
                continue
            
            ret_window = returns.iloc[max(0, i-self.vol_lookback):i]
            inv_vol_weights = self.calculate_inverse_vol_weights(ret_window)
            
            for col in raw_signals.columns:
                raw_signal = raw_signals[col].iloc[i]
                weight = inv_vol_weights.get(col, 1.0 / len(raw_signals.columns))
                # Scale signal by relative weight (multiply by n_assets to maintain scale)
                weighted[col].iloc[i] = raw_signal * weight * len(raw_signals.columns)
        
        return weighted


# =============================================================================
# 3. VALUE + MOMENTUM COMBINATION
# =============================================================================

class ValueMomentumCombo:
    """
    Combines Value and Momentum factors for enhanced risk-adjusted returns.
    
    Value and Momentum are famously negatively correlated - when one
    underperforms, the other often outperforms. Combining them yields
    higher Sharpe than either alone.
    
    Per Asness et al. (2013): 50/50 value-momentum achieved Sharpe 1.88
    vs ~1.1 for momentum-only.
    """
    
    def __init__(self, 
                 momentum_lookback: int = 252,
                 value_lookback: int = 252,
                 value_weight: float = 0.5):
        self.momentum_lookback = momentum_lookback
        self.value_lookback = value_lookback
        self.value_weight = value_weight
        self.momentum_weight = 1.0 - value_weight
    
    def calculate_value_score(self, prices: pd.Series) -> float:
        """
        Calculate value score as price relative to long-term moving average.
        
        Lower price vs. MA = higher value (asset is "cheap").
        This is a simplified proxy for valuation.
        """
        if len(prices) < self.value_lookback:
            return 0.0
        
        current = prices.iloc[-1]
        ma = prices.rolling(self.value_lookback).mean().iloc[-1]
        
        if ma <= 0:
            return 0.0
        
        # Value score: negative means overvalued, positive means undervalued
        value_ratio = (ma - current) / ma
        return np.clip(value_ratio, -0.5, 0.5)  # Clip to reasonable range
    
    def calculate_momentum_score(self, prices: pd.Series) -> float:
        """
        Calculate momentum score as trailing return.
        """
        if len(prices) < self.momentum_lookback:
            return 0.0
        
        momentum = (prices.iloc[-1] / prices.iloc[-self.momentum_lookback]) - 1
        return np.clip(momentum, -1.0, 1.0)
    
    def calculate_composite_score(self, prices: pd.Series) -> float:
        """
        Calculate combined value + momentum composite score.
        """
        value = self.calculate_value_score(prices)
        momentum = self.calculate_momentum_score(prices)
        
        composite = (self.value_weight * value + 
                     self.momentum_weight * momentum)
        return composite
    
    def apply(self, prices: pd.DataFrame, raw_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply value/momentum combination to modify signals.
        
        Enhances signals for assets with both momentum AND value support.
        Reduces signals for expensive momentum names.
        """
        adjusted = raw_signals.copy()
        
        for i in range(self.value_lookback, len(prices)):
            scores = {}
            for col in prices.columns:
                price_slice = prices[col].iloc[:i+1]
                scores[col] = self.calculate_composite_score(price_slice)
            
            # Rank assets by composite score
            scored_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n = len(scored_assets)
            
            for rank, (col, score) in enumerate(scored_assets):
                raw_signal = raw_signals[col].iloc[i]
                
                # Boost top-ranked assets, reduce bottom-ranked
                rank_adjustment = 1.0 + 0.5 * (1 - 2 * rank / (n - 1)) if n > 1 else 1.0
                adjusted[col].iloc[i] = raw_signal * rank_adjustment * (1 + score)
        
        return adjusted


# =============================================================================
# 4. STATISTICAL ARBITRAGE OVERLAY
# =============================================================================

class StatArbOverlay:
    """
    Market-neutral pairs trading overlay.
    
    Identifies highly correlated asset pairs and trades mean-reversion
    when their spread diverges beyond a threshold. This generates returns
    regardless of market direction.
    
    Performs best in range-bound/choppy markets where momentum struggles.
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.7,
                 zscore_entry: float = 2.0,
                 zscore_exit: float = 0.5,
                 lookback: int = 60):
        self.correlation_threshold = correlation_threshold
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback = lookback
        self.active_pairs: List[Tuple[str, str]] = []
    
    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Find highly correlated asset pairs suitable for pairs trading.
        """
        returns = prices.pct_change().dropna()
        if len(returns) < self.lookback:
            return []
        
        recent_returns = returns.iloc[-self.lookback:]
        corr_matrix = recent_returns.corr()
        
        pairs = []
        cols = list(prices.columns)
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = corr_matrix.loc[cols[i], cols[j]]
                if corr >= self.correlation_threshold:
                    pairs.append((cols[i], cols[j]))
        
        return pairs
    
    def calculate_spread_zscore(self, 
                                prices1: pd.Series, 
                                prices2: pd.Series) -> float:
        """
        Calculate z-score of the price spread between two assets.
        """
        if len(prices1) < self.lookback or len(prices2) < self.lookback:
            return 0.0
        
        # Use log price ratio as spread
        spread = np.log(prices1 / prices2)
        recent_spread = spread.iloc[-self.lookback:]
        
        current = spread.iloc[-1]
        mean = recent_spread.mean()
        std = recent_spread.std()
        
        if std < 1e-6:
            return 0.0
        
        return (current - mean) / std
    
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market-neutral pairs trading signals.
        
        Returns DataFrame with positions for each asset from pairs trades.
        """
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        pairs = self.find_pairs(prices)
        if not pairs:
            return signals
        
        # Trade each pair
        for asset1, asset2 in pairs:
            for i in range(self.lookback, len(prices)):
                price_slice1 = prices[asset1].iloc[:i+1]
                price_slice2 = prices[asset2].iloc[:i+1]
                
                zscore = self.calculate_spread_zscore(price_slice1, price_slice2)
                
                # Entry: spread is extended
                if zscore > self.zscore_entry:
                    # Spread is high: short asset1, long asset2
                    signals[asset1].iloc[i] -= 0.5 / len(pairs)
                    signals[asset2].iloc[i] += 0.5 / len(pairs)
                elif zscore < -self.zscore_entry:
                    # Spread is low: long asset1, short asset2
                    signals[asset1].iloc[i] += 0.5 / len(pairs)
                    signals[asset2].iloc[i] -= 0.5 / len(pairs)
                # Exit (mean reversion): zscore near zero - no position
                # (Positions close naturally as zscore returns to normal)
        
        return signals


# =============================================================================
# 5. VOLATILITY RISK PREMIUM OVERLAY
# =============================================================================

class VRPOverlay:
    """
    Volatility Risk Premium harvesting overlay.
    
    Exploits the tendency of implied volatility to exceed realized volatility.
    Simulates systematic short volatility exposure (like selling options or
    short VIX futures).
    
    Provides steady carry returns in calm markets but can suffer large losses
    in volatility spikes. Uses VIX-based regime filter to reduce exposure
    when volatility is elevated.
    """
    
    def __init__(self, 
                 vix_threshold: float = 25.0,
                 vol_lookback: int = 20,
                 target_allocation: float = 0.15):
        self.vix_threshold = vix_threshold
        self.vol_lookback = vol_lookback
        self.target_allocation = target_allocation
    
    def estimate_vrp(self, prices: pd.DataFrame, vix: Optional[pd.Series] = None) -> pd.Series:
        """
        Estimate volatility risk premium returns.
        
        VRP ≈ Implied Vol - Realized Vol
        Returns are positive when implied > realized (most of the time).
        """
        returns = prices.pct_change()
        
        # Calculate realized volatility for each asset
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # If VIX is provided, use it as implied vol proxy
        # Otherwise, estimate implied as 1.2x realized (typical VRP premium)
        if vix is not None and len(vix) == len(prices):
            implied_vol = vix / 100.0  # VIX is in percentage points
            # Broadcast VIX to all columns
            implied_vol_df = pd.DataFrame({col: implied_vol for col in prices.columns})
        else:
            # Estimate: implied is typically 20% higher than realized
            implied_vol_df = realized_vol * 1.2
        
        # VRP return = (implied - realized) * allocation * time_decay_factor
        vrp_returns = (implied_vol_df - realized_vol) * 0.05 / 252  # Daily carry
        
        return vrp_returns
    
    def apply_regime_filter(self, 
                            vrp_signals: pd.DataFrame, 
                            vix: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Reduce VRP exposure when VIX is elevated above threshold.
        """
        filtered = vrp_signals.copy()
        
        if vix is None:
            return filtered
        
        for i in range(len(filtered)):
            if i < len(vix) and vix.iloc[i] > self.vix_threshold:
                # High VIX: reduce exposure significantly
                scale = max(0.1, 1.0 - (vix.iloc[i] - self.vix_threshold) / 50.0)
                filtered.iloc[i] *= scale
        
        return filtered
    
    def generate_signals(self, 
                         prices: pd.DataFrame, 
                         vix: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate VRP overlay signals.
        
        Returns DataFrame with small positive allocations to each asset
        representing short volatility exposure.
        """
        n_assets = len(prices.columns)
        base_allocation = self.target_allocation / n_assets
        
        # Create base signals (equal allocation to all assets)
        signals = pd.DataFrame(
            base_allocation, 
            index=prices.index, 
            columns=prices.columns
        )
        
        # Apply regime filter if VIX available
        signals = self.apply_regime_filter(signals, vix)
        
        return signals


# =============================================================================
# 6. ALL-WEATHER ENSEMBLE
# =============================================================================

@dataclass
class AllWeatherConfig:
    """Configuration for All-Weather Ensemble strategy."""
    # Component weights (should sum to ~1.0)
    ems_weight: float = 0.50       # Core EMS momentum
    trend_weight: float = 0.20     # Trend filter impact
    stat_arb_weight: float = 0.15  # Stat arb overlay
    vrp_weight: float = 0.10       # VRP overlay
    value_mom_weight: float = 0.05 # Value/Momentum adjustment
    
    # Apply risk parity on top of everything
    apply_risk_parity: bool = True
    
    # Component parameters
    trend_sma_lookback: int = 200
    trend_momentum_lookback: int = 252
    risk_parity_lookback: int = 60
    stat_arb_zscore_entry: float = 2.0
    vrp_vix_threshold: float = 25.0


class AllWeatherEnsemble:
    """
    Complete All-Weather Ensemble Momentum Strategy.
    
    Combines:
    1. Core EMS with trend filter (50%)
    2. Trend following overlay (20%)
    3. Statistical arbitrage (15%)
    4. Volatility risk premium (10%)
    5. Optional value/momentum enhancement
    6. Risk parity weighting for position sizing
    
    Target: Maximize Sharpe ratio while limiting max drawdown to <15%
    """
    
    def __init__(self, config: AllWeatherConfig = None):
        self.config = config or AllWeatherConfig()
        
        # Initialize components
        self.trend_filter = TrendFollowingFilter(
            momentum_lookback=self.config.trend_momentum_lookback,
            sma_lookback=self.config.trend_sma_lookback
        )
        self.risk_parity = RiskParityWeighter(
            vol_lookback=self.config.risk_parity_lookback
        )
        self.value_momentum = ValueMomentumCombo()
        self.stat_arb = StatArbOverlay(
            zscore_entry=self.config.stat_arb_zscore_entry
        )
        self.vrp = VRPOverlay(
            vix_threshold=self.config.vrp_vix_threshold
        )
    
    def generate_ems_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate core EMS momentum signals (simplified Lasso-style).
        """
        returns = prices.pct_change()
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for col in prices.columns:
            # Multi-horizon momentum
            r1 = prices[col].pct_change(1).fillna(0)
            r5 = prices[col].pct_change(5).fillna(0)
            r20 = prices[col].pct_change(20).fillna(0)
            
            # Volatility (inverse weight)
            vol = returns[col].rolling(20).std().fillna(0.01)
            inv_vol = 1 / (vol + 0.01)
            
            # Combined signal
            raw_signal = (0.3 * r1 + 0.3 * r5 + 0.4 * r20) * inv_vol
            
            # Normalize
            norm_factor = raw_signal.abs().rolling(60, min_periods=20).mean().fillna(0.01) + 0.001
            signals[col] = (raw_signal / norm_factor).clip(-2, 2).fillna(0)
        
        return signals
    
    def generate_trend_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend following signals (SMA crossover style).
        """
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for col in prices.columns:
            fast_ma = prices[col].rolling(20).mean()
            slow_ma = prices[col].rolling(60).mean()
            
            raw_signal = np.sign(fast_ma - slow_ma)
            strength = ((prices[col] - slow_ma) / (slow_ma * 0.01 + 0.001)).clip(-2, 2)
            
            signals[col] = raw_signal * (0.5 + 0.5 * strength.abs().clip(0, 1))
        
        return signals.fillna(0)
    
    def generate_all_weather_signals(self, 
                                      prices: pd.DataFrame,
                                      vix: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate combined All-Weather ensemble signals.
        
        Combines all strategy components with configured weights.
        """
        # 1. Generate core EMS signals
        ems_signals = self.generate_ems_signals(prices)
        
        # 2. Apply trend filter to EMS
        ems_filtered = self.trend_filter.apply(prices, ems_signals)
        
        # 3. Generate trend following signals
        trend_signals = self.generate_trend_signals(prices)
        
        # 4. Generate stat arb signals
        stat_arb_signals = self.stat_arb.generate_signals(prices)
        
        # 5. Generate VRP signals
        vrp_signals = self.vrp.generate_signals(prices, vix)
        
        # 6. Apply value/momentum enhancement to EMS
        if self.config.value_mom_weight > 0:
            ems_enhanced = self.value_momentum.apply(prices, ems_filtered)
        else:
            ems_enhanced = ems_filtered
        
        # Combine with weights
        combined = (
            self.config.ems_weight * ems_enhanced +
            self.config.trend_weight * trend_signals +
            self.config.stat_arb_weight * stat_arb_signals +
            self.config.vrp_weight * vrp_signals
        )
        
        # 7. Apply risk parity weighting
        if self.config.apply_risk_parity:
            returns = prices.pct_change()
            combined = self.risk_parity.apply(returns, combined)
        
        # Normalize final signals
        abs_sum = combined.abs().sum(axis=1).replace(0, 1)
        normalized = combined.div(abs_sum, axis=0)
        
        return normalized


# =============================================================================
# STRATEGY FUNCTIONS FOR BACKTESTING
# =============================================================================

def strategy_baseline_ems(prices: pd.DataFrame) -> pd.DataFrame:
    """Baseline EMS (Lasso Momentum) without enhancements."""
    ensemble = AllWeatherEnsemble()
    return ensemble.generate_ems_signals(prices)


def strategy_ems_with_trend(prices: pd.DataFrame) -> pd.DataFrame:
    """EMS with trend following filter."""
    ensemble = AllWeatherEnsemble()
    ems_signals = ensemble.generate_ems_signals(prices)
    return ensemble.trend_filter.apply(prices, ems_signals)


def strategy_ems_with_risk_parity(prices: pd.DataFrame) -> pd.DataFrame:
    """EMS with risk parity weighting."""
    ensemble = AllWeatherEnsemble()
    ems_signals = ensemble.generate_ems_signals(prices)
    returns = prices.pct_change()
    return ensemble.risk_parity.apply(returns, ems_signals)


def strategy_ems_with_value_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """EMS with value/momentum combination."""
    ensemble = AllWeatherEnsemble()
    ems_signals = ensemble.generate_ems_signals(prices)
    return ensemble.value_momentum.apply(prices, ems_signals)


def strategy_ems_trend_risk_parity(prices: pd.DataFrame) -> pd.DataFrame:
    """EMS with trend filter AND risk parity."""
    ensemble = AllWeatherEnsemble()
    ems_signals = ensemble.generate_ems_signals(prices)
    filtered = ensemble.trend_filter.apply(prices, ems_signals)
    returns = prices.pct_change()
    return ensemble.risk_parity.apply(returns, filtered)


def strategy_stat_arb_only(prices: pd.DataFrame) -> pd.DataFrame:
    """Pure statistical arbitrage (market-neutral)."""
    stat_arb = StatArbOverlay()
    return stat_arb.generate_signals(prices)


def strategy_vrp_only(prices: pd.DataFrame, vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """Pure volatility risk premium overlay."""
    vrp = VRPOverlay()
    return vrp.generate_signals(prices, vix)


def strategy_all_weather(prices: pd.DataFrame, vix: Optional[pd.Series] = None) -> pd.DataFrame:
    """Full All-Weather Ensemble strategy."""
    ensemble = AllWeatherEnsemble()
    return ensemble.generate_all_weather_signals(prices, vix)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'TrendFollowingFilter',
    'RiskParityWeighter',
    'ValueMomentumCombo',
    'StatArbOverlay',
    'VRPOverlay',
    'AllWeatherEnsemble',
    'AllWeatherConfig',
    'strategy_baseline_ems',
    'strategy_ems_with_trend',
    'strategy_ems_with_risk_parity',
    'strategy_ems_with_value_momentum',
    'strategy_ems_trend_risk_parity',
    'strategy_stat_arb_only',
    'strategy_vrp_only',
    'strategy_all_weather',
]
