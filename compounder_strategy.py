"""
Compounder Detection Strategy (EMS)
====================================

A clean implementation of the Compounder Detection Strategy as specified in
the EMS(Levered)_updated_v3 PDF.

This is NOT a generic momentum system. It is a compounder detection engine
that uses ensemble machine learning to identify structural winners and
concentrate exposure in them (e.g., NVDA, AAPL, LLY).

Key Components:
1. Ensemble ML Signal Generation (Lasso/Ridge/ElasticNet style)
2. 20% Max Position Cap
3. Regime Overlay (VIX > 25 OR SPY < 200MA = risk-off)
4. Volatility Targeting
5. Optional Confidence-Scaled Leverage (1.0x-1.5x in green regimes)

Author: Strategy Recreation from PDF
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

LOGGER = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CompounderConfig:
    """Configuration for the Compounder Detection Strategy."""
    
    # Position limits
    max_position_pct: float = 0.20  # 20% max per position (from PDF)
    
    # Regime overlay thresholds
    vix_threshold: float = 25.0     # Risk-off when VIX > 25
    sma_lookback: int = 200         # SPY 200-day MA for trend filter
    
    # Volatility targeting
    target_volatility: float = 0.15  # 15% annual target vol
    vol_lookback: int = 20           # Days for realized vol calculation
    
    # Leverage settings
    enable_leverage: bool = False    # Optional leverage overlay
    min_leverage: float = 1.0        # Minimum leverage
    max_leverage: float = 1.5        # Maximum leverage (from PDF: 1.0x-1.5x)
    
    # Signal generation
    momentum_horizons: Tuple[int, ...] = (1, 5, 20, 60)  # Multi-horizon momentum
    signal_smoothing: int = 5        # EMA smoothing for signals
    
    # Rebalancing
    rebalance_threshold: float = 0.05  # Min change to trigger rebalance


# =============================================================================
# REGIME DETECTION
# =============================================================================

class RegimeDetector:
    """
    Detects market regime for the kill switch.
    
    From PDF:
    - Bull (SPY > 200MA): CAGR +94.8%, Sharpe 2.19
    - Bear (SPY < 200MA): CAGR -52.8%, Sharpe -1.62
    - High VIX (>25): CAGR -10.9%, Sharpe -0.14
    
    The regime overlay is CRITICAL for investable behavior.
    """
    
    def __init__(self, vix_threshold: float = 25.0, sma_lookback: int = 200):
        self.vix_threshold = vix_threshold
        self.sma_lookback = sma_lookback
    
    def get_regime(self, 
                   spy_prices: pd.Series, 
                   vix: Optional[pd.Series] = None) -> str:
        """
        Determine current market regime.
        
        Returns:
            'BULL': SPY > 200MA and VIX <= 25 (full exposure)
            'CAUTION': Either condition violated (reduced exposure)
            'BEAR': Both conditions violated (minimal/zero exposure)
        """
        if len(spy_prices) < self.sma_lookback:
            return 'CAUTION'
        
        if isinstance(spy_prices, pd.DataFrame):
            current_price = spy_prices.iloc[-1].iloc[0]
            sma_200 = spy_prices.iloc[:,0].rolling(self.sma_lookback).mean().iloc[-1]
        else:
            current_price = spy_prices.iloc[-1]
            sma_200 = spy_prices.rolling(self.sma_lookback).mean().iloc[-1]
        
        above_sma = current_price > sma_200
        
        if vix is not None and not vix.empty:
            if isinstance(vix, pd.DataFrame):
                current_vix = vix.iloc[-1].iloc[0]
            else:
                current_vix = vix.iloc[-1]
            low_vix = current_vix <= self.vix_threshold
        else:
            low_vix = True  # Assume OK if no VIX data
        
        if above_sma and low_vix:
            return 'BULL'
        elif not above_sma and not low_vix:
            return 'BEAR'
        else:
            return 'CAUTION'
    
    def get_exposure_multiplier(self, regime: str) -> float:
        """
        Get exposure multiplier based on regime.
        
        From PDF: "Reduce / cut risk when VIX > 25 OR SPY < 200MA"
        """
        if regime == 'BULL':
            return 1.0
        elif regime == 'CAUTION':
            return 0.5  # Reduced exposure
        else:  # BEAR
            return 0.0  # Risk-off to cash/bonds


# =============================================================================
# ENSEMBLE SIGNAL GENERATOR
# =============================================================================

from ml_engine import PredictiveAlphaEngine

class EnsembleSignalGenerator:
    """
    Generates ensemble momentum signals.
    
    Uses multi-horizon momentum combined with inverse volatility weighting
    to identify compounders - stocks with persistent outperformance.
    """
    
    def __init__(self, 
                 momentum_horizons: Tuple[int, ...] = (1, 5, 20, 60),
                 smoothing: int = 5):
        self.horizons = momentum_horizons
        self.smoothing = smoothing
        self.engine = None
    
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using the PredictiveAlphaEngine (Lasso/RLS).
        Correctly iterates through time to prevent lookahead bias.
        """
        assets = prices.columns.tolist()
        if self.engine is None:
            self.engine = PredictiveAlphaEngine(assets)
            
        # We need a window for features (32 days for 31-day features)
        min_window = 32
        n_rows = len(prices)
        
        signals = pd.DataFrame(0.0, index=prices.index, columns=assets)
        
        # Pre-calculate returns
        returns_df = prices.pct_change().fillna(0)
        
        print(f"Training ML Ensemble on {len(assets)} assets...")
        
        # Iterate through history
        # Start at min_window
        for i in range(min_window, n_rows):
            # 1. Update Models with NEWEST KNOWN data (up to i)
            # We know P[i]. We want to predict P[i+1].
            # But first we learn from the translation P[i-1] -> P[i]
            
            # Returns target for previous step: R[i] = (P[i] - P[i-1])/P[i-1]
            # we already computed returns_df.iloc[i] which IS that return.
            target_ret = returns_df.iloc[i]
            
            # Features for that target were based on window ending at i-1
            # price window: prices[i-32 : i] -> (32 days ending at i-1)
            # _extract_features uses the LAST value as 'n'.
            
            prev_window_end = i
            prev_window_start = max(0, i - 32)
            
            for asset in assets:
                # TRAIN: Learn from the move that just happened
                hist_slice = prices[asset].values[prev_window_start : prev_window_end]
                if len(hist_slice) >= 31:
                    feats = self.engine._extract_features(hist_slice)
                    actual = target_ret[asset]
                    self.engine.models[asset].update(feats, actual)
                
                # PREDICT: Predict the move that WILL happen (i -> i+1)
                # We use window ending at i (today)
                curr_window_end = i + 1
                curr_window_start = max(0, i + 1 - 32)
                
                curr_slice = prices[asset].values[curr_window_start : curr_window_end]
                if len(curr_slice) >= 31:
                    curr_feats = self.engine._extract_features(curr_slice)
                    pred = self.engine.models[asset].predict(curr_feats)
                    
                    # Store signal for this date
                    signals.loc[prices.index[i], asset] = pred
        
        # Normalize signals
        # 1. Volatility scaling (inverse vol)
        vol = prices.pct_change().rolling(20).std() * np.sqrt(252)
        inv_vol = 1.0 / (vol + 0.01)
        
        # 2. Combine prediction * inv_vol
        raw_signal = signals * inv_vol
        
        # 3. Cross-sectional normalization (Z-scoreish)
        # We want to rank them.
        # But let's stick to the momentum-like output for compatibility
        # Clip outliers
        clipped = raw_signal.clip(-3, 3).fillna(0)
        
        # 4. Smooth
        smoothed = clipped.ewm(span=self.smoothing).mean()
        
        return smoothed

# =============================================================================
# POSITION SIZING
# =============================================================================

class PositionSizer:
    """
    Handles position sizing with caps, volatility targeting, and leverage.
    
    From PDF:
    - 20% max position cap
    - Volatility targeting
    - Dynamic 1.0x-1.5x leverage only in green regimes
    """
    
    def __init__(self, config: CompounderConfig):
        self.config = config
    
    def apply_position_cap(self, weights: pd.Series) -> pd.Series:
        """Apply 20% max position cap."""
        capped = weights.clip(upper=self.config.max_position_pct)
        capped = capped.clip(lower=-self.config.max_position_pct)
        return capped
    
    def apply_volatility_targeting(self, 
                                    weights: pd.Series, 
                                    returns: pd.DataFrame,
                                    lookback: int = None) -> pd.Series:
        """
        Scale positions to target portfolio volatility.
        """
        if lookback is None:
            lookback = self.config.vol_lookback
        
        if len(returns) < lookback:
            return weights
        
        # Estimate portfolio volatility
        recent_returns = returns.iloc[-lookback:]
        cov_matrix = recent_returns.cov() * 252
        
        # Current portfolio variance
        w = weights.values
        port_var = w @ cov_matrix.values @ w
        port_vol = np.sqrt(max(port_var, 1e-8))
        
        # Scale to target
        if port_vol > 0:
            scale = self.config.target_volatility / port_vol
            scale = min(scale, 2.0)  # Cap scaling at 2x
            scaled_weights = weights * scale
        else:
            scaled_weights = weights
        
        return scaled_weights
    
    def calculate_leverage(self, 
                           signal_strength: float, 
                           regime: str) -> float:
        """
        Calculate confidence-scaled leverage.
        
        From PDF: Dynamic 1.0x-1.5x only in green regimes
        """
        if not self.config.enable_leverage:
            return 1.0
        
        if regime != 'BULL':
            return 1.0  # No leverage outside bull regime
        
        # Scale leverage with signal confidence
        # signal_strength should be 0-1 representing ensemble conviction
        leverage = self.config.min_leverage + \
                   signal_strength * (self.config.max_leverage - self.config.min_leverage)
        
        return min(leverage, self.config.max_leverage)
    
    def normalize_weights(self, weights: pd.Series) -> pd.Series:
        """Normalize weights to sum to target exposure."""
        total = weights.abs().sum()
        if total > 0:
            return weights / total
        return weights


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class CompounderStrategy:
    """
    Compounder Detection Strategy - main entry point.
    
    Implements the full strategy as specified in the PDF:
    - Ensemble ML signal generation
    - 20% max position cap
    - Regime overlay (kill switch)
    - Volatility targeting
    - Optional confidence-scaled leverage
    """
    
    def __init__(self, config: CompounderConfig = None):
        self.config = config or CompounderConfig()
        
        self.regime_detector = RegimeDetector(
            vix_threshold=self.config.vix_threshold,
            sma_lookback=self.config.sma_lookback
        )
        
        self.signal_generator = EnsembleSignalGenerator(
            momentum_horizons=self.config.momentum_horizons,
            smoothing=self.config.signal_smoothing
        )
        
        self.position_sizer = PositionSizer(self.config)
    
    def generate_weights(self,
                         prices: pd.DataFrame,
                         spy_prices: pd.Series = None,
                         vix: pd.Series = None) -> pd.DataFrame:
        """
        Generate portfolio weights for all dates.
        
        Args:
            prices: Price DataFrame for tradable assets
            spy_prices: SPY prices for regime detection (can be in prices)
            vix: VIX levels for regime detection
            
        Returns:
            DataFrame of portfolio weights
        """
        # Generate raw signals
        signals = self.signal_generator.generate_signals(prices)
        
        # Get SPY for regime detection
        if spy_prices is None:
            spy_cols = [c for c in prices.columns if 'SPY' in c or 'S&P' in c]
            if spy_cols:
                spy_prices = prices[spy_cols[0]]
            else:
                spy_prices = prices.iloc[:, 0]  # Fallback to first column
        
        returns = prices.pct_change()
        weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        
        for i in range(self.config.sma_lookback, len(prices)):
            date_idx = prices.index[i]
            
            # Get regime
            spy_slice = spy_prices.iloc[:i+1]
            vix_slice = vix.iloc[:i+1] if vix is not None else None
            regime = self.regime_detector.get_regime(spy_slice, vix_slice)
            exposure = self.regime_detector.get_exposure_multiplier(regime)
            
            if exposure == 0:
                # Risk-off: go to cash
                weights.loc[date_idx] = 0.0
                continue
            
            # Get signals for this date
            date_signals = signals.loc[date_idx]
            
            # Convert signals to weights (long top performers, short bottom)
            raw_weights = self.position_sizer.normalize_weights(date_signals)
            
            # Apply position cap
            capped_weights = self.position_sizer.apply_position_cap(raw_weights)
            
            # Apply volatility targeting
            if i > self.config.vol_lookback:
                ret_window = returns.iloc[i-self.config.vol_lookback:i]
                sized_weights = self.position_sizer.apply_volatility_targeting(
                    capped_weights, ret_window
                )
            else:
                sized_weights = capped_weights
            
            # Apply regime exposure
            final_weights = sized_weights * exposure
            
            # Apply leverage if enabled
            if self.config.enable_leverage and regime == 'BULL':
                signal_strength = min(1.0, signals.loc[date_idx].abs().mean())
                leverage = self.position_sizer.calculate_leverage(signal_strength, regime)
                final_weights = final_weights * leverage
            
            # Re-apply cap after all adjustments
            final_weights = self.position_sizer.apply_position_cap(final_weights)
            
            weights.loc[date_idx] = final_weights
        
        # T+1 Execution: Apply 1-day lag as specified in PDF
        # "rebalancing with T+1 execution"
        weights = weights.shift(1).fillna(0)
        
        return weights


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKTESTING
# =============================================================================

def compounder_strategy(prices: pd.DataFrame, 
                        vix: pd.Series = None) -> pd.DataFrame:
    """
    Generate Compounder Strategy weights.
    
    This is the main function to use for backtesting.
    """
    strategy = CompounderStrategy()
    return strategy.generate_weights(prices, vix=vix)


def compounder_strategy_levered(prices: pd.DataFrame,
                                 vix: pd.Series = None) -> pd.DataFrame:
    """
    Generate Compounder Strategy weights WITH leverage overlay.
    
    From PDF: Dynamic 1.0x-1.5x only in green regimes
    """
    config = CompounderConfig(enable_leverage=True)
    strategy = CompounderStrategy(config)
    return strategy.generate_weights(prices, vix=vix)


def compounder_strategy_no_overlay(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Compounder Strategy weights WITHOUT regime overlay.
    
    Use this to compare with/without the kill switch.
    """
    config = CompounderConfig(
        vix_threshold=100,  # Effectively disable VIX check
        sma_lookback=1      # Effectively disable SMA check
    )
    strategy = CompounderStrategy(config)
    return strategy.generate_weights(prices)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CompounderConfig',
    'CompounderStrategy',
    'RegimeDetector',
    'EnsembleSignalGenerator',
    'PositionSizer',
    'compounder_strategy',
    'compounder_strategy_levered',
    'compounder_strategy_no_overlay',
]
