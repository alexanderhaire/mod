"""
Unit Tests for All-Weather Strategy Components
===============================================

Tests each strategy component to verify correct behavior.

Run with: python -m pytest test_all_weather.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from all_weather_strategy import (
    TrendFollowingFilter,
    RiskParityWeighter,
    ValueMomentumCombo,
    StatArbOverlay,
    VRPOverlay,
    AllWeatherEnsemble,
    AllWeatherConfig,
    strategy_baseline_ems,
    strategy_ems_with_trend,
    strategy_all_weather,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    n_assets = 5
    
    # Create trending prices
    prices = np.zeros((500, n_assets))
    prices[0] = 100
    
    for t in range(1, 500):
        for j in range(n_assets):
            # Add trend for first 3 assets, mean-revert for last 2
            if j < 3:
                drift = 0.0005
            else:
                drift = 0.0
            noise = np.random.randn() * 0.015
            prices[t, j] = prices[t-1, j] * (1 + drift + noise)
    
    return pd.DataFrame(prices, index=dates, columns=[f'Asset_{i}' for i in range(n_assets)])


@pytest.fixture
def uptrend_prices():
    """Generate clearly uptrending price data."""
    np.random.seed(123)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
    
    # Strong uptrend
    t = np.arange(300)
    prices = 100 * np.exp(0.001 * t + 0.01 * np.random.randn(300).cumsum())
    
    return pd.DataFrame({'Asset_0': prices}, index=dates)


@pytest.fixture
def downtrend_prices():
    """Generate clearly downtrending price data."""
    np.random.seed(456)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
    
    # Strong downtrend
    t = np.arange(300)
    prices = 100 * np.exp(-0.001 * t + 0.01 * np.random.randn(300).cumsum())
    
    return pd.DataFrame({'Asset_0': prices}, index=dates)


@pytest.fixture
def correlated_prices():
    """Generate two highly correlated price series."""
    np.random.seed(789)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
    
    # Common factor
    common = np.random.randn(300).cumsum() * 0.02
    
    # Two correlated series
    prices1 = 100 * np.exp(common + np.random.randn(300) * 0.005)
    prices2 = 100 * np.exp(common + np.random.randn(300) * 0.005)
    
    return pd.DataFrame({'Asset_0': prices1, 'Asset_1': prices2}, index=dates)


# =============================================================================
# TREND FOLLOWING FILTER TESTS
# =============================================================================

class TestTrendFollowingFilter:
    """Tests for TrendFollowingFilter class."""
    
    def test_init(self):
        """Test initialization with default and custom parameters."""
        tf = TrendFollowingFilter()
        assert tf.momentum_lookback == 252
        assert tf.sma_lookback == 200
        
        tf_custom = TrendFollowingFilter(momentum_lookback=126, sma_lookback=50)
        assert tf_custom.momentum_lookback == 126
        assert tf_custom.sma_lookback == 50
    
    def test_uptrend_detection(self, uptrend_prices):
        """Test that uptrend is correctly detected."""
        tf = TrendFollowingFilter(sma_lookback=50, momentum_lookback=60)
        signal = tf.get_trend_signal(uptrend_prices['Asset_0'])
        
        # Should detect positive trend
        assert signal > 0, f"Expected positive trend signal, got {signal}"
    
    def test_downtrend_detection(self, downtrend_prices):
        """Test that downtrend is correctly detected."""
        tf = TrendFollowingFilter(sma_lookback=50, momentum_lookback=60)
        signal = tf.get_trend_signal(downtrend_prices['Asset_0'])
        
        # Should detect negative trend
        assert signal < 0, f"Expected negative trend signal, got {signal}"
    
    def test_filter_blocks_contrarian_trades(self, uptrend_prices):
        """Test that filter blocks short signals in uptrend."""
        tf = TrendFollowingFilter(sma_lookback=50, momentum_lookback=60)
        
        # Create short signals (negative)
        raw_signals = pd.DataFrame(-1.0, index=uptrend_prices.index, 
                                   columns=uptrend_prices.columns)
        
        filtered = tf.apply(uptrend_prices, raw_signals)
        
        # After warmup, short signals should be blocked (set to 0)
        late_signals = filtered.iloc[100:]['Asset_0']
        assert (late_signals == 0).mean() > 0.5, "Filter should block contrarian shorts in uptrend"
    
    def test_filter_allows_trend_aligned_trades(self, uptrend_prices):
        """Test that filter allows long signals in uptrend."""
        tf = TrendFollowingFilter(sma_lookback=50, momentum_lookback=60)
        
        # Create long signals (positive)
        raw_signals = pd.DataFrame(1.0, index=uptrend_prices.index, 
                                   columns=uptrend_prices.columns)
        
        filtered = tf.apply(uptrend_prices, raw_signals)
        
        # After warmup, long signals should be allowed (stay at 1.0)
        late_signals = filtered.iloc[100:]['Asset_0']
        assert (late_signals == 1.0).mean() > 0.5, "Filter should allow trend-aligned longs"


# =============================================================================
# RISK PARITY WEIGHTER TESTS
# =============================================================================

class TestRiskParityWeighter:
    """Tests for RiskParityWeighter class."""
    
    def test_inverse_vol_weights_sum_to_one(self, sample_prices):
        """Test that inverse volatility weights sum to 1."""
        rp = RiskParityWeighter(vol_lookback=60)
        returns = sample_prices.pct_change().dropna()
        
        weights = rp.calculate_inverse_vol_weights(returns)
        
        assert abs(weights.sum() - 1.0) < 0.001, f"Weights should sum to 1, got {weights.sum()}"
    
    def test_high_vol_asset_gets_lower_weight(self):
        """Test that higher volatility assets get lower weights."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='B')
        
        # Create two assets: one low vol, one high vol
        low_vol = 100 + np.random.randn(100) * 0.5  # Low vol
        high_vol = 100 + np.random.randn(100) * 5.0  # High vol
        
        prices = pd.DataFrame({'LowVol': low_vol, 'HighVol': high_vol}, index=dates)
        returns = prices.pct_change().dropna()
        
        rp = RiskParityWeighter(vol_lookback=60)
        weights = rp.calculate_inverse_vol_weights(returns)
        
        assert weights['LowVol'] > weights['HighVol'], \
            f"Low vol should get higher weight: {weights['LowVol']:.3f} vs {weights['HighVol']:.3f}"
    
    def test_apply_scales_signals(self, sample_prices):
        """Test that apply() correctly scales signals by inverse vol."""
        rp = RiskParityWeighter(vol_lookback=60)
        returns = sample_prices.pct_change()
        
        # Create uniform signals
        raw_signals = pd.DataFrame(1.0, index=sample_prices.index, 
                                   columns=sample_prices.columns)
        
        weighted = rp.apply(returns, raw_signals)
        
        # After warmup, signals should be different (scaled by inv vol)
        late_signals = weighted.iloc[100:]
        
        # Verify signals are not all the same
        assert late_signals.std().mean() > 0, "Weighted signals should vary across assets"


# =============================================================================
# VALUE MOMENTUM COMBO TESTS
# =============================================================================

class TestValueMomentumCombo:
    """Tests for ValueMomentumCombo class."""
    
    def test_composite_score_calculation(self, sample_prices):
        """Test that composite score is calculated correctly."""
        vm = ValueMomentumCombo(momentum_lookback=60, value_lookback=60)
        
        score = vm.calculate_composite_score(sample_prices['Asset_0'])
        
        # Score should be bounded
        assert -2 < score < 2, f"Composite score should be bounded, got {score}"
    
    def test_value_score_cheap_assets(self):
        """Test that cheap (below MA) assets get positive value score."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
        
        # Price that drops below its average
        prices = pd.Series(
            [100] * 150 + [80] * 150,  # Drops from 100 to 80
            index=dates
        )
        
        vm = ValueMomentumCombo(value_lookback=100)
        value = vm.calculate_value_score(prices)
        
        # Below MA = positive value (cheap)
        assert value > 0, f"Cheap asset should have positive value score, got {value}"
    
    def test_momentum_score(self, uptrend_prices):
        """Test that uptrending assets get positive momentum score."""
        vm = ValueMomentumCombo(momentum_lookback=60)
        
        momentum = vm.calculate_momentum_score(uptrend_prices['Asset_0'])
        
        assert momentum > 0, f"Uptrending asset should have positive momentum, got {momentum}"


# =============================================================================
# STATISTICAL ARBITRAGE TESTS
# =============================================================================

class TestStatArbOverlay:
    """Tests for StatArbOverlay class."""
    
    def test_pair_correlation_detection(self, correlated_prices):
        """Test that highly correlated pairs are detected."""
        stat_arb = StatArbOverlay(correlation_threshold=0.5)
        
        pairs = stat_arb.find_pairs(correlated_prices)
        
        assert len(pairs) > 0, "Should find at least one correlated pair"
        assert ('Asset_0', 'Asset_1') in pairs or ('Asset_1', 'Asset_0') in pairs
    
    def test_zscore_calculation(self, correlated_prices):
        """Test z-score calculation for spread."""
        stat_arb = StatArbOverlay(lookback=30)
        
        zscore = stat_arb.calculate_spread_zscore(
            correlated_prices['Asset_0'],
            correlated_prices['Asset_1']
        )
        
        # Z-score should be bounded
        assert -5 < zscore < 5, f"Z-score should be reasonable, got {zscore}"
    
    def test_market_neutrality(self, correlated_prices):
        """Test that stat arb signals are approximately market neutral."""
        stat_arb = StatArbOverlay(correlation_threshold=0.5)
        
        signals = stat_arb.generate_signals(correlated_prices)
        
        # Sum of signals across assets should be close to 0 (market neutral)
        signal_sum = signals.sum(axis=1)
        avg_sum = signal_sum.iloc[100:].abs().mean()
        
        assert avg_sum < 0.5, f"Stat arb should be market neutral, avg abs sum = {avg_sum}"


# =============================================================================
# VRP OVERLAY TESTS
# =============================================================================

class TestVRPOverlay:
    """Tests for VRPOverlay class."""
    
    def test_vix_regime_filter(self, sample_prices):
        """Test that high VIX reduces VRP exposure."""
        vrp = VRPOverlay(vix_threshold=20)
        
        # Create high VIX series
        high_vix = pd.Series(35.0, index=sample_prices.index)
        low_vix = pd.Series(15.0, index=sample_prices.index)
        
        signals_high = vrp.generate_signals(sample_prices, high_vix)
        signals_low = vrp.generate_signals(sample_prices, low_vix)
        
        # High VIX should have lower average signal
        avg_high = signals_high.abs().mean().mean()
        avg_low = signals_low.abs().mean().mean()
        
        assert avg_high < avg_low, f"High VIX should reduce exposure: {avg_high} vs {avg_low}"
    
    def test_positive_signals(self, sample_prices):
        """Test that VRP generates positive (long) signals."""
        vrp = VRPOverlay()
        
        signals = vrp.generate_signals(sample_prices)
        
        # VRP is a long strategy (selling vol = getting premium)
        assert (signals >= 0).all().all(), "VRP signals should be non-negative"


# =============================================================================
# ALL-WEATHER ENSEMBLE TESTS
# =============================================================================

class TestAllWeatherEnsemble:
    """Tests for AllWeatherEnsemble class."""
    
    def test_strategy_combination(self, sample_prices):
        """Test that ensemble combines all strategy components."""
        config = AllWeatherConfig()
        ensemble = AllWeatherEnsemble(config)
        
        signals = ensemble.generate_all_weather_signals(sample_prices)
        
        # Should have signals for all assets
        assert signals.shape[1] == sample_prices.shape[1]
        
        # Should have signals for most dates (after warmup)
        non_zero_pct = (signals.iloc[200:].abs().sum(axis=1) > 0.01).mean()
        assert non_zero_pct > 0.5, f"Should have signals most of the time, got {non_zero_pct:.1%}"
    
    def test_weight_allocation(self, sample_prices):
        """Test that weights are properly normalized."""
        ensemble = AllWeatherEnsemble()
        
        signals = ensemble.generate_all_weather_signals(sample_prices)
        
        # Signals should be normalized (abs sum ≈ 1)
        abs_sums = signals.iloc[200:].abs().sum(axis=1)
        valid_sums = abs_sums[abs_sums > 0.01]
        
        if len(valid_sums) > 0:
            assert (valid_sums - 1.0).abs().mean() < 0.5, "Signals should be approximately normalized"
    
    def test_config_affects_output(self, sample_prices):
        """Test that different configs produce different signals."""
        config1 = AllWeatherConfig(ems_weight=1.0, trend_weight=0.0, stat_arb_weight=0.0, vrp_weight=0.0)
        config2 = AllWeatherConfig(ems_weight=0.0, trend_weight=1.0, stat_arb_weight=0.0, vrp_weight=0.0)
        
        ensemble1 = AllWeatherEnsemble(config1)
        ensemble2 = AllWeatherEnsemble(config2)
        
        signals1 = ensemble1.generate_all_weather_signals(sample_prices)
        signals2 = ensemble2.generate_all_weather_signals(sample_prices)
        
        # Different configs should produce different signals
        diff = (signals1 - signals2).iloc[200:].abs().mean().mean()
        assert diff > 0.01, "Different configs should produce different signals"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStrategyFunctions:
    """Integration tests for strategy functions."""
    
    def test_baseline_ems_generates_signals(self, sample_prices):
        """Test that baseline EMS generates signals."""
        signals = strategy_baseline_ems(sample_prices)
        
        assert signals.shape == sample_prices.shape
        assert not signals.iloc[50:].isna().all().all()
    
    def test_ems_with_trend_filters_signals(self, sample_prices):
        """Test that trend filter modifies baseline signals."""
        baseline = strategy_baseline_ems(sample_prices)
        filtered = strategy_ems_with_trend(sample_prices)
        
        # Filtered should have fewer non-zero signals (some blocked by filter)
        baseline_activity = (baseline.iloc[200:].abs() > 0.01).sum().sum()
        filtered_activity = (filtered.iloc[200:].abs() > 0.01).sum().sum()
        
        assert filtered_activity <= baseline_activity, \
            "Trend filter should reduce or maintain activity"
    
    def test_all_weather_runs_without_error(self, sample_prices):
        """Test that all-weather strategy runs without error."""
        signals = strategy_all_weather(sample_prices)
        
        assert signals.shape == sample_prices.shape
        assert not signals.isna().all().all()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
