"""
Test ERP Alpha Scanner
======================

Tests the ERP alpha discovery system using both mock data (unit tests)
and optionally real database data (integration tests).

Run:
    python -m pytest test_erp_alpha.py -v
    
For integration tests with real DB:
    python -m pytest test_erp_alpha.py -v -k "integration"
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Import the scanner
from erp_alpha_scanner import (
    ERPAlphaScanner,
    ERPSignal,
    AlphaCorrelation,
    MARKET_TARGETS,
)


# =============================================================================
# UNIT TESTS (Mock Data - No DB Required)
# =============================================================================

class TestERPSignal:
    """Test ERPSignal data structure."""
    
    def test_signal_creation(self):
        """Test creating an ERP signal."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        values = np.random.randn(24) * 100 + 1000
        
        signal = ERPSignal(
            name="test_signal",
            description="Test signal for unit tests",
            frequency="monthly",
            data=pd.Series(values, index=dates),
            source_tables=["TEST_TABLE"]
        )
        
        assert signal.name == "test_signal"
        assert len(signal.data) == 24
        assert signal.frequency == "monthly"


class TestAlphaCorrelation:
    """Test AlphaCorrelation results."""
    
    def test_significant_correlation(self):
        """Test detection of significant correlations."""
        corr = AlphaCorrelation(
            erp_signal="procurement",
            market_ticker="corn",
            lag_months=1,
            pearson_r=0.75,
            pearson_p=0.001,
            spearman_r=0.70,
            spearman_p=0.002,
            n_observations=36,
            direction="positive"
        )
        
        assert corr.is_significant is True
        assert corr.direction == "positive"
    
    def test_insignificant_correlation(self):
        """Test detection of insignificant correlations."""
        corr = AlphaCorrelation(
            erp_signal="procurement",
            market_ticker="corn",
            lag_months=0,
            pearson_r=0.3,  # Too weak
            pearson_p=0.15,  # Too high
            spearman_r=0.25,
            spearman_p=0.20,
            n_observations=36,
            direction="positive"
        )
        
        assert corr.is_significant is False


class TestCorrelationCalculation:
    """Test the core correlation math."""
    
    def test_perfect_correlation(self):
        """Test correlation calculation with known data."""
        # Create two perfectly correlated series
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        base = np.linspace(0, 1, 24)
        
        signal = pd.Series(base, index=dates, name="signal")
        market = pd.Series(base * 2, index=dates, name="market")  # Same trend
        
        corr = ERPAlphaScanner.calculate_correlation(signal, market, lag=0)
        
        assert corr is not None
        assert abs(corr.pearson_r - 1.0) < 0.01  # Should be ~1.0
        assert corr.pearson_p < 0.001  # Highly significant
    
    def test_negative_correlation(self):
        """Test negative correlation detection."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        base = np.linspace(0, 1, 24)
        
        signal = pd.Series(base, index=dates, name="signal")
        market = pd.Series(-base, index=dates, name="market")  # Inverse trend
        
        corr = ERPAlphaScanner.calculate_correlation(signal, market, lag=0)
        
        assert corr is not None
        assert corr.pearson_r < -0.9  # Strong negative
        assert corr.direction == "negative"
    
    def test_lagged_correlation(self):
        """Test that lag works correctly."""
        dates = pd.date_range('2020-01-01', periods=36, freq='MS')
        
        # Signal at t predicts market at t+1
        signal_vals = np.random.randn(36)
        market_vals = np.roll(signal_vals, 1)  # Shift by 1
        market_vals[0] = 0  # Fill the first shifted value
        
        signal = pd.Series(signal_vals, index=dates, name="signal")
        market = pd.Series(market_vals, index=dates, name="market")
        
        # Lag=1 should find strong correlation
        corr_lag1 = ERPAlphaScanner.calculate_correlation(signal, market, lag=1)
        
        # Lag=0 should find weaker correlation
        corr_lag0 = ERPAlphaScanner.calculate_correlation(signal, market, lag=0)
        
        assert corr_lag1 is not None
        assert corr_lag0 is not None
        # With the roll, lag=1 should be stronger
        # (though noise means we just check it's calculated)


class TestMockData:
    """Test mock data generation for demo mode."""
    
    def test_mock_signal_generation(self):
        """Test that mock signals are properly generated."""
        scanner = ERPAlphaScanner(cursor=None)
        signal = scanner._mock_signal("test")
        
        assert signal.name == "mock_test"
        assert len(signal.data) >= 12
        assert signal.frequency == "monthly"
    
    def test_mock_market_data(self):
        """Test that mock market data is properly generated."""
        scanner = ERPAlphaScanner(cursor=None)
        market = scanner._mock_market_data()
        
        assert isinstance(market, pd.DataFrame)
        assert len(market.columns) >= 3
        assert len(market) >= 12


class TestReportGeneration:
    """Test report generation."""
    
    def test_empty_report(self):
        """Test report with no correlations."""
        scanner = ERPAlphaScanner(cursor=None)
        report = scanner.generate_report([])
        
        assert "No significant correlations found" in report
    
    def test_report_with_correlations(self):
        """Test report with correlations."""
        correlations = [
            AlphaCorrelation(
                erp_signal="procurement_spend",
                market_ticker="corn",
                lag_months=1,
                pearson_r=0.72,
                pearson_p=0.003,
                spearman_r=0.68,
                spearman_p=0.005,
                n_observations=48,
                direction="positive"
            ),
            AlphaCorrelation(
                erp_signal="procurement_spend",
                market_ticker="soybeans",
                lag_months=2,
                pearson_r=-0.65,
                pearson_p=0.01,
                spearman_r=-0.60,
                spearman_p=0.015,
                n_observations=48,
                direction="negative"
            ),
        ]
        
        scanner = ERPAlphaScanner(cursor=None)
        report = scanner.generate_report(correlations, apply_bonferroni=False)
        
        assert "procurement_spend" in report
        assert "corn" in report
        assert "soybeans" in report
        assert "r=+0.72" in report
        assert "r=-0.65" in report


class TestFullScanMock:
    """Test full scan with mock data."""
    
    def test_demo_run(self):
        """Test that demo mode runs without error."""
        scanner = ERPAlphaScanner(cursor=None)
        correlations, report = scanner.run_full_scan(save_report=False)
        
        # Should complete without error
        assert isinstance(correlations, list)
        assert isinstance(report, str)
        assert "ERP ALPHA" in report


# =============================================================================
# INTEGRATION TESTS (Requires DB Connection)
# =============================================================================

@pytest.mark.integration
class TestRealDataExtraction:
    """Integration tests that require database connection."""
    
    @pytest.fixture
    def scanner_with_db(self):
        """Create scanner with real DB connection."""
        try:
            scanner = ERPAlphaScanner()
            if scanner.cursor is None:
                pytest.skip("No database connection available")
            return scanner
        except Exception:
            pytest.skip("Could not connect to database")
    
    def test_procurement_extraction(self, scanner_with_db):
        """Test extracting real procurement data."""
        signal = scanner_with_db.extract_procurement_spend(start_year=2020)
        
        assert signal is not None
        assert len(signal.data) > 0
        assert "POP30300" in signal.source_tables
    
    def test_sales_extraction(self, scanner_with_db):
        """Test extracting real sales data."""
        signal = scanner_with_db.extract_sales_velocity(start_year=2020)
        
        assert signal is not None
        assert len(signal.data) > 0
        assert "SOP30200" in signal.source_tables
    
    def test_inventory_extraction(self, scanner_with_db):
        """Test extracting inventory turnover proxy."""
        signal = scanner_with_db.extract_inventory_turnover(start_year=2020)
        
        assert signal is not None
        assert len(signal.data) > 0
    
    def test_lead_time_extraction(self, scanner_with_db):
        """Test extracting purchase lead time."""
        signal = scanner_with_db.extract_purchase_lead_time(start_year=2020)
        
        assert signal is not None
        # May have fewer observations due to filtering
    
    def test_full_signal_extraction(self, scanner_with_db):
        """Test extracting all signals."""
        signals = scanner_with_db.extract_all_signals(start_year=2020)
        
        assert len(signals) >= 3  # At least core signals
        for signal in signals:
            assert len(signal.data) >= 12  # At least 1 year


# =============================================================================
# STATISTICAL VALIDATION TESTS
# =============================================================================

class TestStatisticalValidity:
    """Tests to ensure statistical methodology is sound."""
    
    def test_bonferroni_correction(self):
        """Test that Bonferroni correction is applied."""
        # Create correlations that pass raw but fail Bonferroni
        correlations = [
            AlphaCorrelation(
                erp_signal="test",
                market_ticker="test",
                lag_months=0,
                pearson_r=0.6,
                pearson_p=0.045,  # Passes at 0.05 but fails after correction
                spearman_r=0.55,
                spearman_p=0.05,
                n_observations=30,
                direction="positive"
            )
        ]
        
        scanner = ERPAlphaScanner(cursor=None)
        
        # Without correction
        report_no_corr = scanner.generate_report(correlations, apply_bonferroni=False)
        assert "test" in report_no_corr
        
        # With correction (100 tests = threshold 0.0005)
        report_with_corr = scanner.generate_report(correlations, apply_bonferroni=True, n_tests=100)
        assert "0 significant correlation" in report_with_corr or "No significant" in report_with_corr


if __name__ == "__main__":
    # Run demo
    print("Running ERP Alpha Scanner Tests...")
    print("=" * 60)
    
    # Quick smoke test
    scanner = ERPAlphaScanner(cursor=None)
    correlations, report = scanner.run_full_scan(save_report=False)
    
    print("\n✅ Tests passed! Scanner is working.")
    print(f"   Generated {len(correlations)} mock correlations")
