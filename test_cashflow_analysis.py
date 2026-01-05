"""
Unit tests for the cashflow_analysis module.
"""

import datetime
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import with patching to avoid actual file I/O during tests
import cashflow_analysis as cf


class TestCashFlowError:
    """Tests for the CashFlowError dataclass."""
    
    def test_to_dict(self):
        error = cf.CashFlowError(
            code="TEST_ERROR",
            message="Test message",
            user_message="User-friendly message",
            details="Some details"
        )
        result = error.to_dict()
        
        assert result["error_code"] == "TEST_ERROR"
        assert result["error_message"] == "Test message"
        assert result["user_message"] == "User-friendly message"
        assert result["details"] == "Some details"


class TestPatternsPersistence:
    """Tests for loading and saving patterns."""
    
    def test_load_patterns_empty(self, tmp_path):
        """Test loading patterns when file doesn't exist."""
        with patch.object(cf, 'PATTERNS_FILE', tmp_path / "patterns.json"):
            result = cf._load_patterns()
            assert result == {"vendors": {}, "last_updated": None}
    
    def test_save_and_load_patterns(self, tmp_path):
        """Test round-trip of saving and loading patterns."""
        patterns_file = tmp_path / "data" / "patterns.json"
        with patch.object(cf, 'PATTERNS_FILE', patterns_file):
            test_patterns = {
                "vendors": {"VENDOR001": {"avg_offset": 2.5}},
                "last_updated": "2026-01-01T00:00:00"
            }
            cf._save_patterns(test_patterns)
            
            loaded = cf._load_patterns()
            assert loaded["vendors"]["VENDOR001"]["avg_offset"] == 2.5


class TestGetAccuracyReport:
    """Tests for the accuracy report function."""
    
    def test_empty_predictions(self, tmp_path):
        """Test accuracy report with no predictions."""
        with patch.object(cf, 'PATTERNS_FILE', tmp_path / "patterns.json"):
            with patch.object(cf, 'PREDICTIONS_FILE', tmp_path / "predictions.json"):
                result = cf.get_accuracy_report()
                
                assert result["accuracy_score"] == 0
                assert result["total_predictions"] == 0
                assert result["avg_days_off"] == 0
                assert result["most_predictable_vendors"] == []
                assert result["least_predictable_vendors"] == []
    
    def test_with_predictions(self, tmp_path):
        """Test accuracy report with some predictions."""
        patterns_file = tmp_path / "data" / "patterns.json"
        predictions_file = tmp_path / "data" / "predictions.json"
        
        with patch.object(cf, 'PATTERNS_FILE', patterns_file):
            with patch.object(cf, 'PREDICTIONS_FILE', predictions_file):
                # Create test predictions
                predictions_file.parent.mkdir(parents=True, exist_ok=True)
                test_data = {
                    "predictions": [
                        {"vendor_id": "V001", "days_off": 1},
                        {"vendor_id": "V001", "days_off": 2},
                        {"vendor_id": "V002", "days_off": 5},
                        {"vendor_id": "V002", "days_off": 10},
                    ],
                    "last_updated": "2026-01-01"
                }
                with open(predictions_file, "w") as f:
                    json.dump(test_data, f)
                
                result = cf.get_accuracy_report()
                
                assert result["total_predictions"] == 4
                # V001 predictions (1, 2) are within 3 days - accurate
                # V002 predictions (5, 10) are not - inaccurate
                assert result["accuracy_score"] == 50.0  # 2 out of 4
                assert len(result["most_predictable_vendors"]) > 0


class TestVendorAdjustment:
    """Tests for vendor adjustment calculations."""
    
    def test_no_adjustment(self, tmp_path):
        """Test getting adjustment for unknown vendor."""
        with patch.object(cf, 'PATTERNS_FILE', tmp_path / "patterns.json"):
            result = cf.get_vendor_adjustment("UNKNOWN_VENDOR")
            assert result == 0.0
    
    def test_with_adjustment(self, tmp_path):
        """Test getting adjustment for known vendor."""
        patterns_file = tmp_path / "data" / "patterns.json"
        with patch.object(cf, 'PATTERNS_FILE', patterns_file):
            # Create test patterns
            patterns_file.parent.mkdir(parents=True, exist_ok=True)
            test_data = {
                "vendors": {
                    "V001": {"average_offset": 3.5}
                },
                "last_updated": None
            }
            with open(patterns_file, "w") as f:
                json.dump(test_data, f)
            
            result = cf.get_vendor_adjustment("V001")
            assert result == 3.5


class TestHandleDbError:
    """Tests for database error handling."""
    
    def test_auth_error(self):
        """Test handling of authentication errors."""
        import pyodbc
        mock_error = pyodbc.OperationalError("Login failed for user")
        
        result = cf._handle_db_error(mock_error, "test operation")
        
        assert result.code == "AUTH_ERROR"
        assert "authentication" in result.user_message.lower() or "credentials" in result.user_message.lower()
    
    def test_connection_error(self):
        """Test handling of connection errors."""
        import pyodbc
        mock_error = pyodbc.OperationalError("Network connection failed")
        
        result = cf._handle_db_error(mock_error, "test operation")
        
        assert result.code == "CONNECTION_ERROR"
        assert "connect" in result.user_message.lower() or "network" in result.user_message.lower()
    
    def test_generic_error(self):
        """Test handling of generic errors."""
        mock_error = Exception("Some random error")
        
        result = cf._handle_db_error(mock_error, "test operation")
        
        assert result.code == "DATABASE_ERROR"


class TestCashPositionSummary:
    """Tests for get_cash_position_summary function."""
    
    def test_returns_default_on_error(self, tmp_path):
        """Test that function returns defaults when DB query fails."""
        with patch.object(cf, 'PATTERNS_FILE', tmp_path / "patterns.json"):
            with patch.object(cf, 'PREDICTIONS_FILE', tmp_path / "predictions.json"):
                mock_cursor = MagicMock()
                mock_cursor.execute.side_effect = Exception("DB Error")
                
                result = cf.get_cash_position_summary(mock_cursor)
                
                assert result["next_30_day_outflow"] == 0
                assert result["next_60_day_outflow"] == 0
                assert result["open_po_count"] == 0
                assert result["error"] is not None


class TestRecordPrediction:
    """Tests for recording predictions."""
    
    def test_record_prediction(self, tmp_path):
        """Test recording a new prediction."""
        patterns_file = tmp_path / "data" / "patterns.json"
        predictions_file = tmp_path / "data" / "predictions.json"
        
        with patch.object(cf, 'PATTERNS_FILE', patterns_file):
            with patch.object(cf, 'PREDICTIONS_FILE', predictions_file):
                cf.record_prediction(
                    vendor_id="V001",
                    po_number="PO12345",
                    predicted_date=datetime.date(2026, 1, 15),
                    actual_date=datetime.date(2026, 1, 17)
                )
                
                # Load and verify
                predictions = cf._load_predictions()
                assert len(predictions["predictions"]) == 1
                assert predictions["predictions"][0]["vendor_id"] == "V001"
                assert predictions["predictions"][0]["days_off"] == 2
                
                # Verify patterns were updated
                patterns = cf._load_patterns()
                assert "V001" in patterns["vendors"]
                assert patterns["vendors"]["V001"]["average_offset"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
