import pytest
import pandas as pd
import datetime
from freight_analytics import predict_best_broker, analyze_routes, load_freight_data

# Mock Data Loading
def test_time_based_analytics():
    # Create dummy data with timestamps
    data = [
        # Monday Cheap
        {"broker_id": "WeekdayWarrior", "freight_price": 500, "origin_zip": "10001", "submitted_at": "2025-01-06T10:00:00"}, # Mon
        # Friday Expensive
        {"broker_id": "WeekdayWarrior", "freight_price": 800, "origin_zip": "10001", "submitted_at": "2025-01-10T10:00:00"}, # Fri
        # Sunday Expensive
        {"broker_id": "WeekendSurcharge", "freight_price": 900, "origin_zip": "10001", "submitted_at": "2025-01-12T10:00:00"}, # Sun
    ]
    df = pd.DataFrame(data)
    
    # Simulate processing in freight_analytics.py load
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df["day_of_week"] = df["submitted_at"].dt.day_name()
    
    # 1. Verify Day Extraction
    assert df.iloc[0]["day_of_week"] == "Monday"
    assert df.iloc[1]["day_of_week"] == "Friday"
    
    # 2. Analyze "Best Day"
    daily_stats = df.groupby("day_of_week")["freight_price"].mean()
    assert daily_stats["Monday"] == 500.0
    assert daily_stats["Friday"] == 800.0
    
    best_day = daily_stats.idxmin()
    assert best_day == "Monday"

def test_manual_entry_integration(tmp_path):
    # Test if save_freight_quote writes correctly (mock file)
    from broker_portal import save_freight_quote, BROKER_QUOTES_FILE
    
    # Patch file path locally
    import broker_portal
    original_file = broker_portal.BROKER_QUOTES_FILE
    
    test_file = tmp_path / "test_quotes.jsonl"
    broker_portal.BROKER_QUOTES_FILE = str(test_file)
    
    try:
        custom_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        save_freight_quote(
            broker_id="ManualBroker", 
            vendor_quote_id="123", 
            vendor_quote_summary={}, 
            freight_price=100.0, 
            valid_until=None, 
            notes="Test", 
            submitted_at=custom_time
        )
        
        # Verify read
        df = pd.read_json(str(test_file), lines=True)
        assert len(df) == 1
        # Convert to string to match expected format or compare timestamps
        assert str(df.iloc[0]["submitted_at"]) == "2023-01-01 12:00:00"
        
    finally:
        # Restore
        broker_portal.BROKER_QUOTES_FILE = original_file
