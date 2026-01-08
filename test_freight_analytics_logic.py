import pytest
import pandas as pd
from freight_analytics import predict_best_broker, analyze_routes

# Dummy Data
def get_dummy_data():
    data = [
        {"broker_id": "CheapBroker", "freight_price": 500, "origin_zip": "10001", "dest_zip": "33563", "vendor_quote_summary": {"origin": "New York, NY 10001"}},
        {"broker_id": "ExpensiveBroker", "freight_price": 800, "origin_zip": "10001", "dest_zip": "33563", "vendor_quote_summary": {"origin": "New York, NY 10001"}},
        {"broker_id": "MidBroker", "freight_price": 600, "origin_zip": "10001", "dest_zip": "33563", "vendor_quote_summary": {"origin": "New York, NY 10001"}},
        # Route 2
        {"broker_id": "RegionalKing", "freight_price": 200, "origin_zip": "90210", "dest_zip": "33563", "vendor_quote_summary": {"origin": "Beverly Hills, CA 90210"}},
    ]
    df = pd.DataFrame(data)
    # Ensure helper column created by loader is present or simulated
    df["origin_zip"] = df["origin_zip"] # Already there
    return df

def test_predict_best_broker():
    df = get_dummy_data()
    
    # Case 1: Competitive Route (10001)
    # "CheapBroker" has $500, others higher.
    prediction = predict_best_broker("10001", df_history=df)
    
    assert prediction is not None
    assert prediction["broker_id"] == "CheapBroker"
    assert prediction["predicted_price"] == 500.0

    # Case 2: Single Player Route (90210)
    prediction_2 = predict_best_broker("90210", df_history=df)
    assert prediction_2["broker_id"] == "RegionalKing"
    assert prediction_2["predicted_price"] == 200.0

    # Case 3: Unknown Route
    prediction_none = predict_best_broker("00000", df_history=df)
    assert prediction_none is None

def test_analyze_routes_aggregation():
    df = get_dummy_data()
    stats_df = analyze_routes(df)
    
    # Should have 2 routes
    assert len(stats_df) == 2
    
    # Check 10001 Stats
    row_ny = stats_df[stats_df["origin_zip"] == "10001"].iloc[0]
    assert row_ny["best_broker"] == "CheapBroker"
    assert row_ny["competitors"] == 3 # Cheap, Expensive, Mid
    assert row_ny["best_price"] == 500.0
