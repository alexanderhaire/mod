"""
Test script for the ML Procurement Optimizer.

Run this to verify the module works with your actual GP data.
"""

import pyodbc
from secrets_loader import build_connection_string
from procurement_ml import ProcurementMLOptimizer, BuyWindowPredictor

def main():
    print("=" * 60)
    print("ML PROCUREMENT OPTIMIZER - TEST")
    print("=" * 60)
    
    # Connect to database
    conn_str, server, db, auth = build_connection_string()
    print(f"\nConnecting to {server}/{db} ({auth})...")
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        print("[OK] Connected successfully\n")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return
    
    # Initialize optimizer
    print("Initializing ML Optimizer...")
    optimizer = ProcurementMLOptimizer(cursor)
    
    # Get a sample item to test
    print("\nFinding a sample raw material item...")
    cursor.execute("""
        SELECT TOP 1 l.ITEMNMBR, COUNT(*) as PurchaseCount
        FROM POP30300 h
        JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
        WHERE l.EXTDCOST > 0
        GROUP BY l.ITEMNMBR
        HAVING COUNT(*) >= 5
        ORDER BY PurchaseCount DESC
    """)
    row = cursor.fetchone()
    
    if not row:
        print(f"[FAIL] No items with purchase history found")
        return
    
    test_item = row.ITEMNMBR.strip()
    print(f"[OK] Using test item: {test_item} ({row.PurchaseCount} purchases)")
    
    # Test feature extraction
    print("\n--- Feature Extraction Test ---")
    features = optimizer.feature_builder.build_features(test_item)
    print(f"  Item: {features.item_number}")
    print(f"  Current Price: ${features.current_price:.2f}")
    print(f"  Price 52w Percentile: {features.price_percentile_52w:.1%}")
    print(f"  Price Trend Slope: {features.price_trend_slope:.4f}")
    print(f"  Days of Coverage: {features.days_of_coverage:.1f}")
    print(f"  Usage 30d Avg: {features.usage_30d_avg:.2f}")
    print(f"  Vendor Payment Days: {features.vendor_payment_days}")
    
    # Test prediction (rule-based since model isn't trained yet)
    print("\n--- Prediction Test (Rule-Based Fallback) ---")
    recommendation = optimizer.get_buy_recommendation(test_item)
    print(f"  Buy Score: {recommendation['buy_score']:.0f}/100")
    print(f"  Recommendation: {recommendation['recommendation']}")
    print(f"  Confidence: {recommendation['confidence']:.1%}")
    print(f"  Top Factors: {recommendation['top_factors']}")
    
    # Test training (optional - takes longer)
    print("\n--- Training Test (12 months of data) ---")
    print("Building training dataset...")
    
    try:
        metrics = optimizer.train_model(lookback_months=12)
        if 'error' in metrics:
            print(f"  [WARN] Training warning: {metrics['error']}")
        else:
            print(f"  [OK] Training complete!")
            print(f"  Train R²: {metrics['train_r2']:.3f}")
            print(f"  Test R²: {metrics['test_r2']:.3f}")
            print(f"  Samples: {metrics['n_samples']}")
            print(f"  Top Features: {list(metrics['feature_importance'].keys())}")
            
            # Re-test prediction with trained model
            print("\n--- Prediction Test (Trained Model) ---")
            recommendation = optimizer.get_buy_recommendation(test_item)
            print(f"  Buy Score: {recommendation['buy_score']:.0f}/100")
            print(f"  Recommendation: {recommendation['recommendation']}")
            print(f"  Confidence: {recommendation['confidence']:.1%}")
            
    except Exception as e:
        print(f"  [WARN] Training skipped: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    conn.close()


if __name__ == "__main__":
    main()
