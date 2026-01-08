"""
Quick validation test for ML Procurement Optimizer.
Runs a simplified validation to show results faster.
"""

import pyodbc
from secrets_loader import build_connection_string
from procurement_ml import (
    ProcurementMLOptimizer, 
    ProcurementFeatureBuilder,
    BuyWindowPredictor,
    CriticalBuyFilter
)
import numpy as np
import pandas as pd

print("=" * 60)
print("ML PROCUREMENT OPTIMIZER - QUICK VALIDATION")
print("=" * 60)

# Connect
print("\n[1] Connecting to database...")
conn_str, server, db, _ = build_connection_string()
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()
print(f"    Connected to {server}/{db}")

# Get sample items with purchase history
print("\n[2] Finding items with purchase history...")
cursor.execute("""
    SELECT TOP 10 
        l.ITEMNMBR, 
        COUNT(*) as PurchaseCount,
        AVG(l.UNITCOST) as AvgCost,
        MIN(l.UNITCOST) as MinCost,
        MAX(l.UNITCOST) as MaxCost
    FROM POP30300 h
    JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
    WHERE l.EXTDCOST > 0 AND l.UNITCOST > 0
    GROUP BY l.ITEMNMBR
    HAVING COUNT(*) >= 3
    ORDER BY COUNT(*) DESC
""")
items = cursor.fetchall()
print(f"    Found {len(items)} items with 3+ purchases")

if not items:
    print("    No items found!")
    conn.close()
    exit()

# Show sample items
print("\n[3] Sample Items:")
for i, row in enumerate(items[:5]):
    item = row.ITEMNMBR.strip()
    print(f"    {i+1}. {item}: {row.PurchaseCount} purchases, Avg ${row.AvgCost:.2f} (${row.MinCost:.2f}-${row.MaxCost:.2f})")

# Test feature extraction
print("\n[4] Testing Feature Extraction...")
optimizer = ProcurementMLOptimizer(cursor)
test_item = items[0].ITEMNMBR.strip()

features = optimizer.feature_builder.build_features(test_item)
print(f"    Item: {features.item_number}")
print(f"    Current Price: ${features.current_price:.2f}")
print(f"    Price 52w Percentile: {features.price_percentile_52w:.1%}")
print(f"    Price Trend: {features.price_trend_slope:+.4f}")
print(f"    Days Coverage: {features.days_of_coverage:.0f}")
print(f"    Vendor Payment Days: {features.vendor_payment_days}")

# Test predictions (rule-based)
print("\n[5] Testing Predictions (Rule-Based Fallback)...")
results = []
for row in items:
    item = row.ITEMNMBR.strip()
    try:
        rec = optimizer.get_buy_recommendation(item)
        results.append({
            'Item': item,
            'BuyScore': rec['buy_score'],
            'Recommendation': rec['recommendation'].split(' - ')[0],
            'Confidence': rec['confidence'],
            'Coverage': rec['features']['days_of_coverage']
        })
    except Exception as e:
        print(f"    Error for {item}: {e}")

if results:
    df = pd.DataFrame(results)
    print("\n    Recommendations:")
    print(df.to_string(index=False))

# Test critical filter
print("\n[6] Applying Critical Buy Filter...")
if results:
    all_recs = pd.DataFrame([{
        'ItemNumber': r['Item'],
        'BuyScore': r['BuyScore'],
        'Recommendation': r['Recommendation'],
        'Confidence': r['Confidence'],
        'DaysCoverage': r['Coverage'],
        'Price52wPct': 0.5  # Default
    } for r in results])
    
    critical_filter = CriticalBuyFilter()
    critical = critical_filter.filter_critical(all_recs)
    
    if critical.empty:
        print("    No critical items in sample!")
    else:
        print(f"    {len(critical)} critical items found:")
        print(critical[['ItemNumber', 'BuyScore', 'DaysCoverage']].to_string(index=False))

# Quick accuracy test on historical labels
print("\n[7] Quick Accuracy Test...")
cursor.execute("""
    SELECT TOP 100
        l.ITEMNMBR,
        l.UNITCOST
    FROM POP30300 h
    JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
    WHERE l.EXTDCOST > 0 AND l.UNITCOST > 0
    ORDER BY h.RECEIPTDATE DESC
""")
recent = cursor.fetchall()

# Calculate price percentiles per item
item_prices = {}
for row in recent:
    item = row.ITEMNMBR.strip()
    if item not in item_prices:
        item_prices[item] = []
    item_prices[item].append(float(row.UNITCOST))

# Label: was this a "good buy"? (bottom 30%)
correct = 0
total = 0
for row in recent[:50]:  # Test on 50 most recent
    item = row.ITEMNMBR.strip()
    if item not in item_prices or len(item_prices[item]) < 3:
        continue
    
    price = float(row.UNITCOST)
    p30 = np.percentile(item_prices[item], 30)
    was_good_buy = price <= p30
    
    try:
        rec = optimizer.get_buy_recommendation(item)
        model_says_buy = rec['buy_score'] >= 60
        
        if model_says_buy == was_good_buy:
            correct += 1
        total += 1
    except:
        pass

if total > 0:
    accuracy = correct / total
    print(f"    Tested on {total} recent purchases")
    print(f"    Accuracy: {accuracy:.1%} ({correct}/{total} correct)")
else:
    print("    Not enough data for accuracy test")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

conn.close()
