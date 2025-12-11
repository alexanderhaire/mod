"""Test script to verify all market_insights functions work correctly."""
import sys
import pyodbc
from secrets_loader import build_connection_string

# Database connection
connection_string, server, database, auth_mode = build_connection_string()
print(f"Connecting to {server}/{database} using {auth_mode}")

try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print("✓ Database connected")
except Exception as e:
    print(f"ERROR: Database connection failed: {e}")
    sys.exit(1)

# Test 1: Get a sample raw material item
print("\n--- Test 1: Finding sample raw materials ---")
cursor.execute("""
    SELECT TOP 5 ITEMNMBR, ITEMDESC, CURRCOST 
    FROM IV00101 
    WHERE ITEMTYPE IN (0, 1, 2) 
    AND CURRCOST > 0
    ORDER BY ITEMNMBR
""")
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"  {row[0]}: {row[1]} (${row[2]:.2f})")
    test_item = rows[0][0]
    print(f"\n✓ Using item: {test_item}")
else:
    print("ERROR: No items found")
    sys.exit(1)

# Test 2: fetch_monthly_price_trends
print("\n--- Test 2: fetch_monthly_price_trends ---")
try:
    from market_insights import fetch_monthly_price_trends
    df = fetch_monthly_price_trends(cursor, test_item, months=12)
    if df.empty:
        print(f"⚠ No price trends found for {test_item}")
        # Check if there's any receipt data at all
        cursor.execute("""
            SELECT TOP 5 h.POPRCTNM, h.RECEIPTDATE, l.ITEMNMBR, l.UNITCOST
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE l.ITEMNMBR = ?
            ORDER BY h.RECEIPTDATE DESC
        """, test_item)
        receipts = cursor.fetchall()
        if receipts:
            print(f"  Found {len(receipts)} receipts in POP tables")
            for r in receipts:
                print(f"    {r[1]}: ${r[3]:.2f}")
        else:
            print(f"  No receipts in POP30310 for {test_item}")
    else:
        print(f"✓ Found {len(df)} months of price trends")
        print(df[['Date', 'AvgCost', 'MonthLabel']].to_string())
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: calculate_inventory_runway
print("\n--- Test 3: calculate_inventory_runway ---")
try:
    from market_insights import calculate_inventory_runway
    runway = calculate_inventory_runway(cursor, test_item)
    print(f"✓ Runway: {runway}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: get_volatility_score
print("\n--- Test 4: get_volatility_score ---")
try:
    from market_insights import get_volatility_score
    vol = get_volatility_score(cursor, test_item)
    print(f"✓ Volatility: {vol}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 5: get_seasonal_pattern
print("\n--- Test 5: get_seasonal_pattern ---")
try:
    from market_insights import get_seasonal_pattern
    season = get_seasonal_pattern(cursor, test_item)
    print(f"✓ Seasonal: {season}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 6: calculate_buying_signals
print("\n--- Test 6: calculate_buying_signals ---")
try:
    from market_insights import calculate_buying_signals
    signals = calculate_buying_signals(cursor, test_item)
    print(f"✓ Buying signals: {signals}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All tests completed ===")
cursor.close()
conn.close()
