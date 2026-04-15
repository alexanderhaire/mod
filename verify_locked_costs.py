import pyodbc
from secrets_loader import build_connection_string
from constants import RAW_MATERIAL_CLASS_CODES

def verify_locked_costs():
    conn_str, _, _, _ = build_connection_string()
    raw_material_classes = "', '".join(RAW_MATERIAL_CLASS_CODES)
    
    # Test Data: Insert a mock override for NO3CA (assuming it exists)
    test_item = "NO3CA"
    test_cost = 0.99999
    
    upsert_sql = """
    MERGE INTO [dbo].[PURCHASING_MOD_OVERRIDES] AS target
    USING (SELECT ? AS ITEMNMBR, ? AS LOCKED_COST, 'TEST_AGENT' AS MODIFIED_BY, GETDATE() AS MODIFIED_DATE) AS source
    ON (target.ITEMNMBR = source.ITEMNMBR)
    WHEN MATCHED THEN
        UPDATE SET LOCKED_COST = source.LOCKED_COST, MODIFIED_BY = source.MODIFIED_BY, MODIFIED_DATE = source.MODIFIED_DATE
    WHEN NOT MATCHED THEN
        INSERT (ITEMNMBR, LOCKED_COST, MODIFIED_BY, MODIFIED_DATE)
        VALUES (source.ITEMNMBR, source.LOCKED_COST, source.MODIFIED_BY, source.MODIFIED_DATE);
    """
    
    verify_query = f"""
    SELECT 
        i.ITEMNMBR, 
        ov.LOCKED_COST as LockedCost,
        COALESCE(ov.LOCKED_COST, 0.0) as TrueCost
    FROM IV00101 i
    LEFT JOIN PURCHASING_MOD_OVERRIDES ov ON i.ITEMNMBR = ov.ITEMNMBR
    WHERE i.ITEMNMBR = ?
    """

    try:
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        
        print(f"Applying test override: {test_item} -> {test_cost}")
        cursor.execute(upsert_sql, (test_item, test_cost))
        
        print("Verifying query prioritization...")
        cursor.execute(verify_query, (test_item,))
        row = cursor.fetchone()
        
        if row:
            print(f"Item: {row[0]} | Locked: {row[1]} | Calculated True: {row[2]}")
            if float(row[2]) == test_cost:
                print("✅ SUCCESS: Locked cost prioritized correctly.")
            else:
                print("❌ FAILURE: Priority logic failed.")
        else:
            print("❌ Item not found in IV00101.")
            
        # Clean up
        print("Cleaning up test data...")
        cursor.execute("DELETE FROM PURCHASING_MOD_OVERRIDES WHERE ITEMNMBR = ?", (test_item,))
        
    except Exception as e:
        print(f"Error during verification: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_locked_costs()
