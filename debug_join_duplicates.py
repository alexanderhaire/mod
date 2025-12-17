
import pyodbc
import os
import sys

sys.path.append(os.getcwd())

try:
    from secrets_loader import build_connection_string
    
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("Checking for duplicate joins...")
    # Find a receipt with multiple lines for the same item
    cursor.execute("""
        SELECT TOP 5 
            POPRCTNM, ITEMNMBR, COUNT(*) as LineCount
        FROM POP30310
        GROUP BY POPRCTNM, ITEMNMBR
        HAVING COUNT(*) > 1
    """)
    
    multi_line_receipts = cursor.fetchall()
    
    if multi_line_receipts:
        for r in multi_line_receipts:
            receipt, item, count = r.POPRCTNM, r.ITEMNMBR, r.LineCount
            print(f"\nScanning Receipt {receipt.strip()} Item {item.strip()} (Lines: {count})")
            
            # Check Join Result Count
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM POP30310 p
                JOIN IV10200 iv ON p.ITEMNMBR = iv.ITEMNMBR 
                    AND p.POPRCTNM = iv.RCPTNMBR
                    AND p.LOCNCODE = iv.TRXLOCTN
                WHERE p.POPRCTNM = '{receipt}' AND p.ITEMNMBR = '{item}'
            """)
            join_count = cursor.fetchone()[0]
            print(f"-> Resulting Rows with Current Join: {join_count}")
            
            if join_count > count:
                print("   [!] DUPLICATES DETECTED!")
            
            # Inspect Columns for Linking
            print("   Inspecting POP30310 Lines:")
            cursor.execute(f"SELECT RCPTLNNM, UNITCOST FROM POP30310 WHERE POPRCTNM='{receipt}' AND ITEMNMBR='{item}'")
            for row in cursor.fetchall():
                print(f"     POP Line: {row.RCPTLNNM}, Cost: {row.UNITCOST}")
                
            print("   Inspecting IV10200 Layers:")
            cursor.execute(f"SELECT RCPTSEQNM, UNITCOST FROM IV10200 WHERE RCPTNMBR='{receipt}' AND ITEMNMBR='{item}'")
            for row in cursor.fetchall():
                print(f"     IV Seq: {row.RCPTSEQNM}, Cost: {row.UNITCOST}")
                
    else:
        print("No receipts with multiple lines for same item found (limit 5). Double counting might be rare or data specific.")

    conn.close()
except Exception as e:
    print(f"Error: {e}")
