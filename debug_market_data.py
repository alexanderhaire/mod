import pyodbc
from secrets_loader import build_connection_string
import pandas as pd

def debug_market_fetch():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # This matches the query in _fetch_market_data in app.py
        query = """
        WITH CurrentPrices AS (
            SELECT 
                l.ITEMNMBR, 
                AVG(l.UNITCOST) as CurrentAvgCost,
                COUNT(*) as RecentReceipts
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -3, GETDATE())
              AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR
        ),
        PriorPrices AS (
            SELECT 
                l.ITEMNMBR, 
                AVG(l.UNITCOST) as PriorAvgCost,
                COUNT(*) as PriorReceipts
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -15, GETDATE())
              AND h.RECEIPTDATE < DATEADD(month, -12, GETDATE())
              AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR
        )
        SELECT
            i.ITEMNMBR, 
            i.ITEMDESC, 
            i.ITEMTYPE,
            i.ITMCLSCD, -- THIS IS THE NEW LINE
            i.STNDCOST, 
            i.CURRCOST, 
            i.USCATVLS_1 as Category,
            COALESCE(c.CurrentAvgCost, i.CURRCOST) as CurrentAvgCost,
            p.PriorAvgCost,
            COALESCE(c.CurrentAvgCost, i.CURRCOST) - COALESCE(p.PriorAvgCost, i.STNDCOST) as PriceChange,
            CASE 
                WHEN p.PriorAvgCost > 0 THEN 
                    ((COALESCE(c.CurrentAvgCost, i.CURRCOST) - p.PriorAvgCost) / p.PriorAvgCost) * 100 
                WHEN i.STNDCOST > 0 THEN
                    ((i.CURRCOST - i.STNDCOST) / i.STNDCOST) * 100
                ELSE 0 
            END as PctChange
        FROM IV00101 i
        LEFT JOIN CurrentPrices c ON i.ITEMNMBR = c.ITEMNMBR
        LEFT JOIN PriorPrices p ON i.ITEMNMBR = p.ITEMNMBR
        WHERE i.ITEMTYPE IN (0, 1, 2)
        ORDER BY i.ITEMNMBR
        """
        
        print("Executing Query...")
        cursor.execute(query)
        print("Query Executed Successfully.")
        
        rows = cursor.fetchall()
        print(f"Fetched {len(rows)} rows.")
        if rows:
            print("First row:", rows[0])

    except Exception as e:
        print(f"SQL Execution Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_market_fetch()
