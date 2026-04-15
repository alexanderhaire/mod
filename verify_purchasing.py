import pyodbc
from secrets_loader import build_connection_string
from constants import RAW_MATERIAL_CLASS_CODES

def verify_purchasing_query():
    conn_str, _, _, _ = build_connection_string()
    raw_material_classes = "', '".join(RAW_MATERIAL_CLASS_CODES)
    
    query = f"""
    WITH LatestOpenPO AS (
        SELECT 
            ITEMNMBR, 
            MAX(DEX_ROW_ID) as MaxID 
        FROM POP10110 
        GROUP BY ITEMNMBR
    ),
    OpenPOCost AS (
        SELECT 
            l.ITEMNMBR, 
            l.UNITCOST as OpenCost,
            h.VENDNAME as OpenVendor,
            h.DOCDATE as OpenDate
        FROM POP10110 l
        JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
        JOIN LatestOpenPO lo ON l.ITEMNMBR = lo.ITEMNMBR AND l.DEX_ROW_ID = lo.MaxID
    ),
    LatestHistPO AS (
        SELECT 
            ITEMNMBR, 
            MAX(DEX_ROW_ID) as MaxID 
        FROM POP30110 
        GROUP BY ITEMNMBR
    ),
    HistPOCost AS (
        SELECT 
            l.ITEMNMBR, 
            l.UNITCOST as HistCost,
            h.VENDNAME as HistVendor,
            h.DOCDATE as HistDate
        FROM POP30110 l
        JOIN POP30100 h ON l.PONUMBER = h.PONUMBER
        JOIN LatestHistPO lh ON l.ITEMNMBR = lh.ITEMNMBR AND l.DEX_ROW_ID = lh.MaxID
    )
    SELECT TOP 5
        i.ITEMNMBR, 
        i.ITEMDESC, 
        i.CURRCOST as GPCurrCost,
        o.OpenCost,
        h.HistCost,
        COALESCE(o.OpenCost, h.HistCost, i.CURRCOST) as TrueCost
    FROM IV00101 i
    LEFT JOIN OpenPOCost o ON i.ITEMNMBR = o.ITEMNMBR
    LEFT JOIN HistPOCost h ON i.ITEMNMBR = h.ITEMNMBR
    WHERE i.ITMCLSCD IN ('{raw_material_classes}')
    ORDER BY i.ITEMNMBR
    """

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"{'Item':<15} | {'Desc':<30} | {'GP Cost':<10} | {'Open':<10} | {'Hist':<10} | {'True'}")
        print("-" * 100)
        for row in rows:
            print(f"{str(row[0]):<15} | {str(row[1])[:30]:<30} | {float(row[2]):<10.4f} | "
                  f"{float(row[3]) if row[3] else 0:<10.4f} | {float(row[4]) if row[4] else 0:<10.4f} | {float(row[5]):.4f}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_purchasing_query()
