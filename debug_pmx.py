
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def inspect_pmx_history():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    query = """
    SELECT 
        l.ITEMNMBR,
        h.RECEIPTDATE,
        h.VENDORID,
        h.VENDNAME,
        l.UNITCOST,
        h.POPRCTNM
    FROM POP30310 l
    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
    WHERE l.ITEMNMBR LIKE '%REC-NPKPHOS85%'
    ORDER BY h.RECEIPTDATE DESC
    """
    
    df = pd.read_sql(query, conn)
    print(df.to_string())
    
    if not df.empty:
        print("\nChecking Vendor IDs:")
        for v in df['VENDORID'].unique():
            print(f"'{v}' (Length: {len(v)})")

if __name__ == "__main__":
    inspect_pmx_history()
