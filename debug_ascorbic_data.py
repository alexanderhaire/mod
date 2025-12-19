
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def fetch_data():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    query = """
    SELECT 
        l.POPRCTNM,
        CAST(h.RECEIPTDATE AS DATE) as TransactionDate,
        h.VENDORID,
        h.VENDNAME,
        l.UOFM,
        l.UMQTYINB,
        l.UNITCOST,
        l.EXTDCOST,
        h.ORFRTAMT,
        h.SUBTOTAL,
        (SELECT TOP 1 UNITCOST FROM IV10200 iv WHERE iv.RCPTNMBR = l.POPRCTNM AND iv.ITEMNMBR = l.ITEMNMBR) as InventoryUnitCost
    FROM POP30310 l
    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
    WHERE l.ITEMNMBR = 'GRPASCORBIC'
        AND h.POPTYPE <> 2
        AND h.VOIDSTTS = 0
    ORDER BY h.RECEIPTDATE DESC
    """
    
    df = pd.read_sql(query, conn)
    df = pd.read_sql(query, conn)
    with open("debug_ascorbic_output.txt", "w") as f:
        f.write(df.head(50).to_string())
    print("Done writing file.")

if __name__ == "__main__":
    fetch_data()
