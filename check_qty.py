
import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def check_quantities():
    target_date = date(2025, 9, 30)
    items = ['MISCGLYCEROL', 'GRPAG50Y']
    conn_str, _, _, _ = build_connection_string()
    
    conn = pyodbc.connect(conn_str)
    
    # 1. Base Qty
    q_base = f"""
    SELECT ITEMNMBR, QTYONHND 
    FROM IV00102 
    WHERE ITEMNMBR IN ('MISCGLYCEROL', 'GRPAG50Y') AND LOCNCODE = 'MAIN'
    """
    base = pd.read_sql(q_base, conn)
    
    # 2. History Change
    q_hist = f"""
    SELECT ITEMNMBR, SUM(TRXQTY) as QtyChange
    FROM IV30300
    WHERE ITEMNMBR IN ('MISCGLYCEROL', 'GRPAG50Y')
      AND DOCDATE > ?
      AND TRXLOCTN = 'MAIN'
    GROUP BY ITEMNMBR
    """
    hist = pd.read_sql(q_hist, conn, params=[target_date])
    
    conn.close()
    
    df = pd.merge(base, hist, on='ITEMNMBR', how='left').fillna(0)
    df['Calculated_Qty'] = df['QTYONHND'] - df['QtyChange']
    
    print(df.to_string())

if __name__ == "__main__":
    check_quantities()
