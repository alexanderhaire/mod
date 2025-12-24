import pyodbc
import sys
import os
import pandas as pd

sys.path.append(os.getcwd())
from secrets_loader import build_connection_string

conn_str, _, _, _ = build_connection_string()
conn = pyodbc.connect(conn_str)

# Query specific receipts from the user's issue
receipts = ['RV0000039571', 'RV0000039572', 'RV0000039573']
placeholders = ','.join(['?'] * len(receipts))

query = f"""
SELECT 
    h.POPRCTNM,
    h.RECEIPTDATE,
    h.POPTYPE,
    h.VENDNAME,
    l.ITEMNMBR,
    l.RCPTRETNUM,
    l.RCPTRETLNNUM,
    l.ACTLSHIP,
    l.UNITCOST,
    l.EXTDCOST
FROM POP30300 h
JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
WHERE h.POPRCTNM IN ({placeholders})
"""

print(f"Querying receipts: {receipts}")
df = pd.read_sql(query, conn, params=receipts)

print("\n--- RAW DATA DUMP ---")
print(df.to_string())

print("\n--- ANALYSIS ---")
for _, row in df.iterrows():
    r_num = row['POPRCTNM']
    r_type = row['POPTYPE']
    ret_ref = row['RCPTRETNUM']
    
    print(f"Receipt {r_num}: Type={r_type} ({'Return' if r_type==5 else 'Shipment/Inv'}), Ref={ret_ref}")

conn.close()
