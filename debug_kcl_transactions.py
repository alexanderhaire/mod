import pyodbc
import sys
import os

sys.path.append(os.getcwd())

from secrets_loader import build_connection_string

conn_str, _, _, _ = build_connection_string()
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Specific investigation for KCL on 2025-07-31
query = """
SELECT 
    h.RECEIPTDATE,
    l.POPRCTNM as ReceiptNumber,
    l.RCPTLNNM as ReceiptLineNumber,
    h.VENDORID,
    h.VENDNAME,
    l.ITEMNMBR,
    l.ACTLSHIP as QtyShipped,
    l.UNITCOST,
    l.EXTDCOST,
    l.UOFM,
    l.PONUMBER as PONumber,
    l.LOCNCODE,
    COUNT(*) OVER (PARTITION BY l.POPRCTNM) as LinesInThisReceipt
FROM POP30310 l
JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
WHERE l.ITEMNMBR = 'KCL'
    AND CAST(h.RECEIPTDATE AS DATE) = '2025-07-31'
    AND h.POPTYPE <> 2
    AND h.VOIDSTTS = 0
ORDER BY l.POPRCTNM, l.RCPTLNNM
"""

cursor.execute(query)
rows = cursor.fetchall()

print(f"Found {len(rows)} receipt lines for KCL on 2025-07-31:")
print("=" * 120)

if cursor.description:
    columns = [c[0] for c in cursor.description]
    
    for row in rows:
        data = dict(zip(columns, row))
        print(f"""
Receipt: {data['ReceiptNumber']} (Line {data['ReceiptLineNumber']})
  PO Number: {data['PONumber']}
  Vendor: {data['VENDNAME']} ({data['VENDORID']})
  Qty Shipped: {data['QtyShipped']} {data['UOFM']}
  Location: {data['LOCNCODE']}
  Unit Cost: ${data['UNITCOST']:.4f}
  Extended Cost: ${data['EXTDCOST']:.2f}
  Lines in this receipt: {data['LinesInThisReceipt']}
        """)

print("=" * 120)
print(f"\nUnique Receipt Numbers: {len(set(row[1] for row in rows))}")
print(f"Total Receipt Lines: {len(rows)}")

conn.close()
