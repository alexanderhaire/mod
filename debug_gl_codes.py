import pyodbc
import pandas as pd
from secrets_loader import build_connection_string
from datetime import date

conn_str, _, _, _ = build_connection_string()
conn = pyodbc.connect(conn_str)

target_date = date(2025, 9, 30)

# Run the exact same query as the report
query_base = """
SELECT 
    T1.ITEMNMBR,
    T1.ITEMDESC,
    T1.CURRCOST,
    T2.QTYONHND as CurrentQty,
    GL.ACTNUMST as GLCode
FROM IV00101 T1
JOIN IV00102 T2 ON T1.ITEMNMBR = T2.ITEMNMBR
LEFT JOIN IV40400 ClassAccts ON T1.ITMCLSCD = ClassAccts.ITMCLSCD
LEFT JOIN GL00105 GL ON GL.ACTINDX = (
    CASE 
        WHEN T1.IVIVINDX > 0 THEN T1.IVIVINDX 
        ELSE ClassAccts.IVIVINDX 
    END
)
WHERE T2.LOCNCODE = 'MAIN'
  AND RTRIM(GL.ACTNUMST) LIKE '12100%'
"""

query_history = """
SELECT 
    ITEMNMBR,
    SUM(TRXQTY) as QtyChange
FROM IV30300
WHERE DOCDATE > ? 
  AND TRXLOCTN = 'MAIN'
GROUP BY ITEMNMBR
"""

base_df = pd.read_sql(query_base, conn)
history_df = pd.read_sql(query_history, conn, params=[target_date])

# Merge
df = pd.merge(base_df, history_df, on='ITEMNMBR', how='left')
df['QtyChange'] = df['QtyChange'].fillna(0)
df['Quantity'] = df['CurrentQty'] - df['QtyChange']

# Filter zero quantities (same as report)
df_nonzero = df[df['Quantity'] != 0]

print(f"Total items before zero filter: {len(df)}")
print(f"Total items after zero filter: {len(df_nonzero)}")
print(f"\nGL Code breakdown (after zero filter):")
print(df_nonzero.groupby('GLCode').size())

# Try to compute total value
df_nonzero['ExtCost'] = df_nonzero['Quantity'] * df_nonzero['CURRCOST']
print(f"\nTotal Value: ${df_nonzero['ExtCost'].sum():,.2f}")

conn.close()
