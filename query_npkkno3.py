import pyodbc
from secrets_loader import build_connection_string

conn_str, _, _, _ = build_connection_string()
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

print("="*80)
print("NPKKNO3 PURCHASE HISTORY")
print("="*80)

# Simpler query with basic columns
query = """
SELECT TOP 20
    h.RECEIPTDATE,
    h.VENDORID,
    h.VENDNAME,
    v.CITY,
    v.STATE,
    v.ZIPCODE,
    h.ORFRTAMT as freight_cost
FROM POP30300 h
JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
LEFT JOIN PM00200 v ON h.VENDORID = v.VENDORID
WHERE l.ITEMNMBR = 'NPKKNO3'
ORDER BY h.RECEIPTDATE DESC
"""

cursor.execute(query)
rows = cursor.fetchall()

print(f"\nFound {len(rows)} NPKKNO3 receipts:")
print("-"*80)
print(f"{'Date':<12} | {'Vendor':<25} | {'Location':<30} | {'Freight':>10}")
print("-"*80)

for row in rows:
    loc = f"{str(row.CITY or '').strip()}, {str(row.STATE or '').strip()} {str(row.ZIPCODE or '').strip()[:5]}"
    freight = f"${row.freight_cost:.2f}" if row.freight_cost else "N/A"
    print(f"{str(row.RECEIPTDATE)[:10]:<12} | {str(row.VENDNAME or '')[:25]:<25} | {loc[:30]:<30} | {freight:>10}")

# Summary
if rows:
    freights = [float(r.freight_cost) for r in rows if r.freight_cost and r.freight_cost > 0]
    if freights:
        print("-"*80)
        print(f"Freight Stats: Min=${min(freights):.2f} | Max=${max(freights):.2f} | Avg=${sum(freights)/len(freights):.2f}")

# Check Cole vendors
print("\n" + "="*80)
print("VENDORS WITH 'COLE' IN NAME:")
print("="*80)
cursor.execute("SELECT VENDORID, VENDNAME, CITY, STATE FROM PM00200 WHERE VENDNAME LIKE '%COLE%'")
for r in cursor.fetchall():
    print(f"  {r.VENDORID} | {r.VENDNAME} | {r.CITY}, {r.STATE}")

conn.close()
