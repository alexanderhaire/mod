"""
Connect the dots: SQM NPKKNO3 purchases + Cole Trucking freight payments
"""
import pyodbc
from secrets_loader import build_connection_string
from datetime import timedelta

conn_str, _, _, _ = build_connection_string()
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

print("="*90)
print("CORRELATING SQM PURCHASES WITH COLE TRUCKING FREIGHT")
print("="*90)

# Step 1: Get SQM NPKKNO3 purchase dates
print("\n>>> Step 1: SQM NPKKNO3 Purchase Dates")
sqm_query = """
SELECT DISTINCT CAST(h.RECEIPTDATE AS DATE) as purchase_date
FROM POP30300 h
JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
WHERE l.ITEMNMBR = 'NPKKNO3'
  AND h.VENDORID LIKE '%SQM%'
ORDER BY purchase_date DESC
"""
cursor.execute(sqm_query)
sqm_dates = [row.purchase_date for row in cursor.fetchall()]
print(f"Found {len(sqm_dates)} SQM purchase dates")

# Step 2: Get Cole Trucking payments from AP
print("\n>>> Step 2: Cole Trucking Payments (PM20000 - AP Open)")
cole_query = """
SELECT 
    DOCDATE,
    VENDORID,
    DOCNUMBR,
    DOCAMNT,
    CURTRXAM
FROM PM20000 
WHERE VENDORID IN ('DAVIDCOLE', 'JESSE COLE TRUC')
ORDER BY DOCDATE DESC
"""
try:
    cursor.execute(cole_query)
    cole_payments = cursor.fetchall()
    print(f"Found {len(cole_payments)} Cole AP transactions:")
    for p in cole_payments[:15]:
        print(f"  {p.DOCDATE} | {p.VENDORID:15} | Doc: {p.DOCNUMBR} | ${p.DOCAMNT:.2f}")
except Exception as e:
    print(f"PM20000 query failed: {e}")
    cole_payments = []

# Step 3: Check PM30200 (AP History)
print("\n>>> Step 3: Cole Trucking History (PM30200 - AP History)")
cole_history_query = """
SELECT TOP 30
    DOCDATE,
    VENDORID,
    DOCNUMBR,
    DOCAMNT
FROM PM30200 
WHERE VENDORID IN ('DAVIDCOLE', 'JESSE COLE TRUC')
ORDER BY DOCDATE DESC
"""
try:
    cursor.execute(cole_history_query)
    cole_history = cursor.fetchall()
    print(f"Found {len(cole_history)} Cole historical AP transactions:")
    
    amounts = []
    for p in cole_history:
        amounts.append(float(p.DOCAMNT))
        print(f"  {p.DOCDATE} | {p.VENDORID:15} | Doc: {p.DOCNUMBR} | ${p.DOCAMNT:.2f}")
    
    if amounts:
        print("-"*60)
        print(f"FREIGHT STATS: Min=${min(amounts):.2f} | Max=${max(amounts):.2f} | Avg=${sum(amounts)/len(amounts):.2f}")
        
        # Check if in $700-900 range
        in_range = [a for a in amounts if 700 <= a <= 900]
        print(f"Payments in $700-900 range: {len(in_range)} of {len(amounts)} ({100*len(in_range)/len(amounts):.0f}%)")
        
except Exception as e:
    print(f"PM30200 query failed: {e}")

# Step 4: Try to match dates
print("\n>>> Step 4: Date Correlation Analysis")
print("Checking if Cole payments align with SQM purchases...")

# For each Cole payment, see if there's an SQM purchase within +/- 3 days
if cole_history and sqm_dates:
    matches = 0
    for p in cole_history:
        pay_date = p.DOCDATE.date() if hasattr(p.DOCDATE, 'date') else p.DOCDATE
        for sqm_date in sqm_dates:
            if abs((pay_date - sqm_date).days) <= 5:
                matches += 1
                print(f"  MATCH: Cole ${p.DOCAMNT:.2f} on {pay_date} ~ SQM on {sqm_date}")
                break
    
    print(f"\nFound {matches} date correlations between Cole payments and SQM purchases")

conn.close()
