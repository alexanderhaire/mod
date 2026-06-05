"""Read-only validation: does the BOM-category fix make implied raw demand reconcile with actual usage?
Compares 12-mo actual raw usage (IV30300) vs sales->BOM-implied, BEFORE and AFTER the fix.
Throwaway audit script (scripts/_audit_*). Does not modify any data."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_pool import get_connection

RAWS = ['NO3MN', 'CHEMONETH', 'SO4BORIC', 'SO4FEDRY19', 'SO4MN32', 'SO4BORON', 'NPKNPHURCDI']


def base(x):
    x = x.strip()
    return x[4:] if x.upper().startswith('REC-') else x


# Original (broken) explosion: no category filter, sums every stored recipe row
BROKEN = """WITH C (tp,ci,q,d,p) AS (
  SELECT PPN_I,CPN_I,CAST(QUANTITY_I AS DECIMAL(38,19)),1,CAST('|'+RTRIM(PPN_I)+'|'+RTRIM(CPN_I)+'|' AS VARCHAR(4000)) FROM BM010115
  UNION ALL
  SELECT c.tp,b.CPN_I,CAST(c.q*b.QUANTITY_I AS DECIMAL(38,19)),c.d+1,CAST(c.p+RTRIM(b.CPN_I)+'|' AS VARCHAR(4000))
  FROM BM010115 b JOIN C c ON b.PPN_I=c.ci WHERE c.d<20 AND c.p NOT LIKE '%|'+RTRIM(b.CPN_I)+'|%')
SELECT RTRIM(tp) tp,RTRIM(ci) ci,SUM(q) q FROM C
WHERE ci IN (SELECT ITEMNMBR FROM IV00101 WHERE ITMCLSCD LIKE 'RAWMAT%')
GROUP BY RTRIM(tp),RTRIM(ci) OPTION (MAXRECURSION 50)"""

# Fixed explosion: active recipe only (BOMCAT_I=1, blank name) on BOTH members,
# plus RAWMAT* items treated as hard leaves (don't recurse into REC-/dilution rows)
FIXED = """WITH C (tp,ci,q,d,p) AS (
  SELECT PPN_I,CPN_I,CAST(QUANTITY_I AS DECIMAL(38,19)),1,CAST('|'+RTRIM(PPN_I)+'|'+RTRIM(CPN_I)+'|' AS VARCHAR(4000))
  FROM BM010115 WHERE BOMCAT_I=1 AND LEN(BOMNAME_I)=0
  UNION ALL
  SELECT c.tp,b.CPN_I,CAST(c.q*b.QUANTITY_I AS DECIMAL(38,19)),c.d+1,CAST(c.p+RTRIM(b.CPN_I)+'|' AS VARCHAR(4000))
  FROM BM010115 b JOIN C c ON b.PPN_I=c.ci
  WHERE c.d<20 AND c.p NOT LIKE '%|'+RTRIM(b.CPN_I)+'|%' AND b.BOMCAT_I=1 AND LEN(b.BOMNAME_I)=0
    AND NOT EXISTS (SELECT 1 FROM IV00101 i WHERE RTRIM(i.ITEMNMBR)=RTRIM(c.ci) AND i.ITMCLSCD LIKE 'RAWMAT%'))
SELECT RTRIM(tp) tp,RTRIM(ci) ci,SUM(q) q FROM C
WHERE ci IN (SELECT ITEMNMBR FROM IV00101 WHERE ITMCLSCD LIKE 'RAWMAT%')
GROUP BY RTRIM(tp),RTRIM(ci) OPTION (MAXRECURSION 50)"""

with get_connection() as conn:
    cur = conn.cursor()
    cur.execute("""SELECT RTRIM(ITEMNMBR) it, SUM(ABS(TRXQTY)) u FROM IV30300
      WHERE TRXQTY<0 AND DOCDATE>=DATEADD(month,-12,GETDATE()) GROUP BY RTRIM(ITEMNMBR)""")
    usage = {}
    for r in cur.fetchall():
        usage[base(r.it)] = usage.get(base(r.it), 0.0) + float(r.u or 0)

    cur.execute("""SELECT RTRIM(l.ITEMNMBR) fg, SUM(CASE WHEN l.SOPTYPE=3 THEN l.QUANTITY ELSE -l.QUANTITY END) u
      FROM SOP30300 l JOIN SOP30200 h ON l.SOPNUMBE=h.SOPNUMBE AND l.SOPTYPE=h.SOPTYPE
      WHERE h.SOPTYPE IN (3,4) AND h.VOIDSTTS=0 AND h.DOCDATE>=DATEADD(month,-12,GETDATE())
      GROUP BY RTRIM(l.ITEMNMBR)""")
    sales = {r.fg: float(r.u or 0) for r in cur.fetchall()}

    def implied(sql):
        cur.execute(sql)
        imp = {}
        for r in cur.fetchall():
            s = sales.get(r.tp, 0.0)
            if s > 0:
                rb = base(r.ci)
                imp[rb] = imp.get(rb, 0.0) + s * float(r.q or 0)
        return imp

    imp_b = implied(BROKEN)
    imp_f = implied(FIXED)

print(f'{"RAW":<13}{"ACTUAL_12mo":>14}{"IMPLIED_now":>15}{"xNOW":>8}{"IMPLIED_fixed":>15}{"xFIXED":>9}')
for r in RAWS:
    a = usage.get(r, 0.0)
    b = imp_b.get(r, 0.0)
    f = imp_f.get(r, 0.0)
    xb = (b / a) if a > 0 else float('nan')
    xf = (f / a) if a > 0 else float('nan')
    print(f'{r:<13}{a:>14,.0f}{b:>15,.0f}{xb:>8.1f}{f:>15,.0f}{xf:>9.2f}')
