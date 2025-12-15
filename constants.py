import logging
import re
from pathlib import Path

# --- Constants ---

RAW_MATERIAL_PREFIXES = (
    'CHEL', 'CHE', 'CITRIC', 'EDTA', 'BORIC', 'MAP', 'MKP', 'UREA',
    'NPK', 'NO3', 'SO4', 'CL2', 'KOH', 'KTS', 'PHOS', 'ACE', 'AGB', 'AQUA'
)
FINISHED_GOOD_PREFIXES = ('EDTAK', 'EDTAC', 'EDTAN', 'PRO', 'DYN', 'DB', 'BULK', 'SOAR', 'ZZ')
RAW_MATERIAL_CATEGORIES = {'CHE', 'EDTA', 'ACID', 'RM', 'NPK', 'NO3', 'SO4', 'CL2', 'THIO'}
FINISHED_GOOD_CATEGORIES = {'DYNAGOLD', 'DIAMONDBACK', 'BULK', 'CUSTOM', 'PRIVATE'}
RAW_MATERIAL_CLASS_CODES: tuple[str, ...] = (
    'RAWMATTNE',
    'RAWMATNTE',
    'RAWMATNT',
    'RAWMATT',
)
RAW_MATERIAL_KEYWORDS = (
    'NITRATE', 'SULFATE', 'ACID', 'HYDROXIDE', 'UREA', 'PHOS', 'CHLORIDE',
    'AMMONIUM', 'BORATE', 'CITRIC', 'GLUCO', 'POTASH', 'PHOSPHORIC',
    'THIOSULFATE', 'MURIATE', 'POTASSIUM', 'MAGNESIUM', 'CALCIUM',
    'ZINC', 'IRON', 'MANGANESE', 'BORON'
)

PRIMARY_LOCATION = "MAIN"

APP_ROOT = Path(__file__).resolve().parent
LOCAL_SECRETS_PATHS = (
    APP_ROOT / ".streamlit" / "secrets.toml",
    APP_ROOT / "secrets.toml",
)
LOGGER = logging.getLogger("reasoning_core")

OPENAI_BEST_MODEL = "gpt-4o"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_DEFAULT_MODEL = OPENAI_BEST_MODEL
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_TIMEOUT_SECONDS = 50
DEFAULT_MODEL_CONTEXT_LIMIT = 128000
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gpt-4.1": 128000,
    "gpt-4.1-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
}

CUSTOM_SQL_MAX_ROWS = 200
CUSTOM_SQL_ALLOWED_TABLES: tuple[str, ...] = (
    "IV00101", "IV00102", "IV30200", "IV30300", "BM00101", "BM00111",
    "BM010115", "POP10100", "POP10110", "POP30100", "POP30110", "POP30200",
    "POP30300", "POP30310", "SOP10100", "SOP10200", "SOP30200", "SOP30300",
    "RM00101", "GL00100", "GL00105", "GL20000", "PM00200", "PM30200",
)

SCHEMA_PRIORITY_COLUMNS: tuple[str, ...] = (
    "CUSTNMBR", "CUSTNAME",  # Customer info
    "ITEMNMBR", "ITEMDESC",  # Item info
    "DOCDATE", "GLPOSTDT",   # Dates
    "QUANTITY", "XTNDPRCE", "DOCAMNT",  # Metrics
    "SOPNUMBE", "SOPTYPE",   # Keys
    "VENDORID", "VENDNAME",  # Vendor info
)

FEW_SHOT_EXAMPLES = """
Here are some examples:

---
Question: "what should we plan to buy in december?"
Context: "intent=planning; month=12; year=2025"
Schema: "SOP10100 (SOPNUMBE, DOCDATE), SOP10200 (SOPNUMBE, ITEMNMBR, QUANTITY), POP10110 (ITEMNMBR, QTYORDER), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH SalesDemand AS (SELECT l.ITEMNMBR, SUM(l.QUANTITY) AS Demand FROM SOP10200 l JOIN SOP10100 h ON l.SOPNUMBE = h.SOPNUMBE WHERE h.DOCDATE >= '2025-12-01' AND h.DOCDATE <= '2025-12-31' GROUP BY l.ITEMNMBR), OpenPurchaseOrders AS (SELECT ITEMNMBR, SUM(QTYORDER) AS Supply FROM POP10110 WHERE POLNESTA = 1 GROUP BY ITEMNMBR) SELECT i.ITEMNMBR, i.ITEMDESC, ISNULL(d.Demand, 0) AS DecemberDemand, ISNULL(s.Supply, 0) AS OpenPOSupply, (ISNULL(d.Demand, 0) - ISNULL(s.Supply, 0)) AS NetRequirement FROM IV00101 i LEFT JOIN SalesDemand d ON i.ITEMNMBR = d.ITEMNMBR LEFT JOIN OpenPurchaseOrders s ON i.ITEMNMBR = s.ITEMNMBR WHERE ISNULL(d.Demand, 0) > 0 ORDER BY NetRequirement DESC",
  "params": [],
  "summary": "Calculates the net requirement for items with demand in December, by comparing sales demand with open purchase order supply.",
  "entities": {"intent": "planning", "month": 12, "year": 2025}
}
---
Question: "what was NO3CA12 usage for last month?"
Context: "intent=report; item=NO3CA12; month=10; year=2025"
Schema: "IV30300 (DOCNUMBR, DOCTYPE, ITEMNMBR, TRXQTY), IV30200 (DOCNUMBR, DOCTYPE, DOCDATE)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT SUM(t.TRXQTY) AS UsageForPeriod FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE t.ITEMNMBR = ? AND t.DOCTYPE = 1 AND t.TRXQTY < 0 AND h.DOCDATE >= ? AND h.DOCDATE <= ?",
  "params": ["NO3CA12", "2025-10-01", "2025-10-31"],
  "summary": "Calculates the total usage for item NO3CA12 for October 2025.",
  "entities": {"item": "NO3CA12", "month": 10, "year": 2025, "intent": "report"}
}
---
Question: "show me sales for AO4ADD in august"
Context: "intent=sales; item=AO4ADD; month=8; year=2025"
Schema: "SOP30300 (SOPTYPE, SOPNUMBE, ITEMNMBR, QUANTITY), SOP30200 (SOPTYPE, SOPNUMBE, DOCDATE)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS SalesQuantity FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE l.ITEMNMBR = ? AND h.DOCDATE >= ? AND h.DOCDATE <= ? AND l.SOPTYPE IN (3, 4)",
  "params": ["AO4ADD", "2025-08-01", "2025-08-31"],
  "summary": "Calculates the total sales for item AO4ADD for August 2025.",
  "entities": {"item": "AO4ADD", "month": 8, "year": 2025, "intent": "sales"}
}
---
Question: "what was the monthly usage for NPKACEK last year?"
Context: "intent=report; item=NPKACEK; year=2024"
Schema: "IV30300 (DOCNUMBR, DOCTYPE, ITEMNMBR, TRXQTY), IV30200 (DOCNUMBR, DOCTYPE, DOCDATE)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT MONTH(h.DOCDATE) AS Month, SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS TotalUsage FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE t.ITEMNMBR = ? AND h.DOCDATE >= ? AND h.DOCDATE <= ? GROUP BY MONTH(h.DOCDATE) ORDER BY Month ASC",
  "params": ["NPKACEK", "2024-01-01", "2024-12-31"],
  "summary": "Calculates the total monthly usage for item NPKACEK for the year 2024.",
  "entities": {"item": "NPKACEK", "year": 2024, "intent": "report"}
}
---
Question: "generate a market dashboard for raw materials"
Context: "intent=dashboard; segment=raw materials"
Schema: "IV00101 (ITEMNMBR, ITEMDESC, CURRCOST)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT TOP 10 ITEMNMBR, ITEMDESC, CURRCOST FROM IV00101 WHERE ITEMNMBR LIKE 'NPK%'",
  "params": [],
  "summary": "Generates a dashboard view for raw materials.",
  "entities": {"intent": "dashboard", "segment": "raw materials"},
  "report_structure": {
    "title": "Raw Material Market Dashboard",
    "sections": [
      {"type": "metric", "title": "Total Items", "value": "150"},
      {"type": "chart", "title": "Top Cost Drivers", "data_source": "sql_result"}
    ]
  }
}
---
Question: "what is the standard cost?"
Context: "item=NPKACEK"
Schema: "IV00101 (ITEMNMBR, STNDCOST)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT STNDCOST FROM IV00101 WHERE ITEMNMBR = ?",
  "params": ["NPKACEK"],
  "summary": "Retrieves the standard cost for item NPKACEK.",
  "entities": {"item": "NPKACEK", "intent": "cost"}
}
---
Question: "what is CHEGLUCOGR50 made of?"
Context: "intent=bom; item=CHEGLUCOGR50"
Schema: "BM00111 (ITEMNMBR, CMPTITNM, Design_Qty)"
Today: 2025-11-19

JSON Response:
{
  "sql": "SELECT COMPONENT.ITEMNMBR, COMPONENT.CMPTITNM AS ComponentDescription, COMPONENT.Design_Qty FROM BM00111 AS COMPONENT WHERE COMPONENT.ITEMNMBR = ?",
  "params": ["CHEGLUCOGR50"],
  "summary": "Lists the bill of materials components and design quantities for CHEGLUCOGR50. For the app view, open the Bill of Materials inquiry page for this item.",
  "entities": {"item": "CHEGLUCOGR50", "intent": "bom"}
}
---
Question: "show me the manufacturing BOM for SOARBLM02 grouped by component"
Context: "intent=bom; item=SOARBLM02"
Schema: "BM010115 (PPN_I, CPN_I, QUANTITY_I, UOFM), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "SELECT b.PPN_I AS ParentItem, ip.ITEMDESC AS ParentDescription, b.CPN_I AS ComponentItem, ic.ITEMDESC AS ComponentDescription, SUM(b.QUANTITY_I) AS QtyPerParent, MAX(b.UOFM) AS ComponentUofM FROM BM010115 b LEFT JOIN IV00101 ip ON ip.ITEMNMBR = b.PPN_I LEFT JOIN IV00101 ic ON ic.ITEMNMBR = b.CPN_I WHERE b.PPN_I = ? GROUP BY b.PPN_I, ip.ITEMDESC, b.CPN_I, ic.ITEMDESC ORDER BY b.CPN_I",
  "params": ["SOARBLM02"],
  "summary": "Aggregates the manufacturing BOM for SOARBLM02 by component, with parent/component descriptions, summed quantities per parent, and the component U of M.",
  "entities": {"item": "SOARBLM02", "intent": "bom"}
}
---
Question: "chart the monthly sales for AO4ADD last year"
Context: "intent=sales; item=AO4ADD; year=2024"
Schema: "SOP30300 (SOPTYPE, SOPNUMBE, ITEMNMBR, QUANTITY), SOP30200 (SOPTYPE, SOPNUMBE, DOCDATE)"
Today: 2025-11-20

JSON Response:
{
  "sql": "SELECT MONTH(h.DOCDATE) AS Month, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS SalesQuantity FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE l.ITEMNMBR = ? AND h.DOCDATE >= ? AND h.DOCDATE <= ? AND l.SOPTYPE IN (3, 4) GROUP BY MONTH(h.DOCDATE) ORDER BY Month ASC",
  "params": ["AO4ADD", "2024-01-01", "2024-12-31"],
  "summary": "Calculates total monthly sales for AO4ADD for 2024 using shipped invoice history.",
  "entities": {"item": "AO4ADD", "year": 2024, "intent": "sales"},
  "chart": {"type": "line", "x": "Month", "y": "SalesQuantity", "title": "Monthly sales for AO4ADD in 2024"}
}
---
Question: "What are the top 5 raw materials used in the last 90 days?"
Context: "intent=usage; period=last 90 days"
Schema: "IV30300 (DOCNUMBR, DOCTYPE, ITEMNMBR, TRXQTY), IV30200 (DOCNUMBR, DOCTYPE, DOCDATE), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "SELECT TOP 5 t.ITEMNMBR, i.ITEMDESC, SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS UsageQty FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP LEFT JOIN IV00101 i ON i.ITEMNMBR = t.ITEMNMBR WHERE h.DOCDATE >= ? AND t.TRXQTY < 0 GROUP BY t.ITEMNMBR, i.ITEMDESC ORDER BY UsageQty DESC",
  "params": ["2025-08-22"],
  "summary": "Ranks the top 5 raw materials by usage quantity over the last 90 days.",
  "entities": {"intent": "usage", "period": "last 90 days"}
}
---
Question: "show total monthly sales value over the last 6 months"
Context: "intent=sales_trend; period=last 6 months"
Schema: "SOP30300 (SOPTYPE, SOPNUMBE, ITEMNMBR, QUANTITY, EXTDCOST), SOP30200 (SOPTYPE, SOPNUMBE, DOCDATE)"
Today: 2025-11-20

JSON Response:
{
  "sql": "SELECT MONTH(h.DOCDATE) AS Month, YEAR(h.DOCDATE) AS Year, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.EXTDCOST) ELSE ABS(l.EXTDCOST) END) AS SalesValue FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE h.DOCDATE >= ? AND l.SOPTYPE IN (3, 4) GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE) ORDER BY Year, Month",
  "params": ["2025-05-20"],
  "summary": "Calculates total monthly sales value over the last 6 months using shipped invoice history.",
  "entities": {"intent": "sales_trend", "period": "last 6 months"},
  "chart": {"type": "line", "x": "Month", "y": "SalesValue", "title": "Monthly sales value (last 6 months)"}
}
---
Question: "Compare usage of NPK3011 vs NPKACEK over the last 3 months"
Context: "intent=comparison; items=NPK3011,NPKACEK; period=last 3 months"
Schema: "IV30300 (ITEMNMBR, TRXQTY), IV30200 (DOCNUMBR, DOCTYPE, DOCDATE)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH UsageData AS (SELECT t.ITEMNMBR, MONTH(h.DOCDATE) AS Month, SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) AS Usage FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE t.ITEMNMBR IN (?, ?) AND h.DOCDATE >= ? GROUP BY t.ITEMNMBR, MONTH(h.DOCDATE)) SELECT Month, SUM(CASE WHEN ITEMNMBR = ? THEN Usage ELSE 0 END) AS NPK3011_Usage, SUM(CASE WHEN ITEMNMBR = ? THEN Usage ELSE 0 END) AS NPKACEK_Usage FROM UsageData GROUP BY Month ORDER BY Month",
  "params": ["NPK3011", "NPKACEK", "2025-08-20", "NPK3011", "NPKACEK"],
  "summary": "Compares monthly usage of NPK3011 vs NPKACEK over the last 3 months using a CTE to pivot the data.",
  "entities": {"items": ["NPK3011", "NPKACEK"], "intent": "comparison"}
}
---
Question: "If SOARBLM02 demand increases 5%, what raw materials are needed?"
Context: "intent=what_if; item=SOARBLM02; change=5% increase"
Schema: "IV30300 (ITEMNMBR, TRXQTY), IV30200 (DOCDATE), BM010115 (PPN_I, CPN_I, QUANTITY_I), IV00102 (ITEMNMBR, QTYONHND, ATYALLOC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH BaselineDemand AS (SELECT ITEMNMBR, ABS(SUM(TRXQTY)) as MonthlyUsage FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE h.DOCDATE >= DATEADD(day, -30, ?) AND t.ITEMNMBR = ? AND t.TRXQTY < 0 GROUP BY ITEMNMBR), ScenarioDemand AS (SELECT ITEMNMBR, MonthlyUsage * 1.05 as NewDemand, MonthlyUsage * 0.05 as Incremental FROM BaselineDemand), BOMComponents AS (SELECT CPN_I as Component, QUANTITY_I as QtyPer FROM BM010115 WHERE PPN_I = ?), RawMaterialNeeds AS (SELECT bc.Component, sd.Incremental * bc.QtyPer as IncrementalNeed FROM BOMComponents bc CROSS JOIN ScenarioDemand sd), CurrentStock AS (SELECT ITEMNMBR, SUM(QTYONHND - ATYALLOC) as FreeStock FROM IV00102 GROUP BY ITEMNMBR) SELECT rm.Component, rm.IncrementalNeed, ISNULL(cs.FreeStock, 0) as CurrentStock, CASE WHEN rm.IncrementalNeed > ISNULL(cs.FreeStock, 0) THEN rm.IncrementalNeed - ISNULL(cs.FreeStock, 0) ELSE 0 END as BuyQty FROM RawMaterialNeeds rm LEFT JOIN CurrentStock cs ON rm.Component = cs.ITEMNMBR WHERE rm.IncrementalNeed > 0 ORDER BY rm.IncrementalNeed DESC",
  "params": ["2025-11-20", "SOARBLM02", "SOARBLM02"],
  "summary": "Calculates incremental raw material needs if SOARBLM02 demand increases by 5% using baseline usage, BOM explosion, and current inventory.",
  "entities": {"item": "SOARBLM02", "scenario": "5% increase"},
  "reasoning": ["Get baseline monthly usage for SOARBLM02", "Apply 5% multiplier for scenario", "Explode BOM to get components", "Calculate incremental material needs", "Compare to current free stock"]
}
---
Question: "Show items with declining sales trend over the last 6 months"
Context: "intent=trend_analysis; metric=sales; trend=declining"
Schema: "SOP30300 (ITEMNMBR, SOPTYPE, QUANTITY), SOP30200 (SOPNUMBE, SOPTYPE, DOCDATE), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH MonthlySales AS (SELECT l.ITEMNMBR, MONTH(h.DOCDATE) AS Month, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS Qty FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE h.DOCDATE >= DATEADD(month, -6, ?) AND l.SOPTYPE IN (3, 4) GROUP BY l.ITEMNMBR, MONTH(h.DOCDATE)), FirstLast AS (SELECT ITEMNMBR, MIN(Month) as FirstMonth, MAX(Month) as LastMonth, AVG(CASE WHEN Month <= (SELECT MIN(Month) + 2 FROM MonthlySales WHERE ITEMNMBR = ms.ITEMNMBR) THEN Qty END) as Early3MonthAvg, AVG(CASE WHEN Month >= (SELECT MAX(Month) - 2 FROM MonthlySales WHERE ITEMNMBR = ms.ITEMNMBR) THEN Qty END) as Recent3MonthAvg FROM MonthlySales ms GROUP BY ITEMNMBR), Trending AS (SELECT ITEMNMBR, Early3MonthAvg, Recent3MonthAvg, (Recent3MonthAvg - Early3MonthAvg) / NULLIF(Early3MonthAvg, 0) * 100 as PercentChange FROM FirstLast WHERE Early3MonthAvg > 0) SELECT TOP 50 t.ITEMNMBR, i.ITEMDESC, t.Early3MonthAvg as EarlyAvg, t.Recent3MonthAvg as RecentAvg, t.PercentChange FROM Trending t LEFT JOIN IV00101 i ON t.ITEMNMBR = i.ITEMNMBR WHERE t.PercentChange < -10 ORDER BY t.PercentChange ASC",
  "params": ["2025-11-20"],
  "summary": "Identifies items with declining sales trends (>10% drop) by comparing early 3-month average vs recent 3-month average over last 6 months.",
  "entities": {"intent": "trend_analysis", "trend": "declining"},
  "reasoning": ["Calculate monthly sales for each item", "Compute early 3-month average", "Compute recent 3-month average", "Calculate percent change", "Filter for >10% decline"]
}
---
Question: "What's the gross margin by item for top 10 sellers last month?"
Context: "intent=profitability; period=last month; limit=10"
Schema: "SOP30300 (ITEMNMBR, QUANTITY, EXTDCOST, XTNDPRCE), SOP30200 (SOPNUMBE, SOPTYPE, DOCDATE), IV00101 (ITEMNMBR, ITEMDESC, STNDCOST)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH Sales AS (SELECT l.ITEMNMBR, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS TotalQty, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.XTNDPRCE) ELSE ABS(l.XTNDPRCE) END) AS Revenue, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.EXTDCOST) ELSE ABS(l.EXTDCOST) END) AS COGS FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE YEAR(h.DOCDATE) = YEAR(DATEADD(month, -1, ?)) AND MONTH(h.DOCDATE) = MONTH(DATEADD(month, -1, ?)) AND l.SOPTYPE IN (3, 4) GROUP BY l.ITEMNMBR), TopSellers AS (SELECT TOP 10 ITEMNMBR, TotalQty, Revenue, COGS FROM Sales ORDER BY TotalQty DESC) SELECT ts.ITEMNMBR, i.ITEMDESC, ts.TotalQty, ts.Revenue, ts.COGS, ts.Revenue - ts.COGS AS GrossProfit, CASE WHEN ts.Revenue > 0 THEN ((ts.Revenue - ts.COGS) / ts.Revenue) * 100 ELSE 0 END AS GrossMarginPct FROM TopSellers ts LEFT JOIN IV00101 i ON ts.ITEMNMBR = i.ITEMNMBR ORDER BY ts.TotalQty DESC",
  "params": ["2025-11-20", "2025-11-20"],
  "summary": "Calculates gross margin for top 10 best-selling items last month using revenue and COGS from shipped invoices.",
  "entities": {"intent": "profitability", "period": "last month"}
}
---
Question: "Find items where current inventory is less than 30 days of usage"
Context: "intent=low_stock; threshold=30 days"
Schema: "IV00102 (ITEMNMBR, QTYONHND, ATYALLOC), IV30300 (ITEMNMBR, TRXQTY), IV30200 (DOCDATE), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH RecentUsage AS (SELECT t.ITEMNMBR, ABS(SUM(t.TRXQTY)) / 30.0 AS AvgDailyUsage FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE h.DOCDATE >= DATEADD(day, -30, ?) AND t.TRXQTY < 0 GROUP BY t.ITEMNMBR HAVING ABS(SUM(t.TRXQTY)) > 0), CurrentStock AS (SELECT ITEMNMBR, SUM(QTYONHND - ATYALLOC) AS FreeStock FROM IV00102 GROUP BY ITEMNMBR), Coverage AS (SELECT cs.ITEMNMBR, cs.FreeStock, ru.AvgDailyUsage, cs.FreeStock / NULLIF(ru.AvgDailyUsage, 0) AS DaysOfCoverage FROM CurrentStock cs INNER JOIN RecentUsage ru ON cs.ITEMNMBR = ru.ITEMNMBR WHERE ru.AvgDailyUsage > 0) SELECT TOP 100 c.ITEMNMBR, i.ITEMDESC, c.FreeStock, c.AvgDailyUsage, c.DaysOfCoverage FROM Coverage c LEFT JOIN IV00101 i ON c.ITEMNMBR = i.ITEMNMBR WHERE c.DaysOfCoverage < ? ORDER BY c.DaysOfCoverage ASC",
  "params": ["2025-11-20", 30],
  "summary": "Identifies items with less than 30 days of inventory coverage based on last 30 days usage rate.",
  "entities": {"intent": "low_stock", "threshold_days": 30}
}
---
Question: "Calculate inventory turnover ratio for each item category"
Context: "intent=kpi; metric=turnover; group_by=category"
Schema: "IV00101 (ITEMNMBR, ITMCLSCD), IV00102 (ITEMNMBR, QTYONHND), IV30300 (ITEMNMBR, TRXQTY), IV30200 (DOCDATE)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH YearlyUsage AS (SELECT t.ITEMNMBR, ABS(SUM(t.TRXQTY)) AS AnnualUsage FROM IV30300 t JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP WHERE h.DOCDATE >= DATEADD(year, -1, ?) AND t.TRXQTY < 0 GROUP BY t.ITEMNMBR), AvgInventory AS (SELECT ITEMNMBR, AVG(QTYONHND) AS AvgQty FROM IV00102 GROUP BY ITEMNMBR), CategoryMetrics AS (SELECT i.ITMCLSCD AS Category, SUM(yu.AnnualUsage) AS TotalUsage, SUM(ai.AvgQty) AS TotalAvgInventory FROM IV00101 i LEFT JOIN YearlyUsage yu ON i.ITEMNMBR = yu.ITEMNMBR LEFT JOIN AvgInventory ai ON i.ITEMNMBR = ai.ITEMNMBR WHERE i.ITMCLSCD IS NOT NULL GROUP BY i.ITMCLSCD) SELECT Category, TotalUsage, TotalAvgInventory, CASE WHEN TotalAvgInventory > 0 THEN TotalUsage / TotalAvgInventory ELSE 0 END AS TurnoverRatio FROM CategoryMetrics WHERE TotalUsage > 0 ORDER BY TurnoverRatio DESC",
  "params": ["2025-11-20"],
  "summary": "Calculates inventory turnover ratio (annual usage / average inventory) for each item category.",
  "entities": {"intent": "kpi", "metric": "turnover"}
}
---
Question: "Show purchase price variance for items bought in the last 3 months"
Context: "intent=variance_analysis; metric=purchase_price; period=last 3 months"
Schema: "POP30110 (ITEMNMBR, UNITCOST, QTYINVCD), POP30100 (PONUMBER, RECEIPTDATE), IV00101 (ITEMNMBR, ITEMDESC, STNDCOST)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH RecentPurchases AS (SELECT l.ITEMNMBR, l.UNITCOST, l.QTYINVCD, h.RECEIPTDATE FROM POP30110 l JOIN POP30100 h ON l.PONUMBER = h.PONUMBER WHERE h.RECEIPTDATE >= DATEADD(month, -3, ?)), WeightedAvgCost AS (SELECT ITEMNMBR, SUM(UNITCOST * QTYINVCD) / NULLIF(SUM(QTYINVCD), 0) AS AvgUnitCost, COUNT(DISTINCT RECEIPTDATE) AS PurchaseCount FROM RecentPurchases GROUP BY ITEMNMBR), Variance AS (SELECT wa.ITEMNMBR, i.ITEMDESC, i.STNDCOST AS StandardCost, wa.AvgUnitCost AS ActualAvgCost, wa.AvgUnitCost - i.STNDCOST AS CostVariance, CASE WHEN i.STNDCOST > 0 THEN ((wa.AvgUnitCost - i.STNDCOST) / i.STNDCOST) * 100 ELSE 0 END AS VariancePct, wa.PurchaseCount FROM WeightedAvgCost wa LEFT JOIN IV00101 i ON wa.ITEMNMBR = i.ITEMNMBR WHERE i.STNDCOST IS NOT NULL) SELECT TOP 100 ITEMNMBR, ITEMDESC, StandardCost, ActualAvgCost, CostVariance, VariancePct, PurchaseCount FROM Variance WHERE ABS(VariancePct) > 5 ORDER BY ABS(VariancePct) DESC",
  "params": ["2025-11-20"],
  "summary": "Identifies items with >5% purchase price variance (actual vs standard) over last 3 months using weighted average.",
  "entities": {"intent": "variance_analysis", "metric": "purchase_price"}
}
---
Question: "Compare current month sales to same month last year for all items"
Context: "intent=yoy_comparison; metric=sales"
Schema: "SOP30300 (ITEMNMBR, SOPTYPE, QUANTITY), SOP30200 (SOPNUMBE, SOPTYPE, DOCDATE), IV00101 (ITEMNMBR, ITEMDESC)"
Today: 2025-11-20

JSON Response:
{
  "sql": "WITH CurrentMonth AS (SELECT l.ITEMNMBR, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS CurrentQty FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE YEAR(h.DOCDATE) = YEAR(?) AND MONTH(h.DOCDATE) = MONTH(?) AND l.SOPTYPE IN (3, 4) GROUP BY l.ITEMNMBR), PriorYear AS (SELECT l.ITEMNMBR, SUM(CASE WHEN l.SOPTYPE = 4 THEN -ABS(l.QUANTITY) ELSE ABS(l.QUANTITY) END) AS PriorQty FROM SOP30300 l JOIN SOP30200 h ON h.SOPTYPE = l.SOPTYPE AND h.SOPNUMBE = l.SOPNUMBE WHERE YEAR(h.DOCDATE) = YEAR(DATEADD(year, -1, ?)) AND MONTH(h.DOCDATE) = MONTH(?) AND l.SOPTYPE IN (3, 4) GROUP BY l.ITEMNMBR) SELECT TOP 100 COALESCE(cm.ITEMNMBR, py.ITEMNMBR) AS ITEMNMBR, i.ITEMDESC, ISNULL(cm.CurrentQty, 0) AS CurrentMonthQty, ISNULL(py.PriorQty, 0) AS PriorYearQty, ISNULL(cm.CurrentQty, 0) - ISNULL(py.PriorQty, 0) AS QtyChange, CASE WHEN ISNULL(py.PriorQty, 0) > 0 THEN ((ISNULL(cm.CurrentQty, 0) - ISNULL(py.PriorQty, 0)) / py.PriorQty) * 100 ELSE NULL END AS PercentChange FROM CurrentMonth cm FULL OUTER JOIN PriorYear py ON cm.ITEMNMBR = py.ITEMNMBR LEFT JOIN IV00101 i ON COALESCE(cm.ITEMNMBR, py.ITEMNMBR) = i.ITEMNMBR WHERE ISNULL(cm.CurrentQty, 0) > 0 OR ISNULL(py.PriorQty, 0) > 0 ORDER BY ABS(ISNULL(cm.CurrentQty, 0) - ISNULL(py.PriorQty, 0)) DESC",
  "params": ["2025-11-20", "2025-11-20", "2025-11-20", "2025-11-20"],
  "summary": "Year-over-year comparison of current month sales vs same month last year for all items with activity.",
  "entities": {"intent": "yoy_comparison"}
}
"""

CUSTOM_SQL_HINTS: tuple[str, ...] = (
    "Inventory adjustments: IV30300 transaction detail holds TRXQTY but not DOCDATE; join IV30200 header (DOCNUMBR + DOCTYPE) to filter by DOCDATE or other header fields.",
    "Sales shipments: SOP30300 line detail holds QUANTITY/EXTDCOST but not DOCDATE; join SOP30200 header (SOPNUMBE + SOPTYPE) whenever you need DOCDATE or posting metadata.",
    "Planning/forecasting: For questions like 'what to buy', analyze demand from open sales orders (SOP10200) and supply from open purchase orders (POP10110). Join with IV00101 for item details.",
    "BOM normalization: Many finished good BOMs are stored under 00 parents (e.g., SOARBLM02 → SOARBLM00). Normalize the finished good to its 00 parent when exploding BOM demand and prefer the manufacturing BOM (BM010115) with standard BOM (BM00111) as fallback.",
    "Payables/Vendors: PM00200 is the Vendor Master (VENDORID, VENDNAME). PM30200 is Paid Transaction History; join to PM00200 on VENDORID. PM30200.DOCAMNT is the document amount.",
)

ITEM_RESOURCE_PLANNING_UI_REFERENCE = """
Great Plains Item Resource Planning Maintenance screen reference:
- Header: Item Number lookup with Description field; Buyer ID and Planner ID lookups.
- Sites: Default Values vs Site ID/Description selector; checkbox 'Calculate MRP for this item/site'.
- MRP suggestions: Generate MRP Suggestions for this item/site with Move In, Move Out, Cancel toggles.
- Ordering policy: Order Policy dropdown plus Fixed Order Qty, Order Point Qty, Order-Up-To Level, Number of Days inputs.
- Order Quantity Modifiers group: Minimum, Maximum, Multiple.
- Replenishment and timing: Replenishment Method dropdown (Buy shown), Item Shrinkage Factor %, Purchasing Lead Time, Mfg Fixed Lead Time, Mfg Variable Lead Time, Planning Time Fence, Move-Out Fence (all in days).
- Reference-only box: Reorder Variance, Safety Stock, with note 'Safety Stock should be included in Order Point Qty'.
"""

BILL_OF_MATERIALS_UI_REFERENCE = """
Great Plains Bill of Materials inquiry reference:
- The Bill of Materials page (Inventory > Inquiry > Bill of Materials) shows the parent item with its component items, descriptions, units, and design quantities.
- Use this screen to verify component structure when a query returns no rows or to drill into related components.
- The same parent/component relationship is stored in tables like BM00111/BM010115; querying those mirrors what the BOM page shows.
"""

SQL_SCHEMA_CACHE_KEY = "default"
SQL_SCHEMA_CACHE: dict[str, dict[str, list[dict]]] = {}
SQL_TABLE_TOKEN_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z0-9_\.\[\]]+)", re.IGNORECASE)

PLATFORM_CAPABILITIES = """
You are powering the "Chemical Market Terminal", a comprehensive procurement intelligence platform.
Your user interface includes the following key modules:

1.  **Market Monitor**: The main dashboard. Displays live pricing trends, "Top Movers" (highest price changes), and market volatility indices.
2.  **Buy Calendar**: A purchasing schedule. Recommendations are based on "Inventory Runway" (days left) and "Optimal Price Windows". It strictly filters for items purchased in the last 2 years from valid vendors.
3.  **Procurement Cockpit (Command Center)**: High-level metrics for executives.
4.  **Product Insights**: A deep-dive page for a single item (price history, usage burn rate, seasonality).
5.  **Broker Portal**: A restricted area for Freight Brokers to view "Priority Opportunities" (high-demand routes) and submit freight bids.
6.  **Vendor Portal**: A restricted area for Vendors to view "Open Requests" and submit material quotes.

When answering "how to" questions, guide the user to these specific tabs/modules.
"""
