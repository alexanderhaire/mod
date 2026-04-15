from decimal import Decimal
import datetime
import pyodbc
from constants import LOGGER, PRIMARY_LOCATION
from parsing_utils import decimal_or_zero
from sql_utils import format_sql_preview

def fetch_completed_production(cursor: pyodbc.Cursor, target_date: datetime.date) -> dict:
    """
    Fetch completed production (Canning and Mixing sheets) for a specific date.
    
    Source: WO010032 (Quick MO Entry) JOIN IV30300 (Inventory Transaction History)
    
    Logic: 
      - MO must be in WO010032 (Quick MO entry table)
      - MO must have generated a Posted Inventory Transaction (IV30300) on the target date.
      - We link WO -> MOP10213 (Receipts) -> IV30300 (History) via IVDOCNBR.
      - We sum the TRXQTY from IV30300 (Posted Quantity) for the Parent Item.
      - This filters out "Unposted" or "Mistake" MOs that exist in WO010032 but not in Inventory History.
      
    Args:
      target_date: The date the inventory transaction occurred (DOCDATE).
    """
    # Convert date to datetime for robust SQL comparison
    start_dt = datetime.datetime.combine(target_date, datetime.time.min)
    end_dt = start_dt + datetime.timedelta(days=1)
    
    # Query updates:
    # 1. Join MOP10213 to get IVDOCNBR
    # 2. Join IV30300 to verify it is posted and get true date/qty
    # 3. Filter by IV30300.DOCDATE (Inventory Date)
    # 4. Filter for Positive Sum(TRXQTY) to ensure we get the output, not consumption.
    
    query = """
        SELECT 
            w.MANUFACTUREORDER_I,
            w.ITEMNMBR,
            w.DSCRIPTN,
            SUM(iv.TRXQTY) as EndQty, -- Sum posted quantity (handle potential splits)
            MAX(i.SELNGUOM) as UOFM
        FROM WO010032 w
        JOIN MOP10213 m ON w.MANUFACTUREORDER_I = m.MANUFACTUREORDER_I
        JOIN IV30300 iv ON m.IVDOCNBR = iv.DOCNUMBR
        LEFT JOIN IV00101 i ON w.ITEMNMBR = i.ITEMNMBR
        WHERE iv.DOCDATE >= ? AND iv.DOCDATE < ?
          AND iv.ITEMNMBR = w.ITEMNMBR -- Match parent item only (ignore component lines if they share doc)
          AND (w.MANUFACTUREORDER_I LIKE 'C%' OR w.MANUFACTUREORDER_I LIKE 'X%')
        GROUP BY w.MANUFACTUREORDER_I, w.ITEMNMBR, w.DSCRIPTN
        HAVING SUM(iv.TRXQTY) > 0
        ORDER BY w.MANUFACTUREORDER_I
    """
    params = [start_dt, end_dt]
    sql_preview = format_sql_preview(query, [str(p) for p in params])
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        canning = []
        mixing = []
        
        for r in rows:
            entry = {
                "mo_number": r.MANUFACTUREORDER_I.strip(),
                "item_number": r.ITEMNMBR.strip(),
                "description": r.DSCRIPTN.strip(),
                "quantity": decimal_or_zero(r.EndQty),
                "uofm": r.UOFM.strip() if r.UOFM else ""
            }
            if entry["mo_number"].upper().startswith('C'):
                canning.append(entry)
            else:
                mixing.append(entry)
                
        return {
            "canning": canning,
            "mixing": mixing,
            "sql_preview": sql_preview
        }
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch completed production: %s", err)
        return {"canning": [], "mixing": [], "sql_preview": sql_preview, "error": str(err)}

def fetch_open_orders_buckets(cursor: pyodbc.Cursor, today: datetime.date) -> dict:
    # ... (Same as before) ...
    """
    Fetch all open orders and organize them into 4 priority buckets:
    1. Past Due (REQSHIPDATE < Today)
    2. Due Today (REQSHIPDATE == Today)
    3. Due Tomorrow (REQSHIPDATE == Today + 1)
    4. Future (REQSHIPDATE > Today + 1)
    
    Source: SOP10100 (Header) join SOP10200 (Line)
    Filter: SOPTYPE = 2 (Order) and QTYREMAI > 0
    """
    
    tomorrow = today + datetime.timedelta(days=1)
    
    query = """
        SELECT 
            h.SOPNUMBE,
            h.CUSTNAME,
            h.REQSHIPDATE,
            d.ITEMNMBR,
            d.ITEMDESC,
            d.QTYREMAI, -- Remaining Quantity to Ship
            d.UOFM,
            h.DOCID 
        FROM SOP10200 d
        JOIN SOP10100 h ON d.SOPNUMBE = h.SOPNUMBE AND d.SOPTYPE = h.SOPTYPE
        WHERE h.SOPTYPE = 2 -- Order
          AND d.QTYREMAI > 0
          AND d.ITEMNMBR NOT LIKE 'FREIGHT%'
        ORDER BY h.REQSHIPDATE ASC, h.SOPNUMBE
    """
    
    sql_preview = format_sql_preview(query, [])
    
    buckets = {
        "past_due": [],
        "due_today": [],
        "due_tomorrow": [],
        "future": []
    }
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for r in rows:
            req_date = r.REQSHIPDATE.date() if r.REQSHIPDATE else datetime.date(2100, 1, 1) # Null date -> Future
            
            entry = {
                "order_number": r.SOPNUMBE.strip(),
                "customer": r.CUSTNAME.strip(),
                "req_date": req_date.strftime('%Y-%m-%d'),
                "item_number": r.ITEMNMBR.strip(),
                "item_desc": r.ITEMDESC.strip(),
                "quantity": decimal_or_zero(r.QTYREMAI),
                "uofm": r.UOFM.strip(),
                "doc_id": r.DOCID.strip()
            }
            
            if req_date < today:
                buckets["past_due"].append(entry)
            elif req_date == today:
                buckets["due_today"].append(entry)
            elif req_date == tomorrow:
                buckets["due_tomorrow"].append(entry)
            else:
                buckets["future"].append(entry)
                
        return {"buckets": buckets, "sql_preview": sql_preview}

    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch open orders: %s", err)
        return {"buckets": buckets, "sql_preview": sql_preview, "error": str(err)}
