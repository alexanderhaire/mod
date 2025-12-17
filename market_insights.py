"""Market insights data retrieval and analysis for the Chemical Market Terminal."""
import bisect
import calendar
import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyodbc

from constants import (
    LOGGER,
    PRIMARY_LOCATION,
    RAW_MATERIAL_CLASS_CODES,
)

RAW_MATERIAL_CLASS_LIST = "', '".join(RAW_MATERIAL_CLASS_CODES)
RAW_MATERIAL_CLASS_FILTER_SQL = (
    f"UPPER(LTRIM(RTRIM(i.ITMCLSCD))) IN ('{RAW_MATERIAL_CLASS_LIST}')"
)


def classify_item_segment(itm_class: str | None) -> str:
    """
    Classify an item into Raw Material vs Finished Good using the same logic as the market monitor.
    """
    code = (itm_class or "").strip().upper()
    return "Raw Material" if code in RAW_MATERIAL_CLASS_CODES else "Finished Good"


def fetch_product_price_history(cursor: pyodbc.Cursor, item_number: str, days: int = 3650) -> list[dict[str, Any]]:
    """
    Fetch historical price data for a product.
    For raw materials, use purchase receipt data from POP30300.
    For finished goods, use inventory transaction data from IV30300.
    Default: 10 years of history to capture older purchase receipts.
    """
    try:
        # Get current item details
        query = """
        SELECT 
            ITEMNMBR,
            ITEMDESC,
            STNDCOST,
            CURRCOST,
            USCATVLS_1 as Category,
            ITMCLSCD
        FROM IV00101
        WHERE ITEMNMBR = ?
        """
        cursor.execute(query, item_number)
        item_row = cursor.fetchone()
        
        if not item_row:
            return []
        
        columns = [c[0] for c in cursor.description]
        item_data = dict(zip(columns, item_row))
        
        # Determine if this is a raw material (based on GP Item Class code)
        itm_class = str(item_data.get('ITMCLSCD', '')).strip().upper()
        
        is_raw_material = itm_class in RAW_MATERIAL_CLASS_CODES
        
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)
        
        history = []
        
        # 1. Try fetching Purchase Receipt History (for Raw Materials)
        if is_raw_material:
            # For raw materials, get purchase receipt cost history with Vendor and UofM details
            # Fetch raw transactions to allow vendor filtering and unit normalization
            receipt_query = """
            SELECT 
                CAST(h.RECEIPTDATE AS DATE) as TransactionDate,
                h.VENDORID,
                h.VENDNAME,
                l.UOFM,
                l.UMQTYINB,
                l.UNITCOST,
                l.EXTDCOST
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE l.ITEMNMBR = ?
                AND h.RECEIPTDATE >= ?
                AND h.RECEIPTDATE <= ?
                AND l.UNITCOST > 0
                AND h.POPTYPE <> 2
                AND h.VOIDSTTS = 0
                AND l.NONINVEN = 0
            ORDER BY h.RECEIPTDATE
            """
            cursor.execute(receipt_query, item_number, start_date, end_date)
            history_rows = cursor.fetchall()
            
            if cursor.description and history_rows:
                hist_columns = [c[0] for c in cursor.description]
                
                for row in history_rows:
                    r = dict(zip(hist_columns, row))
                    qty_in_base = float(r.get('UMQTYINB') or 1)
                    unit_cost = float(r.get('UNITCOST') or 0)
                    ext_cost = float(r.get('EXTDCOST') or 0)
                    
                    # Normalize cost to Base Unit (e.g., Cost Per Pound)
                    if qty_in_base > 0:
                        norm_cost = unit_cost / qty_in_base
                    else:
                        norm_cost = unit_cost
                        
                    # Calculate Quantity based on Cost
                    # Quantity (Purchase UofM) = EXTDCOST / UNITCOST
                    # Quantity (Base UofM) = Quantity (Purchase) * UMQTYINB
                    qty_purchased = 0
                    if unit_cost > 0:
                         qty_purchased = (ext_cost / unit_cost) * qty_in_base
                    
                    history.append({
                        'TransactionDate': r['TransactionDate'],
                        'AvgCost': norm_cost, # Normalized cost
                        'VendorID': str(r.get('VENDORID', '')).strip(),
                        'VendorName': str(r.get('VENDNAME', '')).strip(),
                        'UofM': str(r.get('UOFM', '')).strip(),
                        'OriginalCost': unit_cost,
                        'ExtendedCost': ext_cost,
                        'TransactionCount': 1,
                        'Quantity': qty_purchased
                    })
            else:
                # Fallback: If no purchase history found (e.g. manufactured RM, or old stock), 
                # treat as standard inventory item
                is_raw_material = False # Flip flag to indicate we are showing internal history

        # 2. Fetch Inventory Transaction History (For FGs OR Fallback)
        if not is_raw_material:
            # For finished goods, use inventory transaction data (IV30300) to track cost
            # This captures manufacturing costs/adjustments which is more relevant for "Cost" than selling price
            inv_query = """
            SELECT 
                CAST(h.DOCDATE AS DATE) as TransactionDate,
                AVG(t.UNITCOST) as AvgCost,
                COUNT(*) as TransactionCount,
                SUM(t.TRXQTY) as Quantity
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE t.ITEMNMBR = ?
                AND h.DOCDATE >= ?
                AND h.DOCDATE <= ?
                AND t.UNITCOST > 0
            GROUP BY CAST(h.DOCDATE AS DATE)
            ORDER BY TransactionDate
            """
            cursor.execute(inv_query, item_number, start_date, end_date)
            history_rows = cursor.fetchall()
            
            if cursor.description:
                hist_columns = [c[0] for c in cursor.description]
                history = []
                for row in history_rows:
                    r = dict(zip(hist_columns, row))
                    r['AvgCost'] = float(r.get('AvgCost') or 0)
                    r['Quantity'] = float(r.get('Quantity') or 0)
                    history.append(r)
        
        # Return real data only - no synthetic fallback
        # If sparse, the UI should show "insufficient data" rather than fake smooth lines
        
        # Add item context
        for record in history:
            record['ITEMNMBR'] = item_number
            record['ITEMDESC'] = item_data.get('ITEMDESC', '')
            record['IsRawMaterial'] = is_raw_material
        
        return history
        
    except pyodbc.Error as err:
        LOGGER.warning(f"Price history fetch failed for {item_number}: {err}")
        return []


def get_inventory_distribution(
    cursor: pyodbc.Cursor,
    limit: int = 20,
    segment: str = "Raw Material",
) -> list[dict[str, Any]]:
    """
    Get inventory distribution by value and category for a segment.
    Useful for visualizing portfolio composition (Raw Materials vs Finished Goods).
    Returns: [{'category': 'Chemicals', 'value': 150000, 'items': 12}, ...]
    """
    try:
        normalized_segment = (segment or "Raw Material").strip().lower()
        segment_filter = ""

        if normalized_segment.startswith("raw"):
            segment_filter = f"AND {RAW_MATERIAL_CLASS_FILTER_SQL}"
        elif normalized_segment.startswith("finished"):
            segment_filter = (
                f"AND (NOT ({RAW_MATERIAL_CLASS_FILTER_SQL}) OR i.ITMCLSCD IS NULL)"
            )

        query = f"""
        SELECT 
            i.ITEMNMBR,
            i.USCATVLS_1 as Category,
            SUM(ISNULL(q.QTYONHND, 0)) as OnHand,
            AVG(i.CURRCOST) as UnitCost
        FROM IV00101 i
        LEFT JOIN IV00102 q ON i.ITEMNMBR = q.ITEMNMBR AND q.LOCNCODE = ?
        WHERE i.ITEMTYPE IN (0, 1, 2)
            {segment_filter}
        GROUP BY i.ITEMNMBR, i.USCATVLS_1
        HAVING SUM(ISNULL(q.QTYONHND, 0)) > 0
        """
        cursor.execute(query, PRIMARY_LOCATION)
        rows = cursor.fetchall()
        
        if not rows:
            return []
            
        # Aggregate in Python to ensure clean categories
        stats = {}
        
        for row in rows:
            cat = str(row[1] or 'Uncategorized').strip()
            qty = float(row[2])
            cost = float(row[3])
            val = qty * cost
            
            if cat not in stats:
                stats[cat] = {'value': 0.0, 'items': 0}
            
            stats[cat]['value'] += val
            stats[cat]['items'] += 1
            
        # Convert to list
        results = [
            {'Category': k, 'Value': v['value'], 'ItemCount': v['items']} 
            for k, v in stats.items()
        ]
        
        # Sort by value
        results.sort(key=lambda x: x['Value'], reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        LOGGER.warning(f"Inventory distribution fetch failed: {e}")
        return []



def fetch_product_usage_history(
    cursor: pyodbc.Cursor,
    item_number: str,
    days: int = 180,
    location: str | None = PRIMARY_LOCATION,
    fallback_all_locations: bool = True,
    group_by: str = "month",
) -> list[dict[str, Any]]:
    """
    Fetch usage/consumption history for a product from inventory transactions.
    
    Args:
        cursor: Database cursor.
        item_number: Item to fetch history for.
        days: Lookback window.
        location: Optional location filter.
        fallback_all_locations: Whether to retry without location on empty result.
        group_by: "month" (default) for monthly aggregates or "day" for daily detail.
    """
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=days)

        group_by_mode = (group_by or "month").lower()
        group_by_month = group_by_mode != "day"
        
        if group_by_month:
            select_clause = """
            YEAR(h.DOCDATE) as Year,
            MONTH(h.DOCDATE) as Month,
            SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) as UsageQty,
            SUM(CASE WHEN t.TRXQTY > 0 THEN t.TRXQTY ELSE 0 END) as ReceiptQty,
            COUNT(*) as TransactionCount
            """
            group_clause = "GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE)"
            order_clause = "ORDER BY Year, Month"
        else:
            select_clause = """
            YEAR(h.DOCDATE) as Year,
            MONTH(h.DOCDATE) as Month,
            DAY(h.DOCDATE) as Day,
            h.DOCDATE as UsageDate,
            SUM(CASE WHEN t.TRXQTY < 0 THEN -t.TRXQTY ELSE 0 END) as UsageQty,
            SUM(CASE WHEN t.TRXQTY > 0 THEN t.TRXQTY ELSE 0 END) as ReceiptQty,
            COUNT(*) as TransactionCount
            """
            group_clause = "GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE), DAY(h.DOCDATE), h.DOCDATE"
            order_clause = "ORDER BY UsageDate"
        
        location_clause = "AND t.LOCNCODE = ?" if location else ""
        query = f"""
        SELECT 
            {select_clause}
        FROM IV30300 t
        JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
        WHERE t.ITEMNMBR = ?
            {location_clause}
            AND h.DOCDATE >= ?
            AND h.DOCDATE <= ?
        {group_clause}
        {order_clause}
        """
        
        params = [item_number]
        if location:
            params.append(location)
        params.extend([start_date, end_date])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not cursor.description or not rows:
            if fallback_all_locations and location:
                return fetch_product_usage_history(cursor, item_number, days=days, location=None, fallback_all_locations=False)
            return []
        
        columns = [c[0] for c in cursor.description]
        usage_data = [dict(zip(columns, row)) for row in rows]
        
        # Add month names for better display
        for record in usage_data:
            month_num = record.get('Month', 1)
            record['MonthName'] = calendar.month_name[month_num] if 1 <= month_num <= 12 else 'Unknown'
            record['ITEMNMBR'] = item_number
            if not group_by_month and 'UsageDate' in record and record['UsageDate']:
                # Preserve date for charting; pyodbc can return date or datetime
                record['UsageDate'] = record['UsageDate']
        
        return usage_data
        
    except pyodbc.Error:
        # Silently return empty list, but try fallback when allowed
        if fallback_all_locations and location:
            return fetch_product_usage_history(cursor, item_number, days=days, location=None, fallback_all_locations=False)
        return []


def fetch_product_inventory_trends(cursor: pyodbc.Cursor, item_number: str) -> dict[str, Any]:
    """
    Fetch current inventory status and trends for a product.
    """
    try:
        # Fetch Inventory from IV00102 for MAIN location to avoid double counting
        query = """
        SELECT 
            i.ITEMNMBR,
            i.ITEMDESC,
            SUM(q.QTYONHND) as TotalOnHand,
            SUM(q.ATYALLOC) as TotalAllocated,
            SUM(q.QTYONHND - q.ATYALLOC) as Available,
            SUM(q.QTYONORD) as OnOrder,
            i.STNDCOST,
            i.CURRCOST,
            i.USCATVLS_1 as Category
        FROM IV00101 i
        LEFT JOIN IV00102 q ON i.ITEMNMBR = q.ITEMNMBR
        WHERE i.ITEMNMBR = ? AND q.LOCNCODE = ?
        GROUP BY i.ITEMNMBR, i.ITEMDESC, i.STNDCOST, i.CURRCOST, i.USCATVLS_1
        """
        
        cursor.execute(query, item_number, PRIMARY_LOCATION)
        row = cursor.fetchone()
        
        if not row or not cursor.description:
            return {}
        
        columns = [c[0] for c in cursor.description]
        inventory = dict(zip(columns, row))
        
        # Calculate On Order from Open Purchase Orders (POP10110)
        # Status 1=New, 2=Released, 3=Change Order. 
        # Exclude 4=Received, 5=Closed, 6=Canceled.
        try:
            po_query = """
            SELECT COALESCE(SUM(l.QTYORDER - l.QTYCANCE), 0) as POOnOrder
            FROM POP10110 l
            JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
            WHERE l.ITEMNMBR = ?
              AND h.POSTATUS IN (1, 2, 3)
              AND l.POLNESTA IN (1, 2, 3)
              AND l.LOCNCODE = ?
            """
            cursor.execute(po_query, item_number, PRIMARY_LOCATION)
            po_row = cursor.fetchone()
            po_on_order = float(po_row[0]) if po_row and po_row[0] else 0
            
            # Use the calculated PO On Order
            inventory['OnOrder'] = po_on_order
            
        except Exception as e:
            # Fallback to IV00102 OnOrder if PO query fails, but likely 0 if MAIN filtered
            pass
        
        # Calculate inventory metrics
        total_on_hand = float(inventory.get('TotalOnHand', 0) or 0)
        total_allocated = float(inventory.get('TotalAllocated', 0) or 0)
        on_order_val = float(inventory.get('OnOrder', 0) or 0)

        # Normalize numeric fields back onto the inventory dict to avoid Decimal/float collisions downstream
        inventory['TotalOnHand'] = total_on_hand
        inventory['TotalAllocated'] = total_allocated
        inventory['OnOrder'] = on_order_val

        # Recalculate Available based on OnHand - Allocated
        available = total_on_hand - total_allocated
        inventory['Available'] = available
        
        inventory['InventoryValue'] = total_on_hand * float(inventory.get('CURRCOST', 0) or 0)
        inventory['StockStatus'] = 'Low' if available < 100 else 'Normal' if available < 500 else 'High'
        
        return inventory
        
    except pyodbc.Error as err:
        return {}


def get_product_details(cursor: pyodbc.Cursor, item_number: str) -> dict[str, Any]:
    """
    Get comprehensive product details combining price, usage, and inventory data.
    """
    # Fetch a longer window so price charts and summaries cover multi-year trends
    price_history = fetch_product_price_history(cursor, item_number, days=3650)  # 10 years
    # Increase usage history to 2 years (730 days) to allow Year-Over-Year seasonality analysis
    usage_history = fetch_product_usage_history(cursor, item_number, days=730)
    inventory_status = fetch_product_inventory_trends(cursor, item_number)
    
    return {
        'item_number': item_number,
        'item_desc': inventory_status.get('ITEMDESC', item_number),
        'price_history': price_history,
        'usage_history': usage_history,
        'inventory_status': inventory_status,
        'category': inventory_status.get('Category', 'Unknown'),
        'buying_signal': calculate_buying_signals(cursor, item_number),
        'demand_forecast': forecast_demand(cursor, item_number)
    }


def calculate_buying_signals(cursor: pyodbc.Cursor, item_number: str, runway_days: float | None = None) -> dict[str, Any]:
    """
    Analyze price history AND inventory status to determine if now is a good time to buy.
    Returns a score (0-100) and reasoning.
    
    Weighting Strategy:
    - 80% Inventory Score (Lower Runway = Higher Urgency)
    - 20% Price Score (Lower Price percentile = Higher Opportunity)
    """
    try:
        # 1. Price Score Calculation (0-100)
        # Get price history (last 2 years)
        history = fetch_product_price_history(cursor, item_number, days=730)
        
        price_score = 50
        price_reason = "Price is stable."
        current_cost = 0
        avg_6mo = 0
        avg_12mo = 0
        percentile = 50
        
        if history and len(history) >= 1:
            costs = [float(r['AvgCost']) for r in history if r.get('AvgCost')]
            if costs:
                current_cost = costs[-1]
                avg_6mo = sum(costs[-6:]) / len(costs[-6:]) if len(costs) >= 6 else sum(costs) / len(costs)
                avg_12mo = sum(costs[-12:]) / len(costs[-12:]) if len(costs) >= 12 else sum(costs) / len(costs)
                
                sorted_costs = sorted(costs)
                rank = bisect.bisect_left(sorted_costs, current_cost)
                percentile = (rank / len(costs)) * 100
                
                # Percentile Logic
                reasons = []
                if percentile < 10:
                    price_score += 40
                    reasons.append("Price is in the bottom 10%.")
                elif percentile < 25:
                    price_score += 20
                    reasons.append("Price is in the bottom 25%.")
                elif percentile > 90:
                    price_score -= 30
                    reasons.append("Price is high (>90th pct).")
                    
                # Moving Average Logic
                if current_cost < avg_6mo:
                    price_score += 10
                else:
                    price_score -= 10
                    
                price_score = max(0, min(100, price_score))
                if reasons:
                    price_reason = " ".join(reasons)
        
        # 2. Inventory Score Calculation (0-100)
        # If runway_days is not provided, fetch it
        if runway_days is None:
            # We don't want to infinite loop if this function is called inside runway calc, 
            # but usually runway calc calls simple logic. 
            # Safest is to calculate if missing.
            from market_insights import calculate_inventory_runway
            runway_data = calculate_inventory_runway(cursor, item_number)
            runway_days = runway_data.get('runway_days', 999)
            
        inv_score = 0
        inv_reason = ""
        
        if runway_days < 30:
            inv_score = 100
            inv_reason = f"Critical low stock ({runway_days:.0f}d)."
        elif runway_days < 60:
            inv_score = 75
            inv_reason = f"Low stock ({runway_days:.0f}d)."
        elif runway_days < 90:
            inv_score = 50
            inv_reason = f"Adequate stock ({runway_days:.0f}d)."
        else:
            inv_score = 25 # Low urgency
            inv_reason = "High stock levels."
            
        # 3. Final Weighted Score
        # User requested 80% weight on Inventory, 20% on Price
        weighted_score = (inv_score * 0.8) + (price_score * 0.2)
        
        signal = "Strong Buy" if weighted_score >= 80 else "Buy" if weighted_score >= 60 else "Hold" if weighted_score >= 40 else "Wait"
        
        final_reasons = []
        if inv_score > 50:
             final_reasons.append(inv_reason)
        if price_score > 50:
             final_reasons.append(price_reason)
        if not final_reasons:
             final_reasons.append("Conditions normal.")
             
        return {
            'score': weighted_score,
            'signal': signal,
            'current_cost': current_cost,
            'avg_6mo': avg_6mo,
            'avg_12mo': avg_12mo,
            'percentile': percentile,
            'reason': " ".join(final_reasons),
            'subscores': {'inventory': inv_score, 'price': price_score}
        }
        
    except Exception as e:
        LOGGER.error(f"Error calculating buying signal for {item_number}: {e}")
        return {'score': 0, 'signal': 'Error', 'reason': str(e)}


def forecast_demand(cursor: pyodbc.Cursor, item_number: str) -> dict[str, Any]:
    """
    Simple demand forecasting using 3-month moving average.
    """
    try:
        # Get usage history
        usage = fetch_product_usage_history(cursor, item_number, days=365)
        if not usage:
            return {'forecast_next_3mo': 0, 'trend': 'Unknown'}
            
        # Extract usage quantities (ensure positive)
        usage_qtys = [float(r['UsageQty']) for r in usage]
        
        if not usage_qtys:
             return {'forecast_next_3mo': 0, 'trend': 'No Usage'}

        # Calculate 3-month moving average
        last_3_months = usage_qtys[-3:]
        avg_usage = sum(last_3_months) / len(last_3_months) if last_3_months else 0
        
        # Simple trend
        trend = "Stable"
        if len(usage_qtys) >= 6:
            prev_3_months = usage_qtys[-6:-3]
            prev_avg = sum(prev_3_months) / len(prev_3_months) if prev_3_months else 0
            if avg_usage > prev_avg * 1.1:
                trend = "Increasing"
            elif avg_usage < prev_avg * 0.9:
                trend = "Decreasing"
                
        return {
            'forecast_monthly_avg': avg_usage,
            'forecast_next_3mo': avg_usage * 3,
            'trend': trend,
            'basis': f"Based on {len(last_3_months)} months of data"
        }

    except Exception as e:
        LOGGER.error(f"Error forecasting demand for {item_number}: {e}")
        return {'forecast_next_3mo': 0, 'trend': 'Error'}


def fetch_monthly_price_trends(cursor: pyodbc.Cursor, item_number: str, months: int = 12) -> pd.DataFrame:
    """
    Fetch monthly aggregated price trends for charting.
    Returns a DataFrame with Month, AvgCost, MinCost, MaxCost columns.
    """
    try:
        # Determine if this is a raw material based on strict GP Class
        check_query = "SELECT ITMCLSCD FROM IV00101 WHERE ITEMNMBR = ?"
        cursor.execute(check_query, item_number)
        row = cursor.fetchone()
        
        is_raw_material = False
        if row:
            itm_class = str(row[0]).strip().upper()
            is_raw_material = itm_class in RAW_MATERIAL_CLASS_CODES

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=months * 30)
        
        if is_raw_material:
            query = """
            SELECT 
                YEAR(h.RECEIPTDATE) as Year,
                MONTH(h.RECEIPTDATE) as Month,
                AVG(l.UNITCOST) as AvgCost,
                MIN(l.UNITCOST) as MinCost,
                MAX(l.UNITCOST) as MaxCost,
                COUNT(*) as Receipts
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE l.ITEMNMBR = ?
                AND h.RECEIPTDATE >= ?
                AND h.RECEIPTDATE <= ?
                AND l.UNITCOST > 0
                AND h.POPTYPE <> 2
                AND h.VOIDSTTS = 0
                AND l.NONINVEN = 0
            GROUP BY YEAR(h.RECEIPTDATE), MONTH(h.RECEIPTDATE)
            ORDER BY Year, Month
            """
        else:
            # Finished Goods - use Inventory History (IV30300) for Cost Trends
            query = """
            SELECT 
                YEAR(h.DOCDATE) as Year,
                MONTH(h.DOCDATE) as Month,
                AVG(t.UNITCOST) as AvgCost,
                MIN(t.UNITCOST) as MinCost,
                MAX(t.UNITCOST) as MaxCost,
                COUNT(*) as Receipts
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE t.ITEMNMBR = ?
                AND h.DOCDATE >= ?
                AND h.DOCDATE <= ?
                AND t.UNITCOST > 0
            GROUP BY YEAR(h.DOCDATE), MONTH(h.DOCDATE)
            ORDER BY Year, Month
            """
        
        cursor.execute(query, item_number, start_date, end_date)
        rows = cursor.fetchall()
        
        if not cursor.description or not rows:
            return pd.DataFrame()
        
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Cast numeric columns to float to avoid Altair Decimal warnings
        for col in ['AvgCost', 'MinCost', 'MaxCost']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Create proper date column for charting
        df['Date'] = pd.to_datetime(df.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}-01", axis=1))
        df['MonthLabel'] = df['Date'].dt.strftime('%b %Y')
        
        return df
        
    except Exception as e:
        LOGGER.warning(f"Monthly price trends fetch failed for {item_number}: {e}")
        return pd.DataFrame()


def calculate_inventory_runway(cursor: pyodbc.Cursor, item_number: str, fallback_all_locations: bool = True) -> dict[str, Any]:
    """
    Calculate inventory "runway" - days of supply remaining.
    Combines current stock with usage trends and open POs.
    """
    try:
        # Get current inventory at the primary location
        inv_query = """
        SELECT 
            SUM(QTYONHND) as OnHand,
            SUM(QTYONORD) as OnOrder,
            SUM(QTYBKORD) as BackOrdered
        FROM IV00102
        WHERE ITEMNMBR = ?
          AND LOCNCODE = ?
        """
        cursor.execute(inv_query, item_number, PRIMARY_LOCATION)
        inv_row = cursor.fetchone()
        
        on_hand = float(inv_row[0] or 0) if inv_row else 0
        on_order = float(inv_row[1] or 0) if inv_row else 0

        # Add open POs at the primary location (status not closed/received/canceled)
        try:
            po_query = """
            SELECT COALESCE(SUM(l.QTYORDER - l.QTYCANCE), 0) as POOnOrder
            FROM POP10110 l
            JOIN POP10100 h ON l.PONUMBER = h.PONUMBER
            WHERE l.ITEMNMBR = ?
              AND h.POSTATUS IN (1, 2, 3)
              AND l.POLNESTA IN (1, 2, 3)
              AND l.LOCNCODE = ?
            """
            cursor.execute(po_query, item_number, PRIMARY_LOCATION)
            po_row = cursor.fetchone()
            if po_row and po_row[0]:
                on_order += float(po_row[0] or 0)
        except Exception as e:
            LOGGER.warning(f"PO on-order fetch failed for {item_number}: {e}")
        
        # Get average daily usage from last 90 days
        usage = fetch_product_usage_history(cursor, item_number, days=90, location=PRIMARY_LOCATION)
        if fallback_all_locations and not usage:
            usage = fetch_product_usage_history(cursor, item_number, days=90, location=None)
        if usage:
            total_usage = sum(float(r.get('UsageQty', 0)) for r in usage)
            days_covered = 90
            daily_usage = total_usage / days_covered if days_covered > 0 else 0
        else:
            daily_usage = 0
        
        # Calculate runway
        # STRICT RULE: Only count On-Hand for runway to avoid false security
        available_stock = on_hand 
        # previously: available_stock = on_hand + on_order
        if daily_usage > 0:
            # SAFETY BUFFER: Subtract 7 days to ensure emergency stock
            raw_runway = available_stock / daily_usage
            runway_days = max(0, raw_runway - 7)
        else:
            runway_days = 999  # Essentially infinite if no usage
        
        # Classify urgency
        if runway_days < 30:
            urgency = "CRITICAL"
            color = "#dc322f"  # Red
        elif runway_days < 60:
            urgency = "WARNING"
            color = "#b58900"  # Amber
        else:
            urgency = "OK"
            color = "#859900"  # Green
        
        return {
            'on_hand': on_hand,
            'on_order': on_order,
            'available': available_stock,
            'daily_usage': daily_usage,
            'runway_days': min(runway_days, 365),  # Cap at 1 year
            'urgency': urgency,
            'color': color
        }
        
    except Exception as e:
        LOGGER.warning(f"Inventory runway calculation failed for {item_number}: {e}")
        return {'runway_days': 0, 'urgency': 'UNKNOWN', 'color': '#839496'}


def calculate_seasonal_burn_metrics(
    usage_history: list[dict[str, Any]],
    on_hand: float = 0,
    on_order: float = 0,
    today: datetime.date | None = None,
) -> dict[str, float | None]:
    """
    Estimate burn rate with seasonality and recency decay.

    - Uses exponential decay (15% drop per month) to weight recent usage higher.
    - Applies a seasonal factor based on the current month's historical usage.
    - Days of coverage uses ONLY On Hand stock (strict View).
    """
    today = today or datetime.date.today()
    # STRICT RULE: Only count On-Hand for runway
    available_stock = float(on_hand or 0)

    if not usage_history:
        return {
            'avg_daily_usage': 0.0,
            'decayed_daily_usage': 0.0,
            'seasonal_burn_rate': 0.0,
            'seasonal_factor': 1.0,
            'days_of_coverage': None,
            'available_stock': available_stock,
        }

    total_qty = 0.0
    total_days = 0.0
    weighted_usage = 0.0
    weight_sum = 0.0
    month_totals: dict[int, float] = {}

    for record in usage_history:
        year = record.get('Year')
        month = record.get('Month')
        qty = float(record.get('UsageQty', 0) or 0)

        if not year or not month:
            continue

        try:
            month_year = int(year)
            month_num = int(month)
            month_date = datetime.date(month_year, month_num, 1)
        except Exception:
            continue

        days_in_month = calendar.monthrange(month_date.year, month_date.month)[1]
        daily_usage = qty / days_in_month if days_in_month else 0.0

        months_ago = max((today.year - month_date.year) * 12 + (today.month - month_date.month), 0)
        decay = 0.85 ** months_ago  # Dynamic materials-per-day decay

        weighted_usage += daily_usage * decay
        weight_sum += decay

        total_qty += qty
        total_days += days_in_month

        month_totals[month_num] = month_totals.get(month_num, 0.0) + qty

    avg_daily_usage = total_qty / total_days if total_days > 0 else 0.0
    decayed_daily_usage = weighted_usage / weight_sum if weight_sum > 0 else avg_daily_usage

    if month_totals:
        avg_monthly_usage = sum(month_totals.values()) / len(month_totals)
        seasonal_month_usage = month_totals.get(today.month, avg_monthly_usage)
        seasonal_factor = (seasonal_month_usage / avg_monthly_usage) if avg_monthly_usage > 0 else 1.0
    else:
        seasonal_factor = 1.0

    seasonal_factor = max(0.25, min(seasonal_factor, 3.0))
    seasonal_burn_rate = decayed_daily_usage * seasonal_factor
    
    # SAFETY BUFFER: Subtract 7 days
    if seasonal_burn_rate > 0:
        raw_days = available_stock / seasonal_burn_rate
        days_of_coverage = max(0, raw_days - 7)
    else:
        days_of_coverage = None

    return {
        'avg_daily_usage': avg_daily_usage,
        'decayed_daily_usage': decayed_daily_usage,
        'seasonal_burn_rate': seasonal_burn_rate,
        'seasonal_factor': seasonal_factor,
        'days_of_coverage': days_of_coverage,
        'available_stock': available_stock,
    }


def _prepare_daily_usage_series(usage_history: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize usage history into a daily time series for modeling.

    Returns a DataFrame with Date and daily_usage columns. Falls back to an empty
    DataFrame when data is missing so callers can handle insufficiency gracefully.
    """
    if not usage_history:
        return pd.DataFrame()

    usage_df = pd.DataFrame(usage_history)
    if usage_df.empty or 'UsageQty' not in usage_df.columns:
        return pd.DataFrame()

    working = usage_df.copy()
    if 'UsageDate' in working.columns:
        working['Date'] = pd.to_datetime(working['UsageDate'], errors='coerce')
        working['daily_usage'] = pd.to_numeric(working['UsageQty'], errors='coerce')
    elif {'Year', 'Month'}.issubset(working.columns):
        # Monthly aggregates – convert to daily rate using days in month
        def _daily_from_month(row) -> float:
            try:
                year = int(row.get('Year'))
                month = int(row.get('Month'))
                days_in_month = calendar.monthrange(year, month)[1] or 1
                return float(row.get('UsageQty', 0) or 0) / days_in_month
            except Exception:
                return 0.0

        working['Date'] = pd.to_datetime(
            working[['Year', 'Month']].assign(Day=1), errors='coerce'
        )
        working['daily_usage'] = working.apply(_daily_from_month, axis=1)
    else:
        return pd.DataFrame()

    working = working.dropna(subset=['Date', 'daily_usage'])
    return working[['Date', 'daily_usage']].sort_values('Date')


def recommend_optimal_buy_window(
    price_history: list[dict[str, Any]],
    usage_history: list[dict[str, Any]],
    coverage_days: float | None = None,
    available_stock: float | None = None,
    today: datetime.date | None = None,
    on_order: float | None = None,
) -> dict[str, Any]:
    """
    Recommend the best day (within remaining coverage) to place a buy.

    Uses a lightweight linear regression on historical prices to forecast daily
    prices, and a usage trend line to keep the recommendation inside the safety
    window before stock-out. This keeps the recommendation local (no external
    services) while still using learned trends instead of fixed thresholds.
    The window start is delayed when coverage/on-order are high to avoid
    recommending an immediate buy unless the price trend justifies it.
    """
    today = today or datetime.date.today()

    if not price_history:
        return {
            'status': 'insufficient',
            'reason': 'No price history available',
            'days_from_now': 0,
            'expected_price': 0.0,
            'current_price': 0.0,
            'price_delta': 0.0,
            'latest_safe_day': 0,
            'confidence': 0.0,
        }

    price_df = pd.DataFrame(price_history)
    if price_df.empty or 'TransactionDate' not in price_df.columns or 'AvgCost' not in price_df.columns:
        return {
            'status': 'insufficient',
            'reason': 'Missing price fields',
            'days_from_now': 0,
            'expected_price': 0.0,
            'current_price': 0.0,
            'price_delta': 0.0,
            'latest_safe_day': 0,
            'confidence': 0.0,
        }

    price_df['TransactionDate'] = pd.to_datetime(price_df['TransactionDate'], errors='coerce')
    price_df['AvgCost'] = pd.to_numeric(price_df['AvgCost'], errors='coerce')
    price_df = price_df.dropna(subset=['TransactionDate', 'AvgCost']).sort_values('TransactionDate')
    if price_df.empty:
        return {
            'status': 'insufficient',
            'reason': 'Unusable price records',
            'days_from_now': 0,
            'expected_price': 0.0,
            'current_price': 0.0,
            'price_delta': 0.0,
            'latest_safe_day': 0,
            'confidence': 0.0,
        }

    base_price_date = price_df['TransactionDate'].min()
    price_df['t'] = (price_df['TransactionDate'] - base_price_date).dt.days
    current_price = float(price_df['AvgCost'].iloc[-1])

    if price_df['t'].nunique() >= 2:
        slope, intercept = np.polyfit(price_df['t'], price_df['AvgCost'], 1)
    else:
        slope, intercept = 0.0, current_price

    today_t = int(max(price_df['t'].max(), (pd.Timestamp(today) - base_price_date).days))

    # Usage trend to keep recommendation inside a safe window
    usage_df = _prepare_daily_usage_series(usage_history)
    base_daily_usage = float(usage_df['daily_usage'].mean()) if not usage_df.empty else 0.0
    usage_slope = 0.0
    if not usage_df.empty:
        usage_df['t'] = (usage_df['Date'] - usage_df['Date'].min()).dt.days
        if usage_df['t'].nunique() >= 2:
            usage_slope, usage_intercept = np.polyfit(usage_df['t'], usage_df['daily_usage'], 1)
            base_daily_usage = max(base_daily_usage, float(usage_intercept))

    if base_daily_usage == 0 and coverage_days and available_stock:
        base_daily_usage = float(available_stock) / max(float(coverage_days), 1.0)

    available_stock_val = float(available_stock or 0.0)
    horizon = int(max(7, min(int(coverage_days) if coverage_days else 90, 180)))
    future_days = np.arange(0, horizon + 1)
    future_t = today_t + future_days

    pred_prices = np.clip(intercept + slope * future_t, a_min=0.0, a_max=None)

    # Estimate when we run out, factoring in usage trend
    def _cumulative_usage(day: int) -> float:
        # Sum of arithmetic series for rising/falling usage: u0*d + 0.5*slope*d*(d+1)
        return max(0.0, (base_daily_usage * day) + 0.5 * usage_slope * day * (day + 1))

    stockout_day = None
    if available_stock_val > 0 and base_daily_usage > 0:
        for d in future_days:
            if _cumulative_usage(int(d)) >= available_stock_val:
                stockout_day = int(d)
                break

    if stockout_day is None and coverage_days:
        stockout_day = int(coverage_days)
    if stockout_day is None:
        stockout_day = horizon
        
    # SAFETY BUFFER: Shift stockout date 7 days earlier
    stockout_day = max(0, stockout_day - 7)

    safety_buffer = 3 if stockout_day < 45 else 7
    latest_day = max(0, min(stockout_day, horizon) - safety_buffer)

    min_start_day = 0
    coverage_for_delay = coverage_days if coverage_days is not None else stockout_day
    if coverage_for_delay and coverage_for_delay > 30:
        extra_coverage = coverage_for_delay - 30
        min_start_day = min(
            latest_day,
            max(min_start_day, int(round(extra_coverage * 0.35)))
        )

    on_order_days = None
    if on_order is not None and base_daily_usage > 0:
        on_order_days = float(on_order) / max(base_daily_usage, 1e-6)
        if on_order_days > 7:
            min_start_day = min(
                latest_day,
                max(min_start_day, int(round(on_order_days * 0.25)))
            )

    if slope < 0:
        min_start_day = max(0, min_start_day - 3)

    min_start_day = min(min_start_day, latest_day)

    mask = (future_days >= min_start_day) & (future_days <= latest_day)
    if not mask.any():
        mask = future_days <= min(stockout_day, horizon)

    candidates = future_days[mask]
    candidate_prices = pred_prices[mask]

    if candidates.size == 0:
        best_day = 0
        expected_price = current_price
    else:
        # User requested 80% weight on Inventory (Safety) and 20% on Price
        # Inventory Score: Buying earlier is safer (1.0 at day 0, 0.0 at latest_day)
        # Price Score: Lower price is better (1.0 at min_price, 0.0 at max_price)
        
        # 1. Calculate Price Score
        min_p = candidate_prices.min()
        max_p = candidate_prices.max()
        price_range = max_p - min_p
        
        if price_range > 0:
            price_scores = 1.0 - ((candidate_prices - min_p) / price_range)
        else:
            price_scores = np.ones_like(candidate_prices) # All prices same -> Score 1.0
            
        # 2. Calculate Inventory Score (Safety)
        # Normalize day 0 to latest_day
        day_range = latest_day - min_start_day
        if day_range > 0:
             # Earlier days = Higher Safety Score
             inventory_scores = 1.0 - ((candidates - min_start_day) / day_range)
        else:
             inventory_scores = np.ones_like(candidates)
             
        # 3. Combined Weighted Score
        # 80% Inventory, 20% Price
        final_scores = (0.8 * inventory_scores) + (0.2 * price_scores)
        
        # 4. Pick Best Day
        best_idx = int(np.argmax(final_scores))
        best_day = int(candidates[best_idx])
        expected_price = float(candidate_prices[best_idx])

    buy_date = today + datetime.timedelta(days=best_day)
    trend_direction = "down" if slope < 0 else "up" if slope > 0 else "flat"

    volatility = float(price_df['AvgCost'].std() or 0.0)
    stability = 1 / (1 + (volatility / (current_price + 1e-6)))
    confidence = max(0.25, min(0.95, (len(price_df) / 24) * stability))

    reason = (
        f"Price trend {trend_direction} with projected ${expected_price:,.2f} at day {best_day}; "
        f"keeps {safety_buffer}d buffer before stock-out at ~day {stockout_day}."
    )
    if min_start_day > 0:
        reason += f" Window starts at day {min_start_day} to use existing coverage/on-order."

    return {
        'status': 'ok',
        'days_from_now': best_day,
        'buy_date': buy_date,
        'expected_price': round(expected_price, 2),
        'current_price': round(current_price, 2),
        'price_delta': round(expected_price - current_price, 2),
        'latest_safe_day': int(latest_day),
        'confidence': round(confidence, 2),
        'trend_direction': trend_direction,
        'reason': reason,
    }
def get_batch_volatility_scores(cursor: pyodbc.Cursor, limit: int = 50) -> list[dict[str, Any]]:
    """
    Calculate price volatility for all raw materials in a single batch query.
    Much faster than calling get_volatility_score per item.
    Returns list of dicts with ITEMNMBR, volatility_score, volatility_label, color.
    """
    try:
        # Single query to get all price variance data for raw materials
        query = """
        WITH ItemPriceStats AS (
            SELECT 
                l.ITEMNMBR,
                AVG(l.UNITCOST) as MeanCost,
                STDEV(l.UNITCOST) as StdDevCost,
                COUNT(*) as DataPoints
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(year, -1, GETDATE())
              AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR
            HAVING COUNT(*) >= 3  -- Need at least 3 data points
        )
        SELECT TOP {limit}
            ips.ITEMNMBR,
            ips.MeanCost,
            ips.StdDevCost,
            ips.DataPoints,
            -- Coefficient of Variation as a percentage
            CASE 
                WHEN ips.MeanCost > 0 THEN (ips.StdDevCost / ips.MeanCost) * 100
                ELSE 0 
            END as CoefVariation
        FROM ItemPriceStats ips
        JOIN IV00101 i ON ips.ITEMNMBR = i.ITEMNMBR
        WHERE i.ITEMTYPE IN (0, 1, 2)
        ORDER BY CoefVariation DESC
        """.format(limit=limit * 2)  # Fetch more to filter
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not cursor.description or not rows:
            return []
        
        columns = [c[0] for c in cursor.description]
        results = []
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            cv = float(row_dict.get('CoefVariation', 0) or 0)
            
            # Scale to 0-100 (most items have CV < 20%)
            volatility_score = min(cv * 5, 100)
            
            if volatility_score < 20:
                label, color = "LOW", "#859900"
            elif volatility_score < 50:
                label, color = "MODERATE", "#b58900"
            else:
                label, color = "HIGH", "#dc322f"
            
            results.append({
                'ITEMNMBR': row_dict.get('ITEMNMBR'),
                'VolatilityScore': round(volatility_score, 1),
                'MeanCost': float(row_dict.get('MeanCost', 0) or 0),
                'TransactionCount': int(row_dict.get('DataPoints', 0) or 0),
                'Label': label,
                'color': color
            })
            
            if len(results) >= limit:
                break
        
        return results
        
    except Exception as e:
        LOGGER.warning(f"Batch volatility calculation failed: {e}")
        return []

def find_optimal_hedging_asset(
    internal_history: list[dict[str, Any]],
    pool_data: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """
    Iterate through a pool of market data to find the asset with the highest correlation.
    Returns the best fit asset name and its full metrics.
    """
    best_asset = None
    best_corr = -1.0
    best_metrics = {}

    for asset_name, asset_data in pool_data.items():
        futures_history = asset_data.get('data', [])
        if not futures_history:
            continue
            
        metrics = calculate_hedge_metrics(internal_history, futures_history)
        if not metrics:
            continue
            
        # Use absolute correlation because a strong negative correlation is also a valid hedge (just reverse position)
        # But typically we want positive correlation for direct hedging, or negative for inverse.
        # calculate_hedge_metrics returns distinct optimal_hedge_ratio which handles sign.
        # So we just look for strength of relationship.
        corr = abs(metrics.get('correlation', 0))
        
        if corr > best_corr:
            best_corr = corr
            best_asset = asset_name
            best_metrics = metrics
            
    if best_asset:
        return {
            'best_asset': best_asset,
            'correlation': best_metrics.get('correlation', 0),
            'metrics': best_metrics,
            'data': pool_data[best_asset] # Return the full external data for the winner
        }
        
    return {}


def get_volatility_score(cursor: pyodbc.Cursor, item_number: str) -> dict[str, Any]:
    """
    Calculate price volatility using coefficient of variation.
    Returns score 0-100 where higher = more volatile.
    """
    try:
        price_history = fetch_product_price_history(cursor, item_number, days=365)
        
        if not price_history or len(price_history) < 3:
            return {'volatility_score': 0, 'volatility_label': 'Unknown', 'color': '#839496'}
        
        costs = [float(r.get('AvgCost', 0)) for r in price_history if r.get('AvgCost')]
        if not costs:
            return {'volatility_score': 0, 'volatility_label': 'Unknown', 'color': '#839496'}
        
        mean_cost = sum(costs) / len(costs)
        if mean_cost == 0:
            return {'volatility_score': 0, 'volatility_label': 'Unknown', 'color': '#839496'}
        
        variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
        std_dev = variance ** 0.5
        coef_variation = (std_dev / mean_cost) * 100  # Convert to percentage
        
        # Scale to 0-100 (most items have CV < 20%)
        volatility_score = min(coef_variation * 5, 100)
        
        if volatility_score < 20:
            volatility_label = "LOW"
            color = "#859900"  # Green
        elif volatility_score < 50:
            volatility_label = "MODERATE"
            color = "#b58900"  # Amber
        else:
            volatility_label = "HIGH"
            color = "#dc322f"  # Red
        
        return {
            'volatility_score': round(volatility_score, 1),
            'volatility_label': volatility_label,
            'color': color,
            'std_dev': round(std_dev, 2),
            'mean': round(mean_cost, 2)
        }
        
    except Exception as e:
        LOGGER.warning(f"Volatility calculation failed for {item_number}: {e}")
        return {'volatility_score': 0, 'volatility_label': 'Error', 'color': '#839496'}


def get_seasonal_pattern(cursor: pyodbc.Cursor, item_number: str) -> dict[str, Any]:
    """
    Detect seasonal usage patterns by analyzing month-over-month variations.
    Returns peak and low months plus seasonality strength indicator.
    """
    try:
        usage = fetch_product_usage_history(cursor, item_number, days=730)  # 2 years
        
        if not usage or len(usage) < 6:
            return {'has_pattern': False, 'pattern': 'Insufficient data'}
        
        # Aggregate by month number (1-12)
        month_totals = {}
        for r in usage:
            month = r.get('Month', 0)
            qty = float(r.get('UsageQty', 0))
            month_totals[month] = month_totals.get(month, 0) + qty
        
        if not month_totals:
            return {'has_pattern': False, 'pattern': 'No usage'}
        
        # Find peak and low months
        peak_month = max(month_totals, key=month_totals.get)
        low_month = min(month_totals, key=month_totals.get)
        avg_usage = sum(month_totals.values()) / len(month_totals)
        
        peak_ratio = (month_totals[peak_month] / avg_usage - 1) * 100 if avg_usage > 0 else 0
        
        # Determine if pattern is significant (peak > 30% above average)
        has_pattern = peak_ratio > 30
        
        return {
            'has_pattern': has_pattern,
            'peak_month': calendar.month_name[peak_month] if 1 <= peak_month <= 12 else 'Unknown',
            'low_month': calendar.month_name[low_month] if 1 <= low_month <= 12 else 'Unknown',
            'peak_ratio': round(peak_ratio, 1),
            'monthly_data': month_totals,
            'pattern': 'Seasonal' if has_pattern else 'Stable'
        }
        
    except Exception as e:
        LOGGER.warning(f"Seasonal pattern detection failed for {item_number}: {e}")
        return {'has_pattern': False, 'pattern': 'Error'}


def get_priority_raw_materials(
    cursor: pyodbc.Cursor,
    limit: int = 25,
    location: str | None = PRIMARY_LOCATION,
    fallback_all_locations: bool = True,
    require_purchase_history: bool = False,
) -> pd.DataFrame:
    """
    Identify high-priority raw materials that a purchasing analyst should focus on.
    
    Priority criteria:
    1. Has significant cost (> $0.01) - not placeholder items
    2. Has recent usage (inventory movement in last 180 days)
    3. Has purchase history (bought from vendors)
    4. Sorted by total annual spend (cost * usage)
    
    Args:
        require_purchase_history: If True, STRICTLY enforces that the item must have 
                                  been purchased in the last 2 years (PurchaseCount > 0).
    """
    try:
        # Dynamic filter for purchase history
        # Default behavior (False): (Usage > 0 OR Purchase > 0)
        # Strict behavior (True): AND PurchaseCount > 0
        
        purchase_filter = ""
        if require_purchase_history:
             purchase_filter = "AND p.PurchaseCount > 0"

        # Lookback Period (User requested 2.5 years)
        months_back = -30 if require_purchase_history else -24

        # Query raw materials with actual purchasing activity
        query = f"""
        WITH RecentUsage AS (
            SELECT 
                t.ITEMNMBR,
                SUM(ABS(t.TRXQTY)) as UsageQty180
            FROM IV30300 t
            JOIN IV30200 h ON t.DOCNUMBR = h.DOCNUMBR AND t.DOCTYPE = h.IVDOCTYP
            WHERE h.DOCDATE >= DATEADD(day, -180, GETDATE())
            GROUP BY t.ITEMNMBR
        ),
        RecentPurchases AS (
            SELECT 
                l.ITEMNMBR,
                MAX(h.RECEIPTDATE) as LastPurchase,
                AVG(l.UNITCOST) as AvgPurchaseCost,
                COUNT(*) as PurchaseCount
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, {months_back}, GETDATE())
              AND h.VENDORID IS NOT NULL 
              AND RTRIM(h.VENDORID) <> ''
              AND h.POPTYPE <> 2
              AND h.VOIDSTTS = 0
              AND l.NONINVEN = 0
            GROUP BY l.ITEMNMBR
        )
        SELECT TOP {limit if limit else 5000}
            i.ITEMNMBR,
            RTRIM(i.ITEMDESC) as ITEMDESC,
            i.CURRCOST,
            i.STNDCOST,
            RTRIM(i.ITMCLSCD) as ITMCLSCD,
            RTRIM(i.USCATVLS_1) as Category,
            i.ITEMTYPE,
            ISNULL(u.UsageQty180, 0) as UsageQty180,
            p.LastPurchase,
            ISNULL(p.AvgPurchaseCost, i.CURRCOST) as AvgPurchaseCost,
            ISNULL(p.PurchaseCount, 0) as PurchaseCount,
            -- Calculate annual spend estimate
            i.CURRCOST * ISNULL(u.UsageQty180, 0) * 2 as EstAnnualSpend
        FROM IV00101 i
        LEFT JOIN RecentUsage u ON i.ITEMNMBR = u.ITEMNMBR
        LEFT JOIN RecentPurchases p ON i.ITEMNMBR = p.ITEMNMBR
        WHERE 
            -- Strict Raw Material Definition (Item Class Codes)
            {RAW_MATERIAL_CLASS_FILTER_SQL}
            AND i.CURRCOST > 0.01
            -- EXCLUDE Receiving/Intermediate Items (User Feedback)
            AND i.ITEMNMBR NOT LIKE 'REC-%'
            AND i.ITEMNMBR NOT LIKE 'PMX%'
            AND (
                -- Has recent usage OR recent purchases
                u.UsageQty180 > 0
                OR p.PurchaseCount > 0
            )
            {purchase_filter}
        ORDER BY 
            -- Prioritize by annual spend AND recent activity
            CASE WHEN u.UsageQty180 > 0 THEN 1 ELSE 2 END,
            i.CURRCOST * ISNULL(u.UsageQty180, 0) DESC,
            i.CURRCOST DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if (not cursor.description or not rows) and fallback_all_locations and location:
            # Retry without location filter
            return get_priority_raw_materials(cursor, limit=limit, location=None, fallback_all_locations=False)
        if not cursor.description or not rows:
            return pd.DataFrame()
        
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)

        # Normalize numeric fields and guard against zero/blank standard costs
        df['CURRCOST'] = pd.to_numeric(df['CURRCOST'], errors='coerce')
        df['STNDCOST'] = pd.to_numeric(df['STNDCOST'], errors='coerce')

        # Add segment classification using the same logic as the market monitor
        df['ITMCLSCD'] = df['ITMCLSCD'].fillna('').astype(str).str.strip()
        df['Segment'] = df['ITMCLSCD'].apply(classify_item_segment)

        # Calculate % change safely (fallback to 0 when standard cost is missing/zero)
        df['PctChange'] = 0.0
        valid_cost_mask = df['STNDCOST'].notna() & df['STNDCOST'].ne(0)
        df.loc[valid_cost_mask, 'PctChange'] = (
            (df.loc[valid_cost_mask, 'CURRCOST'] - df.loc[valid_cost_mask, 'STNDCOST'])
            / df.loc[valid_cost_mask, 'STNDCOST'] * 100
        )
        df['PctChange'] = df['PctChange'].fillna(0)
        
        return df
        
    except Exception as e:
        LOGGER.warning(f"Priority raw materials query failed: {e}")
        return pd.DataFrame()


def get_items_needing_attention(cursor: pyodbc.Cursor, df_items: pd.DataFrame) -> list[dict]:
    """
    For a given list of items, identify which ones need immediate attention.
    
    Returns list of attention items with reasons:
    - Low inventory runway (<30 days)
    - Price near 2-year high (wait to buy)
    - Strong buying opportunity (price at historical low)
    - High volatility (unpredictable pricing)
    """
    attention_items = []
    
    for _, row in df_items.iterrows():
        item_num = row['ITEMNMBR']
        item_desc = row.get('ITEMDESC', '')
        curr_cost = float(row.get('CURRCOST', 0))
        
        alerts = []
        priority = 0  # Higher = more urgent
        
        # Check inventory runway
        runway = calculate_inventory_runway(cursor, item_num)
        runway_days = runway.get('runway_days', 999)
        
        if runway_days < 30:
            alerts.append({
                'type': 'INVENTORY_CRITICAL',
                'icon': '🔴',
                'message': f'Only {runway_days:.0f} days of supply remaining',
                'action': 'ORDER NOW'
            })
            priority += 100
        elif runway_days < 60:
            alerts.append({
                'type': 'INVENTORY_WARNING',
                'icon': '🟡',
                'message': f'{runway_days:.0f} days of supply - monitor closely',
                'action': 'PLAN REORDER'
            })
            priority += 50
        
        # Check buying signals
        signals = calculate_buying_signals(cursor, item_num, runway_days=runway_days)
        score = signals.get('score', 50)
        
        if score >= 80:
            alerts.append({
                'type': 'BUY_OPPORTUNITY',
                'icon': '💰',
                'message': signals.get('reason', 'Price at historical low'),
                'action': 'BUY NOW'
            })
            priority += 40
        elif score <= 20:
            alerts.append({
                'type': 'PRICE_WARNING',
                'icon': '⚠️',
                'message': 'Price near 2-year high',
                'action': 'DELAY IF POSSIBLE'
            })
            priority += 20
        
        # Check volatility
        vol = get_volatility_score(cursor, item_num)
        if vol.get('volatility_score', 0) > 50:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'icon': '📊',
                'message': f"High price volatility ({vol.get('volatility_label')})",
                'action': 'HEDGE/LOCK PRICE'
            })
            priority += 30
        
        # Only include items with alerts
        if alerts:
            attention_items.append({
                'Item': item_num,
                'Description': item_desc,
                'Cost': curr_cost,
                'Priority': priority,
                'Runway': runway_days,
                'BuyScore': score,
                'Alerts': alerts
            })
    
    # Sort by priority (highest first)
    attention_items.sort(key=lambda x: x['Priority'], reverse=True)
    
    return attention_items


def get_top_movers_raw_materials(cursor: pyodbc.Cursor, limit: int = 15) -> pd.DataFrame:
    """
    Identify Raw Materials with the largest price changes (positive or negative)
    over the last year (Current 3 months vs Same 3 months last year).
    """
    try:
        query = f"""
        WITH RawMaterialReceipts AS (
            SELECT 
                l.ITEMNMBR,
                h.RECEIPTDATE,
                l.UNITCOST
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE l.UNITCOST > 0
                AND h.POPTYPE <> 2
                AND h.VOIDSTTS = 0
                AND l.NONINVEN = 0
                AND h.RECEIPTDATE >= DATEADD(month, -15, GETDATE()) -- Look back 15 months to cover both periods
        ),
        CurrentPeriod AS (
            SELECT 
                ITEMNMBR,
                AVG(UNITCOST) as CurrentCost
            FROM RawMaterialReceipts
            WHERE RECEIPTDATE >= DATEADD(month, -3, GETDATE())
            GROUP BY ITEMNMBR
        ),
        PriorPeriod AS (
            SELECT 
                ITEMNMBR,
                AVG(UNITCOST) as PriorCost
            FROM RawMaterialReceipts
            WHERE RECEIPTDATE >= DATEADD(month, -15, GETDATE())
                AND RECEIPTDATE <= DATEADD(month, -12, GETDATE())
            GROUP BY ITEMNMBR
        )
        SELECT TOP {limit}
            i.ITEMNMBR,
            RTRIM(i.ITEMDESC) as ITEMDESC,
            c.CurrentCost,
            p.PriorCost,
            ((c.CurrentCost - p.PriorCost) / p.PriorCost) * 100 as PctChange,
            i.USCATVLS_1 as Category
        FROM CurrentPeriod c
        JOIN PriorPeriod p ON c.ITEMNMBR = p.ITEMNMBR
        JOIN IV00101 i ON c.ITEMNMBR = i.ITEMNMBR
        WHERE p.PriorCost > 0
            -- Strict Raw Material Definition
            AND {RAW_MATERIAL_CLASS_FILTER_SQL}
        ORDER BY ABS(((c.CurrentCost - p.PriorCost) / p.PriorCost) * 100) DESC
        """  # No need to fetch extra valid items anymore
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not cursor.description or not rows:
            return pd.DataFrame()
        
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Sort by absolute change again after filtering and take top N
        df['AbsChange'] = df['PctChange'].abs()
        df = df.sort_values('AbsChange', ascending=False).head(limit)
        
        return df
        
    except Exception as e:
        LOGGER.warning(f"Top movers query failed: {e}")
        return pd.DataFrame()


def get_raw_material_time_series(cursor: pyodbc.Cursor, months: int = 24) -> dict[str, pd.DataFrame]:
    """
    Fetch aggregated time series data for Raw Materials.
    Returns multiple DataFrames for different visualizations:
    - monthly_volume: Monthly purchase quantities
    - monthly_cost: Monthly average costs
    - yoy_comparison: Year-over-year cost comparison
    """
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=months * 30)
        
        # 1. Monthly Purchase Volume and Cost by Item
        query = f"""
        WITH MonthlyData AS (
            SELECT 
                l.ITEMNMBR,
                YEAR(h.RECEIPTDATE) as Year,
                MONTH(h.RECEIPTDATE) as Month,
                SUM(l.UMQTYINB) as TotalQty,
                AVG(l.UNITCOST) as AvgCost,
                SUM(l.EXTDCOST) as TotalSpend
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.RECEIPTDATE >= ?
                AND h.RECEIPTDATE <= ?
                AND l.UNITCOST > 0
            GROUP BY l.ITEMNMBR, YEAR(h.RECEIPTDATE), MONTH(h.RECEIPTDATE)
        )
        SELECT 
            md.ITEMNMBR,
            RTRIM(i.ITEMDESC) as ITEMDESC,
            md.Year,
            md.Month,
            md.TotalQty,
            md.AvgCost,
            md.TotalSpend,
            i.USCATVLS_1 as Category
        FROM MonthlyData md
        JOIN IV00101 i ON md.ITEMNMBR = i.ITEMNMBR
        WHERE {RAW_MATERIAL_CLASS_FILTER_SQL} -- Strict Filter
        ORDER BY md.ITEMNMBR, md.Year, md.Month
        """
        
        cursor.execute(query, start_date, end_date)
        rows = cursor.fetchall()
        
        if not cursor.description or not rows:
            return {'monthly_volume': pd.DataFrame(), 'monthly_cost': pd.DataFrame(), 'cost_index': pd.DataFrame()}
        
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        # Strict SQL filtering handles the RM check now.
        
        if df.empty:
            return {'monthly_volume': pd.DataFrame(), 'monthly_cost': pd.DataFrame(), 'cost_index': pd.DataFrame()}
        
        # Create Date column
        df['Date'] = pd.to_datetime(df.apply(lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}-01", axis=1))
        df['TotalQty'] = pd.to_numeric(df['TotalQty'], errors='coerce').fillna(0)
        df['TotalSpend'] = pd.to_numeric(df['TotalSpend'], errors='coerce').fillna(0)
        df['AvgCost'] = pd.to_numeric(df['AvgCost'], errors='coerce').fillna(0)
        
        # 1. Aggregate Monthly Volume (sum across all items)
        monthly_volume = df.groupby('Date').agg({
            'TotalQty': 'sum',
            'TotalSpend': 'sum'
        }).reset_index()
        monthly_volume.columns = ['Date', 'TotalQty', 'TotalSpend']
        
        # 2. Aggregate Monthly Cost (weighted average)
        cost_agg = df.groupby('Date', as_index=False).agg({
            'TotalQty': 'sum',
            'TotalSpend': 'sum'
        })
        cost_agg['AvgCost'] = cost_agg.apply(
            lambda r: (r['TotalSpend'] / r['TotalQty']) if r['TotalQty'] > 0 else 0,
            axis=1
        )
        monthly_cost = cost_agg[['Date', 'AvgCost']]
        
        # 3. Cost Index for top items (normalized to 100)
        # Get top 5 items by total spend
        spend_by_item = df.groupby('ITEMNMBR')['TotalSpend'].sum()
        spend_by_item = pd.to_numeric(spend_by_item, errors='coerce').fillna(0)
        top_items = spend_by_item.sort_values(ascending=False).head(5).index.tolist()
        df_top = df[df['ITEMNMBR'].isin(top_items)].copy()
        
        cost_index = pd.DataFrame()
        for item in top_items:
            item_df = df_top[df_top['ITEMNMBR'] == item][['Date', 'AvgCost']].copy()
            if not item_df.empty:
                first_cost = item_df.iloc[0]['AvgCost']
                if first_cost > 0:
                    item_df['Index'] = (item_df['AvgCost'] / first_cost) * 100
                    item_df['Item'] = item
                    cost_index = pd.concat([cost_index, item_df[['Date', 'Item', 'Index']]])
        
        if not cost_index.empty:
            cost_index = cost_index.sort_values(['Item', 'Date']).reset_index(drop=True)
        
        return {
            'monthly_volume': monthly_volume,
            'monthly_cost': monthly_cost,
            'cost_index': cost_index
        }
        
    except Exception as e:
        LOGGER.warning(f"Raw material time series query failed: {e}")
        return {'monthly_volume': pd.DataFrame(), 'monthly_cost': pd.DataFrame(), 'cost_index': pd.DataFrame()}

def calculate_hedge_metrics(
    raw_history: list[dict[str, Any]],
    futures_history: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Calculate hedging efficiency metrics:
    - Correlation
    - Optimal Hedge Ratio (Min Variance and Max Sharpe)
    - Current vs Optimal Sharpe Ratio
    - Efficient Frontier points
    """
    try:
        if not raw_history or not futures_history:
            return {}

        # Convert to DataFrames
        df_raw = pd.DataFrame(raw_history)
        df_fut = pd.DataFrame(futures_history)

        # Ensure dates are datetime
        if 'TransactionDate' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['TransactionDate'])
            df_raw = df_raw.set_index('Date').sort_index()
            # Resample to monthly to align
            raw_series = df_raw['AvgCost'].resample('ME').mean().ffill()
        elif 'Date' in df_raw.columns: # If passed from external data format
             df_raw['Date'] = pd.to_datetime(df_raw['Date'])
             df_raw = df_raw.set_index('Date').sort_index()
             raw_series = df_raw['Price'].resample('ME').mean().ffill()
        else:
            return {}

        if 'date' in df_fut.columns:
            df_fut['Date'] = pd.to_datetime(df_fut['date'])
            df_fut = df_fut.set_index('Date').sort_index()
            # Resample to monthly
            fut_series = df_fut['price_index'].resample('ME').mean().ffill()
        else:
            return {}

        # Align series
        aligned = pd.concat([raw_series, fut_series], axis=1, join='inner').dropna()
        if len(aligned) < 6: # Need some history
            return {}
        
        aligned.columns = ['Raw', 'Fut']
        
        # Calculate Returns
        returns = aligned.pct_change().dropna()
        
        if len(returns) < 3:
            return {}

        # Stats
        mu_raw = returns['Raw'].mean() * 12 # Annualized
        sigma_raw = returns['Raw'].std() * np.sqrt(12)
        
        mu_fut = returns['Fut'].mean() * 12
        sigma_fut = returns['Fut'].std() * np.sqrt(12)
        
        # Avoid correlation calculation if variance is zero (constant price)
        if sigma_raw == 0 or sigma_fut == 0:
            corr = 0.0
        else:
            try:
                corr = returns['Raw'].corr(returns['Fut'])
                if np.isnan(corr): corr = 0.0
            except:
                corr = 0.0
        
        # Optimal Hedge Ratio (Minimum Variance)
        # h* = rho * (sigma_spot / sigma_fut)
        # But we are hedging a long position in Raw with a Short position in Fut.
        # Portfolio P = Raw - h * Fut
        # Var(P) = sigma_r^2 + h^2*sigma_f^2 - 2*h*rho*sigma_r*sigma_f
        # dVar/dh = 2h*sigma_f^2 - 2*rho*sigma_r*sigma_f = 0 => h = rho * (sigma_r / sigma_fut)
        
        if sigma_fut > 0:
            min_var_hedge_ratio = corr * (sigma_raw / sigma_fut)
        else:
            min_var_hedge_ratio = 0
            
        # Max Sharpe Ratio Optimization
        # We want to find weight w (fraction unhedged? or hedge ratio?)
        # Let's stick to the user's definition: "mix of raw material vs futures"
        # "Raw = w, Futures = (1-w) effectively as a short position"
        # This implies a portfolio construction: P = w * Raw_Return + (1-w) * Fut_Return
        # Wait, usually hedging means holding the physical (100%) and shorting futures (h%).
        # The user said: "Cocoa: 60% Physical, 40% Futures" as the recommended mix.
        # This sounds like a portfolio weight problem where we allocate capital. 
        # But for procurement, we hold 100% physical. The "mix" might be "Hedge Ratio".
        # User example: "60% Physical, 40% Futures" -> "40% offset via futures".
        # This implies Hedge Ratio = 0.4.
        # Let's assume we are optimizing the Hedge Ratio h.
        # Portfolio Return = R_raw - h * R_fut (assuming short futures)
        # Actually, if futures return is positive, shorting it loses money.
        # So Return = mu_raw - h * mu_fut.
        # Volatility = sqrt(sigma_raw^2 + h^2*sigma_fut^2 - 2*h*corr*sigma_raw*sigma_fut)
        # We maximize (Return) / Volatility.
        
        best_h = 0.0
        best_sharpe = -999.0
        
        # Simple grid search for h from -1.5 to 1.5 (Allow Long Hedges for negative correlation)
        hedge_ratios = np.linspace(-1.5, 1.5, 200)
        frontier_points = []
        
        # Zero-Drift Assumption:
        # For ex-ante hedging, we assume futures are fair-priced (expected return = 0).
        # We optimize purely for risk-adjusted stability of the *physical* position.
        # This prevents "Bull Market Bias" where the model refuses to hedge because futures went up last year.
        mu_fut_opt = 0.0 
        
        for h in hedge_ratios:
            # Return of hedged portfolio
            # If we short futures, we pay the return.
            # But futures P&L is (Sell Price - Buy Price). 
            # If prices go up, we lose on short.
            # So yes, Return = R_raw - h * R_fut.
            
            p_ret = mu_raw - h * mu_fut_opt # Use zero drift for optimization decision
            
            var_p = (sigma_raw**2) + (h**2 * sigma_fut**2) - (2 * h * corr * sigma_raw * sigma_fut)
            p_vol = np.sqrt(var_p) if var_p > 0 else 0.001
            
            sharpe = p_ret / p_vol
            
            frontier_points.append({
                "HedgeRatio": float(h),
                "Return": float(p_ret),
                "Risk": float(p_vol),
                "Sharpe": float(sharpe)
            })
            
            # Store the volatility for the best h to calc reduction later
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_h = h
                best_metrics_vol = p_vol
                
        
        # Determine Optimization Strategy
        # If the asset has negative expected returns (mu_raw < 0), maximizing Sharpe (Return/Risk)
        # leads to maximizing Risk (to allow the ratio to be 'less negative' or just weird behavior).
        # In these cases, we FORCE Minimum Variance Optimization.
        force_min_var = (mu_raw < 0) or (best_h < 0.05 and abs(corr) > 0.3 and min_var_hedge_ratio > 0.05)
        
        if force_min_var:
             # Force consideration of MinVar hedge
            h_alt = min(1.5, abs(min_var_hedge_ratio))
            if h_alt < 0.01: h_alt = 0.0 # Avoid dust
            
            p_ret_alt = mu_raw - h_alt * mu_fut_opt
            var_p_alt = (sigma_raw**2) + (h_alt**2 * sigma_fut**2) - (2 * h_alt * corr * sigma_raw * sigma_fut)
            p_vol_alt = np.sqrt(var_p_alt) if var_p_alt > 0 else 0.001
            
            # If original return was negative, risk reduction makes Sharpe "worse" (more negative).
            # This is mathematically correct but confusing to users ("Why did you make my Sharpe -11?").
            # We keep the raw calculation but rely on 'Volatility Reduction' to sell the hedge utility.
            sharpe_alt = p_ret_alt / p_vol_alt
            
            best_h = h_alt
            best_sharpe = sharpe_alt
            best_metrics_vol = p_vol_alt

                
        # Current Portfolio (Unhedged, h=0)
        current_ret = mu_raw
        current_vol = sigma_raw
        current_sharpe = current_ret / current_vol if current_vol > 0 else 0
        
        current_sharpe = current_ret / current_vol if current_vol > 0 else 0
        
        # Recalculate Best Metrics using ACTUAL historical drift for display accuracy?
        # Or stick to the Zero Drift "Proforma" view?
        # User wants to know if "Sharpe increases". Using Zero Drift is the specific "Hedged Portfolio Structure" view.
        # Let's return the metrics consistent with the optimization assumption.
        
        vol_reduction = ((current_vol - best_metrics_vol) / current_vol) * 100 if current_vol > 0 else 0
        
        return {
            "correlation": corr,
            "raw_volatility": sigma_raw,
            "futures_volatility": sigma_fut,
            "optimal_hedge_ratio": best_h,
            "optimal_sharpe": best_sharpe,
            "min_var_hedge_ratio": min_var_hedge_ratio,
            "volatility_reduction": vol_reduction
        }

    except Exception as e:
        LOGGER.error(f"Error calculating hedge metrics: {e}")
        return {}


def simulate_portfolio_variance(
    current_value: float,
    unhedged_vol: float,
    hedged_vol: float,
    mu: float = 0.05, # Assumed 5% drift for simulation context
    months: int = 6,
    num_simulations: int = 1000
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for portfolio value under Unhedged vs Hedged volatility.
    Uses Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
    """
    dt = 1/252 # Daily steps
    days = months * 21
    
    # 1. Unhedged Simulation
    # Vectorized simulation
    # Returns: Matrix of [sims, days]
    drift = (mu - 0.5 * unhedged_vol**2) * dt
    shock_unhedged = unhedged_vol * np.sqrt(dt) * np.random.normal(0, 1, (num_simulations, days))
    
    # Starting value
    # Calculate returns (t=1 to t=days)
    returns_unhedged = np.exp(np.cumsum(drift + shock_unhedged, axis=1))
    
    # Prepend starting state (t=0, return=1.0)
    initial_state = np.ones((num_simulations, 1))
    path_multipliers_u = np.hstack([initial_state, returns_unhedged])
    
    # Paths shape: [sims, days+1]
    paths_unhedged = current_value * path_multipliers_u
    
    # 2. Hedged Simulation
    drift_h = (mu - 0.5 * hedged_vol**2) * dt
    shock_hedged = hedged_vol * np.sqrt(dt) * np.random.normal(0, 1, (num_simulations, days))
    
    returns_hedged = np.exp(np.cumsum(drift_h + shock_hedged, axis=1))
    path_multipliers_h = np.hstack([initial_state, returns_hedged])
    
    paths_hedged = current_value * path_multipliers_h
    
    # Collect final values for distribution plot
    final_values_unhedged = paths_unhedged[:, -1]
    final_values_hedged = paths_hedged[:, -1]
    
    # Create DataFrame for Altair (Distribution)
    df_unhedged = pd.DataFrame({'Value': final_values_unhedged, 'Scenario': 'Unhedged'})
    df_hedged = pd.DataFrame({'Value': final_values_hedged, 'Scenario': 'Hedged'})
    
    dist_df = pd.concat([df_unhedged, df_hedged], ignore_index=True)
    
    # Create DataFrame for Time Series (Aggregated)
    # Calculate stats per day
    days_range = np.arange(days + 1)
    
    unhedged_mean = np.mean(paths_unhedged, axis=0)
    unhedged_p05 = np.percentile(paths_unhedged, 5, axis=0)
    unhedged_p95 = np.percentile(paths_unhedged, 95, axis=0)
    
    hedged_mean = np.mean(paths_hedged, axis=0)
    hedged_p05 = np.percentile(paths_hedged, 5, axis=0)
    hedged_p95 = np.percentile(paths_hedged, 95, axis=0)
    
    # Flatten for Altair
    # Format: [Day, Scenario, Mean, Lower, Upper]
    ts_data = []
    
    # Downsample for charting if needed (every 7 days to reduce size), but 180 points is fine.
    for i in range(0, days + 1, 1):
        ts_data.append({'Day': i, 'Scenario': 'Unhedged', 'Mean': unhedged_mean[i], 'Lower': unhedged_p05[i], 'Upper': unhedged_p95[i]})
        ts_data.append({'Day': i, 'Scenario': 'Hedged', 'Mean': hedged_mean[i], 'Lower': hedged_p05[i], 'Upper': hedged_p95[i]})
        
    ts_df = pd.DataFrame(ts_data)
    
    return {
        'distribution': dist_df,
        'timeseries': ts_df
    }

def optimize_capital_allocation(candidates: list[dict[str, Any]], total_capital: float = 10000.0) -> pd.DataFrame:
    """
    Allocates a fixed capital budget across the best hedging opportunities to maximize Portfolio Sharpe.
    Implements a 'Smart Concentration' strategy:
    1. Filter for positive Sharpe Gain.
    2. Rank by Sharpe Efficiency (Sharpe Gain / Vol Reduction).
    3. Allocate capital proportionally to conviction.
    """
    if not candidates:
        return pd.DataFrame()
        
    df = pd.DataFrame(candidates)
    
    # Check required columns
    required_cols = ['Item', 'Best Hedge', 'Sharpe Gain', 'Vol Reduction']
    for col in required_cols:
         if col not in df.columns:
             return pd.DataFrame() # Missing data

    # 1. Filter for viable trades
    # Ensure numeric
    df['Sharpe Gain'] = pd.to_numeric(df['Sharpe Gain'], errors='coerce').fillna(0)
    df['Vol Reduction'] = pd.to_numeric(df['Vol Reduction'], errors='coerce').fillna(0)
    
    df = df[df['Sharpe Gain'] > 0].copy()
    if df.empty:
        return pd.DataFrame()
        
    # 2. Score Opportunities
    # Score = Sharpe Gain * (Vol Reduction / 100)
    # This rewards high impact hedges.
    df['Score'] = df['Sharpe Gain'] * (df['Vol Reduction'] / 100.0)
    
    # 3. Select Top Picks (Constraint: Max 5 positions for $10k to ensure meaningful size)
    # Concentration is key for high Sharpe.
    df = df.sort_values('Score', ascending=False).head(5)
    
    # 4. Allocate Capital
    # Proportional to Score
    total_score = df['Score'].sum()
    if total_score > 0:
        df['Allocation %'] = df['Score'] / total_score
    else:
        df['Allocation %'] = 1.0 / len(df)
        
    df['Allocated Capital'] = df['Allocation %'] * total_capital
    
    return df[['Item', 'Best Hedge', 'Sharpe Gain', 'Allocated Capital', 'Allocation %']]


def calculate_black_scholes_put(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float
) -> float:
    """
    Calculate Black-Scholes Price for a Put Option.
    Used to value the 'Implied Insurance' of the portfolio.
    
    S: Current Price (Portfolio Value)
    K: Strike Price (Target Value, usually S for ATM)
    T: Time to expiration (years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    """
    import math
    
    def norm_cdf(x):
        """Cumulative distribution function for the standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    
    if sigma <= 0 or T <= 0:
        return 0.0
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    return float(put_price)
