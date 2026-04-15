"""
Reorder Math Module

Mathematical calculations for inventory reorder management, replacing
manual "eyeballing" with data-driven decisions based on actual usage patterns.

Key Formulas:
- Reorder Point (ROP) = (Avg Daily Usage × Lead Time) + Safety Stock
- Suggested Order Qty = Order Up To Qty - Current Available - On Order
- Days Until Stockout = Available Qty / Avg Daily Usage
- Must Order By = Today + Days Until Stockout - Lead Time - Safety Days
"""

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pandas as pd
import pyodbc

from constants import PRIMARY_LOCATION, RAW_MATERIAL_CLASS_CODES

LOGGER = logging.getLogger(__name__)


@dataclass
class ReorderRecommendation:
    """Recommendation for a single item."""
    item_number: str
    item_description: str
    item_class: str
    
    # Current inventory state
    qty_on_hand: float
    qty_allocated: float
    qty_on_order: float
    qty_available: float  # (on_hand - allocated) + on_order
    
    # GP's configured values
    gp_order_point: float
    gp_order_up_to: float
    
    # Calculated values from actual usage
    avg_daily_usage: float
    usage_lookback_days: int
    seasonal_burn_rate: float   # avg_daily_usage × current-month seasonal factor
    seasonal_factor: float      # ratio of current-month usage to annual monthly avg
    days_of_coverage: float     # available / seasonal_burn_rate (falls back to avg_daily_usage)
    
    # Calculated reorder point
    lead_time_days: int
    lead_time_source: str  # "Historical" or "Configured"
    lead_time_samples: int  # Number of POs used for calculation
    safety_days: int
    calculated_rop: float
    
    # Recommendations
    suggested_order_qty: float
    must_order_by: datetime.date
    urgency: str  # "Critical", "Soon", "OK"
    
    # Vendor info
    vendor_id: str
    vendor_name: str


def calculate_average_daily_usage(
    cursor: pyodbc.Cursor, 
    item_number: str, 
    lookback_days: int = 90,
    location: str = PRIMARY_LOCATION
) -> float:
    """
    Calculate average daily usage from historical inventory transactions.
    
    Uses IV30300 (Inventory Transaction History) to find outbound transactions
    (negative TRXQTY = consumption/sales).
    
    Args:
        cursor: Database cursor
        item_number: GP item number
        lookback_days: Days of history to analyze (default 90)
        location: Inventory location (default MAIN)
    
    Returns:
        Average daily usage (units per day)
    """
    query = """
    SELECT 
        SUM(ABS(TRXQTY)) AS TotalUsage,
        COUNT(DISTINCT CAST(DOCDATE AS DATE)) AS DaysWithUsage
    FROM IV30300
    WHERE ITEMNMBR = ?
      AND TRXLOCTN = ?
      AND TRXQTY < 0  -- Outbound transactions only
      AND DOCDATE >= DATEADD(day, ?, GETDATE())
    """
    
    try:
        cursor.execute(query, (item_number, location, -lookback_days))
        row = cursor.fetchone()
        
        if row and row.TotalUsage:
            total_usage = float(row.TotalUsage)
            return total_usage / lookback_days
        return 0.0
        
    except Exception as e:
        LOGGER.warning(f"Error calculating usage for {item_number}: {e}")
        return 0.0


def calculate_batch_usage(
    cursor: pyodbc.Cursor,
    item_numbers: list[str],
    lookback_days: int = 90,
    location: str = PRIMARY_LOCATION
) -> dict[str, float]:
    """
    Calculate average daily usage for multiple items in a single query.
    
    Returns:
        Dictionary mapping item_number -> avg_daily_usage
    """
    if not item_numbers:
        return {}
    
    placeholders = ", ".join("?" for _ in item_numbers)
    query = f"""
    SELECT 
        ITEMNMBR,
        SUM(ABS(TRXQTY)) / ? AS AvgDailyUsage
    FROM IV30300
    WHERE ITEMNMBR IN ({placeholders})
      AND TRXLOCTN = ?
      AND TRXQTY < 0
      AND DOCDATE >= DATEADD(day, ?, GETDATE())
    GROUP BY ITEMNMBR
    """
    
    params = [lookback_days, *item_numbers, location, -lookback_days]
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return {row.ITEMNMBR.strip(): float(row.AvgDailyUsage or 0) for row in rows}
    except Exception as e:
        LOGGER.warning(f"Error in batch usage calculation: {e}")
        return {}


def calculate_historical_lead_time(
    cursor: pyodbc.Cursor,
    item_number: str,
    vendor_id: str = None,
    lookback_years: int = 2
) -> tuple[float, int]:
    """
    Calculate average lead time from historical PO data.
    
    Lead Time = Receipt Date - PO Date
    
    Uses:
    - POP30300 (Receipt Header) for receipt date
    - POP30310 (Receipt Lines) to link item to receipt
    - POP30100 (PO History Header) for PO placement date
    
    Args:
        cursor: Database cursor
        item_number: GP item number
        vendor_id: Optional vendor filter
        lookback_years: Years of history to analyze (default 2)
    
    Returns:
        Tuple of (average_lead_time_days, sample_count)
        Returns (0, 0) if no data found
    """
    vendor_filter = "AND r_head.VENDORID = ?" if vendor_id else ""
    params = [item_number, lookback_years]
    if vendor_id:
        params.append(vendor_id)
    
    query = f"""
    SELECT 
        AVG(DATEDIFF(day, po_head.DOCDATE, r_head.RECEIPTDATE)) AS AvgLeadTime,
        COUNT(*) AS SampleCount
    FROM POP30310 r_line  -- Receipt Lines
    JOIN POP30300 r_head ON r_line.POPRCTNM = r_head.POPRCTNM  -- Receipt Header
    JOIN POP30100 po_head ON r_line.PONUMBER = po_head.PONUMBER  -- PO Header
    WHERE r_line.ITEMNMBR = ?
      AND r_head.RECEIPTDATE >= DATEADD(year, -?, GETDATE())
      AND r_line.PONUMBER <> ''  -- Must have linked PO
      AND r_head.RECEIPTDATE >= po_head.DOCDATE  -- Valid dates only
    {vendor_filter}
    """
    
    try:
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row and row.AvgLeadTime is not None and row.SampleCount > 0:
            return float(row.AvgLeadTime), int(row.SampleCount)
        return 0.0, 0
        
    except Exception as e:
        LOGGER.warning(f"Error calculating lead time for {item_number}: {e}")
        return 0.0, 0


def calculate_batch_lead_times(
    cursor: pyodbc.Cursor,
    item_numbers: list[str],
    lookback_years: int = 2
) -> dict[str, tuple[float, int]]:
    """
    Calculate average lead time for multiple items in a single query.
    
    Returns:
        Dictionary mapping item_number -> (avg_lead_time, sample_count)
    """
    if not item_numbers:
        return {}
    
    placeholders = ", ".join("?" for _ in item_numbers)
    query = f"""
    SELECT 
        r_line.ITEMNMBR,
        AVG(DATEDIFF(day, po_head.DOCDATE, r_head.RECEIPTDATE)) AS AvgLeadTime,
        COUNT(*) AS SampleCount
    FROM POP30310 r_line
    JOIN POP30300 r_head ON r_line.POPRCTNM = r_head.POPRCTNM
    JOIN POP30100 po_head ON r_line.PONUMBER = po_head.PONUMBER
    WHERE r_line.ITEMNMBR IN ({placeholders})
      AND r_head.RECEIPTDATE >= DATEADD(year, -?, GETDATE())
      AND r_line.PONUMBER <> ''
      AND r_head.RECEIPTDATE >= po_head.DOCDATE
    GROUP BY r_line.ITEMNMBR
    """
    
    params = [*item_numbers, lookback_years]
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return {
            row.ITEMNMBR.strip(): (float(row.AvgLeadTime or 0), int(row.SampleCount or 0))
            for row in rows
        }
    except Exception as e:
        LOGGER.warning(f"Error in batch lead time calculation: {e}")
        return {}


def calculate_reorder_point(
    avg_daily_usage: float, 
    lead_time_days: int, 
    safety_days: int = 7
) -> float:
    """
    Calculate the reorder point (ROP).
    
    ROP = (Avg Daily Usage × Lead Time) + Safety Stock
    Safety Stock = Avg Daily Usage × Safety Days
    
    Args:
        avg_daily_usage: Units consumed per day
        lead_time_days: Days from order to delivery
        safety_days: Buffer days for variability (default 7)
    
    Returns:
        Quantity at which to reorder
    """
    lead_time_demand = avg_daily_usage * lead_time_days
    safety_stock = avg_daily_usage * safety_days
    return lead_time_demand + safety_stock


def calculate_days_of_coverage(qty_available: float, burn_rate: float) -> float:
    """
    Calculate how many days the current inventory will last.

    Args:
        qty_available: On-hand + on-order quantity
        burn_rate: Effective daily burn rate (prefer seasonal_burn_rate for accuracy)

    Returns:
        Days until stockout (999 if no usage)
    """
    if burn_rate <= 0:
        return 999.0  # No usage = infinite coverage
    return qty_available / burn_rate


def _compute_seasonal_burn_rate(
    cursor: pyodbc.Cursor,
    item_number: str,
    avg_daily_usage: float,
    location: str = PRIMARY_LOCATION,
    today: datetime.date = None,
) -> tuple[float, float]:
    """
    Compute a seasonally-adjusted burn rate using 24 months of monthly usage.

    Ported from c:/Users/alexh/email/reorder_math.py. Algorithm:
      - Exponential decay (15% drop per month) weights recent history higher.
      - Seasonal factor = current-month historical usage / avg monthly usage.
      - Factor is clamped to [0.25, 3.0].

    Returns:
        (seasonal_burn_rate, seasonal_factor)
        Falls back to (avg_daily_usage, 1.0) if there is insufficient data.
    """
    import calendar as _cal

    today = today or datetime.date.today()

    query = """
    SELECT
        YEAR(DOCDATE)  AS Yr,
        MONTH(DOCDATE) AS Mo,
        SUM(ABS(TRXQTY)) AS UsageQty
    FROM IV30300
    WHERE ITEMNMBR = ?
      AND TRXLOCTN = ?
      AND TRXQTY < 0
      AND DOCDATE >= DATEADD(month, -24, GETDATE())
    GROUP BY YEAR(DOCDATE), MONTH(DOCDATE)
    """

    try:
        cursor.execute(query, (item_number, location))
        rows = cursor.fetchall()
    except Exception as e:
        LOGGER.warning(f"Seasonal factor query failed for {item_number}: {e}")
        return avg_daily_usage, 1.0

    if not rows:
        return avg_daily_usage, 1.0

    weighted_usage = 0.0
    weight_sum = 0.0
    month_totals: dict[int, float] = {}

    for row in rows:
        year, month, qty = int(row[0]), int(row[1]), float(row[2] or 0)
        month_date = datetime.date(year, month, 1)
        days_in_month = _cal.monthrange(year, month)[1]
        daily = qty / days_in_month if days_in_month else 0.0

        months_ago = max(
            (today.year - month_date.year) * 12 + (today.month - month_date.month), 0
        )
        decay = 0.85 ** months_ago
        weighted_usage += daily * decay
        weight_sum += decay

        month_totals[month] = month_totals.get(month, 0.0) + qty

    decayed_daily = weighted_usage / weight_sum if weight_sum > 0 else avg_daily_usage

    if month_totals:
        avg_monthly = sum(month_totals.values()) / len(month_totals)
        this_month_usage = month_totals.get(today.month, avg_monthly)
        seasonal_factor = (this_month_usage / avg_monthly) if avg_monthly > 0 else 1.0
    else:
        seasonal_factor = 1.0

    seasonal_factor = max(0.25, min(seasonal_factor, 3.0))
    seasonal_burn_rate = decayed_daily * seasonal_factor

    return seasonal_burn_rate, seasonal_factor


def calculate_order_quantity(
    qty_on_hand: float,
    qty_on_order: float,
    order_up_to_qty: float,
    calculated_rop: float
) -> float:
    """
    Calculate suggested order quantity.
    
    Order Qty = Order Up To - (On Hand + On Order)
    Only returns positive values; 0 if no order needed.
    
    Args:
        qty_on_hand: Current on-hand inventory
        qty_on_order: Quantity already on open POs
        order_up_to_qty: Target inventory level
        calculated_rop: Calculated reorder point
    
    Returns:
        Quantity to order (0 if sufficient inventory)
    """
    available = qty_on_hand + qty_on_order
    
    # Only suggest ordering if below ROP
    if available >= calculated_rop:
        return 0.0
    
    suggested = order_up_to_qty - available
    return max(0.0, suggested)


def calculate_must_order_by(
    days_of_coverage: float,
    lead_time_days: int,
    safety_days: int = 7,
    as_of_date: datetime.date = None
) -> datetime.date:
    """
    Calculate the latest date to place an order to avoid stockout.
    
    Must Order By = Today + Days of Coverage - Lead Time - Safety Buffer
    
    Args:
        days_of_coverage: Days until stockout
        lead_time_days: Days from order to delivery
        safety_days: Buffer days (default 7)
        as_of_date: Base date (default today)
    
    Returns:
        Date by which order must be placed
    """
    as_of_date = as_of_date or datetime.date.today()
    
    # Days until we MUST order (accounting for lead time + safety)
    days_until_order = days_of_coverage - lead_time_days - safety_days
    
    # Clamp to today if already past due
    days_until_order = max(0, int(days_until_order))
    
    # Cap at 10 years to prevent date overflow/nonsense dates
    days_until_order = min(days_until_order, 3650)
    
    return as_of_date + datetime.timedelta(days=days_until_order)


def get_urgency_level(must_order_by: datetime.date, as_of_date: datetime.date = None) -> str:
    """
    Determine urgency level based on must-order date.
    
    Returns:
        "Critical" (past due or today), "Soon" (within 7 days), or "OK"
    """
    as_of_date = as_of_date or datetime.date.today()
    days_until = (must_order_by - as_of_date).days
    
    if days_until <= 0:
        return "Critical"
    elif days_until <= 7:
        return "Soon"
    else:
        return "OK"


def get_reorder_recommendations(
    cursor: pyodbc.Cursor,
    item_numbers: list[str] = None,
    include_classes: list[str] = None,
    lookback_days: int = 90,
    safety_days: int = 7,
    location: str = PRIMARY_LOCATION,
    only_below_rop: bool = False
) -> pd.DataFrame:
    """
    Get reorder recommendations for items.
    
    Args:
        cursor: Database cursor
        item_numbers: Specific items to analyze (None = all raw materials)
        include_classes: Item class codes to include (default: RAW_MATERIAL_CLASS_CODES)
        lookback_days: Days of usage history to analyze
        safety_days: Safety buffer days
        location: Inventory location
        only_below_rop: Only return items below reorder point
    
    Returns:
        DataFrame with recommendations for each item
    """
    include_classes = include_classes or RAW_MATERIAL_CLASS_CODES
    
    # Build the query to get inventory status and item info
    if item_numbers:
        placeholders = ", ".join("?" for _ in item_numbers)
        item_filter = f"AND i.ITEMNMBR IN ({placeholders})"
        params = [location, *item_numbers]
    else:
        class_placeholders = ", ".join("?" for _ in include_classes)
        item_filter = f"AND i.ITMCLSCD IN ({class_placeholders})"
        params = [location, *include_classes]
    
    query = f"""
    SELECT 
        i.ITEMNMBR,
        i.ITEMDESC,
        i.ITMCLSCD,
        COALESCE(loc.QTYONHND, 0) AS QtyOnHand,
        COALESCE(loc.ATYALLOC, 0) AS QtyAllocated,
        COALESCE(loc.QTYONORD, 0) AS QtyOnOrder,
        COALESCE(loc.ORDRPNTQTY, 0) AS OrderPointQty,
        COALESCE(NULLIF(loc.ORDRUPTOLVL, 0), loc.ORDRPNTQTY * 2, 0) AS OrderUpToQty,
        COALESCE(pv.VENDORID, '') AS VendorID,
        COALESCE(pv.VENDNAME, '') AS VendorName,
        COALESCE(pv.PLANNINGLEADTIME, 14) AS LeadTimeDays
    FROM IV00101 i
    LEFT JOIN IV00102 loc ON i.ITEMNMBR = loc.ITEMNMBR AND loc.LOCNCODE = ?
    OUTER APPLY (
        -- Get primary vendor (most recent order date) - only items with vendors
        SELECT TOP 1 iv.VENDORID, v.VENDNAME, iv.PLANNINGLEADTIME
        FROM IV00103 iv
        LEFT JOIN PM00200 v ON iv.VENDORID = v.VENDORID
        WHERE iv.ITEMNMBR = i.ITEMNMBR
        ORDER BY iv.LSTORDDT DESC
    ) pv
    WHERE i.ITEMTYPE = 1  -- Inventory items
      AND i.INACTIVE = 0  -- Active items only
      {item_filter}
    """
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    except Exception as e:
        LOGGER.error(f"Error fetching inventory data: {e}")
        return pd.DataFrame()
    
    if not rows:
        return pd.DataFrame()
    
    # Collect all item numbers for batch queries
    all_items = [row.ITEMNMBR.strip() for row in rows]
    
    # Batch calculate usage and lead times (2 queries instead of 2N)
    usage_map = calculate_batch_usage(cursor, all_items, lookback_days, location)
    lead_time_map = calculate_batch_lead_times(cursor, all_items)
    
    recommendations = []
    today = datetime.date.today()
    
    for row in rows:
        item_number = row.ITEMNMBR.strip()
        vendor_id = (row.VendorID or "").strip()
        
        # Get usage from batch results (default 0)
        avg_daily_usage = usage_map.get(item_number, 0.0)
        
        qty_on_hand = float(row.QtyOnHand or 0)
        qty_allocated = float(row.QtyAllocated or 0)
        qty_on_order = float(row.QtyOnOrder or 0)
        qty_available = (qty_on_hand - qty_allocated) + qty_on_order
        
        # Get lead time from batch results
        historical_lead_time, lead_time_samples = lead_time_map.get(item_number, (0.0, 0))
        
        # Use historical if available, otherwise fall back to GP configured
        configured_lead_time = int(row.LeadTimeDays or 14)
        if historical_lead_time > 0 and lead_time_samples > 0:
            lead_time_days = int(round(historical_lead_time))
            lead_time_source = "Historical"
        else:
            lead_time_days = configured_lead_time
            lead_time_source = "Configured"
            lead_time_samples = 0
        
        # GP's configured values
        gp_order_point = float(row.OrderPointQty or 0)
        gp_order_up_to = float(row.OrderUpToQty or gp_order_point * 2)
        
        # Calculate our values
        calculated_rop = calculate_reorder_point(avg_daily_usage, lead_time_days, safety_days)
        seasonal_burn_rate, seasonal_factor = _compute_seasonal_burn_rate(
            cursor, item_number, avg_daily_usage, location=location, today=today
        )
        # Use the seasonally-adjusted burn rate for coverage so current-month
        # urgency reflects peak/trough months, not a flat 90-day average.
        days_of_coverage = calculate_days_of_coverage(qty_available, seasonal_burn_rate)
        suggested_order_qty = calculate_order_quantity(
            qty_on_hand, qty_on_order, gp_order_up_to, calculated_rop
        )
        must_order_by = calculate_must_order_by(days_of_coverage, lead_time_days, safety_days, today)
        
        # Urgency based on GP's order point (matches GP's "Items Below Order Point" list)
        # STRICT MODE: Only alert if the user has actually set an Order Point in GP.
        # If GP Order Point is 0, we assume the user doesn't want to track it, regardless of usage.
        
        if gp_order_point <= 0:
            urgency = "OK"
        elif qty_available < gp_order_point:
            urgency = "Critical"  # Below the hard GP line
        elif calculated_rop > 0 and qty_available < calculated_rop:
            urgency = "Soon"      # Above GP line, but below calculated safety safety
        else:
            urgency = "OK"
        
        # Skip items not below ROP if filter is on
        if only_below_rop and urgency == "OK":
            continue
        
        rec = ReorderRecommendation(
            item_number=item_number,
            item_description=(row.ITEMDESC or "").strip(),
            item_class=(row.ITMCLSCD or "").strip(),
            qty_on_hand=qty_on_hand,
            qty_allocated=qty_allocated,
            qty_on_order=qty_on_order,
            qty_available=qty_available,
            gp_order_point=gp_order_point,
            gp_order_up_to=gp_order_up_to,
            avg_daily_usage=avg_daily_usage,
            usage_lookback_days=lookback_days,
            seasonal_burn_rate=seasonal_burn_rate,
            seasonal_factor=seasonal_factor,
            days_of_coverage=days_of_coverage,
            lead_time_days=lead_time_days,
            lead_time_source=lead_time_source,
            lead_time_samples=lead_time_samples,
            safety_days=safety_days,
            calculated_rop=calculated_rop,
            suggested_order_qty=suggested_order_qty,
            must_order_by=must_order_by,
            urgency=urgency,
            vendor_id=vendor_id,
            vendor_name=(row.VendorName or "").strip(),
        )
        recommendations.append(rec)
    
    # Convert to DataFrame
    if not recommendations:
        return pd.DataFrame()
    
    df = pd.DataFrame([vars(r) for r in recommendations])
    
    # Sort by urgency (Critical first) then by must_order_by
    urgency_order = {"Critical": 0, "Soon": 1, "OK": 2}
    df["_urgency_rank"] = df["urgency"].map(urgency_order)
    df = df.sort_values(["_urgency_rank", "must_order_by"]).drop(columns=["_urgency_rank"])
    
    return df
