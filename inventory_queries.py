from decimal import Decimal

import pyodbc

from constants import LOGGER, PRIMARY_LOCATION
from parsing_utils import decimal_or_zero
from sql_utils import format_sql_preview


def fetch_on_hand_by_item(cursor: pyodbc.Cursor, items: list[str], location: str = PRIMARY_LOCATION) -> tuple[dict[str, Decimal], str]:
    """Return on-hand quantities for the given items, filtered to a specific location (default MAIN)."""
    if not items:
        return {}, ""
    filtered = [itm for itm in items if itm]
    if not filtered:
        return {}, ""

    placeholders = ", ".join("?" for _ in filtered)
    params = [*filtered, location] if location else filtered
    location_clause = " AND LOCNCODE = ?" if location else ""
    query = f"""
        SELECT ITEMNMBR, SUM(QTYONHND) AS OnHand
        FROM IV00102
        WHERE ITEMNMBR IN ({placeholders}){location_clause}
        GROUP BY ITEMNMBR
    """
    sql_preview = format_sql_preview(query, params)
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        on_hand = {row.ITEMNMBR: decimal_or_zero(row.OnHand) for row in rows}
        return on_hand, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch on-hand quantities: %s", err)
        return {}, sql_preview


def fetch_open_po_supply(cursor: pyodbc.Cursor, items: list[str], location: str = PRIMARY_LOCATION) -> tuple[dict[str, Decimal], str]:
    """Return open purchase order quantities for items (POLNESTA = 1), filtered to a location (default MAIN)."""
    if not items:
        return {}, ""
    filtered = [itm for itm in items if itm]
    if not filtered:
        return {}, ""

    placeholders = ", ".join("?" for _ in filtered)
    params = [*filtered, location] if location else filtered
    location_clause = " AND LOCNCODE = ?" if location else ""
    query = f"""
        SELECT ITEMNMBR, SUM(QTYORDER) AS OpenPOQty
        FROM POP10110
        WHERE ITEMNMBR IN ({placeholders})
          AND POLNESTA = 1{location_clause}
        GROUP BY ITEMNMBR
    """
    sql_preview = format_sql_preview(query, params)
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        supply = {row.ITEMNMBR: decimal_or_zero(row.OpenPOQty) for row in rows}
        return supply, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch open PO supply: %s", err)
        return {}, sql_preview


def fetch_mfg_bom_grouped_by_component(cursor: pyodbc.Cursor, parent_item: str) -> tuple[list, str]:
    """
    Fetch the manufacturing BOM for a given parent item, grouped by component.

    This query retrieves the components for a given parent item from the Bill of Materials,
    summarizing the quantity for each component.
    """
    if not parent_item:
        return [], ""

    query = """
        SELECT
            b.PPN_I AS ParentItem,
            ip.ITEMDESC AS ParentDescription,
            b.CPN_I AS ComponentItem,
            ic.ITEMDESC AS ComponentDescription,
            SUM(b.QUANTITY_I) AS QtyPerParent, -- total qty per component
            MAX(b.UOFM) AS ComponentUofM
        FROM BM010115 b
        LEFT JOIN IV00101 ip ON ip.ITEMNMBR = b.PPN_I
        LEFT JOIN IV00101 ic ON ic.ITEMNMBR = b.CPN_I
        WHERE b.PPN_I = ?
        GROUP BY b.PPN_I, ip.ITEMDESC, b.CPN_I, ic.ITEMDESC
        ORDER BY b.CPN_I
    """
    sql_preview = format_sql_preview(query, [parent_item])
    try:
        cursor.execute(query, parent_item)
        rows = cursor.fetchall()
        return rows, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch BOM for item %s: %s", parent_item, err)
        return [], sql_preview


def fetch_recursive_bom_for_item(cursor: pyodbc.Cursor, parent_item: str) -> tuple[list, str]:
    """
    Fetch the recursive BOM for a given parent item.

    This query uses a recursive CTE to traverse the entire BOM hierarchy for a given parent item.
    It then calculates the total quantity of each component required for one unit of the parent.
    """
    if not parent_item:
        return [], ""

    query = """
        WITH BOM_CTE (TopLevelParent, ParentItem, ComponentItem, Quantity, Depth) AS (
            SELECT
                PPN_I AS TopLevelParent,
                PPN_I AS ParentItem,
                CPN_I AS ComponentItem,
                CAST(QUANTITY_I AS DECIMAL(38, 19)) AS Quantity,
                1 AS Depth
            FROM BM010115
            WHERE PPN_I = ?

            UNION ALL

            SELECT
                cte.TopLevelParent,
                b.PPN_I AS ParentItem,
                b.CPN_I AS ComponentItem,
                CAST(cte.Quantity * b.QUANTITY_I AS DECIMAL(38, 19)) AS Quantity,
                cte.Depth + 1
            FROM BM010115 b
            INNER JOIN BOM_CTE cte ON b.PPN_I = cte.ComponentItem
        )
        SELECT
            ComponentItem AS RawMaterial,
            SUM(Quantity) AS Design_Qty
        FROM BOM_CTE
        GROUP BY ComponentItem
        OPTION (MAXRECURSION 0)
    """
    params = [parent_item]
    sql_preview = format_sql_preview(query, params)
    try:
        cursor.execute(query, *params)
        rows = cursor.fetchall()
        return rows, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch recursive BOM for item %s: %s", parent_item, err)
        return [], sql_preview
