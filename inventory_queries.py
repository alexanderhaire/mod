import datetime
from decimal import Decimal

import pyodbc

from constants import LOGGER, PRIMARY_LOCATION
from parsing_utils import decimal_or_zero
from sql_utils import format_sql_preview


def fetch_last_vendor_receipt_map(
    cursor: pyodbc.Cursor,
    items: list[str] | None = None,
) -> dict[str, datetime.date | None]:
    """
    Return ``{item: last_receipt_date}`` from POP30310/POP30300 excluding
    returns (POPTYPE 4/5) and voided receipts.

    If ``items`` is provided, restrict to that list (and pre-seed the result
    dict with ``None`` for items that have no receipt history so callers can
    distinguish "never bought" from "not queried"). If ``items`` is ``None`` or
    empty, fetch the map for every item that has ever been received — cheaper
    than thousands of IN-clause lookups when the caller will need many items.
    """
    if items:
        filtered = sorted({(i or "").strip() for i in items if i and (i or "").strip()})
        if not filtered:
            return {}
        placeholders = ", ".join("?" for _ in filtered)
        query = f"""
            SELECT RTRIM(l.ITEMNMBR) AS item,
                   MAX(CAST(h.RECEIPTDATE AS DATE)) AS last_rcpt
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE RTRIM(l.ITEMNMBR) IN ({placeholders})
              AND h.POPTYPE NOT IN (4, 5)
              AND h.VOIDSTTS = 0
            GROUP BY l.ITEMNMBR
        """
        params: tuple = tuple(filtered)
        result: dict[str, datetime.date | None] = {item: None for item in filtered}
    else:
        query = """
            SELECT RTRIM(l.ITEMNMBR) AS item,
                   MAX(CAST(h.RECEIPTDATE AS DATE)) AS last_rcpt
            FROM POP30310 l
            JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
            WHERE h.POPTYPE NOT IN (4, 5)
              AND h.VOIDSTTS = 0
            GROUP BY l.ITEMNMBR
        """
        params = ()
        result = {}
    try:
        cursor.execute(query, params)
        for r in cursor.fetchall():
            result[r.item] = r.last_rcpt
    except pyodbc.Error as err:
        LOGGER.warning("fetch_last_vendor_receipt_map failed: %s", err)
    return result


def fetch_obsolete_item_set(cursor: pyodbc.Cursor) -> set[str]:
    """
    Return items considered obsolete: GP Discontinued (ITEMTYPE = 2) or whose
    description shouts "DO NOT USE" / "OBSOLETE". Used to prune the kanban
    pipeline before BOM explosion so obsolete items never become buy candidates
    and never propagate through parent BOMs.
    """
    query = """
        SELECT RTRIM(ITEMNMBR) AS item
        FROM IV00101
        WHERE ITEMTYPE = 2
           OR UPPER(ITEMDESC) LIKE '%DO NOT USE%'
           OR UPPER(ITEMDESC) LIKE '%OBSOLETE%'
    """
    try:
        cursor.execute(query)
        return {(r.item or "").strip() for r in cursor.fetchall() if r.item}
    except pyodbc.Error as err:
        LOGGER.warning("fetch_obsolete_item_set failed: %s", err)
        return set()


def fetch_dilution_proxy_map(
    cursor: pyodbc.Cursor,
) -> dict[str, list[tuple[str, float]]]:
    """
    Derive a map of *leaf item* → list of ``(parent_item, conversion_factor)``
    entries. A "dilution proxy" is a BM010115 parent whose recipe is dominated
    by a single non-water component — e.g. ``NO3FE`` is the parent, its BOM is
    ~0.96 LBS ``REC-NO3FE`` + ~0.04 LBS ``H2OCOLD``, so after the leaf-only
    filter ``REC-NO3FE`` becomes the buy-list item but the real on-hand stock
    sits under ``NO3FE``.

    For each such relationship we record::

        { 'REC-NO3FE': [('NO3FE', 0.96)] }

    meaning "for every 1 unit of NO3FE on hand, credit 0.96 units toward
    REC-NO3FE's supply". ``conversion_factor`` is the BM010115 Design_Qty of
    the leaf per unit of the parent, so ``parent_on_hand * factor`` is the
    leaf-equivalent stock.

    Heuristic for "dilution parent":
      - parent appears in BM010115 as PPN_I
      - parent has exactly one non-water component (H2OCOLD is ignored)
      - the non-water component accounts for >= 80% of the recipe mass
    """
    query = """
        WITH totals AS (
            SELECT RTRIM(PPN_I) AS parent,
                   RTRIM(CPN_I) AS comp,
                   AVG(CAST(QUANTITY_I AS FLOAT)) AS qty
            FROM BM010115
            GROUP BY PPN_I, CPN_I
        ),
        non_water AS (
            SELECT parent, comp, qty FROM totals WHERE comp <> 'H2OCOLD'
        ),
        parent_stats AS (
            SELECT parent,
                   COUNT(*) AS non_water_count,
                   SUM(qty) AS non_water_total
            FROM non_water
            GROUP BY parent
        ),
        dominant AS (
            -- A dilution is a recipe where (a) exactly one non-water component
            -- is consumed, and (b) that component is between 80% and 100% of
            -- one unit of parent — i.e. parent is mostly-leaf-plus-water. A
            -- factor > 1.0 indicates a mixing ratio ("parent uses N LBS of
            -- leaf"), which is NOT reversible on-hand credit.
            SELECT nw.parent, nw.comp, nw.qty, ps.non_water_count
            FROM non_water nw
            JOIN parent_stats ps ON nw.parent = ps.parent
            WHERE ps.non_water_count = 1
              AND nw.qty >= 0.80
              AND nw.qty <= 1.00
        )
        SELECT parent, comp, qty FROM dominant
    """
    out: dict[str, list[tuple[str, float]]] = {}
    try:
        cursor.execute(query)
        for r in cursor.fetchall():
            parent = (r.parent or "").strip()
            leaf = (r.comp or "").strip()
            if not parent or not leaf:
                continue
            factor = float(r.qty or 0)
            if factor <= 0:
                continue
            out.setdefault(leaf, []).append((parent, factor))
    except pyodbc.Error as err:
        LOGGER.warning("fetch_dilution_proxy_map failed: %s", err)
    return out


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
        on_hand = {(row.ITEMNMBR or "").strip(): decimal_or_zero(row.OnHand) for row in rows}
        return on_hand, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch on-hand quantities: %s", err)
        return {}, sql_preview


def fetch_open_po_supply(cursor: pyodbc.Cursor, items: list[str], location: str = PRIMARY_LOCATION) -> tuple[dict[str, Decimal], str]:
    """Return open purchase order quantities for items, filtered to a location (default MAIN).

    Includes PO line statuses 1 (New), 2 (Released/Change Order), and 3
    (Received but not fully closed). Subtracts cancelled quantity.
    """
    if not items:
        return {}, ""
    filtered = [itm for itm in items if itm]
    if not filtered:
        return {}, ""

    placeholders = ", ".join("?" for _ in filtered)
    params = [*filtered, location] if location else filtered
    location_clause = " AND LOCNCODE = ?" if location else ""
    query = f"""
        SELECT ITEMNMBR, SUM(QTYORDER - QTYCANCE) AS OpenPOQty
        FROM POP10110
        WHERE ITEMNMBR IN ({placeholders})
          AND POLNESTA IN (1, 2, 3){location_clause}
        GROUP BY ITEMNMBR
    """
    sql_preview = format_sql_preview(query, params)
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        supply = {(row.ITEMNMBR or "").strip(): decimal_or_zero(row.OpenPOQty) for row in rows}
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
          AND b.BOMCAT_I = 1 AND LEN(b.BOMNAME_I) = 0  -- active recipe only (exclude archived/named batch recipes)
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
        WITH BOM_CTE (TopLevelParent, ParentItem, ComponentItem, Quantity, Depth, Path) AS (
            SELECT
                PPN_I AS TopLevelParent,
                PPN_I AS ParentItem,
                CPN_I AS ComponentItem,
                CAST(QUANTITY_I AS DECIMAL(38, 19)) AS Quantity,
                1 AS Depth,
                CAST('|' + RTRIM(PPN_I) + '|' + RTRIM(CPN_I) + '|' AS VARCHAR(4000)) AS Path
            FROM BM010115
            WHERE PPN_I = ?
              AND BOMCAT_I = 1 AND LEN(BOMNAME_I) = 0  -- active recipe only

            UNION ALL

            SELECT
                cte.TopLevelParent,
                b.PPN_I AS ParentItem,
                b.CPN_I AS ComponentItem,
                CAST(cte.Quantity * b.QUANTITY_I AS DECIMAL(38, 19)) AS Quantity,
                cte.Depth + 1,
                CAST(cte.Path + RTRIM(b.CPN_I) + '|' AS VARCHAR(4000)) AS Path
            FROM BM010115 b
            INNER JOIN BOM_CTE cte ON b.PPN_I = cte.ComponentItem
            WHERE cte.Depth < 20
              AND cte.Path NOT LIKE '%|' + RTRIM(b.CPN_I) + '|%'
              AND b.BOMCAT_I = 1 AND LEN(b.BOMNAME_I) = 0  -- active recipe only
              -- treat raw materials as hard leaves: don't recurse into their REC-/dilution recipes
              AND NOT EXISTS (
                  SELECT 1 FROM IV00101 i
                  WHERE RTRIM(i.ITEMNMBR) = RTRIM(cte.ComponentItem)
                    AND i.ITMCLSCD LIKE 'RAWMAT%'
              )
        )
        SELECT
            ComponentItem AS RawMaterial,
            SUM(Quantity) AS Design_Qty
        FROM BOM_CTE
        -- a row is a leaf if it is a raw material (hard leaf) or it is not an active parent
        WHERE ComponentItem IN (SELECT ITEMNMBR FROM IV00101 WHERE ITMCLSCD LIKE 'RAWMAT%')
           OR ComponentItem NOT IN (
                  SELECT DISTINCT PPN_I FROM BM010115 WHERE BOMCAT_I = 1 AND LEN(BOMNAME_I) = 0
              )
        GROUP BY ComponentItem
        OPTION (MAXRECURSION 50)
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


def fetch_parent_items_for_component(cursor: pyodbc.Cursor, component_item: str) -> tuple[list, str]:
    """
    Fetch all parent items that use the given component (Reverse BOM).
    Useful for identifying derived demand for raw materials.
    """
    if not component_item:
        return [], ""

    query = """
        SELECT 
            b.PPN_I AS ParentItem, 
            i.ITEMDESC AS ParentDescription,
            b.QUANTITY_I AS QtyPerParent,
            b.UOFM AS UofM,
            ISNULL(s.TotalSales, 0) as Volume
        FROM BM010115 b
        JOIN IV00101 i ON b.PPN_I = i.ITEMNMBR
        OUTER APPLY (
            SELECT SUM(d.QTYFULFI) as TotalSales
            FROM SOP30300 d
            JOIN SOP30200 h ON d.SOPNUMBE = h.SOPNUMBE
            WHERE d.ITEMNMBR = b.PPN_I
              AND h.DOCDATE >= DATEADD(year, -1, GETDATE())
              AND h.SOPTYPE = 3 -- Invoice
        ) s
        WHERE b.CPN_I = ?
        ORDER BY s.TotalSales DESC, b.PPN_I
    """
    sql_preview = format_sql_preview(query, [component_item])
    try:
        cursor.execute(query, component_item)
        rows = cursor.fetchall()
        return rows, sql_preview
    except pyodbc.Error as err:
        LOGGER.warning("Failed to fetch parents for component %s: %s", component_item, err)
        return [], sql_preview
