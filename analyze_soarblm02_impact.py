"""
Analyze how increasing SOARBLM02 sales by 1000 gallons affects raw material order points.

Steps:
1. Get BOM components for SOARBLM02 (or SOARBLM00 parent)
2. Calculate additional raw material needs for 1000 gallons
3. Compare against current inventory, reorder points, and avg daily usage
4. Show impact on when each raw material needs reordering
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db_pool import get_cursor
from reorder_math import (
    calculate_reorder_point,
    calculate_days_of_coverage,
    calculate_must_order_by,
    get_urgency_level,
)
from constants import PRIMARY_LOCATION
import datetime

INCREASE_GALLONS = 1000


def get_bom_components(cursor):
    """Get BOM for SOARBLM02, trying BM010115 first, then BM00111, checking both 02 and 00 parents."""
    # Try manufacturing BOM (BM010115) first - preferred
    for parent in ("SOARBLM02", "SOARBLM00"):
        cursor.execute("""
            SELECT b.PPN_I AS ParentItem, 
                   b.CPN_I AS ComponentItem, 
                   ic.ITEMDESC AS ComponentDescription,
                   ic.ITMCLSCD AS ComponentClass,
                   SUM(b.QUANTITY_I) AS QtyPerParent, 
                   MAX(b.UOFM) AS ComponentUofM
            FROM BM010115 b
            LEFT JOIN IV00101 ic ON ic.ITEMNMBR = b.CPN_I
            WHERE b.PPN_I = ?
            GROUP BY b.PPN_I, b.CPN_I, ic.ITEMDESC, ic.ITMCLSCD
            ORDER BY b.CPN_I
        """, (parent,))
        rows = cursor.fetchall()
        if rows:
            print(f"[BOM Source] Manufacturing BOM (BM010115) for parent: {parent}")
            return rows, parent

    # Fallback to standard BOM (BM00111)
    for parent in ("SOARBLM02", "SOARBLM00"):
        cursor.execute("""
            SELECT b.ITEMNMBR AS ParentItem,
                   b.CMPTITNM AS ComponentItem,
                   ic.ITEMDESC AS ComponentDescription,
                   ic.ITMCLSCD AS ComponentClass,
                   SUM(b.Design_Qty) AS QtyPerParent,
                   '' AS ComponentUofM
            FROM BM00111 b
            LEFT JOIN IV00101 ic ON ic.ITEMNMBR = b.CMPTITNM
            WHERE b.ITEMNMBR = ?
            GROUP BY b.ITEMNMBR, b.CMPTITNM, ic.ITEMDESC, ic.ITMCLSCD
            ORDER BY b.CMPTITNM
        """, (parent,))
        rows = cursor.fetchall()
        if rows:
            print(f"[BOM Source] Standard BOM (BM00111) for parent: {parent}")
            return rows, parent

    return [], None


def get_inventory_and_reorder_data(cursor, component_items):
    """Get current inventory, on-order, reorder points, and usage for components."""
    if not component_items:
        return {}

    placeholders = ", ".join("?" for _ in component_items)

    # Current inventory and GP order points
    cursor.execute(f"""
        SELECT 
            i.ITEMNMBR,
            i.ITEMDESC,
            i.ITMCLSCD,
            COALESCE(loc.QTYONHND, 0) AS QtyOnHand,
            COALESCE(loc.QTYONORD, 0) AS QtyOnOrder,
            COALESCE(loc.ORDRPNTQTY, 0) AS OrderPointQty,
            COALESCE(NULLIF(loc.ORDRUPTOLVL, 0), loc.ORDRPNTQTY * 2, 0) AS OrderUpToQty
        FROM IV00101 i
        LEFT JOIN IV00102 loc ON i.ITEMNMBR = loc.ITEMNMBR AND loc.LOCNCODE = ?
        WHERE i.ITEMNMBR IN ({placeholders})
    """, [PRIMARY_LOCATION, *component_items])
    inv_rows = cursor.fetchall()

    # Usage over last 90 days
    cursor.execute(f"""
        SELECT 
            ITEMNMBR,
            SUM(ABS(TRXQTY)) / 90.0 AS AvgDailyUsage
        FROM IV30300
        WHERE ITEMNMBR IN ({placeholders})
          AND TRXLOCTN = ?
          AND TRXQTY < 0
          AND DOCDATE >= DATEADD(day, -90, GETDATE())
        GROUP BY ITEMNMBR
    """, [*component_items, PRIMARY_LOCATION])
    usage_rows = cursor.fetchall()
    usage_map = {r.ITEMNMBR.strip(): float(r.AvgDailyUsage or 0) for r in usage_rows}

    # Lead times (historical)
    cursor.execute(f"""
        SELECT 
            r_line.ITEMNMBR,
            AVG(DATEDIFF(day, po_head.DOCDATE, r_head.RECEIPTDATE)) AS AvgLeadTime,
            COUNT(*) AS SampleCount
        FROM POP30310 r_line
        JOIN POP30300 r_head ON r_line.POPRCTNM = r_head.POPRCTNM
        JOIN POP30100 po_head ON r_line.PONUMBER = po_head.PONUMBER
        WHERE r_line.ITEMNMBR IN ({placeholders})
          AND r_head.RECEIPTDATE >= DATEADD(year, -2, GETDATE())
          AND r_line.PONUMBER <> ''
          AND r_head.RECEIPTDATE >= po_head.DOCDATE
        GROUP BY r_line.ITEMNMBR
    """, component_items)
    lt_rows = cursor.fetchall()
    lt_map = {r.ITEMNMBR.strip(): (float(r.AvgLeadTime or 14), int(r.SampleCount or 0)) for r in lt_rows}

    result = {}
    for row in inv_rows:
        item = row.ITEMNMBR.strip()
        result[item] = {
            "description": (row.ITEMDESC or "").strip(),
            "class": (row.ITMCLSCD or "").strip(),
            "qty_on_hand": float(row.QtyOnHand or 0),
            "qty_on_order": float(row.QtyOnOrder or 0),
            "gp_order_point": float(row.OrderPointQty or 0),
            "gp_order_up_to": float(row.OrderUpToQty or 0),
            "avg_daily_usage": usage_map.get(item, 0.0),
            "lead_time_days": int(lt_map.get(item, (14, 0))[0]),
            "lead_time_samples": lt_map.get(item, (14, 0))[1],
        }
    return result


def main():
    today = datetime.date.today()
    safety_days = 7

    with get_cursor() as cursor:
        # Step 1: Get BOM
        bom_rows, parent_used = get_bom_components(cursor)
        if not bom_rows:
            print("ERROR: No BOM found for SOARBLM02 or SOARBLM00")
            return

        print(f"\n{'='*100}")
        print(f"IMPACT ANALYSIS: Increasing SOARBLM02 Sales by {INCREASE_GALLONS:,} Gallons")
        print(f"{'='*100}")

        # Step 2: Calculate additional raw material needs
        print(f"\n--- BOM Components for {parent_used} ---")
        print(f"{'Component':<22} {'Description':<45} {'Qty/Gal':>10} {'Extra Need':>12} {'UoM':<6}")
        print("-" * 100)

        component_items = []
        extra_needs = {}
        for row in bom_rows:
            comp = row.ComponentItem.strip()
            desc = (row.ComponentDescription or "").strip()[:44]
            qty_per = float(row.QtyPerParent or 0)
            extra = qty_per * INCREASE_GALLONS
            uom = (row.ComponentUofM or "").strip()
            component_items.append(comp)
            extra_needs[comp] = extra
            print(f"{comp:<22} {desc:<45} {qty_per:>10.4f} {extra:>12.2f} {uom:<6}")

        # Step 3: Get current inventory/reorder data
        inv_data = get_inventory_and_reorder_data(cursor, component_items)

        # Step 4: Impact analysis
        print(f"\n{'='*100}")
        print("RAW MATERIAL REORDER IMPACT")
        print(f"{'='*100}")
        print(f"{'Component':<22} {'On Hand':>10} {'On Order':>10} {'GP ROP':>10} {'Curr Usage':>12} "
              f"{'New Usage':>12} {'New ROP':>10} {'ROP Chg':>8} {'Days Cover':>10} {'Urgency':<10}")
        print("-" * 130)

        impacts = []
        for comp in component_items:
            extra = extra_needs[comp]
            data = inv_data.get(comp)
            if not data:
                print(f"{comp:<22} ** No inventory data found **")
                continue

            # Current state
            qty_on_hand = data["qty_on_hand"]
            qty_on_order = data["qty_on_order"]
            qty_available = qty_on_hand + qty_on_order
            gp_rop = data["gp_order_point"]
            current_daily_usage = data["avg_daily_usage"]
            lead_time = data["lead_time_days"]

            # New state: add 1000 gal demand spread over ~30 days as additional daily consumption
            additional_daily = extra / 30.0  # spread 1000-gal batch over ~30 days
            new_daily_usage = current_daily_usage + additional_daily

            # Calculate current and new ROPs
            current_rop = calculate_reorder_point(current_daily_usage, lead_time, safety_days)
            new_rop = calculate_reorder_point(new_daily_usage, lead_time, safety_days)
            rop_change = new_rop - current_rop

            # Days of coverage with new usage rate
            new_days_coverage = qty_available / new_daily_usage if new_daily_usage > 0 else 999

            # Urgency with new ROP
            must_order = calculate_must_order_by(new_days_coverage, lead_time, safety_days, today)
            urgency = get_urgency_level(must_order, today)

            impacts.append({
                "component": comp,
                "description": data["description"],
                "extra_need": extra,
                "qty_on_hand": qty_on_hand,
                "qty_on_order": qty_on_order,
                "gp_rop": gp_rop,
                "current_daily": current_daily_usage,
                "new_daily": new_daily_usage,
                "current_rop": current_rop,
                "new_rop": new_rop,
                "rop_change": rop_change,
                "new_days_coverage": new_days_coverage,
                "urgency": urgency,
                "must_order_by": must_order,
            })

            print(f"{comp:<22} {qty_on_hand:>10.1f} {qty_on_order:>10.1f} {gp_rop:>10.1f} "
                  f"{current_daily_usage:>12.2f} {new_daily_usage:>12.2f} {new_rop:>10.1f} "
                  f"{rop_change:>+8.1f} {new_days_coverage:>10.1f} {urgency:<10}")

        # Summary
        print(f"\n{'='*100}")
        print("SUMMARY")
        print(f"{'='*100}")
        critical = [i for i in impacts if i["urgency"] == "Critical"]
        soon = [i for i in impacts if i["urgency"] == "Soon"]
        ok = [i for i in impacts if i["urgency"] == "OK"]

        print(f"\nTotal BOM components: {len(impacts)}")
        print(f"  🔴 Critical (need to order NOW): {len(critical)}")
        for i in critical:
            print(f"     - {i['component']}: {i['description'][:50]}")
            print(f"       Extra need: {i['extra_need']:,.1f} | On Hand: {i['qty_on_hand']:,.1f} | "
                  f"New ROP: {i['new_rop']:,.1f} (was {i['current_rop']:,.1f}, +{i['rop_change']:,.1f})")

        print(f"  🟡 Soon (order within 7 days): {len(soon)}")
        for i in soon:
            print(f"     - {i['component']}: {i['description'][:50]} | Must order by: {i['must_order_by']}")

        print(f"  🟢 OK: {len(ok)}")
        for i in ok:
            print(f"     - {i['component']}: {i['description'][:50]} | {i['new_days_coverage']:.0f} days coverage")


if __name__ == "__main__":
    main()
