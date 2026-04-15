"""
Synthetic BOM reconstruction from MO picklist history.

Used by the Kanban reorder pipeline when an item is made in-house but has no
entry in BM010115 as a parent — we reverse-engineer a per-unit component recipe
from the last 12 months of manufacturing orders in MOP1016.

MOP1016 is the MO Pick Document table. For a given MANUFACTUREORDER_I it
contains one row per item touched by the MO — both the end item (with a real
DATERECD when the MO was received) and each component (with DATERECD matching
the issue date). QTYRECVD stores the qty issued for components and the qty
received for the end item. We identify the end item by asking the caller
(we're starting from "synthesize the BOM for X", so X is known) and treat
every OTHER item on the same MO as a component.

Returned rows expose a ``.RawMaterial`` and ``.Design_Qty`` attribute so the
existing BOM-explosion loop in kanban_reorder.py can consume them with no
special-casing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pyodbc

LOGGER = logging.getLogger(__name__)

# Minimum number of MOs we need to trust a synthetic BOM. Anything below this
# is still returned (so the buy list isn't empty) but gets flagged in the UI.
MIN_CONFIDENT_MOS = 3

# How far back to look for MO history.
LOOKBACK_MONTHS = 12


@dataclass(frozen=True)
class SyntheticBomRow:
    """Mimics the shape of BM010115 rows consumed by explode_demand_to_raw_materials."""
    RawMaterial: str
    Design_Qty: float
    n_mos_observed: int  # extra — used for confidence display


def reconstruct_synthetic_bom(
    cursor: pyodbc.Cursor,
    end_item: str,
    lookback_months: int = LOOKBACK_MONTHS,
) -> list[SyntheticBomRow]:
    """
    Reverse-engineer a per-unit component recipe for ``end_item`` using the
    last ``lookback_months`` of MO history in MOP1016.

    Returns an empty list if we have no MO evidence at all — the caller should
    treat that as "BOM missing, surface for manual review".
    """
    end_item = (end_item or "").strip()
    if not end_item:
        return []

    # Step 1: find MOs that PRODUCED this end item recently.
    # Filter DATERECD > '1900-01-01' to skip the "pending receipt" marker rows
    # GP inserts before the MO is closed out.
    try:
        cursor.execute(
            """
            SELECT RTRIM(MANUFACTUREORDER_I) AS mo, SUM(QTYRECVD) AS end_qty
            FROM MOP1016
            WHERE RTRIM(ITEMNMBR) = ?
              AND DATERECD >= DATEADD(month, ?, GETDATE())
              AND DATERECD > '1900-01-01'
              AND QTYRECVD > 0
            GROUP BY MANUFACTUREORDER_I
            """,
            end_item,
            -lookback_months,
        )
        mo_rows = cursor.fetchall()
    except pyodbc.Error as err:
        LOGGER.warning("synthetic BOM step 1 failed for %s: %s", end_item, err)
        return []

    if not mo_rows:
        return []

    mo_to_end_qty: dict[str, float] = {r.mo: float(r.end_qty or 0) for r in mo_rows}
    mo_to_end_qty = {mo: q for mo, q in mo_to_end_qty.items() if q > 0}
    if not mo_to_end_qty:
        return []

    total_end = sum(mo_to_end_qty.values())
    if total_end <= 0:
        return []

    # Step 2: sum component issues for those MOs (excluding the end item itself).
    mo_list = list(mo_to_end_qty.keys())
    placeholders = ", ".join("?" for _ in mo_list)
    try:
        cursor.execute(
            f"""
            SELECT RTRIM(ITEMNMBR) AS comp_item,
                   SUM(QTYRECVD)   AS total_qty
            FROM MOP1016
            WHERE MANUFACTUREORDER_I IN ({placeholders})
              AND RTRIM(ITEMNMBR) != ?
              AND QTYRECVD > 0
              AND DATERECD > '1900-01-01'
            GROUP BY ITEMNMBR
            """,
            *mo_list,
            end_item,
        )
        component_rows = cursor.fetchall()
    except pyodbc.Error as err:
        LOGGER.warning("synthetic BOM step 2 failed for %s: %s", end_item, err)
        return []

    n_mos = len(mo_list)
    out: list[SyntheticBomRow] = []
    for r in component_rows:
        comp = (r.comp_item or "").strip()
        if not comp:
            continue
        total_comp = float(r.total_qty or 0)
        if total_comp <= 0:
            continue
        per_unit = total_comp / total_end
        out.append(SyntheticBomRow(
            RawMaterial=comp,
            Design_Qty=per_unit,
            n_mos_observed=n_mos,
        ))

    return out


def is_low_confidence(synth: Iterable[SyntheticBomRow]) -> bool:
    """Return True if the synthetic BOM is based on fewer than MIN_CONFIDENT_MOS MOs."""
    rows = list(synth)
    if not rows:
        return True
    return rows[0].n_mos_observed < MIN_CONFIDENT_MOS
