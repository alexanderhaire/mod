"""Integration tests for the recursive BOM explosion (read-only, hits live Dynamics GP).

These lock in the fix for the recipe-version-summing bug. BM010115 stores the active
manufacturing BOM (BOMCAT_I=1, blank BOMNAME_I) alongside archived/named batch recipes
(BOMCAT_I=4, and named BOMCAT_I=1 rows like 'TALL F JUG'). The explosion must:
  1. use ONLY the active recipe (not SUM every stored version), and
  2. treat raw materials (ITMCLSCD LIKE 'RAWMAT%') as hard leaves, rather than recursing
     into their REC-/dilution sub-recipes.

If the GP database is unreachable, these tests skip rather than fail.
"""
from __future__ import annotations

import pytest

try:
    from db_pool import get_connection
    from inventory_queries import (
        fetch_recursive_bom_for_item,
        fetch_mfg_bom_grouped_by_component,
    )
except Exception as exc:  # pragma: no cover - import/env issues
    pytest.skip(f"app modules unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def cursor():
    try:
        cm = get_connection()
        conn = cm.__enter__()
    except Exception as exc:  # pragma: no cover - no DB in this environment
        pytest.skip(f"GP database unavailable: {exc}")
    try:
        yield conn.cursor()
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def _explode(cursor, parent):
    rows, _ = fetch_recursive_bom_for_item(cursor, parent)
    return {r.RawMaterial.strip(): float(r.Design_Qty) for r in rows}


def _active_qty(cursor, parent, component):
    """The per-unit quantity from ONLY the active recipe row (BOMCAT_I=1, blank name)."""
    cursor.execute(
        "SELECT SUM(QUANTITY_I) FROM BM010115 "
        "WHERE RTRIM(PPN_I)=? AND RTRIM(CPN_I)=? AND BOMCAT_I=1 AND LEN(BOMNAME_I)=0",
        parent, component,
    )
    val = cursor.fetchone()[0]
    return float(val) if val is not None else None


def test_recursive_explosion_uses_active_recipe_only(cursor):
    # TRITOPS00 stores 13 SO4BORIC recipe rows (summing to ~48.03); only the single
    # active recipe (~3.51) is the real consumption.
    expected = _active_qty(cursor, "TRITOPS00", "SO4BORIC")
    assert expected is not None and expected < 10, "expected active SO4BORIC recipe ~3.5 lb/gal"
    exploded = _explode(cursor, "TRITOPS00")
    assert "SO4BORIC" in exploded
    assert exploded["SO4BORIC"] == pytest.approx(expected, rel=0.001)


def test_recursive_explosion_treats_raw_materials_as_leaves(cursor):
    # GROMN02 lists NO3MN directly. The explosion must return NO3MN (the purchasable raw),
    # NOT recurse into its dilution code REC-NO3MN.
    expected = _active_qty(cursor, "GROMN02", "NO3MN")
    assert expected is not None and expected < 30, "expected active NO3MN recipe ~13 lb/unit"
    exploded = _explode(cursor, "GROMN02")
    assert "NO3MN" in exploded
    assert exploded["NO3MN"] == pytest.approx(expected, rel=0.001)
    assert "REC-NO3MN" not in exploded


def test_grouped_bom_uses_active_recipe_only(cursor):
    # The single-level grouped BOM (used by the chat handler) has the same SUM bug.
    expected = _active_qty(cursor, "TRITOPS00", "SO4BORIC")
    rows, _ = fetch_mfg_bom_grouped_by_component(cursor, "TRITOPS00")
    by_comp = {r.ComponentItem.strip(): float(r.QtyPerParent) for r in rows}
    assert "SO4BORIC" in by_comp
    assert by_comp["SO4BORIC"] == pytest.approx(expected, rel=0.001)
