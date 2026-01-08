"""
Database optimization script for ML Procurement queries.

Creates indexes to improve query performance.
Run this once against your GP database.
"""

import pyodbc
from secrets_loader import build_connection_string

# Index creation statements for GP tables
INDEX_STATEMENTS = [
    # POP30310 - Receipt Lines (most queried)
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_POP30310_ITEMNMBR_EXTDCOST')
    CREATE NONCLUSTERED INDEX IX_POP30310_ITEMNMBR_EXTDCOST 
    ON POP30310 (ITEMNMBR, EXTDCOST) 
    INCLUDE (POPRCTNM, UNITCOST)
    """,
    
    # POP30300 - Receipt Headers
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_POP30300_RECEIPTDATE')
    CREATE NONCLUSTERED INDEX IX_POP30300_RECEIPTDATE 
    ON POP30300 (RECEIPTDATE) 
    INCLUDE (POPRCTNM, VENDORID)
    """,
    
    # IV30300 - Inventory Transactions (for usage history)
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_IV30300_ITEMNMBR_DOCTYPE')
    CREATE NONCLUSTERED INDEX IX_IV30300_ITEMNMBR_DOCTYPE 
    ON IV30300 (ITEMNMBR, DOCTYPE, DOCDATE) 
    INCLUDE (TRXQTY)
    """,
    
    # IV00102 - Item Location Quantities
    """
    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_IV00102_ITEMNMBR_LOCNCODE')
    CREATE NONCLUSTERED INDEX IX_IV00102_ITEMNMBR_LOCNCODE 
    ON IV00102 (ITEMNMBR, LOCNCODE) 
    INCLUDE (QTYONHND, QTYONORD, ORDRPNTQTY)
    """,
]


def apply_indexes(dry_run: bool = True):
    """Apply database indexes to improve ML query performance."""
    conn_str, server, db, _ = build_connection_string()
    print(f"Connecting to {server}/{db}...")
    
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        print("[OK] Connected\n")
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return
    
    print("=" * 60)
    print(f"{'DRY RUN' if dry_run else 'APPLYING'} DATABASE INDEXES")
    print("=" * 60)
    
    for i, stmt in enumerate(INDEX_STATEMENTS, 1):
        # Extract index name from statement
        idx_name = "Unknown"
        if "CREATE" in stmt in stmt:
            parts = stmt.split("INDEX")
            if len(parts) > 1:
                idx_name = parts[1].split()[0].strip()
        
        print(f"\n[{i}/{len(INDEX_STATEMENTS)}] {idx_name}")
        
        if dry_run:
            print("  [DRY RUN] Would execute:")
            print(f"  {stmt[:100]}...")
        else:
            try:
                cursor.execute(stmt)
                conn.commit()
                print("  [OK] Index created/verified")
            except Exception as e:
                print(f"  [WARN] {e}")
    
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE. Run with --apply to actually create indexes.")
    else:
        print("INDEX CREATION COMPLETE")
    print("=" * 60)
    
    conn.close()


if __name__ == "__main__":
    import sys
    apply_flag = "--apply" in sys.argv
    apply_indexes(dry_run=not apply_flag)
