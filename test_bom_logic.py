import pyodbc
from db_pool import get_connection
from inventory_queries import fetch_parent_items_for_component

def test_bom_logic():
    print("--- Testing BOM Logic (Volume-Based Sorting) ---")
    
    # Pick a raw material known to be in many things (e.g. Flour substitute or common chemical)
    # Using 'NPK3011' or similar common item if known, or just find one
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Find a component with many parents
        print("Finding a component with multiple parents...")
        cursor.execute("""
            SELECT TOP 1 CPN_I, COUNT(*) as ParentCount 
            FROM BM010115 
            GROUP BY CPN_I 
            HAVING COUNT(*) > 3 
            ORDER BY ParentCount DESC
        """)
        row = cursor.fetchone()
        
        if not row:
            print("No suitable component found for testing.")
            return

        component = row.CPN_I.strip()
        count = row.ParentCount
        print(f"Testing with component: {component} (Used in {count} products)")
        
        # Test the query
        print("Fetching parents (Should be ordered by Volume)...")
        parents, sql = fetch_parent_items_for_component(cursor, component)
        
        print(f"Found {len(parents)} parents.")
        for i, p in enumerate(parents[:5]):
            print(f"{i+1}. {p.ParentItem} - {p.ParentDescription} (Vol: {p.Volume})")
            
        # Verify order
        volumes = [p.Volume for p in parents]
        if volumes == sorted(volumes, reverse=True):
            print("\nSUCCESS: Parents are sorted by volume descending!")
        else:
            print("\nFAILURE: Parents are NOT sorted by volume.")

if __name__ == "__main__":
    test_bom_logic()
