
import pyodbc
from market_insights import get_priority_raw_materials, calculate_inventory_runway
from database import get_db_connection

def find_critical():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("Scanning for critical items...")
        df = get_priority_raw_materials(cursor, limit=50)
        
        target = None
        for _, row in df.iterrows():
            item = row['ITEMNMBR'].strip()
            runway = calculate_inventory_runway(cursor, item)
            days = runway.get('runway_days', 999)
            
            if days < 30:
                print(f"FOUND TARGET: {item} (Runway: {days:.1f} days)")
                target = item
                break
        
        if not target:
            print("No critical items found. Defaulting to CHEACETIC for test.")
            target = "CHEACETIC"
            
        return target
    except Exception as e:
        print(f"Error: {e}")
        return "CHEACETIC"

if __name__ == "__main__":
    find_critical()
