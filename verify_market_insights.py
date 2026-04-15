import pyodbc
from secrets_loader import build_connection_string
from market_insights import get_inventory_distribution
from constants import LOGGER

def test_market_insights():
    print("Testing get_inventory_distribution...")
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Test Raw Materials
        print("\n--- Raw Material Distribution ---")
        distribution = get_inventory_distribution(cursor, segment="Raw Material")
        for d in distribution:
            print(f"Category: {d['Category']}, Value: ${d['Value']:,.2f}, Items: {d['ItemCount']}")
            
        # Test Finished Goods
        print("\n--- Finished Good Distribution ---")
        distribution_fg = get_inventory_distribution(cursor, segment="Finished Good")
        for d in distribution_fg:
            print(f"Category: {d['Category']}, Value: ${d['Value']:,.2f}, Items: {d['ItemCount']}")
            
        total_val = sum(d['Value'] for d in distribution) + sum(d['Value'] for d in distribution_fg)
        print(f"\nTotal Portfolio Value (RM + FG): ${total_val:,.2f}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        
if __name__ == "__main__":
    test_market_insights()
