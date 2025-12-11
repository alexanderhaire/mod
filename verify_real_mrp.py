import datetime
import logging
import pyodbc
from handlers import handle_mrp_planning
from secrets_loader import build_connection_string

def test_real_mrp_logic():
    print("\n--- Testing Real MRP Logic ---")
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    today = datetime.date.today()
    prompt = "what should we buy in december 2025"
    
    print(f"Running handle_mrp_planning with prompt: '{prompt}'")
    result = handle_mrp_planning(cursor, prompt, today)
    
    if result:
        print(f"Result keys: {result.keys()}")
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Summary: {result['insights']['summary']}")
        print(f"Row Count: {len(result['data'])}")
        if result['data']:
            print(f"First Row: {result['data'][0]}")
    else:
        print("MRP Handler returned None (Failed to return any data)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_real_mrp_logic()
