import datetime
import logging
import pyodbc
from handlers import handle_what_if_analysis
from secrets_loader import build_connection_string

def test_what_if():
    print("\n--- Testing What-If Analysis ---")
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    today = datetime.date.today()
    
    # Test 1: Specific item with percentage
    prompt1 = "If SOARBLM02 demand increases by 3%, what raw materials should we buy?"
    print(f"\nTest 1: '{prompt1}'")
    result1 = handle_what_if_analysis(cursor, prompt1, today)
    if result1:
        if "error" in result1:
            print(f"ERROR: {result1['error']}")
        else:
            print(f"Summary: {result1['insights']['summary']}")
            print(f"Row Count: {len(result1['data'])}")
            if result1['data']:
                print(f"First Row: {result1['data'][0]}")
    else:
        print("Handler returned None")
    
    # Test 2: General growth scenario
    prompt2 = "What if demand grows 5% across all products?"
    print(f"\nTest 2: '{prompt2}'")
    result2 = handle_what_if_analysis(cursor, prompt2, today)
    if result2:
        if "error" in result2:
            print(f"ERROR: {result2['error']}")
        else:
            print(f"Summary: {result2['insights']['summary']}")
            print(f"Row Count: {len(result2['data'])}")
    else:
        print("Handler returned None")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_what_if()
