import datetime
import logging
import pyodbc
from handlers import handle_scenario_comparison
from secrets_loader import build_connection_string

def test_scenario_comparison():
    print("\n--- Testing Scenario Comparison ---")
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    today = datetime.date.today()
    
    # Test 1: Multiple percentage comparison
    prompt1 = "Compare 5% vs 10% vs 15% growth scenarios for SOARBLM02"
    print(f"\nTest 1: '{prompt1}'")
    result1 = handle_scenario_comparison(cursor, prompt1, today)
    if result1:
        if "error" in result1:
            print(f"ERROR: {result1['error']}")
        else:
            print(f"Summary:\n{result1['insights']['summary']}")
            print(f"\nRow Count: {len(result1['data'])}")
            if result1['data']:
                print(f"First Row: {result1['data'][0]}")
    else:
        print("Handler returned None")
    
    # Test 2: General comparison
    prompt2 = "Compare 3% versus 7% increase scenarios"
    print(f"\nTest 2: '{prompt2}'")
    result2 = handle_scenario_comparison(cursor, prompt2, today)
    if result2:
        print(f"Summary:\n{result2['insights']['summary']}")
        print(f"Row Count: {len(result2['data'])}")
    else:
        print("Handler returned None")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_scenario_comparison()
