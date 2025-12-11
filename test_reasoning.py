import datetime
import logging
import pyodbc
from reasoning_coordinator import execute_reasoning_chain, should_use_reasoning_coordinator
from secrets_loader import build_connection_string

def test_reasoning_coordinator():
    print("\n--- Testing Reasoning Coordinator ---")
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        return

    today = datetime.date.today()
    
    # Test 1: Should trigger coordinator
    prompt1 = "Find items with declining sales, then show their BOM components"
    print(f"\nTest 1: '{prompt1}'")
    print(f"Should use coordinator? {should_use_reasoning_coordinator(prompt1)}")
    
    result1 = execute_reasoning_chain(cursor, prompt1, today)
    if result1:
        print(f"Result summary: {result1['insights'].get('summary', 'No summary')}")
        if "reasoning_chain" in result1["insights"]:
            print(f"Reasoning chain:\n{result1['insights']['reasoning_chain']}")
    else:
        print("Coordinator returned None (single-step question)")
    
    # Test 2: Should NOT trigger coordinator (single-step complex)
    prompt2 = "If SOARBLM02 demand increases 3%, what raw materials should we buy?"
    print(f"\nTest 2: '{prompt2}'")
    print(f"Should use coordinator? {should_use_reasoning_coordinator(prompt2)}")
    
    result2 = execute_reasoning_chain(cursor, prompt2, today)
    if result2:
        print("Coordinator executed (unexpected)")
    else:
        print("Coordinator correctly identified as single-step")
    
    # Test 3: Multi-step with "what uses X, then forecast"
    prompt3 = "What finished goods use NPK3011, and what's their demand forecast?"
    print(f"\nTest 3: '{prompt3}'")
    print(f"Should use coordinator? {should_use_reasoning_coordinator(prompt3)}")
    
    result3 = execute_reasoning_chain(cursor, prompt3, today)
    if result3:
        print(f"Result summary: {result3['insights'].get('summary', 'No summary')}")
    else:
        print("Coordinator returned None")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_reasoning_coordinator()
