import datetime
import sys
import os
import pyodbc
from secrets_loader import build_connection_string
from openai_clients import call_openai_question_router
from handlers import handle_mrp_planning, handle_order_point_gap

# Add current directory to path
sys.path.append(os.getcwd())

def reproduce_issue():
    print("--- Reproducing 'No Data' Issue ---")
    
    # 1. Connect to DB
    try:
        conn_str, _, _, _ = build_connection_string()
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        print("✅ DB Connected")
    except Exception as e:
        print(f"❌ DB Connection Failed: {e}")
        return

    # 2. Simulate User Query
    prompt = "what items should I look at"
    today = datetime.date.today()
    print(f"Query: '{prompt}'")
    print(f"Date: {today}")

    # 3. Check Routing (Simulated)
    # We can't easily call the full router without OpenAI key if it uses LLM, 
    # but we can check the deterministic handlers directly or try the router if it's available.
    # Let's try to call the router if possible, otherwise we guess the handler.
    
    # "what items should I look at" is vague. It might hit 'planning' or 'inventory'.
    # Let's try the likely handlers directly first to see if they return data.
    
    print("\n--- Testing handle_mrp_planning ---")
    mrp_result = handle_mrp_planning(cursor, prompt, today)
    if mrp_result:
        print(f"MRP Result: {mrp_result.get('insights', {}).get('summary')}")
        print(f"Rows: {len(mrp_result.get('data', []))}")
    else:
        print("MRP Handler returned None (Not triggered)")

    print("\n--- Testing handle_order_point_gap ---")
    # "look at" might not trigger order point gap keywords like "order point"
    op_result = handle_order_point_gap(cursor, prompt, today)
    if op_result:
        print(f"Order Point Result: {op_result.get('insights', {}).get('summary')}")
        print(f"Rows: {len(op_result.get('data', []))}")
    else:
        print("Order Point Handler returned None (Not triggered)")

    # If neither triggered, it goes to SQL generation.
    # We can't easily reproduce SQL generation here without the full context, 
    # but we can see if the handlers *should* have triggered.

if __name__ == "__main__":
    reproduce_issue()
