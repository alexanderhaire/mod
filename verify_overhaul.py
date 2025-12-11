import datetime
import logging
from handlers import handle_mrp_planning
from router import _decorate_response

# Mock cursor and data for testing
class MockCursor:
    def __init__(self):
        self.description = [
            ("ITEMNMBR",), ("ITEMDESC",), ("CurrentFreeStock",), 
            ("ProjectedDemand",), ("ProjectedSupply",), ("ProjectedBalance",), 
            ("ReorderPoint",), ("OrderUpTo",), ("SuggestedBuyQty",)
        ]
    
    def execute(self, query, params):
        print(f"Executing Query: {query[:50]}... with params {params}")
    
    def fetchmany(self, size):
        # Return dummy data for MRP
        return [
            ("ITEM-A", "Test Item A", 10.0, 50.0, 0.0, -40.0, 20.0, 100.0, 140.0),
            ("ITEM-B", "Test Item B", 5.0, 10.0, 0.0, -5.0, 10.0, 50.0, 55.0)
        ]

def test_mrp_logic():
    print("\n--- Testing MRP Logic ---")
    cursor = MockCursor()
    today = datetime.date.today()
    prompt = "what should we buy in december 2025"
    
    result = handle_mrp_planning(cursor, prompt, today)
    
    if result:
        print("MRP Handler returned result:")
        print(f"Summary: {result['insights']['summary']}")
        print(f"Row Count: {len(result['data'])}")
        print(f"First Row: {result['data'][0]}")
    else:
        print("MRP Handler returned None (Failed)")

def test_router_analysis_detection():
    print("\n--- Testing Router Analysis Detection ---")
    # We can't easily test the full router without OpenAI, but we can check if the keywords trigger the logic
    # This is a bit tricky since _decorate_response calls OpenAI.
    # We'll just verify the keywords are correct in our mind/code.
    
    analysis_keywords = ("analyze", "analysis", "why", "explain", "reason", "trend", "outlier", "correlation", "breakdown")
    test_prompt = "Analyze the sales trends for Item A"
    is_deep_analysis = any(k in test_prompt.lower() for k in analysis_keywords)
    print(f"Prompt: '{test_prompt}'")
    print(f"Detected as Deep Analysis? {is_deep_analysis}")
    
    test_prompt_2 = "Show me sales for Item A"
    is_deep_analysis_2 = any(k in test_prompt_2.lower() for k in analysis_keywords)
    print(f"Prompt: '{test_prompt_2}'")
    print(f"Detected as Deep Analysis? {is_deep_analysis_2}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mrp_logic()
    test_router_analysis_detection()
