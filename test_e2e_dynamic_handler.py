"""
End-to-end test to verify the fix for edge case handling in dynamic handlers.
This test simulates the actual scenario from the user's screenshot.
"""
import json
from pathlib import Path
from dynamic_handler import DynamicHandlerService, HandlerStore, EmbeddingService, HandlerScorer

def test_real_scenario():
    """Test the actual scenario from the user's screenshot."""
    
    # Setup
    store = HandlerStore(Path("dynamic_handlers.json"))
    embeddings = EmbeddingService()
    scorer = HandlerScorer(embeddings)
    service = DynamicHandlerService(store, embeddings, scorer)
    
    # Load handlers
    handlers = service.load_handlers()
    
    # The problematic prompt from the screenshot
    prompt = "are there any items where our standard cost is less than our current cost"
    
    # Find handler
    matched_handler = service.find_handler(prompt)
    
    if matched_handler:
        handler_name = matched_handler.get("name", "Unknown")
        handler_sql = matched_handler.get("sql", "")
        match_score = matched_handler.get("match_score", 0)
        
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}")
        print(f"\nMatched Handler: {handler_name}")
        print(f"Match Score: {match_score}")
        print(f"\nSQL Preview:")
        print(f"{handler_sql[:200]}..." if len(handler_sql) > 200 else handler_sql)
        print(f"\n{'='*80}")
        
        # Check if it's the generic insights handler (the wrong one)
        is_generic_insights = "Provides insights into the total number of items" in handler_name
        
        if is_generic_insights:
            print("\n❌ FAILED: Matched the generic insights handler (incorrect)")
            print("   This handler doesn't support comparison queries.")
            return False
        else:
            print("\n✅ PASSED: Did NOT match the generic insights handler")
            print("   The handler appears to be more relevant to the comparison query.")
            return True
    else:
        print(f"\n✅ PASSED: No handler matched (will generate new SQL)")
        print("   This is correct behavior - the generic handler was skipped.")
        return True

if __name__ == "__main__":
    import sys
    success = test_real_scenario()
    sys.exit(0 if success else 1)
