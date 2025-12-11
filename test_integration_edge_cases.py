"""
Integration test: Test real edge case prompts against the dynamic handler system.
This verifies that our fixes work end-to-end.
"""
from dynamic_handler import DynamicHandlerService, HandlerStore, EmbeddingService, HandlerScorer
from pathlib import Path

def test_edge_case_prompts():
    """Test a selection of edge case prompts to verify proper handler matching."""
    
    # Setup
    store = HandlerStore(Path("dynamic_handlers.json"))
    embeddings = EmbeddingService()
    scorer = HandlerScorer(embeddings)
    service = DynamicHandlerService(store, embeddings, scorer)
    
    # Test cases: (prompt, should_match_handler, reason)
    test_cases = [
        (
            "show items where standard cost is less than current cost",
            False,  # Should NOT match - needs special comparison logic
            "Cost comparison requires both STNDCOST and CURRCOST"
        ),
        (
            "what is the standard cost of SOARBLM02?",
            True,  # Could match if handler exists for this item
            "Simple standard cost query for specific item"
        ),
        (
            "show items NOT in inventory",
            False,  # Should NOT match simple handlers
            "Negation requires NOT IN or LEFT JOIN NULL pattern"
        ),
        (
            "items with missing cost information",
            False,  # Should NOT match unless handler has NULL logic
            "Missing/NULL detection requires IS NULL check"
        ),
        (
            "what is our usage on npkacek?",  # lowercase
            True,  # Should match - case already handled
            "Case-insensitive item matching"
        ),
    ]
    
    print("\n" + "="*80)
    print("EDGE CASE INTEGRATION TESTS")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for prompt, should_match, reason in test_cases:
        matched_handler = service.find_handler(prompt)
        did_match = matched_handler is not None
        
        test_passed = did_match == should_match
        status = "[PASS]" if test_passed else "[FAIL]"
        
        print(f"\n{status}: {prompt}")
        print(f"  Expected: {'Match' if should_match else 'No Match'}")
        print(f"  Actual: {'Match' if did_match else 'No Match'}")
        print(f"  Reason: {reason}")
        
        if did_match:
            handler_name = matched_handler.get("name", "Unknown")
            match_score = matched_handler.get("match_score", 0)
            print(f"  Handler: {handler_name[:80]}...")
            print(f"  Score: {match_score}")
        
        if test_passed:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*80)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    success = test_edge_case_prompts()
    sys.exit(0 if success else 1)
