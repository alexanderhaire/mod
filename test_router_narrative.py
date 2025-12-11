"""
Unit test for router narrative generation logic.
Verifies that narratives are generated when the route is correctly passed.
"""
import unittest
from unittest.mock import MagicMock, patch
from router import _decorate_response, _attach_routing_metadata

class TestRouterNarrative(unittest.TestCase):
    def test_decorate_response_missing_route(self):
        """Test that narrative is SKIPPED when route is missing from result/insights."""
        result = {
            "data": [{"col": 1}],
            "insights": {"summary": "Some summary"},
            # No route here
        }
        prompt = "show me data"
        context = {}
        
        # Mock OpenAI call to ensure it's NOT called
        with patch("router.call_openai_data_narrative") as mock_narrative:
            decorated = _decorate_response(result, prompt, context)
            
            # Should NOT call narrative generation because route is missing
            mock_narrative.assert_not_called()
            self.assertNotIn("narrative", decorated["insights"])

    def test_decorate_response_with_route(self):
        """Test that narrative is GENERATED when route is present."""
        result = {
            "data": [{"col": 1}],
            "insights": {"summary": "Some summary", "route": "erp_sql"},
            "route": "erp_sql" # Route present
        }
        prompt = "show me data"
        context = {}
        
        # Mock OpenAI call to return a story
        with patch("router.call_openai_data_narrative") as mock_narrative:
            mock_narrative.return_value = {"narrative": "This is a data story."}
            
            decorated = _decorate_response(result, prompt, context)
            
            # Should call narrative generation
            mock_narrative.assert_called_once()
            self.assertEqual(decorated["insights"]["narrative"], "This is a data story.")

if __name__ == "__main__":
    unittest.main()
