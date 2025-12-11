
import unittest
from unittest.mock import MagicMock
from dynamic_handler import DynamicHandlerService, HandlerStore, EmbeddingService, HandlerScorer, HandlerTextProcessor

class TestDynamicHandlerEdgeCases(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock(spec=HandlerStore)
        self.embeddings = MagicMock(spec=EmbeddingService)
        self.scorer = MagicMock(spec=HandlerScorer)
        self.service = DynamicHandlerService(self.store, self.embeddings, self.scorer)

        # Mock embeddings to return dummy vectors
        self.embeddings.get_prompt_embedding.return_value = [0.1, 0.2, 0.3]
        self.embeddings.normalize_vector.return_value = [0.1, 0.2, 0.3]
        self.embeddings.cosine_similarity.return_value = 0.9

        # Mock scorer to return a high score
        self.scorer.compute_handler_score.return_value = 0.95

    def test_should_skip_single_item_handler_mismatch(self):
        # Handler targets "ITEM123"
        handler_data = {
            "name": "Handler for ITEM123",
            "sql": "SELECT * FROM table WHERE item = 'ITEM123'",
            "entities": {"item": "ITEM123"},
            "params": ["ITEM123"]
        }
        
        # Prompt does NOT mention ITEM123
        prompt = "Show me sales for ITEM456"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip handler for ITEM123 when prompt is for ITEM456")

    def test_should_not_skip_single_item_handler_match(self):
        # Handler targets "ITEM123"
        handler_data = {
            "name": "Handler for ITEM123",
            "sql": "SELECT * FROM table WHERE item = 'ITEM123'",
            "entities": {"item": "ITEM123"},
            "params": ["ITEM123"]
        }
        
        # Prompt mentions ITEM123
        prompt = "Show me sales for ITEM123"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertFalse(should_skip, "Should NOT skip handler for ITEM123 when prompt matches")

    def test_should_skip_single_item_handler_for_multi_item_prompt(self):
        # Handler targets "ITEM123"
        handler_data = {
            "name": "Handler for ITEM123",
            "sql": "SELECT * FROM table WHERE item = 'ITEM123'",
            "entities": {"item": "ITEM123"},
            "params": ["ITEM123"]
        }
        
        # Prompt asks for top items (multi-item)
        prompt = "What are the top selling items?"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=True
        )
        self.assertTrue(should_skip, "Should skip single-item handler for multi-item prompt")

    def test_should_skip_if_breakdown_required_but_not_supported(self):
        # Handler SQL does NOT support breakdown (no group by item)
        handler_data = {
            "name": "Total Sales",
            "sql": "SELECT SUM(sales) FROM table",
            "entities": {}
        }
        
        # Prompt asks for breakdown
        prompt = "Show sales per item"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=True, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip if SQL doesn't support required breakdown")

    def test_item_extraction_edge_cases(self):
        # Test item extraction from handler data
        handler_data = {"entities": {"item": "ITEM-123"}}
        self.assertEqual(HandlerTextProcessor.extract_handler_item_code(handler_data), "ITEM-123")
        
        handler_data = {"params": ["ITEM-123"]}
        self.assertEqual(HandlerTextProcessor.extract_handler_item_code(handler_data), "ITEM-123")

    def test_prompt_mentions_token_edge_cases(self):
        # Test punctuation handling
        self.assertTrue(HandlerTextProcessor.prompt_mentions_token("Sales for ITEM123?", "ITEM123"))
        self.assertTrue(HandlerTextProcessor.prompt_mentions_token("Sales for (ITEM123)", "ITEM123"))
        self.assertTrue(HandlerTextProcessor.prompt_mentions_token("ITEM123 sales", "ITEM123"))
        
        # Test partial match (should be false if strict word boundary is needed, but implementation might be loose)
        # The current implementation uses `in` or regex with \b
        self.assertFalse(HandlerTextProcessor.prompt_mentions_token("Sales for ITEM1234", "ITEM123"))

    def test_should_skip_handler_with_item_and_dates_mismatch(self):
        # Handler targets "ITEM123" with dates
        handler_data = {
            "name": "Sales for ITEM123",
            "sql": "SELECT * FROM table WHERE item = ? AND date BETWEEN ? AND ?",
            "entities": {"item": "ITEM123"},
            "params": ["ITEM123", "2023-01-01", "2023-01-31"]
        }
        
        # Prompt asks for ITEM456
        prompt = "Sales for ITEM456"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip handler for ITEM123 (with dates) when prompt is for ITEM456")

    def test_should_skip_handler_with_item_and_dates_mismatch_no_entities(self):
        # Handler targets "ITEM123" with dates, NO entities
        handler_data = {
            "name": "Sales for ITEM123",
            "sql": "SELECT * FROM table WHERE item = ? AND date BETWEEN ? AND ?",
            "entities": {},
            "params": ["ITEM123", "2023-01-01", "2023-01-31"]
        }
        
        # Prompt asks for ITEM456
        prompt = "Sales for ITEM456"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip handler for ITEM123 (with dates, no entities) when prompt is for ITEM456")

    def test_should_not_skip_date_range_handler(self):
        # Handler targets a date range, NO entities
        handler_data = {
            "name": "Sales Jan 2023",
            "sql": "SELECT * FROM table WHERE date BETWEEN ? AND ?",
            "entities": {},
            "params": ["2023-01-01", "2023-01-31"]
        }
        
        # Prompt asks for something else (or same thing), shouldn't skip based on item logic
        prompt = "Sales Feb 2023"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertFalse(should_skip, "Should NOT skip date-range handler (treated as generic)")

    def test_should_skip_generic_handler_for_specific_comparison(self):
        # Generic handler for "standard cost" insights
        handler_data = {
            "name": "Standard Cost Insights",
            "sql": "SELECT AVG(STNDCOST) FROM items",
            "keywords": ["standard", "cost", "items", "average", "insights"],
            "params": []
        }
        
        # Prompt asks for a comparison ("less than current cost")
        # "current" and "less" are NOT in the handler keywords
        prompt = "are there any items where our standard cost is less than our current cost"
        
        # This is what we WANT to happen (skip or low score). 
        # For now, let's test if it is skipped by _should_skip_for_prompt.
        # If it's not skipped, we might need to adjust the scorer or the skipper.
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=True
        )
        
        # Currently, I expect this might FAIL (return False) because we haven't implemented the logic yet.
        # But this confirms the behavior.
        self.assertTrue(should_skip, "Should skip generic handler when prompt asks for specific comparison (less than current)")

    def test_should_skip_handler_without_standard_cost(self):
        # Handler doesn't mention standard cost
        handler_data = {
            "name": "Sales Insights",
            "sql": "SELECT AVG(SALES) FROM items",
            "keywords": ["sales", "average", "items"],
            "params": []
        }
        
        # Prompt asks for standard cost
        prompt = "show items with high standard cost"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip handler without standard cost when prompt requires it")

    def test_should_skip_handler_without_negation_support(self):
        # Handler doesn't have NOT or exclusion logic
        handler_data = {
            "name": "All Items",
            "sql": "SELECT * FROM items",
            "keywords": ["items", "all"],
            "params": []
        }
        
        # Prompt asks for NOT/exclusion
        prompt = "show items not in inventory"
        
        should_skip = self.service._should_skip_for_prompt(
            prompt, handler_data, requires_item_breakdown=False, multi_item_prompt=False
        )
        self.assertTrue(should_skip, "Should skip handler without negation support when prompt uses NOT")




 

if __name__ == '__main__':
    unittest.main()
