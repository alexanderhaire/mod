import unittest
from unittest.mock import MagicMock
import sys
import os

# Add the current directory to sys.path so we can import the module
sys.path.append(os.getcwd())

from market_insights import calculate_buying_signals, forecast_demand

class TestProcurementLogic(unittest.TestCase):

    def setUp(self):
        self.mock_cursor = MagicMock()

    def test_buying_signal_strong_buy(self):
        # Mock price history: decreasing trend, current price is lowest
        # Last 10 prices: 100, 90, 80, ..., 10
        mock_history = [{'AvgCost': float(i)} for i in range(100, 0, -10)]
        # Add dates if needed by the function, but currently it only uses AvgCost for calculation
        # The function calls fetch_product_price_history, so we need to mock that or mock the return of the cursor if we were testing the fetch.
        # But calculate_buying_signals calls fetch_product_price_history. 
        # Since fetch_product_price_history is in the same module, we should mock it or mock the cursor behavior it relies on.
        # However, calculate_buying_signals calls fetch_product_price_history(cursor, ...).
        # To test calculate_buying_signals in isolation without DB, we should mock fetch_product_price_history.
        pass

    # Since calculate_buying_signals calls another function in the same module, 
    # it's easier to mock the cursor if we want to test the full flow, 
    # OR mock the internal function call. 
    # Let's try to mock the cursor results for fetch_product_price_history.
    # fetch_product_price_history executes a query. 
    # It's complex to mock the cursor for that specific query structure.
    # Better to mock the 'market_insights.fetch_product_price_history' function.

from unittest.mock import patch

class TestProcurementLogicMocked(unittest.TestCase):

    @patch('market_insights.fetch_product_price_history')
    def test_buying_signal_strong_buy(self, mock_fetch):
        # Setup mock return value: 2 years of data (approx 24 points if monthly, or just a list of costs)
        # Let's say we have 20 data points.
        # Current cost (last item) is 10. Previous were higher.
        costs = [100.0] * 10 + [50.0] * 5 + [10.0] 
        mock_fetch.return_value = [{'AvgCost': c} for c in costs]
        
        mock_cursor = MagicMock()
        result = calculate_buying_signals(mock_cursor, 'TEST-ITEM')
        
        print(f"Strong Buy Result: {result}")
        self.assertGreaterEqual(result['score'], 80)
        self.assertEqual(result['signal'], 'Strong Buy')
        self.assertIn("bottom 10%", result['reason'])

    @patch('market_insights.fetch_product_price_history')
    def test_buying_signal_high_price(self, mock_fetch):
        # Current cost is high (100), previous were low (10)
        costs = [10.0] * 15 + [100.0]
        mock_fetch.return_value = [{'AvgCost': c} for c in costs]
        
        mock_cursor = MagicMock()
        result = calculate_buying_signals(mock_cursor, 'TEST-ITEM')
        
        print(f"High Price Result: {result}")
        self.assertLess(result['score'], 50)
        self.assertIn("near 2-year high", result['reason'])

    @patch('market_insights.fetch_product_usage_history')
    def test_forecast_demand_stable(self, mock_fetch):
        # Usage last 6 months: 100, 100, 100, 100, 100, 100
        mock_fetch.return_value = [{'UsageQty': 100.0} for _ in range(6)]
        
        mock_cursor = MagicMock()
        result = forecast_demand(mock_cursor, 'TEST-ITEM')
        
        print(f"Forecast Result: {result}")
        self.assertEqual(result['forecast_monthly_avg'], 100.0)
        self.assertEqual(result['forecast_next_3mo'], 300.0)
        self.assertEqual(result['trend'], 'Stable')

    @patch('market_insights.fetch_product_usage_history')
    def test_forecast_demand_increasing(self, mock_fetch):
        # Usage last 6 months: 10, 10, 10 (avg 10) -> 100, 100, 100 (avg 100)
        # 100 > 10 * 1.1 -> Increasing
        usage_data = [{'UsageQty': 10.0} for _ in range(3)] + [{'UsageQty': 100.0} for _ in range(3)]
        mock_fetch.return_value = usage_data
        
        mock_cursor = MagicMock()
        result = forecast_demand(mock_cursor, 'TEST-ITEM')
        
        print(f"Increasing Forecast Result: {result}")
        self.assertEqual(result['forecast_monthly_avg'], 100.0)
        self.assertEqual(result['trend'], 'Increasing')

if __name__ == '__main__':
    unittest.main()
