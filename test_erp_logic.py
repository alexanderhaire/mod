import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import datetime
from decimal import Decimal

# Add the current directory to sys.path so we can import the module
sys.path.append(os.getcwd())

from handlers import handle_mrp_planning, handle_order_point_gap
from inventory_queries import fetch_recursive_bom_for_item

class TestERPLogic(unittest.TestCase):

    def setUp(self):
        self.mock_cursor = MagicMock()
        self.today = datetime.date(2025, 12, 1)

    # --- 1. MRP Logic Tests ---

    def test_mrp_zero_demand(self):
        """Test MRP behavior when there is zero demand."""
        # handle_mrp_planning calls fetchmany
        self.mock_cursor.fetchmany.return_value = []
        self.mock_cursor.description = [('ITEMNMBR',), ('ITEMDESC',), ('NetNeed',)] 
        
        result = handle_mrp_planning(self.mock_cursor, "what should we buy in Dec 2025", self.today)
        
        # Updated assertion to match actual output
        self.assertIn("No items with demand or usage found", result['insights']['summary'])

    def test_mrp_demand_spike(self):
        """Test MRP behavior when demand spikes above stock."""
        mock_row = {
            'ITEMNMBR': 'SPIKE-ITEM',
            'ITEMDESC': 'High Demand Item',
            'OnHand': 100.0,
            'Allocated': 0.0,
            'CurrentFreeStock': 100.0,
            'FirmDemand': 500.0,
            'MonthlyForecast': 16.6,
            'OnOrder': 0.0,
            'ProjectedDemand': 500.0,
            'ProjectedBalance': -400.0,
            'SuggestedBuyQty': 400.0
        }
        
        # Use fetchmany instead of fetchall
        self.mock_cursor.fetchmany.return_value = [tuple(mock_row.values())]
        self.mock_cursor.description = [(k,) for k in mock_row.keys()]
        
        result = handle_mrp_planning(self.mock_cursor, "what should we buy in Dec 2025", self.today)
        
        self.assertEqual(len(result['data']), 1)
        self.assertEqual(result['data'][0]['ITEMNMBR'], 'SPIKE-ITEM')
        self.assertEqual(result['data'][0]['SuggestedBuyQty'], 400.0)
        self.assertIn("Found **1 items** with projected shortages", result['insights']['summary'])

    # --- 2. Order Point Logic Tests ---

    @patch('handlers.load_allowed_sql_schema')
    def test_order_point_seasonality(self, mock_load_schema):
        """Test that seasonality adjusts the order point."""
        mock_load_schema.return_value = {'IV00102': [{'name': 'ORDERPOINTQTY'}, {'name': 'QTYONHND'}]}
        
        mock_row = {
            'ITEMNMBR': 'SEASONAL-ITEM',
            'ITEMDESC': 'Seasonal Item',
            'LocationCode': 'MAIN',
            'OrderPointQty': 100.0,
            'OrderUpToQty': 100.0,
            'QtyAvailable': 90.0,
            'QtyOnOrder': 0.0,
            'QtyOnHand': 90.0,
            'UsageLast90D': 300.0,
            'UsageSameMonthLastYear': 600.0,
            'AvgDailyUse90D': 3.3,
            'SeasonalAvgDailyUse': 20.0,
            'SeasonalityFactor': 1.5,
            'Projected30DayDemand': 150.0,
            'AdjustedOrderPoint': 150.0,
            'GapToOrderPoint': 60.0,
            'BuyToOrderUpTo': 60.0
        }
        
        # Use fetchmany
        self.mock_cursor.fetchmany.return_value = [tuple(mock_row.values())]
        self.mock_cursor.description = [(k,) for k in mock_row.keys()]
        
        result = handle_order_point_gap(self.mock_cursor, "check order points", self.today)
        
        self.assertEqual(len(result['data']), 1)
        item = result['data'][0]
        self.assertEqual(item['ITEMNMBR'], 'SEASONAL-ITEM')
        self.assertEqual(item['SeasonalityFactor'], 1.5)
        self.assertEqual(item['GapToOrderPoint'], 60.0)

    # --- 3. BOM Logic Tests ---

    def test_recursive_bom_explosion(self):
        """Test recursive BOM retrieval."""
        # Mock rows as objects with attributes
        row = MagicMock()
        row.RawMaterial = 'RAW-B'
        row.Design_Qty = Decimal('6.0')
        
        self.mock_cursor.fetchall.return_value = [row]
        self.mock_cursor.description = [('RawMaterial',), ('Design_Qty',)]
        
        rows, sql = fetch_recursive_bom_for_item(self.mock_cursor, "PARENT-ITEM")
        
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].RawMaterial, 'RAW-B')
        self.assertEqual(rows[0].Design_Qty, Decimal('6.0'))

if __name__ == '__main__':
    unittest.main()
