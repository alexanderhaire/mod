
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from constants import RAW_MATERIAL_CLASS_CODES
from market_insights import classify_item_segment

class TestRawMaterialClassification(unittest.TestCase):
    def test_rawmatntb_is_included(self):
        """Verify RAWMATNTB is in the constants."""
        self.assertIn('RAWMATNTB', RAW_MATERIAL_CLASS_CODES)
        
    def test_classification_logic(self):
        """Verify logic classifies RAWMATNTB correctly."""
        self.assertEqual(classify_item_segment('RAWMATNTB'), 'Raw Material')
        self.assertEqual(classify_item_segment('RAWMATNT'), 'Raw Material')
        self.assertEqual(classify_item_segment('FINISHED'), 'Finished Good')

if __name__ == '__main__':
    unittest.main()
