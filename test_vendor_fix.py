
from unittest.mock import MagicMock
from vendor_portal import fetch_vendors

def test_fetch_vendors_fix():
    # Mock Cursor
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    
    fetch_vendors(mock_cursor)
    
    # Check query now uses VENDSTTS
    mock_cursor.execute.assert_called_with("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDSTTS = 1 ORDER BY VENDNAME")
    print("PASS: Correct column VENDSTTS used")

if __name__ == "__main__":
    test_fetch_vendors_fix()
