
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from secrets_loader import load_project_secrets
from notifications.teams_notifier import send_raw_material_alert

def test_alert():
    secrets = load_project_secrets()
    webhook_url = secrets.get("teams", {}).get("webhook_url")
    
    if not webhook_url:
        print("❌ No webhook URL found in secrets.toml")
        return

    # Mock item data based on user's screenshot for CHEACETIC
    mock_item = {
        'ItemNumber': 'CHEACETIC',
        'ItemDescription': 'Glacial Acetic Acid',
        'QtyOnHand': 110.0,
        'QtyAllocated': 0.0,
        'OrderPointQty': 250.0,
        'QtyOnOrder': 0.0
    }
    
    print(f"Sending test alert for {mock_item['ItemNumber']}...")
    success = send_raw_material_alert(mock_item, webhook_url)
    
    if success:
        print("✅ Alert sent successfully!")
    else:
        print("❌ Failed to send alert.")

if __name__ == "__main__":
    test_alert()
