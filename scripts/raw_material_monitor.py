"""
Raw Material Inventory Monitor

Monitors raw materials (RAWMATT, RAWMATNTE, RAWMATNT) and sends Teams alerts
when Available Quantity drops below Order Point Quantity.

Implements a "High Water Mark" style logic using a state file:
- Alerts are sent ONLY when an item *newly* appears on the low-stock list.
- Uses reorder_math to calculate usage/coverage and teams_notifier for standard alerts.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from secrets_loader import load_project_secrets
from db_pool import get_connection
import reorder_math
from notifications.teams_notifier import send_reorder_alert, send_raw_material_alert, send_test_message
from logging_config import get_logger

LOGGER = get_logger(__name__)

# State file to track which items we've already alerted on
STATE_FILE = Path(__file__).parent.parent / "data" / "raw_material_state.json"

RAW_MATERIAL_CLASSES = ['RAWMATT', 'RAWMATNTE', 'RAWMATNT']


class ItemObject:
    """Helper to convert dictionary to object for teams_notifier compatibility."""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_low_raw_materials() -> list:
    """
    Query GP for raw materials where Available < Order Point using reorder_math.
    Returns value-added objects with usage/coverage data.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Get all raw materials
        # We don't use only_below_rop=True inside reorder_math because its logic 
        # for 'urgency' might differ slightly from our strict 'GP Order Point' requirement.
        # We'll filter manually.
        df = reorder_math.get_reorder_recommendations(
            cursor,
            include_classes=RAW_MATERIAL_CLASSES,
            lookback_days=90,
            safety_days=7,
            location="MAIN" 
        )
        
    if df.empty:
        return []

    results = []
    
    # Filter for items below GP Order Point using (On Hand - Allocated)
    # AND GP Order Point > 0
    for _, row in df.iterrows():
        gp_order_point = row['gp_order_point']
        
        # User requested logic: Available = On Hand - Allocated (Ignore On Order for alert trigger)
        # This matches the "Items Below Order Point" SmartList view provided.
        qty_available_alert = row['qty_on_hand'] - row['qty_allocated']
        
        if gp_order_point > 0 and qty_available_alert < gp_order_point:
            # Create object compatible with send_reorder_alert
            # The row keys match the ReorderRecommendation dataclass attributes
            item = ItemObject(**row.to_dict())
            results.append(item)
            
    return results


def load_state() -> dict:
    """Load the state of previously alerted items."""
    if not STATE_FILE.exists():
        return {}
    
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error(f"Error loading state file: {e}")
        return {}


def save_state(state: dict):
    """Save the current state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        LOGGER.error(f"Error saving state file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Raw Material Inventory Monitor")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    parser.add_argument("--test", action="store_true", help="Send test notification")
    args = parser.parse_args()

    # Load secrets
    secrets = load_project_secrets()
    teams_config = secrets.get("teams", {})
    webhook_url = teams_config.get("webhook_url")
    
    if args.test:
        if not webhook_url:
            print("❌ No Teams webhook URL configured in secrets.toml")
            return
        print("Sending test message...")
        if send_test_message(webhook_url):
            print("✅ Test message sent!")
        else:
            print("❌ Failed to send test message.")
        return

    if not webhook_url:
        LOGGER.error("No Teams webhook URL configured. Exiting.")
        return

    LOGGER.info("Starting Raw Material Monitor...")

    while True:
        try:
            # 1. Get current low items (rich objects)
            current_low_items = get_low_raw_materials()
            current_low_map = {item.item_number: item for item in current_low_items}
            
            # 2. Load previous state
            previous_state = load_state()
            
            # 3. Compare to find NEW items
            new_alerts = []
            new_state = {}
            
            for item_code, item_obj in current_low_map.items():
                if item_code not in previous_state:
                    new_alerts.append(item_obj)
                    new_state[item_code] = datetime.now().isoformat()
                else:
                    new_state[item_code] = previous_state[item_code]
            
            # 4. Handle Recovered Items
            for old_item_code in previous_state:
                if old_item_code not in new_state:
                    LOGGER.info(f"Item RECOVERED: {old_item_code}")

            # 5. Send Alerts
            if new_alerts:
                LOGGER.info(f"Found {len(new_alerts)} NEW items below order point.")
                for item in new_alerts:
                    LOGGER.info(f"Sending alert for: {item.item_number}")
                    
                    # Ensure qty_available is consistent with the trigger logic (On Hand - Allocated)
                    # This ensures the alert shows the same "Available" number as the user's list.
                    # We modify the object before sending.
                    item.qty_available = item.qty_on_hand - item.qty_allocated
                    
                    # Use standard reorder alert (Standard Format with Usage/Coverage/Top Sellers)
                    send_reorder_alert(item, webhook_url)
                    time.sleep(2) 
            else:
                LOGGER.info(f"No new alerts. {len(current_low_items)} items are currently low.")

            # 6. Save new state
            save_state(new_state)

        except Exception as e:
            LOGGER.error(f"Error in monitor loop: {e}", exc_info=True)
        
        if args.once:
            break
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
