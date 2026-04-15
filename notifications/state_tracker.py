"""
State tracking for reorder point notifications.

Tracks which items are below reorder point and detects state transitions
to prevent duplicate notifications and only alert on new issues.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from logging_config import get_logger

LOGGER = get_logger(__name__)

# State file location
STATE_FILE = Path(__file__).parent.parent / "data" / "reorder_state_history.json"


def load_previous_state() -> Dict:
    """
    Load the previous reorder point state from JSON file.

    Returns:
        Dictionary with 'last_updated' timestamp and 'items' dict
        Returns empty state if file doesn't exist or is corrupted
    """
    try:
        if not STATE_FILE.exists():
            LOGGER.info("No previous state file found, starting fresh")
            return {"last_updated": None, "items": {}}

        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            LOGGER.info(
                f"Loaded previous state: {len(state.get('items', {}))} items tracked, "
                f"last updated {state.get('last_updated', 'unknown')}"
            )
            return state

    except json.JSONDecodeError as e:
        LOGGER.error(f"State file corrupted: {e}. Starting with empty state.")
        return {"last_updated": None, "items": {}}

    except Exception as e:
        LOGGER.error(f"Error loading state file: {e}. Starting with empty state.")
        return {"last_updated": None, "items": {}}


def save_current_state(items: List) -> None:
    """
    Save the current reorder point state to JSON file.

    Args:
        items: List of ReorderRecommendation objects (all items, not just below ROP)
    """
    try:
        # Ensure data directory exists
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Build state dictionary
        state = {
            "last_updated": datetime.now().isoformat(),
            "items": {}
        }

        for item in items:
            state["items"][item.item_number] = {
                "urgency": item.urgency,
                "qty_available": float(item.qty_available),
                "days_of_coverage": float(item.days_of_coverage)
            }

        # Write to file
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        LOGGER.info(f"Saved state for {len(state['items'])} items to {STATE_FILE}")

    except Exception as e:
        LOGGER.error(f"Failed to save state file: {e}")


def should_notify(item, previous_urgency: Optional[str]) -> bool:
    """
    Determine if a notification should be sent for this item.

    Notification is sent when:
    - Item is new (not in previous state) AND below reorder point
    - Item was OK, now Soon or Critical
    - Item was Soon, now Critical (escalation)

    Notification is NOT sent when:
    - Item is OK (not below reorder point)
    - Item remains at same urgency (already notified)
    - Item is improving (Critical → Soon, Soon → OK)

    Args:
        item: ReorderRecommendation object
        previous_urgency: Previous urgency state ("OK", "Soon", "Critical", or None)

    Returns:
        True if notification should be sent, False otherwise
    """
    current = item.urgency

    # Not below reorder point - never notify
    if current == "OK":
        return False

    # New item below reorder point
    if previous_urgency is None:
        LOGGER.debug(f"{item.item_number}: New item at {current} - will notify")
        return True

    # Was OK, now below reorder point
    if previous_urgency == "OK" and current in ["Soon", "Critical"]:
        LOGGER.debug(f"{item.item_number}: Transitioned OK → {current} - will notify")
        return True

    # Escalation: Soon → Critical
    if previous_urgency == "Soon" and current == "Critical":
        LOGGER.debug(f"{item.item_number}: Escalated Soon → Critical - will notify")
        return True

    # Already notified (no change or improving)
    if previous_urgency == current:
        LOGGER.debug(f"{item.item_number}: Remains at {current} - already notified")
        return False

    # Improving (Critical → Soon, Soon → OK, Critical → OK)
    LOGGER.debug(f"{item.item_number}: Improving {previous_urgency} → {current} - no notification")
    return False


def detect_new_reorder_items(current_items: List, previous_state: Dict) -> List:
    """
    Detect items that newly crossed the reorder point threshold.

    Compares current item states against previous state to find items
    that require notification based on state transition rules.

    Args:
        current_items: List of current ReorderRecommendation objects
        previous_state: Previous state dict from load_previous_state()

    Returns:
        List of ReorderRecommendation objects that should trigger notifications
    """
    items_to_notify = []
    previous_items = previous_state.get("items", {})

    for item in current_items:
        previous_urgency = previous_items.get(item.item_number, {}).get("urgency")

        if should_notify(item, previous_urgency):
            items_to_notify.append(item)
            LOGGER.info(
                f"Will notify for {item.item_number}: {previous_urgency or 'NEW'} → {item.urgency}"
            )

    if items_to_notify:
        LOGGER.info(f"Found {len(items_to_notify)} items requiring notification")
    else:
        LOGGER.info("No new items below reorder point")

    return items_to_notify


def get_state_summary(state: Dict) -> Dict:
    """
    Get a summary of the current state for logging/debugging.

    Args:
        state: State dictionary from load_previous_state()

    Returns:
        Dictionary with counts by urgency level
    """
    items = state.get("items", {})

    summary = {
        "total": len(items),
        "critical": sum(1 for i in items.values() if i.get("urgency") == "Critical"),
        "soon": sum(1 for i in items.values() if i.get("urgency") == "Soon"),
        "ok": sum(1 for i in items.values() if i.get("urgency") == "OK")
    }

    return summary
