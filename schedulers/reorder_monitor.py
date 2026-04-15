"""
Reorder point monitoring orchestration.

Continuously checks inventory levels, detects state transitions,
and sends Teams notifications when items hit reorder points.
"""

import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_pool import get_connection
from reorder_math import get_reorder_recommendations
from notifications.teams_notifier import send_reorder_alert, send_error_notification
from notifications.state_tracker import (
    load_previous_state,
    save_current_state,
    detect_new_reorder_items,
    get_state_summary
)
from logging_config import get_logger

LOGGER = get_logger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    LOGGER.info("Shutdown signal received, will exit after current cycle")
    _shutdown_requested = True


def load_config() -> dict:
    """
    Load Teams notification configuration from secrets.toml.

    Returns:
        Dictionary with webhook_url, enabled, and check_interval_minutes
    """
    try:
        from secrets_loader import load_project_secrets

        secrets = load_project_secrets()
        teams_config = secrets.get("teams", {})

        config = {
            "webhook_url": teams_config.get("webhook_url"),
            "enabled": teams_config.get("enabled", True),
            "check_interval_minutes": teams_config.get("check_interval_minutes", 30)
        }

        if not config["webhook_url"]:
            LOGGER.warning("Teams webhook URL not configured in secrets.toml")
            config["enabled"] = False

        return config

    except Exception as e:
        LOGGER.error(f"Error loading configuration: {e}")
        return {
            "webhook_url": None,
            "enabled": False,
            "check_interval_minutes": 30
        }


def check_and_notify(webhook_url: Optional[str] = None) -> bool:
    """
    Main monitoring function - check inventory and send notifications for new items.

    This function:
    1. Queries current reorder recommendations from database
    2. Loads previous state from history file
    3. Detects items that newly crossed reorder threshold
    4. Sends Teams notification for each new item
    5. Updates state history

    Args:
        webhook_url: Teams webhook URL (loads from config if not provided)

    Returns:
        True if check completed successfully (even if no notifications sent)
        False if critical error occurred
    """
    LOGGER.info("=" * 60)
    LOGGER.info(f"Starting reorder point check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    LOGGER.info("=" * 60)

    try:
        # Load configuration
        config = load_config()
        webhook_url = webhook_url or config["webhook_url"]

        if not config["enabled"]:
            LOGGER.info("Teams notifications disabled in configuration")
            return True

        if not webhook_url:
            LOGGER.warning("No webhook URL configured, skipping notifications")
            return True

        # Step 1: Query current reorder recommendations
        LOGGER.info("Querying current reorder recommendations from database...")

        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                recommendations_df = get_reorder_recommendations(
                    cursor=cursor,
                    lookback_days=90,
                    safety_days=7,
                    only_below_rop=False,  # Get ALL items to track state transitions
                    location="MAIN"
                )

            if recommendations_df.empty:
                LOGGER.info("No items found in database")
                return True

            LOGGER.info(f"Retrieved {len(recommendations_df)} items from database")

            # Convert DataFrame to list of objects for easier handling
            items = recommendations_df.to_dict('records')

            # Count urgency levels
            critical_count = sum(1 for item in items if item['urgency'] == 'Critical')
            soon_count = sum(1 for item in items if item['urgency'] == 'Soon')
            ok_count = sum(1 for item in items if item['urgency'] == 'OK')

            LOGGER.info(
                f"Current status: {critical_count} Critical, {soon_count} Soon, {ok_count} OK"
            )

        except Exception as e:
            LOGGER.error(f"Database query failed: {e}", exc_info=True)
            send_error_notification(f"Database connection failed: {str(e)}", webhook_url)
            return False

        # Step 2: Load previous state
        LOGGER.info("Loading previous state...")
        previous_state = load_previous_state()
        previous_summary = get_state_summary(previous_state)
        LOGGER.info(
            f"Previous state: {previous_summary['critical']} Critical, "
            f"{previous_summary['soon']} Soon, {previous_summary['ok']} OK"
        )

        # Step 3: Detect new items below reorder point
        LOGGER.info("Detecting state transitions...")

        # Convert dict items to objects for easier comparison
        class Item:
            def __init__(self, item_dict):
                for key, value in item_dict.items():
                    setattr(self, key, value)

        item_objects = [Item(item) for item in items]
        items_to_notify = detect_new_reorder_items(item_objects, previous_state)

        if not items_to_notify:
            LOGGER.info("No new items require notification")
        else:
            LOGGER.info(f"Found {len(items_to_notify)} items requiring notification")

            # Step 4: Send Teams notifications
            success_count = 0
            fail_count = 0

            for item in items_to_notify:
                LOGGER.info(
                    f"Sending notification for {item.item_number} "
                    f"({item.urgency}, {item.days_of_coverage:.1f} days coverage)"
                )

                if send_reorder_alert(item, webhook_url):
                    success_count += 1
                else:
                    fail_count += 1

                # Small delay between notifications to avoid rate limiting
                if len(items_to_notify) > 1:
                    time.sleep(2)

            LOGGER.info(
                f"Notification results: {success_count} sent successfully, {fail_count} failed"
            )

        # Step 5: Update state history
        LOGGER.info("Updating state history...")
        save_current_state(item_objects)

        LOGGER.info("Check cycle completed successfully")
        return True

    except Exception as e:
        LOGGER.error(f"Unexpected error in check_and_notify: {e}", exc_info=True)
        try:
            send_error_notification(f"Unexpected error: {str(e)}", webhook_url)
        except:
            pass  # Don't fail if error notification fails
        return False


def run_continuous_monitor(check_interval_minutes: int = 30) -> None:
    """
    Run continuous monitoring loop with periodic checks.

    This function runs indefinitely, checking reorder points at the specified
    interval until shutdown is requested (Ctrl+C or SIGTERM).

    Args:
        check_interval_minutes: Minutes between checks (default 30)
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    LOGGER.info("=" * 60)
    LOGGER.info("Starting Reorder Point Monitor")
    LOGGER.info(f"Check interval: {check_interval_minutes} minutes")
    LOGGER.info("Press Ctrl+C to stop")
    LOGGER.info("=" * 60)

    check_count = 0

    while not _shutdown_requested:
        check_count += 1
        LOGGER.info(f"\n[Check #{check_count}] Starting monitoring cycle")

        try:
            success = check_and_notify()

            if success:
                LOGGER.info(f"Cycle #{check_count} completed successfully")
            else:
                LOGGER.warning(f"Cycle #{check_count} completed with errors")

        except Exception as e:
            LOGGER.error(f"Cycle #{check_count} failed with exception: {e}", exc_info=True)

        # Wait for next cycle (with periodic checks for shutdown signal)
        if not _shutdown_requested:
            next_check = datetime.now().timestamp() + (check_interval_minutes * 60)
            LOGGER.info(
                f"Next check in {check_interval_minutes} minutes "
                f"at {datetime.fromtimestamp(next_check).strftime('%H:%M:%S')}"
            )

            # Sleep in smaller intervals to check shutdown flag more frequently
            sleep_interval = 10  # Check every 10 seconds
            total_sleep = check_interval_minutes * 60

            for _ in range(int(total_sleep / sleep_interval)):
                if _shutdown_requested:
                    break
                time.sleep(sleep_interval)

    LOGGER.info("Shutdown requested, exiting gracefully...")
    LOGGER.info(f"Completed {check_count} monitoring cycles")


if __name__ == "__main__":
    # Allow running directly for testing
    import argparse

    parser = argparse.ArgumentParser(description="Reorder Point Monitor")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing or scheduled tasks)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in minutes (default: 30)"
    )

    args = parser.parse_args()

    if args.once:
        LOGGER.info("Running in single-check mode")
        success = check_and_notify()
        sys.exit(0 if success else 1)
    else:
        run_continuous_monitor(check_interval_minutes=args.interval)
