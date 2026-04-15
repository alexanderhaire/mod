"""
Microsoft Teams webhook integration for reorder point alerts.

Sends formatted Adaptive Card messages to Teams channels when items
hit reorder points or when errors occur.
"""

import requests
import time
from typing import Optional
from logging_config import get_logger
from db_pool import get_connection
from inventory_queries import fetch_parent_items_for_component

LOGGER = get_logger(__name__)


def send_reorder_alert(item, webhook_url: str, max_retries: int = 2) -> bool:
    """
    Send a reorder point alert to Microsoft Teams.

    Args:
        item: ReorderRecommendation object with item details
        webhook_url: Teams incoming webhook URL
        max_retries: Number of retry attempts on failure

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not webhook_url:
        LOGGER.error("Teams webhook URL not configured")
        return False

    # Determine color and icon based on urgency
    if item.urgency == "Critical":
        color = "FF0000"  # Red
        icon = "🚨"
        urgency_text = "Critical - Order Today"
    elif item.urgency == "Soon":
        color = "FFA500"  # Orange
        icon = "⚠️"
        urgency_text = "Order Soon (Within 7 Days)"
    else:
        # Shouldn't happen, but handle gracefully
        color = "808080"  # Gray
        icon = "ℹ️"
        urgency_text = "Information"

    # Query top 3 selling items that use this raw material
    top_sellers_text = "N/A"
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # First, get direct parent items from BOM
            parent_items, _ = fetch_parent_items_for_component(cursor, item.item_number)

            if parent_items and len(parent_items) > 0:
                # Collect base names to search for variants
                base_names = set()
                for row in parent_items:
                    parent_item = row.ParentItem.strip() if hasattr(row, 'ParentItem') else str(row[0]).strip()

                    # Extract base name (remove 00/02/250/30 suffix)
                    base_name = parent_item
                    if parent_item.endswith('00') or parent_item.endswith('02') or parent_item.endswith('30'):
                        base_name = parent_item[:-2]
                    elif parent_item.endswith('250'):
                        base_name = parent_item[:-3]

                    base_names.add(base_name)

                # Now query for ALL variants of these base items and their sales
                family_sales = {}
                for base_name in base_names:
                    # Query for all items starting with this base name and get their sales
                    variant_query = """
                        SELECT
                            i.ITEMNMBR AS ItemNumber,
                            i.ITEMDESC AS ItemDescription,
                            ISNULL(s.TotalSales, 0) as Volume
                        FROM IV00101 i
                        OUTER APPLY (
                            SELECT SUM(d.QTYFULFI) as TotalSales
                            FROM SOP30300 d
                            JOIN SOP30200 h ON d.SOPNUMBE = h.SOPNUMBE
                            WHERE d.ITEMNMBR = i.ITEMNMBR
                              AND h.DOCDATE >= DATEADD(year, -1, GETDATE())
                              AND h.SOPTYPE = 3
                        ) s
                        WHERE i.ITEMNMBR LIKE ? + '%'
                          AND i.ITEMTYPE = 1
                          AND i.INACTIVE = 0
                    """

                    cursor.execute(variant_query, base_name)
                    variant_rows = cursor.fetchall()

                    if variant_rows:
                        family_sales[base_name] = {
                            'base_name': base_name,
                            'total_volume': 0,
                            'variants': []
                        }

                        for variant_row in variant_rows:
                            item_num = variant_row.ItemNumber.strip() if hasattr(variant_row, 'ItemNumber') else str(variant_row[0]).strip()
                            volume = float(variant_row.Volume) if hasattr(variant_row, 'Volume') else float(variant_row[2] or 0)

                            family_sales[base_name]['total_volume'] += volume
                            if volume > 0:
                                family_sales[base_name]['variants'].append((item_num, volume))

                # Sort families by total sales volume
                sorted_families = sorted(family_sales.values(), key=lambda x: x['total_volume'], reverse=True)

                # Get top 3 families
                top_3 = sorted_families[:3]
                sellers = []
                for family in top_3:
                    base = family['base_name']
                    total = family['total_volume']
                    # Show the base name with total sales
                    sellers.append(f"{base} family ({total:,.0f} units sold)")

                top_sellers_text = "; ".join(sellers) if sellers else "No sales in past year"
    except Exception as e:
        LOGGER.warning(f"Failed to fetch top sellers for {item.item_number}: {e}")
        top_sellers_text = "Unable to retrieve"

    # Build Adaptive Card message
    message = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": color,
        "summary": f"{icon} {item.item_number} has hit reorder point",
        "sections": [
            {
                "activityTitle": f"{icon} Reorder Alert - {item.urgency}",
                "activitySubtitle": "Item has crossed reorder threshold",
                "facts": [
                    {"name": "Item", "value": f"{item.item_number} - {item.item_description}"},
                    {"name": "On Hand", "value": f"{item.qty_on_hand:,.1f} units"},
                    {"name": "On Order", "value": f"{item.qty_on_order:,.1f} units"},
                    {"name": "Available", "value": f"{item.qty_available:,.1f} units"},
                    {"name": "Avg Usage/Day", "value": f"{item.avg_daily_usage:,.1f} units/day"},
                    {"name": "Days of Coverage", "value": f"{item.days_of_coverage:.1f} days"},
                    {"name": "Order Point Qty", "value": f"{item.gp_order_point:,.1f} units"},
                    {"name": "Top 3 Selling Items", "value": top_sellers_text}
                ]
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "View Reorder Dashboard",
                "targets": [
                    {"os": "default", "uri": "http://localhost:8501/reorder_recommendations"}
                ]
            }
        ]
    }

    # Attempt to send with retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                webhook_url,
                json=message,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            LOGGER.info(
                f"Successfully sent Teams alert for {item.item_number} "
                f"(urgency: {item.urgency}, attempt {attempt + 1})"
            )
            return True

        except requests.exceptions.RequestException as e:
            LOGGER.warning(
                f"Attempt {attempt + 1} of {max_retries} failed for {item.item_number}: {e}"
            )

            # Wait before retrying (except on last attempt)
            if attempt < max_retries - 1:
                time.sleep(30)

    LOGGER.error(
        f"Failed to send Teams alert for {item.item_number} after {max_retries} attempts"
    )
    return False


def send_error_notification(error_message: str, webhook_url: Optional[str] = None) -> bool:
    """
    Send an error notification to Teams.

    Args:
        error_message: Description of the error
        webhook_url: Teams webhook URL (optional, loads from config if not provided)

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not webhook_url:
        LOGGER.warning("Cannot send error notification: no webhook URL provided")
        return False

    message = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": "FF0000",
        "summary": "⚠️ Reorder Monitor Error",
        "sections": [
            {
                "activityTitle": "⚠️ Reorder Monitor Error",
                "activitySubtitle": "The reorder point monitoring system encountered an error",
                "facts": [
                    {"name": "Error", "value": error_message},
                    {"name": "Status", "value": "Monitor will retry on next cycle"}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            json=message,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        LOGGER.info("Sent error notification to Teams")
        return True

    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Failed to send error notification to Teams: {e}")
        return False


def send_test_message(webhook_url: str) -> bool:
    """
    Send a test message to verify webhook configuration.

    Args:
        webhook_url: Teams incoming webhook URL

    Returns:
        True if test message sent successfully, False otherwise
    """
    message = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": "00FF00",
        "summary": "✅ Reorder Monitor Test",
        "sections": [
            {
                "activityTitle": "✅ Reorder Monitor Configuration Test",
                "activitySubtitle": "This is a test message to verify Teams webhook integration",
                "facts": [
                    {"name": "Status", "value": "Webhook configured correctly"},
                    {"name": "System", "value": "Reorder Point Notification System"}
                ]
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            json=message,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        LOGGER.info("Test message sent successfully")
        return True

    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Failed to send test message: {e}")
        return False



def send_raw_material_alert(item: dict, webhook_url: str) -> bool:
    """
    Send a raw material shortage alert to Microsoft Teams.

    Args:
        item: Dictionary containing item details (ItemNumber, ItemDescription, etc.)
        webhook_url: Teams incoming webhook URL

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not webhook_url:
        LOGGER.error("Teams webhook URL not configured")
        return False

    item_number = item.get('ItemNumber', 'Unknown')
    description = item.get('ItemDescription', 'Unknown')
    on_hand = float(item.get('QtyOnHand', 0))
    allocated = float(item.get('QtyAllocated', 0))
    available = on_hand - allocated
    order_point = float(item.get('OrderPointQty', 0))
    
    # Calculate deficit
    shortage_amt = order_point - available
    
    # On Order is provided, handle potentially missing key gracefully
    on_order = float(item.get('QtyOnOrder', 0))
    
    # Helper to format numbers nicely
    def fmt(n):
        return f"{n:,.2f}".rstrip('0').rstrip('.')

    # Construct Teams Message Card (Red Alert)
    message = {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "themeColor": "FF0000", # Red for critical raw material shortage
        "summary": f"🚨 RAW MATERIAL ALERT: {item_number}",
        "sections": [
            {
                "activityTitle": f"🚨 RAW MATERIAL SHORTAGE: {item_number}",
                "activitySubtitle": description,
                "facts": [
                    {"name": "Status", "value": "BELOW ORDER POINT"},
                    {"name": "Available", "value": f"{fmt(available)}"},
                    {"name": "Order Point", "value": f"{fmt(order_point)}"},
                    {"name": "Deficit", "value": f"-{fmt(shortage_amt)}"},
                    {"name": "On Order", "value": f"{fmt(on_order)}"},
                    {"name": "On Hand", "value": f"{fmt(on_hand)}"},
                    {"name": "Allocated", "value": f"{fmt(allocated)}"}
                ],
                "markdown": True
            }
        ],
        "potentialAction": [
            {
                "@type": "OpenUri",
                "name": "View Item in GP",
                "targets": [
                    {"os": "default", "uri": f"https://kpi.com/item/{item_number}"} # Placeholder URI
                ]
            }
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            json=message,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        LOGGER.info(f"Sent Raw Material alert for {item_number}")
        return True

    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Failed to send Raw Material alert for {item_number}: {e}")
        return False
