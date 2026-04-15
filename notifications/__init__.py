"""
Notification system for reorder point alerts.

This package provides Teams webhook integration and state tracking
to send real-time alerts when items cross reorder point thresholds.
"""

from .teams_notifier import send_reorder_alert, send_error_notification
from .state_tracker import (
    load_previous_state,
    save_current_state,
    detect_new_reorder_items,
    should_notify
)

__all__ = [
    'send_reorder_alert',
    'send_error_notification',
    'load_previous_state',
    'save_current_state',
    'detect_new_reorder_items',
    'should_notify'
]
