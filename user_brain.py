"""
User Brain - The Learning Intelligence System

Tracks every user interaction, learns preferences from behavior and ERP data,
and calculates a "Brain Health" score representing how well the system knows the user.

Usage:
    from user_brain import UserBrain
    
    brain = UserBrain(user_id="alex")
    brain.log_event("product_view", {"item": "NPK3011"})
    health = brain.get_brain_health()
"""

import json
import logging
import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

LOGGER = logging.getLogger(__name__)

# Storage location
DATA_DIR = Path(__file__).parent / "data"
EVENTS_FILE = DATA_DIR / "user_events.json"
PROFILES_FILE = DATA_DIR / "user_profiles.json"

# Minimum events needed before brain starts "knowing" you
MIN_EVENTS_FOR_LEARNING = 5
MAX_EVENTS_PER_USER = 1000  # Rolling window to prevent unbounded growth


class UserBrain:
    """
    The learning brain for a specific user.
    Tracks interactions, learns patterns, calculates confidence.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id or "anonymous"
        self._ensure_storage()
        self._events = self._load_events()
        self._profile = self._load_profile()
    
    def _ensure_storage(self) -> None:
        """Ensure data directory and files exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not EVENTS_FILE.exists():
            EVENTS_FILE.write_text("{}")
        if not PROFILES_FILE.exists():
            PROFILES_FILE.write_text("{}")
    
    def _load_events(self) -> list[dict]:
        """Load user's event history."""
        try:
            all_events = json.loads(EVENTS_FILE.read_text())
            return all_events.get(self.user_id, [])
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_events(self) -> None:
        """Save events to disk."""
        try:
            all_events = json.loads(EVENTS_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            all_events = {}
        
        # Keep only the last MAX_EVENTS_PER_USER events
        all_events[self.user_id] = self._events[-MAX_EVENTS_PER_USER:]
        EVENTS_FILE.write_text(json.dumps(all_events, indent=2, default=str))
    
    def _load_profile(self) -> dict:
        """Load learned user profile."""
        try:
            all_profiles = json.loads(PROFILES_FILE.read_text())
            return all_profiles.get(self.user_id, self._default_profile())
        except (json.JSONDecodeError, FileNotFoundError):
            return self._default_profile()
    
    def _save_profile(self) -> None:
        """Save profile to disk."""
        try:
            all_profiles = json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            all_profiles = {}
        
        all_profiles[self.user_id] = self._profile
        PROFILES_FILE.write_text(json.dumps(all_profiles, indent=2, default=str))
    
    def _default_profile(self) -> dict:
        """Default empty profile."""
        return {
            # UI-learned preferences
            "favorite_items": [],
            "favorite_categories": [],
            "favorite_vendors": [],
            "search_patterns": [],
            "active_hours": [],
            
            # ERP-learned patterns (THE DEEP STUFF)
            "buying_patterns": {
                "preferred_po_days": [],         # Days of week you typically do POs (0=Mon, 4=Fri)
                "preferred_po_week": [],         # Week of month (1-4)
                "avg_lead_time_preference": 0,   # How early you like to order
                "reorder_thresholds": {},        # Item -> days of runway when you reorder
                "vendor_preferences": {},        # Item -> preferred vendor
                "seasonal_patterns": {},         # Month -> items typically ordered
                "order_frequency": {},           # Item -> avg days between orders
                "typical_quantities": {},        # Item -> typical order quantity
            },
            
            # PO History Summary
            "po_history": {
                "total_pos": 0,
                "total_value": 0,
                "last_po_date": None,
                "items_ordered": [],
            },
            
            # Prediction tracking
            "prediction_hits": 0,
            "prediction_attempts": 0,
            
            # The Brain Center - Personal Connection & Insights
            "love_level": 0,         # Tracks affection <3
            "learning_journal": [],  # High-level insights ("I learned that...")
            "conversions": [],       # Track View -> Buy events
            
            # Digital Body Language (The "How")
            "decision_patterns": {
                "avg_decision_time": 0,       # Avg seconds to click an action
                "fastest_decisions": [],      # Items you decide on instantly
                "slowest_decisions": [],      # Items you ponder over
                "chart_interactions": 0,      # Times you engaged with charts
                "interaction_style": "Undetermined" # "Intuitive" or "Analytical"
            },
            
            "last_updated": None,
            "erp_sync_date": None,  # When we last synced from ERP
        }
    
    # =========================================================================
    # EVENT LOGGING
    # =========================================================================

    def log_decision(self, context: str, seconds: float, item: str = None) -> None:
        """
        Log 'Digital Body Language' - how fast did you act?
        """
        if seconds <= 0: return
        
        # Log the raw event
        self.log_event("decision_timing", {
            "context": context, "seconds": seconds, "item": item
        })
        
        # Update patterns
        dp = self._profile.setdefault("decision_patterns", {
            "avg_decision_time": 0, "fastest_decisions": [], 
            "slowest_decisions": [], "chart_interactions": 0,
            "interaction_style": "Undetermined"
        })
        
        # Update rolling average
        n = self._profile.get("prediction_attempts", 1) # simple proxy for N
        current_avg = dp.get("avg_decision_time", 0)
        dp["avg_decision_time"] = (current_avg * n + seconds) / (n + 1)
        
        # Classify speed for this item
        if item:
            if seconds < 5:
                # Fast!
                if item not in dp["fastest_decisions"]:
                    dp["fastest_decisions"].append(item)
                    dp["fastest_decisions"] = dp["fastest_decisions"][-10:] # Keep last 10
            elif seconds > 30:
                # Slow/Considered
                if item not in dp["slowest_decisions"]:
                    dp["slowest_decisions"].append(item)
                    dp["slowest_decisions"] = dp["slowest_decisions"][-10:]
            
        # Determine Style
        avg = dp["avg_decision_time"]
        if avg < 10:
            dp["interaction_style"] = "Intuitive / Decisive"
        elif avg > 30:
            dp["interaction_style"] = "Analytical / Deliberate"
        else:
            dp["interaction_style"] = "Balanced"
            
        self._save_profile()
    
    def log_event(self, event_type: str, payload: dict = None) -> None:
        """
        Log a user interaction event.
        
        Event Types:
            - product_view: User viewed a product
            - product_search: User searched for something
            - chat_question: User asked a question
            - buy_calendar_view: User checked buy calendar
            - vendor_view: User looked at vendor info
            - category_filter: User filtered by category
            - page_view: User navigated to a page
        """
        event = {
            "type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "hour": datetime.datetime.now().hour,
            "payload": payload or {}
        }
        
        self._events.append(event)
        self._save_events()
        
        # Update profile based on event
        self._learn_from_event(event)
        
        LOGGER.debug(f"Brain logged: {event_type} for user {self.user_id}")
    
    def _learn_from_event(self, event: dict) -> None:
        """Update user profile based on new event."""
        event_type = event["type"]
        payload = event.get("payload", {})
        
        # Track active hours
        hour = event.get("hour", 12)
        if hour not in self._profile["active_hours"]:
            self._profile["active_hours"].append(hour)
        
        # Learn from specific event types
        if event_type == "product_view":
            item = payload.get("item")
            if item and item not in self._profile["favorite_items"]:
                self._profile["favorite_items"].append(item)
                # Keep top 20
                self._profile["favorite_items"] = self._profile["favorite_items"][-20:]
        
        elif event_type == "category_filter":
            category = payload.get("category")
            if category and category not in self._profile["favorite_categories"]:
                self._profile["favorite_categories"].append(category)
        
        elif event_type == "vendor_view":
            vendor = payload.get("vendor")
            if vendor and vendor not in self._profile["favorite_vendors"]:
                self._profile["favorite_vendors"].append(vendor)
        
        elif event_type == "product_search":
            query = payload.get("query", "")
            if query:
                self._profile["search_patterns"].append(query.lower())
                # Keep last 50 searches
                self._profile["search_patterns"] = self._profile["search_patterns"][-50:]
        
        self._profile["last_updated"] = datetime.datetime.now().isoformat()
        self._save_profile()
    
    # =========================================================================
    # BRAIN HEALTH CALCULATION
    # =========================================================================
    
    def get_brain_health(self) -> float:
        """
        Calculate brain health as a 0-100% score.
        
        Components:
            - Interaction depth (30%): How many events logged
            - Profile richness (40%): How much we know about preferences
            - Prediction accuracy (30%): How often our predictions are right
        """
        event_count = len(self._events)
        
        # Interaction Depth Score (0-1)
        # Scales from 0 at 0 events to 1 at 100+ events
        interaction_score = min(event_count / 100, 1.0)
        
        # Profile Richness Score (0-1)
        profile_items = (
            len(self._profile.get("favorite_items", [])) +
            len(self._profile.get("favorite_categories", [])) +
            len(self._profile.get("favorite_vendors", [])) +
            len(self._profile.get("search_patterns", []))
        )
        richness_score = min(profile_items / 30, 1.0)  # Max out at 30 items
        
        # Prediction Accuracy Score (0-1)
        attempts = self._profile.get("prediction_attempts", 0)
        hits = self._profile.get("prediction_hits", 0)
        if attempts > 0:
            accuracy_score = hits / attempts
        else:
            accuracy_score = 0.5  # Neutral until we have data
        
        # Weighted combination
        health = (
            interaction_score * 0.30 +
            richness_score * 0.40 +
            accuracy_score * 0.30
        )
        
        # Ensure minimum 10% if user exists, scale to 10-99%
        if event_count > 0:
            health = 0.10 + (health * 0.89)
        else:
            health = 0.10  # New user starts at 10%
        
        return round(health, 2)
    
    # =========================================================================
    # PREDICTIONS & SUGGESTIONS
    # =========================================================================
    
    def get_predictions(self) -> dict:
        """
        Generate predictions about what the user might want.
        
        Returns:
            {
                "likely_items": ["NPK3011", "CHEACETIC"],
                "likely_time": "morning",
                "suggested_action": "Check buy calendar for NPK items"
            }
        """
        predictions = {
            "likely_items": [],
            "likely_time": "unknown",
            "suggested_action": None
        }
        
        # Not enough data yet
        if len(self._events) < MIN_EVENTS_FOR_LEARNING:
            return predictions
        
        # Predict likely items (most viewed)
        item_counts = Counter()
        for event in self._events:
            if event.get("type") == "product_view":
                item = event.get("payload", {}).get("item")
                if item:
                    item_counts[item] += 1
        
        predictions["likely_items"] = [item for item, _ in item_counts.most_common(5)]
        
        # Predict active time
        hour_counts = Counter(self._profile.get("active_hours", []))
        if hour_counts:
            peak_hour = hour_counts.most_common(1)[0][0]
            if 5 <= peak_hour < 12:
                predictions["likely_time"] = "morning"
            elif 12 <= peak_hour < 17:
                predictions["likely_time"] = "afternoon"
            else:
                predictions["likely_time"] = "evening"
        
        # Generate suggestion
        if predictions["likely_items"]:
            top_item = predictions["likely_items"][0]
            predictions["suggested_action"] = f"Check status of {top_item}"
        
        return predictions
    
    def record_prediction_result(self, was_correct: bool) -> None:
        """Record whether a prediction was accurate."""
        self._profile["prediction_attempts"] = self._profile.get("prediction_attempts", 0) + 1
        if was_correct:
            self._profile["prediction_hits"] = self._profile.get("prediction_hits", 0) + 1
        self._save_profile()
    
    # =========================================================================
    # ERP DATA INTEGRATION - THE DEEP LEARNING
    # =========================================================================
    
    def learn_from_po_history(self, cursor) -> dict:
        """
        Learn buying patterns from actual ERP PO history.
        This is where the brain BECOMES YOU.
        
        Learns:
            - What days/weeks you typically place POs
            - Your vendor preferences per item
            - Your typical order quantities
            - Seasonal buying patterns
            - Order frequency per item
        
        Returns:
            Summary of what was learned
        """
        from collections import Counter, defaultdict
        import datetime
        
        # Initialize buying patterns if missing
        if "buying_patterns" not in self._profile:
            self._profile["buying_patterns"] = self._default_profile()["buying_patterns"]
        
        bp = self._profile["buying_patterns"]
        
        # Query PO history (last 2 years)
        query = """
        SELECT 
            h.PONUMBER,
            h.DOCDATE,
            h.VENDORID,
            l.ITEMNMBR,
            l.QTYORDER,
            l.UNITCOST * l.QTYORDER AS LineTotal
        FROM POP30100 h
        JOIN POP30110 l ON h.PONUMBER = l.PONUMBER
        WHERE h.DOCDATE >= DATEADD(year, -2, GETDATE())
        ORDER BY h.DOCDATE DESC
        """
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except Exception as e:
            LOGGER.warning(f"Failed to query PO history: {e}")
            return {"error": str(e)}
        
        if not rows:
            return {"message": "No PO history found"}
        
        # Tracking structures
        po_days = []           # Day of week (0-6)
        po_weeks = []          # Week of month (1-4)
        vendor_by_item = defaultdict(Counter)
        qty_by_item = defaultdict(list)
        month_items = defaultdict(list)
        item_dates = defaultdict(list)
        total_value = 0
        po_numbers = set()
        items_ordered = set()
        last_po_date = None
        
        for row in rows:
            po_number = row[0]
            doc_date = row[1]
            vendor_id = str(row[2]).strip() if row[2] else None
            item = str(row[3]).strip() if row[3] else None
            qty = float(row[4] or 0)
            line_total = float(row[5] or 0)
            
            if not doc_date or not item:
                continue
            
            # Parse date
            if isinstance(doc_date, str):
                try:
                    doc_date = datetime.datetime.fromisoformat(doc_date)
                except ValueError:
                    continue
            
            # Track PO timing patterns
            po_days.append(doc_date.weekday())
            week_of_month = min((doc_date.day - 1) // 7 + 1, 4)
            po_weeks.append(week_of_month)
            
            # Track vendor preferences per item
            if vendor_id:
                vendor_by_item[item][vendor_id] += 1
            
            # Track typical quantities
            if qty > 0:
                qty_by_item[item].append(qty)
            
            # Track seasonal patterns (what items in what month)
            month_items[doc_date.month].append(item)
            
            # Track order frequency
            item_dates[item].append(doc_date)
            
            # Aggregate totals
            total_value += line_total
            po_numbers.add(po_number)
            items_ordered.add(item)
            
            if last_po_date is None or doc_date > last_po_date:
                last_po_date = doc_date
        
        # Calculate preferred PO days (most common)
        day_counts = Counter(po_days)
        bp["preferred_po_days"] = [day for day, _ in day_counts.most_common(3)]
        
        # Calculate preferred weeks of month
        week_counts = Counter(po_weeks)
        bp["preferred_po_week"] = [week for week, _ in week_counts.most_common(2)]
        
        # Calculate vendor preferences per item
        bp["vendor_preferences"] = {}
        for item, vendors in vendor_by_item.items():
            if vendors:
                bp["vendor_preferences"][item] = vendors.most_common(1)[0][0]
        
        # Calculate typical quantities
        bp["typical_quantities"] = {}
        for item, qtys in qty_by_item.items():
            if qtys:
                bp["typical_quantities"][item] = round(sum(qtys) / len(qtys), 2)
        
        # Calculate seasonal patterns (top 5 items per month)
        bp["seasonal_patterns"] = {}
        for month, items in month_items.items():
            item_counts = Counter(items)
            bp["seasonal_patterns"][str(month)] = [i for i, _ in item_counts.most_common(5)]
        
        # Calculate order frequency (avg days between orders per item)
        bp["order_frequency"] = {}
        for item, dates in item_dates.items():
            if len(dates) >= 2:
                sorted_dates = sorted(dates)
                gaps = [(sorted_dates[i+1] - sorted_dates[i]).days 
                        for i in range(len(sorted_dates)-1)]
                if gaps:
                    bp["order_frequency"][item] = round(sum(gaps) / len(gaps), 1)
        
        # Update PO history summary
        self._profile["po_history"] = {
            "total_pos": len(po_numbers),
            "total_value": round(total_value, 2),
            "last_po_date": last_po_date.isoformat() if last_po_date else None,
            "items_ordered": list(items_ordered)[:100],  # Keep top 100
        }
        
        # Update favorite items/vendors from ERP
        self._profile["favorite_items"] = list(items_ordered)[:50]
        self._profile["favorite_vendors"] = list(set(
            v for vendors in vendor_by_item.values() 
            for v in vendors.keys()
        ))[:20]
        
        # Mark sync time
        self._profile["erp_sync_date"] = datetime.datetime.now().isoformat()
        self._save_profile()
        
        summary = {
            "pos_analyzed": len(po_numbers),
            "items_learned": len(items_ordered),
            "vendors_learned": len(self._profile["favorite_vendors"]),
            "preferred_days": [["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d] for d in bp["preferred_po_days"]],
            "brain_health": self.get_brain_health()
        }
        
        LOGGER.info(f"Brain learned from {len(po_numbers)} POs: {summary}")
        return summary
    
    def learn_from_erp(self, purchase_history: list[dict]) -> None:
        """
        Learn from a pre-fetched list of purchases (legacy method).
        For full learning, use learn_from_po_history() with a cursor.
        """
        if not purchase_history:
            return
        
        for purchase in purchase_history:
            item = purchase.get("item") or purchase.get("ITEMNMBR")
            vendor = purchase.get("vendor") or purchase.get("VENDORID")
            
            if item and item not in self._profile["favorite_items"]:
                self._profile["favorite_items"].append(item)
            
            if vendor and vendor not in self._profile["favorite_vendors"]:
                self._profile["favorite_vendors"].append(vendor)
        
        self._profile["favorite_items"] = self._profile["favorite_items"][-50:]
        self._profile["favorite_vendors"] = self._profile["favorite_vendors"][-20:]
        self._save_profile()

    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return len(self._events)
    
    def get_profile_summary(self) -> dict:
        """Get a summary of the learned profile."""
        return {
            "event_count": len(self._events),
            "favorite_items": self._profile.get("favorite_items", [])[:5],
            "favorite_vendors": self._profile.get("favorite_vendors", [])[:3],
            "active_hours": sorted(set(self._profile.get("active_hours", [])))[:5],
            "brain_health": self.get_brain_health()
        }
    
    def reset(self) -> None:
        """Reset all learned data for this user."""
        self._events = []
        self._profile = self._default_profile()
        self._save_events()
        self._save_profile()
        LOGGER.info(f"Brain reset for user {self.user_id}")
    
    # =========================================================================
    # THE BRAIN CENTER - EXTENSIVE TRAINING
    # =========================================================================

    def receive_love(self) -> int:
        """User expressed love (<3). Increase love_level and return new value."""
        self._profile["love_level"] = self._profile.get("love_level", 0) + 1
        self._save_profile()
        return self._profile["love_level"]

    def correlate_views_with_purchases(self, cursor) -> list[dict]:
        """
        The Core Training Loop: Connect App Behavior -> Real ERP Actions.
        Finds instances where user Viewed an item -> Then Bought it.
        """
        import datetime
        
        # 1. Get recent app views (last 30 days)
        recent_views = {} # item -> [timestamps]
        cutoff = datetime.datetime.now() - datetime.timedelta(days=30)
        
        for event in self._events:
            try:
                # Handle potential timestamp format issues
                ts = datetime.datetime.fromisoformat(event["timestamp"])
                if ts < cutoff:
                    continue
                
                if event["type"] == "product_view":
                    item = event["payload"].get("item")
                    if item:
                        if item not in recent_views:
                            recent_views[item] = []
                        recent_views[item].append(ts)
            except Exception:
                continue
        
        if not recent_views:
            return []

        # 2. Get recent POs (last 30 days) matching these items
        # Escape matched items for SQL
        items_list = list(recent_views.keys())
        if not items_list:
            return []
            
        # Chunk validation to avoid massive IN clauses
        items_str = "', '".join([str(i).replace("'", "''") for i in items_list[:50]]) 
        
        query = f"""
        SELECT h.PONUMBER, h.DOCDATE, l.ITEMNMBR, l.QTYORDER
        FROM POP30100 h
        JOIN POP30110 l ON h.PONUMBER = l.PONUMBER
        WHERE l.ITEMNMBR IN ('{items_str}')
          AND h.DOCDATE >= DATEADD(day, -30, GETDATE())
        """
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
        except Exception as e:
            LOGGER.error(f"Failed to correlation query: {e}")
            return []

        correlations = []
        
        # 3. Match them up
        for row in rows:
            po_num, po_date_thing, item, qty = row
            item = str(item).strip()
            
            # Convert PO date to datetime (start of day)
            po_date = None
            if isinstance(po_date_thing, str):
                try:
                    po_date = datetime.datetime.fromisoformat(po_date_thing)
                except ValueError:
                    continue
            elif isinstance(po_date_thing, (datetime.date, datetime.datetime)):
                po_date = datetime.datetime(po_date_thing.year, po_date_thing.month, po_date_thing.day)
            
            if not po_date:
                continue
            
            # Check for views BEFORE this PO
            if item in recent_views:
                for view_date in recent_views[item]:
                    # If view was within 7 days BEFORE the PO
                    delta = po_date - view_date
                    # Allow same day (delta 0) to 7 days prior
                    # Note: delta might be negative if view was later in the day than PO timestamp (usually midnight)
                    # Use absolute check or generous window
                    days_diff = delta.total_seconds() / 86400
                    
                    if -1 <= days_diff <= 7: # View happened just before or same day as PO
                        insight = {
                            "item": item,
                            "view_date": view_date.isoformat(),
                            "po_date": po_date.isoformat(),
                            "po_number": po_num,
                            "type": "Actionable",
                            "message": f"You researched {item} on {view_date.strftime('%A')} and bought it on {po_date.strftime('%A')}."
                        }
                        correlations.append(insight)
                        self._log_learning(insight["message"])
                        break # Count once per PO
        
        self._profile["conversions"] = correlations[-50:] # Keep last 50
        self._save_profile()
        return correlations

    def _log_learning(self, message: str) -> None:
        """Add a high-level insight to the journal."""
        journal = self._profile.get("learning_journal", [])
        # Check if message already exists (simple de-dupe)
        if not any(entry.endswith(message) for entry in journal): 
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            entry = f"[{timestamp}] {message}"
            journal.insert(0, entry) # Newest first
            self._profile["learning_journal"] = journal[:100] # Keep last 100
            self._save_profile()

    def get_weekly_report(self) -> str:
        """Generate a natural language report of recent learning."""
        journal = self._profile.get("learning_journal", [])
        love = self._profile.get("love_level", 0)
        
        if not journal:
            return f"I am watching closely. <3 (Love Score: {love})"
            
        report = f"**Training Report** (Love Level: {love} <3)\n\n"
        report += "Here is what I have learned from your actions:\n\n"
        for entry in journal[:5]:
            report += f"- {entry}\n"
            
        return report


# =============================================================================
# THE HIVE MIND - COLLECTIVE INTELLIGENCE 🐝
# =============================================================================

class HiveBrain:
    """
    Aggregates intelligence across ALL users.
    'The Collective Wisdom'
    """
    def __init__(self):
        self._ensure_storage()
    
    def _ensure_storage(self) -> None:
        if not PROFILES_FILE.exists():
            PROFILES_FILE.write_text("{}")
            
    def get_hive_stats(self) -> dict:
        """
        Calculate aggregate stats for the team.
        """
        try:
            profiles = json.loads(PROFILES_FILE.read_text())
        except Exception:
            return {}
            
        stats = {
            "total_users": len(profiles),
            "avg_decision_time": 0,
            "top_vendors": Counter(),
            "top_items": Counter(),
            "interaction_styles": Counter(),
            "total_knowledge_events": 0
        }
        
        timed_users = 0
        
        for user_id, p in profiles.items():
            # Decision Time
            dp = p.get("decision_patterns", {})
            avg_time = dp.get("avg_decision_time", 0)
            if avg_time > 0:
                stats["avg_decision_time"] += avg_time
                timed_users += 1
                
            # Styles
            style = dp.get("interaction_style", "Undetermined")
            stats["interaction_styles"][style] += 1
            
            # Vendors (Favorite Vendors)
            for v in p.get("favorite_vendors", []):
                stats["top_vendors"][v] += 1
                
            # Items
            for i in p.get("favorite_items", []):
                stats["top_items"][i] += 1
            
            # Knowledge
            stats["total_knowledge_events"] += len(p.get("learning_journal", []))
            
        if timed_users > 0:
            stats["avg_decision_time"] = round(stats["avg_decision_time"] / timed_users, 1)
            
        return stats

    def compare_user_to_hive(self, user_id: str) -> dict:
        """
        Compare a specific user to the hive average.
        """
        hive = self.get_hive_stats()
        if not hive: return {}
        
        try:
            profiles = json.loads(PROFILES_FILE.read_text())
            user_profile = profiles.get(user_id, {})
        except:
            return {}
            
        user_dp = user_profile.get("decision_patterns", {})
        user_time = user_dp.get("avg_decision_time", 0)
        
        comparison = {
            "speed_diff_pct": 0,
            "style_match": False,
            "unique_vendors": []
        }
        
        # Speed Comparison
        if hive["avg_decision_time"] > 0 and user_time > 0:
            # +30% means user is 30% SLOWER (took more time)
            diff = (user_time - hive["avg_decision_time"]) / hive["avg_decision_time"]
            comparison["speed_diff_pct"] = round(diff * 100, 1)
            
        # Vendor Uniqueness (Variables user likes but Hive usually doesn't pick)
        # (Simplified: User likes it, but it's not in Top 3 of Hive)
        top_hive_vendors = [v for v, _ in hive["top_vendors"].most_common(3)]
        user_vendors = user_profile.get("favorite_vendors", [])
        comparison["unique_vendors"] = [v for v in user_vendors if v not in top_hive_vendors][:3]
        
        return comparison

# Global Hive Instance
_hive = HiveBrain()

def get_hive() -> HiveBrain:
    return _hive


# =============================================================================
# GLOBAL BRAIN INSTANCE FACTORY
# =============================================================================

_brains: dict[str, UserBrain] = {}

def get_brain(user_id: str = "default") -> UserBrain:
    """
    Get or create a brain instance for a user.
    Uses a simple cache to avoid repeated disk reads.
    """
    if user_id not in _brains:
        _brains[user_id] = UserBrain(user_id)
    return _brains[user_id]
