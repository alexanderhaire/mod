import streamlit as st
from typing import Callable
import auth


# Tour step definitions
TOUR_STEPS = [
    {
        "title": "👋 Welcome to Chemical Market Terminal",
        "content": """
This is your powerful procurement intelligence platform. Let's take a quick tour of the key features.

**What you can do here:**
- Ask questions in natural language
- Analyze product pricing & trends  
- Track usage patterns
- Monitor market conditions
        """,
        "icon": "🚀"
    },
    {
        "title": "💬 AI-Powered Chat",
        "content": """
The chat interface is your main way to interact with the system.

**Try asking things like:**
- "What's the current price of Citric Acid?"
- "Show me my top 10 purchases last month"
- "Which vendors have the best prices?"

Just type naturally - the AI understands your intent!
        """,
        "icon": "🤖"
    },
    {
        "title": "🔍 Product Insights",
        "content": """
Search for any product to get detailed analytics.

**You'll see:**
- Current pricing & cost trends
- Inventory levels & runway
- Vendor information
- Buying signals (BUY NOW / HOLD / WAIT)
        """,
        "icon": "📊"
    },
    {
        "title": "📈 Price History Charts",
        "content": """
Track how costs change over time for any product.

**Features:**
- Adjust date ranges to zoom in/out
- Compare Landed Cost vs Delivered Cost
- See quantity bars overlaid on price lines
- View summary stats (total spend, weighted avg)
        """,
        "icon": "💰"
    },
    {
        "title": "📉 Usage Analytics",
        "content": """
Understand consumption patterns to plan purchases better.

**View modes:**
- Monthly breakdown to spot seasonal trends
- All-time view for complete history
- Date range filtering for specific periods
        """,
        "icon": "📆"
    },
    {
        "title": "🌐 Market Monitor",
        "content": """
Get a bird's-eye view of market conditions.

**Includes:**
- Top price movers (up & down)
- Purchase volume trends
- Cost index tracking
- System alerts for opportunities

**You're all set!** Click 'Finish' to start exploring.
        """,
        "icon": "✅"
    },
]


class OnboardingTour:
    """Manages the onboarding tour state in Streamlit session."""
    
    STATE_KEY = "onboarding_tour"
    
    def __init__(self):
        """Initialize tour state if not exists."""
        if self.STATE_KEY not in st.session_state:
            st.session_state[self.STATE_KEY] = {
                "active": False,
                "step": 0,
                "completed": False,
            }
    
    @property
    def _state(self) -> dict:
        return st.session_state.get(self.STATE_KEY, {})
    
    @property
    def is_active(self) -> bool:
        return self._state.get("active", False)
    
    @property
    def current_step(self) -> int:
        return self._state.get("step", 0)
    
    @property
    def is_completed(self) -> bool:
        return self._state.get("completed", False)
    
    @property
    def total_steps(self) -> int:
        return len(TOUR_STEPS)
    
    @property
    def current_step_data(self) -> dict:
        idx = self.current_step
        if 0 <= idx < len(TOUR_STEPS):
            return TOUR_STEPS[idx]
        return {}
    
    def start(self) -> None:
        """Start the tour from the beginning."""
        st.session_state[self.STATE_KEY] = {
            "active": True,
            "step": 0,
            "completed": False,
        }
    
    def next_step(self) -> bool:
        """Move to the next step. Returns True if there's a next step."""
        if self.current_step < self.total_steps - 1:
            st.session_state[self.STATE_KEY]["step"] = self.current_step + 1
            return True
        return False
    
    def prev_step(self) -> bool:
        """Move to the previous step. Returns True if there's a previous step."""
        if self.current_step > 0:
            st.session_state[self.STATE_KEY]["step"] = self.current_step - 1
            return True
        return False
    
    def complete(self) -> None:
        """Mark tour as completed and deactivate."""
        # Persist completion if user is logged in
        if st.session_state.get("user"):
            auth.set_tour_completed(st.session_state.user)
            
        st.session_state[self.STATE_KEY] = {
            "active": False,
            "step": 0,
            "completed": True,
        }
    
    def skip(self) -> None:
        """Skip the tour (same as complete for now)."""
        self.complete()
    
    def should_show(self) -> bool:
        """Check if tour should be shown (first time user who hasn't completed)."""
        # 1. Check session state first
        if self.is_completed:
            return False
            
        # 2. Check persistent user store
        user = st.session_state.get("user")
        if user:
            if auth.has_completed_tour(user):
                return False
            return True
            
        # 3. No user logged in yet -> Do not show tour
        # The tour is personalized and requires a user context.
        return False


def render_tour_overlay(tour: OnboardingTour) -> None:
    """Render the tour as a prominent panel at the top of the app."""
    if not tour.is_active:
        return
    
    step_data = tour.current_step_data
    step_num = tour.current_step + 1
    total = tour.total_steps
    is_last = step_num == total
    is_first = step_num == 1
    
    # Render a high-visibility container
    with st.container():
        st.markdown("""
        <style>
        .tour-panel {
            background: #0a0a0a;
            border: 2px solid #ffb000;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(255, 176, 0, 0.2);
            font-family: 'Inconsolata', monospace;
        }
        .tour-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            border-bottom: 1px solid #333;
            padding-bottom: 0.5rem;
        }
        .tour-step {
            color: #ffb000;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .tour-title {
            color: #fff;
            font-size: 1.4rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .tour-content {
            color: #ccc;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate progress
        progress = step_num / total
        
        # Panel Content
        content = step_data.get("content", "")
        icon = step_data.get("icon", "")
        title = step_data.get("title", "").replace(icon, "").strip()
        
        st.markdown(f"""
        <div class="tour-panel">
            <div class="tour-header">
                <span class="tour-step">STEP {step_num} OF {total}</span>
                <span style="font-size: 1.5rem;">{icon}</span>
            </div>
            <div class="tour-title">{title}</div>
            <div class="tour-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar aligned with visual style
        st.progress(progress)
        
        # Controls (Native Streamlit Buttons)
        # Use columns to align them
        c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
        
        with c1:
            if not is_first:
                if st.button("← Back", key="tour_nav_prev"):
                    tour.prev_step()
                    st.rerun()
        
        with c3:
            if st.button("Skip Tour", key="tour_nav_skip"):
                tour.skip()
                st.rerun()
                
        with c4:
            if is_last:
                if st.button("Finish ✓", type="primary", key="tour_nav_finish"):
                    tour.complete()
                    st.rerun()
            else:
                if st.button("Next →", type="primary", key="tour_nav_next"):
                    tour.next_step()
                    st.rerun()
        
        st.markdown("---") # Visual separator from main app content


def init_tour_for_new_user() -> OnboardingTour:
    """
    Initialize and return tour instance.
    Automatically starts tour for first-time users.
    """
    tour = OnboardingTour()
    
    # Auto-start for new users who haven't completed
    if tour.should_show() and not tour.is_active:
        tour.start()
    
    return tour
