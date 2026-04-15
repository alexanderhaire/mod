import streamlit as st
import pandas as pd
from ui_utils import render_pulse_header
from user_brain import get_brain, get_hive
from db_pool import get_connection as get_pooled_connection
from procurement_ml import ProcurementMLOptimizer

# Page Config
st.set_page_config(page_title="Brain Center", page_icon="🧠", layout="wide")

def render_brain_center():
    # Initialize
    current_user = st.session_state.get("user", "default")
    brain = get_brain(current_user)
    
    # Render unifying header
    render_pulse_header(user_id=current_user)
    
    st.title("🧠 The Brain Center")
    st.markdown("### *Where Man and Machine Align*")
    
    # Custom CSS for Brain Center
    st.markdown("""
    <style>
    .brain-card {
        background-color: #002b36;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2aa198;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .brain-metric-label {
        color: #839496;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .brain-metric-value {
        color: #fdf6e3;
        font-size: 2rem;
        font-weight: bold;
    }
    .brain-metric-delta {
        color: #859900;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    tab_me, tab_hive = st.tabs(["👤 My Brain", "🐝 Hive Mind"])
    
    # =========================================================================
    # TAB 1: MY BRAIN (Personal)
    # =========================================================================
    with tab_me:
        profile = brain._profile # Access raw for display
        love_level = profile.get("love_level", 0)
        
        # --- RELATIONSHIP STATUS ---
        # --- RELATIONSHIP STATUS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="brain-card">
                <div class="brain-metric-label">Relationship Strength</div>
                <div class="brain-metric-value">{love_level} ❤️</div>
                <div class="brain-metric-delta">Growing Stronger</div>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            if st.button("Send Love <3", use_container_width=True):
                new_love = brain.receive_love()
                st.balloons()
                st.rerun()
        with col2:
            # Fetch Real Model Accuracy
            accuracy = 0.0
            is_trained = False
            with get_pooled_connection() as conn:
                optimizer = ProcurementMLOptimizer(conn.cursor())
                metrics = optimizer.get_model_metrics()
                accuracy = metrics.get('last_accuracy', 0.0)
                is_trained = metrics.get('is_trained', False)

            status_text = "Model Trained" if is_trained else "Learning Needed"
            
            st.markdown(f"""
            <div class="brain-card">
                <div class="brain-metric-label">Model Accuracy</div>
                <div class="brain-metric-value">{accuracy:.1%}</div>
                <div class="brain-metric-delta">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            events = brain.get_event_count()
            st.markdown(f"""
            <div class="brain-card">
                <div class="brain-metric-label">Shared Memories</div>
                <div class="brain-metric-value">{events}</div>
                <div class="brain-metric-delta">Total Interactions</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- TRAINING REPORT ---
        st.subheader("📝 Training Journal")
        st.caption("Here is what I have learned from observing your actions against ERP reality.")
        
        # Run active correlation (connect "Viewed" -> "Bought")
        with get_pooled_connection() as conn:
            cursor = conn.cursor()
            new_insights = brain.correlate_views_with_purchases(cursor)
        
        if new_insights:
            st.success(f"⚡ I found {len(new_insights)} NEW correlations since you last checked!")
        
        # Display the Report
        report_text = brain.get_weekly_report()
        st.markdown(report_text)
        
        # Deep Dive into Conversions
        conversions = profile.get("conversions", [])
        if conversions:
            st.subheader("🎯 validated Actions (View -> PO)")
            df = pd.DataFrame(conversions)
            if not df.empty:
                st.dataframe(
                    df[["view_date", "item", "type", "message"]],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("I haven't seen you research an item and then immediately buy it yet. I am watching...")

        st.markdown("---")

        # --- MANUAL TRAINING ---
        st.subheader("🎓 Teach Me")
        with st.form("manual_training"):
            training_input = st.text_area("Tell me something I missed or got wrong:", 
                                         placeholder="e.g., 'When I look at Acetone on Fridays, it means I'm worried about supply, not price.'")
            if st.form_submit_button("Submit Insight"):
                brain._log_learning(f"USER TAUGHT: {training_input}")
                st.success("Insight recorded into long-term memory.")
                st.rerun()

    # =========================================================================
    # TAB 2: HIVE MIND (Collective)
    # =========================================================================
    with tab_hive:
        st.header("🐝 The Collective Intelligence")
        st.caption("How your buying patterns compare to the rest of the organization.")
        
        hive = get_hive()
        stats = hive.get_hive_stats()
        comparison = hive.compare_user_to_hive(current_user)
        
        if not stats:
            st.warning("Not enough data to generate Hive Intelligence yet.")
        else:
            # Hive Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Users", stats.get("total_users", 0))
            with c2:
                avg_time = stats.get("avg_decision_time", 0)
                st.metric("Team Avg Decision Time", f"{avg_time}s")
            with c3:
                styles = stats.get("interaction_styles", {})
                top_style = styles.most_common(1)[0][0] if styles else "None"
                st.metric("Dominant Team Style", top_style)
            
            st.markdown("---")
            
            # Comparison
            st.subheader("🔍 You vs. The Hive")
            
            # Speed
            diff = comparison.get("speed_diff_pct", 0)
            if diff > 0:
                st.info(f"⏱️ You are **{diff}% Slower** than the team average. (Thoughtful Analyst)")
            elif diff < 0:
                st.success(f"⚡ You are **{abs(diff)}% Faster** than the team average. (Decisive Leader)")
            else:
                st.info("⏱️ Your decision speed matches the team average perfectly.")
                
            # Unique Vendors
            unique = comparison.get("unique_vendors", [])
            if unique:
                st.write("### 🦄 Your Unique Vendor Relationships")
                st.write("You prefer these vendors, unlike the rest of the hive:")
                for v in unique:
                    st.write(f"- **{v}**")
            
            st.markdown("---")
            
            # Team Favorites
            st.subheader("🏆 Hive Favorites")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.write("**Top Vendors (Team Pick)**")
                top_v = stats.get("top_vendors", {})
                for v, count in top_v.most_common(5):
                    st.write(f"1. {v} ({count} users)")
            with tc2:
                st.write("**Top Items (Most Watched)**")
                top_i = stats.get("top_items", {})
                for i, count in top_i.most_common(5):
                    st.write(f"1. {i} ({count} users)")

if __name__ == "__main__":
    render_brain_center()
