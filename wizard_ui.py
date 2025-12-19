"""Wizard-style UI components for multi-step workflows in Streamlit."""
import streamlit as st
from typing import Callable


class WizardFlow:
    """
    Manages multi-step wizard state in Streamlit session.
    
    Usage:
        wizard = WizardFlow("hedge_optimizer", steps=["Select", "Configure", "Review", "Results"])
        
        if wizard.current_step == 0:
            # Render step 1 content
            if st.button("Next"):
                wizard.next_step()
                st.rerun()
    """
    
    def __init__(self, wizard_id: str, steps: list[str]):
        """
        Initialize wizard with unique ID and step labels.
        
        Args:
            wizard_id: Unique identifier for this wizard (used in session state)
            steps: List of step names/labels (e.g., ["Select", "Configure", "Run"])
        """
        self.wizard_id = wizard_id
        self.steps = steps
        self.total_steps = len(steps)
        self._state_key = f"wizard_{wizard_id}_step"
        self._data_key = f"wizard_{wizard_id}_data"
        
        # Initialize state if not exists
        if self._state_key not in st.session_state:
            st.session_state[self._state_key] = 0
        if self._data_key not in st.session_state:
            st.session_state[self._data_key] = {}
    
    @property
    def current_step(self) -> int:
        """Get current step index (0-based)."""
        return st.session_state.get(self._state_key, 0)
    
    @property
    def current_step_name(self) -> str:
        """Get current step label."""
        idx = self.current_step
        return self.steps[idx] if 0 <= idx < len(self.steps) else ""
    
    @property
    def data(self) -> dict:
        """Get wizard data dictionary for storing user selections across steps."""
        return st.session_state.get(self._data_key, {})
    
    def set_data(self, key: str, value) -> None:
        """Store data in wizard state."""
        if self._data_key not in st.session_state:
            st.session_state[self._data_key] = {}
        st.session_state[self._data_key][key] = value
    
    def next_step(self) -> bool:
        """Advance to next step. Returns True if advanced, False if already at end."""
        if self.current_step < self.total_steps - 1:
            st.session_state[self._state_key] = self.current_step + 1
            return True
        return False
    
    def prev_step(self) -> bool:
        """Go back to previous step. Returns True if moved back, False if at start."""
        if self.current_step > 0:
            st.session_state[self._state_key] = self.current_step - 1
            return True
        return False
    
    def go_to_step(self, step: int) -> None:
        """Jump to a specific step."""
        if 0 <= step < self.total_steps:
            st.session_state[self._state_key] = step
    
    def reset(self) -> None:
        """Reset wizard to step 0 and clear data."""
        st.session_state[self._state_key] = 0
        st.session_state[self._data_key] = {}
    
    def is_complete(self) -> bool:
        """Check if wizard is on the final step."""
        return self.current_step >= self.total_steps - 1
    
    def render_progress(self) -> None:
        """Render the step progress indicator."""
        current = self.current_step
        
        # Build step indicator HTML
        steps_html = []
        for i, step_name in enumerate(self.steps):
            if i < current:
                # Completed step
                color = "#859900"  # Green
                icon = "✓"
            elif i == current:
                # Current step
                color = "#ffb000"  # Amber
                icon = "●"
            else:
                # Future step
                color = "#586e75"  # Grey
                icon = "○"
            
            steps_html.append(
                f'<span style="color:{color}; margin-right: 12px;">'
                f'{icon} <strong>{step_name}</strong>'
                f'</span>'
            )
        
        # Progress bar
        progress_pct = ((current + 1) / self.total_steps) * 100
        
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <div style="font-size: 0.85em; margin-bottom: 8px;">
                    {''.join(steps_html)}
                </div>
                <div style="background: #222; border-radius: 4px; height: 6px; overflow: hidden;">
                    <div style="background: #ffb000; width: {progress_pct}%; height: 100%;"></div>
                </div>
                <div style="font-size: 0.75em; color: #839496; margin-top: 4px;">
                    Step {current + 1} of {self.total_steps}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_navigation(
        self,
        next_label: str = "Next →",
        prev_label: str = "← Back",
        next_disabled: bool = False,
        show_prev: bool = True,
        on_complete: Callable | None = None,
    ) -> tuple[bool, bool]:
        """
        Render navigation buttons and handle clicks.
        
        Args:
            next_label: Label for the next/complete button
            prev_label: Label for the back button
            next_disabled: Whether to disable the next button
            show_prev: Whether to show the back button
            on_complete: Optional callback when completing the wizard
            
        Returns:
            Tuple of (next_clicked, prev_clicked)
        """
        col1, col2, col3 = st.columns([1, 2, 1])
        
        prev_clicked = False
        next_clicked = False
        
        with col1:
            if show_prev and self.current_step > 0:
                if st.button(prev_label, key=f"{self.wizard_id}_prev_{self.current_step}"):
                    prev_clicked = True
                    self.prev_step()
                    st.rerun()
        
        with col3:
            # Change label on final step
            is_final = self.is_complete()
            btn_label = "Complete ✓" if is_final else next_label
            
            if st.button(
                btn_label,
                key=f"{self.wizard_id}_next_{self.current_step}",
                disabled=next_disabled,
                type="primary"
            ):
                next_clicked = True
                if is_final and on_complete:
                    on_complete()
                elif not is_final:
                    self.next_step()
                    st.rerun()
        
        return next_clicked, prev_clicked


def wizard_step_container(wizard: WizardFlow, step_index: int):
    """
    Context manager decorator for wizard step content.
    Only renders content if we're on the specified step.
    
    Usage:
        with wizard_step_container(wizard, 0):
            st.write("Step 1 content")
    """
    class StepContainer:
        def __init__(self, wiz: WizardFlow, idx: int):
            self.wiz = wiz
            self.idx = idx
            self.active = wiz.current_step == idx
        
        def __enter__(self):
            return self.active
        
        def __exit__(self, *args):
            pass
    
    return StepContainer(wizard, step_index)
