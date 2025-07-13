# main.py
import streamlit as st

from config import MESSAGES, ss
from profiler import profile_to_file
from ui import (
    make_body_backtesting_mode,
    make_body_optimization_mode,
    make_sidebar,
    show_subheader_according_to_mode,
)
from utils import initialize_session_states, load_strategies


def main_app() -> None:
    """Run the main application.

    Set up the Streamlit page, initialize session state, load strategies,
    and render the UI which includes a sidebar and a main content area that
    switches between 'Backtest' and 'Optimization' modes.
    """
    # --- Page and State Initialization ---
    # Must be the first Streamlit command.
    st.set_page_config(
        page_title=MESSAGES.get("display_texts", {}).get("page_title", "BOST"),
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon=MESSAGES.get("display_texts", {}).get("page_icon", "📈"),
    )

    initialize_session_states()
    # Load strategies once and store in session state.
    if "all_strategies" not in ss or ss.get("all_strategies") == {}:
        ss.all_strategies = load_strategies()

    # --- UI Rendering ---
    # Custom CSS for styling UI elements.
    st.markdown(MESSAGES.get("custom_css", ""), unsafe_allow_html=True)

    # Main application title.
    st.header(MESSAGES.get("display_texts", {}).get("app_title", "BOST"))

    # Display subheader based on the current mode.
    show_subheader_according_to_mode()

    # Render the sidebar for user inputs.
    make_sidebar()

    # Render the main content area based on the selected mode.
    if ss.mode == "backtest":
        make_body_backtesting_mode()
    elif ss.mode == "optimization":
        make_body_optimization_mode()


if __name__ == "__main__":
    with profile_to_file(enabled=False):
        main_app()
