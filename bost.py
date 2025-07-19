# main.py
import streamlit as st

from src.config.config import MESSAGES, ss
from src.utils.profiler import profile_to_file
from src.utils.utils import initialize_session_states, load_strategies


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

    pages = [
        st.Page("pages/01_backtest_mode.py", title="Backtest", icon="📊"),
        st.Page("pages/02_optimization_mode.py", title="Optimization", icon="🎯"),
    ]

    # Configure navigation with hidden sidebar
    page = st.navigation(pages)
    page.run()


if __name__ == "__main__":
    with profile_to_file(enabled=False):
        main_app()
