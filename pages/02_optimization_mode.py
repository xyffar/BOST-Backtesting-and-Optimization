import streamlit as st

from src.config.config import MESSAGES
from src.ui_components.optimization_ui import make_body_optimization_mode
from src.ui_components.ui import make_sidebar

st.header(MESSAGES.get("display_texts", {}).get("app_title", "BOST"))

# Display subheader based on the current mode.
subheader_text = MESSAGES.get("display_texts", {}).get("mode_optimization", "✨ Optimization Mode")
st.subheader(subheader_text)

# Render the sidebar for user inputs.
make_sidebar()

make_body_optimization_mode()
