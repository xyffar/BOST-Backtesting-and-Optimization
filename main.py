# main.py

import streamlit as st

# Import DISPLAY_TEXTS and CUSTOM_CSS
from config import MESSAGES
from ui import (
    make_body_backtesting_mode,
    make_body_optimization_mode,
    make_sidebar,
    show_subheader_according_to_mode,
)
from utils import load_strategies

st.session_state.all_strategies = load_strategies()

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=MESSAGES["display_texts"]["page_title"],
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=MESSAGES["display_texts"]["page_icon"],  # Added crown emoji as page logo
)

# --- Custom CSS to reduce separator line spacing and style mode buttons ---
st.markdown(MESSAGES["custom_css"], unsafe_allow_html=True)

# --- Main Title and Mode Indicator ---
st.header(MESSAGES.get("display_texts").get("app_title"))

# Call the new function to display the subheader based on mode
show_subheader_according_to_mode()

# Make the sidebar
(
    tickers,
    start_date_yf,
    end_date_yf,
    data_interval,
    initial_capital,
    commission_percent,
) = make_sidebar()

# --- Conditional UI for Mode (Main Content Area) ---
if st.session_state.mode == "backtest":
    make_body_backtesting_mode(
        tickers=tickers,
        start_date_yf=start_date_yf,
        end_date_yf=end_date_yf,
        data_interval=data_interval,
        initial_capital=initial_capital,
        commission_percent=commission_percent,
    )

elif st.session_state.mode == "optimization":
    make_body_optimization_mode(
        tickers=tickers,
        start_date=start_date_yf,
        end_date=end_date_yf,
        data_interval=data_interval,
        initial_capital=initial_capital,
        commission_percent=commission_percent,
    )
