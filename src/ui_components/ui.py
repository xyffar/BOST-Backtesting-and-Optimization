# from typing import type

import streamlit as st

from src.config.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    MESSAGES,
    MIN_DATE,
    ss,
)
from src.utils.utils import (
    parse_date,
)


def _render_asset_and_account_inputs() -> None:
    """Render sidebar widgets for ticker, capital, and commission inputs."""
    display_texts = MESSAGES.get("display_texts", {})
    general_settings = MESSAGES.get("general_settings", {})

    tickers_input = (
        st.text_area(
            display_texts.get("ticker_input_label", "Tickers (comma separated)"),
            value="AAPL,MSFT",
            key="tickers_wid",
        )
        .replace(" ", "")
        .upper()
    )

    # Update session state directly from the processed input
    ss.tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    col_capital, col_commission = st.columns(2)
    with col_capital:
        st.number_input(
            display_texts.get("initial_capital_label", "Initial Capital"),
            min_value=100.0,
            value=general_settings.get("initial_capital", 10000.0),
            step=100.0,
            key="initial_capital_wid",
        )
    with col_commission:
        # The widget value is in percent (e.g., 0.20 for 0.20%).
        # The division by 100 is handled in the backtest/optimizer runners.
        st.number_input(
            display_texts.get("commission_percent_label", "Comm. (%)"),
            min_value=0.00,
            max_value=1.00,
            value=general_settings.get("commission_percent", 0.002) * 100,
            step=0.01,
            format="%.2f",
            key="commission_percent_wid",
        )


def _render_period_and_granularity_inputs() -> None:
    """Render sidebar widgets for date range and data interval."""
    display_texts = MESSAGES.get("display_texts", {})
    general_settings = MESSAGES.get("general_settings", {})
    default_intervals = ["1d", "1h", "1wk", "1mo"]
    intervals = general_settings.get("data_intervals", default_intervals)

    st.markdown(f"### {display_texts.get('data_period_granularity_header', 'Period and Data Granularity')}")
    col_start_date, col_end_date = st.columns(2)

    with col_start_date:
        st.date_input(
            display_texts.get("start_date_label", "Start Date"),
            value=parse_date(DEFAULT_START_DATE),
            min_value=parse_date(MIN_DATE),
            format="DD/MM/YYYY",
            key="start_date_wid",
        )

    with col_end_date:
        st.date_input(
            display_texts.get("end_date_label", "End Date"),
            value=parse_date(DEFAULT_END_DATE),
            format="DD/MM/YYYY",
            key="end_date_wid",
        )

    st.selectbox(
        display_texts.get("data_granularity_label", "Data Granularity"),
        intervals,
        index=intervals.index("1d") if "1d" in intervals else 0,
        key="data_interval_wid",
    )


def _render_debug_info(enabled: bool = False) -> None:
    """Display the session state as a JSON object for debugging purposes."""
    if enabled and st.toggle("Show Session State", value=False, key="debug_toggle"):
        st.json(st.session_state)


def make_sidebar() -> None:
    """Render the sidebar UI for user input of global settings.

    This function orchestrates the rendering of all sidebar components,
    including asset selection, account details, data period, granularity,
    and application mode selection. All inputs are stored in the Streamlit
    session state.
    """
    with st.sidebar:
        _render_asset_and_account_inputs()
        st.markdown("---")
        _render_period_and_granularity_inputs()
        st.markdown("---")
        # _render_mode_selection_buttons()
        # st.markdown("---")
        _render_debug_info(enabled=True)  # Set to False to hide in production

    # else:
    #     wfo_n_cycles, wfo_oos_ratio = None, None

    # return run_wfo, wfo_n_cycles, wfo_oos_ratio


# def create_opt_info_area(opt_infos_container: st.delta_generator.DeltaGenerator) -> tuple:
#     """Create and return placeholders for optimization info messages in the UI.

#     Sets up columns and placeholders for progress, success, and failure messages during optimization.

#     Args:
#         opt_infos_container: The Streamlit container for optimization info messages.

#     Returns:
#         tuple: Placeholders for download progress, download success, run progress, run success, download failure, and run failure.

#     """
#     with opt_infos_container:
#         # Placeholders for dynamic messages
#         col_progress, col_success, col_failed = st.columns(3)
#         with col_progress:
#             download_progress_placeholder = st.empty()
#             download_success_placeholder = st.empty()
#         with col_success:
#             run_progress_placeholder = st.empty()
#             run_success_placeholder = st.empty()
#         with col_failed:
#             download_fail_placeholder = st.empty()
#             run_fail_placeholder = st.empty()  # For backtest/optimization success/failure messages

#     return (
#         download_progress_placeholder,
#         download_success_placeholder,
#         run_progress_placeholder,
#         run_success_placeholder,
#         download_fail_placeholder,
#         run_fail_placeholder,
#     )
