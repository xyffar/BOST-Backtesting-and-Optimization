import streamlit as st

from src.calcs.backtest_runner import start_backtest_process
from src.config.config import MESSAGES, ss
from src.ui_components.display_results import display_results


def render_strategy_params() -> None:
    """Dynamically render widgets for the selected strategy's parameters.

    Retrieves the selected strategy from the session state, iterates through its
    parameter definitions (`PARAMS_INFO`), and renders the appropriate
    Streamlit input widget for each one in a grid layout.

    The collected parameter values are stored in `ss.bt_params`. This function
    does not take arguments or return any value.
    """
    display_texts = MESSAGES.get("display_texts", {})

    strategy_name = ss.get("bt_strategy_wid")
    if not strategy_name or strategy_name not in ss.get("all_strategies", {}):
        st.info(display_texts.get("select_strategy_to_display_params", "Select a strategy to see its parameters."))
        ss.bt_params = {}
        return

    strategy_class = ss.all_strategies[strategy_name]
    param_defs: list[dict] = getattr(strategy_class, "PARAMS_INFO", [])

    if not param_defs:
        ss.bt_params = {}  # No parameters to render
        return

    params = {}
    cols_per_row: int = 5
    current_columns = []

    for i, param_def in enumerate(param_defs):
        if i % cols_per_row == 0:
            current_columns = st.columns(cols_per_row)

        with current_columns[i % cols_per_row]:
            param_name = param_def["name"]
            params[param_name] = _render_param_widget(param_def)

    ss.bt_params = params


def render_backtest_settings() -> None:
    """Render UI controls for backtest strategy and Monte Carlo settings.

    Displays a strategy selection dropdown and, if enabled, controls for
    configuring a Monte Carlo simulation. All user inputs are stored directly
    in the Streamlit session state (`ss`). This function takes no arguments
    and returns no value.
    """
    display_texts = MESSAGES.get("display_texts", {})
    cols = st.columns(5, vertical_alignment="bottom")

    with cols[0]:
        st.selectbox(
            display_texts.get("strategy_selection_label", "Strategy"),
            list(ss.get("all_strategies", {}).keys()),
            key="bt_strategy_wid",
        )

    with cols[1]:
        run_mc = st.checkbox(display_texts.get("run_mc_label", "Run MonteCarlo (MC)?"), key="run_mc_wid")

    if run_mc:
        with cols[2]:
            mc_sampling_method = st.selectbox(
                display_texts.get("mc_sampling_method_label", "Sampling Method"),
                options=["resampling_con_reimmissione", "permutazione"],
                help=display_texts.get(
                    "mc_sampling_method_help", "Bootstrap resampling is typically preferred for future scenarios."
                ),
                key="mc_sampling_method_wid",
            )

        with cols[3]:
            st.number_input(
                display_texts.get("mc_num_sims_label", "# Simulations"),
                value=100,
                min_value=2,
                max_value=100000,
                key="mc_n_sims_wid",
            )

        with cols[4]:
            # The simulation length is only relevant for the resampling method.
            is_resampling = mc_sampling_method == "resampling_con_reimmissione"
            st.number_input(
                display_texts.get("mc_sim_length_label", "# Trades per Simulation"),
                value=0,
                min_value=0,
                max_value=1000,
                help=display_texts.get("mc_sim_length_help", "If 0, uses the original number of trades."),
                key="mc_sim_length_wid",
                disabled=not is_resampling,
            )


def make_body_backtesting_mode() -> None:
    """Render the main UI and workflow for backtesting mode.

    This function orchestrates the display of all UI components for the
    backtesting mode. It creates containers for settings, parameters,
    informational messages, and results. It renders the strategy selection,
    parameter inputs, and the main "Run Backtest" button. It also calls
    the function to display results if they are available in the session state.
    """
    # Create containers for different sections of the UI
    settings_container = st.container()
    params_container = st.container()
    infos_container = st.container()
    results_container = st.container()

    # Render strategy and Monte Carlo settings
    with settings_container:
        render_backtest_settings()

    # Render strategy-specific parameters and the run button
    with params_container:
        display_texts = MESSAGES.get("display_texts", {})
        st.subheader(display_texts.get("strategy_params_subheader", "Strategy Parameters"))
        render_strategy_params()

        # st.markdown("---")

        st.button(
            display_texts.get("run_backtest_button", "Run Backtest"),
            key="run_backtest_button_wid",
            type="primary",
            on_click=start_backtest_process,
            args=[infos_container, results_container],
            # use_container_width=True,
        )

    # The results container is initially empty and gets populated by the
    # start_backtest_process callback. It's also used to display
    # results from previous runs if they exist in the session state.
    with results_container:
        display_results(mode="backtest")


def _render_param_widget(param_def: dict) -> any:
    """Render a single Streamlit widget based on a parameter definition.

    This helper function selects and configures the appropriate Streamlit input
    widget (e.g., `st.selectbox`, `st.number_input`) based on the type and
    properties defined in `param_def`. It also handles special formatting,
    like displaying percentage-based parameters correctly.

    Args:
        param_def (dict): A dictionary defining a single strategy parameter,
                          typically from a strategy's `PARAMS_INFO`.

    Returns:
        any: The value returned by the Streamlit widget, converted to the
             correct type.

    """
    param_name = param_def["name"]
    param_type = param_def["type"]
    default_value = param_def["default"]
    display_texts = MESSAGES.get("display_texts", {})

    display_name = param_name.replace("_", " ").title()
    if param_name in ["sl_percent", "tp_percent"]:
        display_name = display_name.replace("Percent", "(%)")

    key = f"param_{param_name}"

    if "options" in param_def:
        return st.selectbox(
            f"{display_name}", param_def["options"], index=param_def["options"].index(default_value), key=key
        )

    if isinstance(param_type, int):
        return st.number_input(
            f"{display_name}",
            min_value=param_def["lowest"],
            max_value=param_def["highest"],
            value=default_value,
            step=param_def["step"],
            key=key,
        )

    if isinstance(param_type, float):
        is_percentage = param_name in ["sl_percent", "tp_percent"]
        multiplier = 100.0 if is_percentage else 1.0

        widget_value = st.number_input(
            f"{display_name}",
            min_value=float(param_def["lowest"] * multiplier),
            max_value=float(param_def["highest"] * multiplier),
            value=float(default_value * multiplier),
            step=float(param_def["step"] * multiplier),
            format="%.2f",
            key=key,
        )
        return widget_value / multiplier if is_percentage else widget_value

    # Fallback for other types (e.g., str)
    text_value = st.text_input(f"{display_name}", value=str(default_value), key=f"{key}_text")
    try:
        return param_type(text_value)
    except (ValueError, TypeError):
        st.warning(
            display_texts.get("param_invalid_value", "Invalid value for '{param_name}'.").format(param_name=param_name)
        )
        return default_value
