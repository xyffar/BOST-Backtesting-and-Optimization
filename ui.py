# from typing import type

import numpy as np
import streamlit as st

from backtest_runner import start_backtest_process
from config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    MESSAGES,
    MIN_DATE,
    ss,
)
from display_results import display_results
from optimizer_runner import start_optimization_process
from utils import (
    _get_opt_button_label,
    calculate_optimization_combinations,
    parse_date,
    set_backtest_mode,
    set_optimization_mode,
)


# Show subheader
def show_subheader_according_to_mode() -> None:
    """Display a subheader in the UI based on the current application mode.

    Retrieves the current mode from the session state and displays the
    corresponding subheader text defined in the MESSAGES configuration.
    This helps users identify whether they are in 'Backtest' or
    'Optimization' mode.

    """
    # 'ss.mode' is guaranteed to be initialized by `initialize_session_states()`
    # at the start of the app, so no need to check for its existence here.
    display_texts = MESSAGES.get("display_texts", {})

    if ss.mode == "backtest":
        subheader_text = display_texts.get("mode_backtest", "📊 Backtest Mode")
        st.subheader(subheader_text)
    elif ss.mode == "optimization":
        subheader_text = display_texts.get("mode_optimization", "✨ Optimization Mode")
        st.subheader(subheader_text)
    else:
        # Fallback for an unexpected mode value
        st.warning(f"No subheader defined for mode: '{ss.mode}'")


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


def _render_mode_button(mode_name: str, button_text: str, on_click_handler: callable, is_active: bool) -> None:
    """Render a single mode button, styled as active or inactive."""
    if is_active:
        st.markdown(f"<p class='mode-button active'>{button_text}</p>", unsafe_allow_html=True)
    else:
        st.button(
            button_text,
            use_container_width=True,
            on_click=on_click_handler,
            key=f"{mode_name}_button_wid",
        )


def _render_mode_selection_buttons() -> None:
    """Render the Backtest/Optimization mode selection buttons."""
    display_texts = MESSAGES.get("display_texts", {})
    st.header(display_texts.get("sidebar_choose_mode", "Choose mode"))

    col1, col2 = st.columns(2)
    with col1:
        _render_mode_button(
            mode_name="backtest",
            button_text=display_texts.get("button_backtest", "Backtest"),
            on_click_handler=set_backtest_mode,
            is_active=(ss.mode == "backtest"),
        )
    with col2:
        _render_mode_button(
            mode_name="optimization",
            button_text=display_texts.get("button_optimization", "Optimization"),
            on_click_handler=set_optimization_mode,
            is_active=(ss.mode == "optimization"),
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
        _render_mode_selection_buttons()
        st.markdown("---")
        _render_debug_info(enabled=True)  # Set to False to hide in production


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

    if param_type is int:
        return st.number_input(
            f"{display_name}",
            min_value=param_def["lowest"],
            max_value=param_def["highest"],
            value=default_value,
            step=param_def["step"],
            key=key,
        )

    if param_type is float:
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
        display_results()


def make_body_optimization_mode() -> None:
    """Render the main UI and workflow for the optimization mode.

    This function orchestrates the display of all UI components specific to the
    optimization mode. It sets up containers for optimization settings,
    parameter ranges, informational messages, and results. It then calls
    helper functions to render these sections, including strategy selection,
    objective function, optimization method, Monte Carlo settings for
    optimization, Walk Forward Optimization settings, and the dynamic
    parameter range inputs.

    The function also manages the display of the "Run Optimization" button
    and the calculated number of parameter combinations.

    All user inputs and generated results are managed via Streamlit's session
    state (`ss`).
    """
    # Create containers for different sections of the UI
    settings_container = st.container()
    params_container = st.container()
    infos_container = st.container()
    results_container = st.container()
    with settings_container:
        render_opt_settings()

        # Mostra le impostazioni per il Monte Carlo dell'ottimizzazione
        render_opt_mc_settings()

        # Mostra le impostazioni per la Walk Forward Optimization
        render_opt_wfo_settings()

    with params_container:
        render_optimization_params(ss.opt_strategy_wid)
        render_opt_button_and_pars_combs([infos_container, results_container])

    with results_container:
        display_results()


def render_opt_button_and_pars_combs(args_for_opt: list) -> None:
    """Render the optimization run button and displays the number of parameter combinations.

    Args:
        args_for_opt (list): A list of arguments to be passed to the
                             `start_optimization_process` function when the button is clicked.

    Returns:
        None

    """
    col_button, col_combs = st.columns([1, 3.8], vertical_alignment="center")
    with col_button:
        st.button(
            _get_opt_button_label(),
            icon="🔄" if ss.opt_results_generated else "▶️",
            type="primary",
            use_container_width=True,
            key="run_optimization_button_wid",
            on_click=start_optimization_process,
            args=args_for_opt,
        )

    with col_combs:
        # Display the number of combinations if optimization parameters are defined.
        if ss.opt_params:
            num_combinations = calculate_optimization_combinations(ss.opt_params)
            st.info(f"""{MESSAGES["display_texts"]["messages"]["number_of_combinations_label"]}: {num_combinations}""")
        else:
            # Inform the user to select a strategy to set optimization parameters.
            st.info(MESSAGES["display_texts"]["messages"]["select_strategy_to_set_opt_params"])


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


def render_opt_settings() -> None:
    """Render UI controls for optimization strategy, objective, and method.

    Displays select boxes for choosing a strategy, an objective function, and
    an optimization method. If the 'SAMBO' method is selected, an additional
    number input for 'max_tries' is rendered.

    All user inputs are stored directly in the Streamlit session state (`ss`).
    This function takes no arguments and returns no value.
    """
    display_texts = MESSAGES.get("display_texts", {})
    opt_settings = MESSAGES.get("optimization_settings", {})

    # --- Optimization Settings in the main area ---
    # Strategy, Objective, and Method on the same row
    cols = st.columns(4, vertical_alignment="bottom")

    with cols[0]:
        st.selectbox(
            display_texts.get("strategy_selection_label", "Strategy"),
            list(ss.get("all_strategies", {}).keys()),
            key="opt_strategy_wid",
        )
    with cols[1]:
        st.selectbox(
            display_texts.get("optimization_objective_label", "Objective"),
            list(opt_settings.get("objectives", {}).keys()),
            key="opt_obj_func_wid",
        )
    with cols[2]:
        opt_methods = list(opt_settings.get("methods", {"Grid Search": "grid", "SAMBO": "sambo"}).keys())
        st.selectbox(
            display_texts.get("optimization_method_label", "Method"),
            opt_methods,
            key="opt_method_wid",
        )

    if ss.get("opt_method_wid") == "SAMBO":
        with cols[3]:
            st.number_input(
                display_texts.get("sambo_max_tries", "SAMBO Max Tries"),
                min_value=1,
                max_value=5000,
                value=150,
                step=10,
                key="max_tries_SAMBO_wid",
            )


def render_opt_mc_settings() -> None:
    """Render UI controls for Monte Carlo settings in optimization mode.

    Displays a checkbox to enable Monte Carlo analysis on the best optimization
    results. If enabled, it shows controls for the number of top combinations
    to test, the sampling method, and other simulation parameters.

    All user inputs are stored directly in the Streamlit session state (`ss`).
    This function takes no arguments and returns no value.
    """
    display_texts = MESSAGES.get("display_texts", {})
    cols = st.columns(5, vertical_alignment="bottom")

    with cols[0]:
        run_mc = st.checkbox(display_texts.get("opt_run_mc_label", "Run Monte Carlo?"), key="opt_run_mc_wid")

    if run_mc:
        with cols[1]:
            st.number_input(
                display_texts.get("mc_promoted_combs_label", "# Promoted combinations"),
                value=10,
                min_value=1,
                max_value=100,
                key="mc_promoted_combs_wid",
            )

        with cols[2]:
            mc_sampling_method = st.selectbox(
                display_texts.get("mc_sampling_method_label", "Sampling Method"),
                options=["resampling_con_reimmissione", "permutazione"],
                help=display_texts.get(
                    "mc_sampling_method_help", "Bootstrap resampling is typically preferred for future scenarios."
                ),
                key="opt_mc_sampling_method_wid",
            )

        with cols[3]:
            st.number_input(
                display_texts.get("mc_num_sims_label", "# Simulations"),
                value=100,
                min_value=2,
                max_value=100000,
                key="opt_mc_n_sims_wid",
            )

        with cols[4]:
            is_resampling = mc_sampling_method == "resampling_con_reimmissione"
            st.number_input(
                display_texts.get("mc_sim_length_label", "# Trades per Simulation"),
                value=0,
                min_value=0,
                max_value=1000,
                help=display_texts.get("mc_sim_length_help", "If 0, uses the original number of trades."),
                key="opt_mc_sim_length_wid",
                disabled=not is_resampling,
            )


def render_opt_wfo_settings() -> None:
    """Render the UI controls for Walk Forward Optimization (WFO) settings.

    Allows the user to enable WFO, set the number of cycles, and specify the out-of-sample ratio.

    Returns:
        tuple: WFO enabled flag, number of cycles, and out-of-sample ratio.

    """
    (col_enable_wfo, col_wfo_n_cycles, col_wfo_oos_ratio, _, _) = st.columns(5, vertical_alignment="bottom")

    with col_enable_wfo:
        run_wfo: bool = st.checkbox("Run WFO?", key="opt_run_wfo_wid")

    if run_wfo:
        with col_wfo_n_cycles:
            st.number_input("Cycles", value=10, min_value=1, max_value=50, key="opt_wfo_n_cycles_wid")

        with col_wfo_oos_ratio:
            st.number_input(
                "Out-Of-Sample ratio",
                value=0.25,
                min_value=0.01,
                max_value=0.75,
                help="Percentuale della porzione Out Of Sample rispetto alla lunghezza totale del ciclo.",
                key="opt_wfo_oos_ratio_wid",
            )

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


def render_optimization_params(
    strategy_name: str | None,
) -> None:
    """Dynamically render widgets for defining strategy optimization ranges.

    Retrieves the selected strategy, iterates through its optimizable parameter
    definitions (`PARAMS_INFO`), and renders UI widgets for defining optimization
    ranges (e.g., min, max, step, or a list of options).

    The collected parameter ranges are stored in `ss.opt_params`. This function
    does not return any value.

    Args:
        strategy_name (str | None): The name of the selected strategy from the UI.

    """
    display_texts = MESSAGES.get("display_texts", {})
    st.subheader(display_texts.get("optimization_params_subheader", "Optimization Parameters"))

    # --- Guard Clauses ---
    if not strategy_name or strategy_name not in ss.get("all_strategies", {}):
        st.info(
            display_texts.get("select_strategy_to_set_opt_params", "Select a strategy to set optimization parameters.")
        )
        ss.opt_params = {}
        return

    strategy_class = ss.all_strategies[strategy_name]
    param_defs = getattr(strategy_class, "PARAMS_INFO", [])

    if not param_defs:
        st.info(f"Strategy '{strategy_name}' has no optimizable parameters defined.")
        ss.opt_params = {}
        return

    # --- Render Parameter Grid ---
    opt_params_ranges = {}
    num_columns = MESSAGES.get("general_settings", {}).get("param_columns_opt_mode", 3)
    num_rows = (len(param_defs) + num_columns - 1) // num_columns
    cols_grid = [st.columns(num_columns, border=True) for _ in range(num_rows)]

    for i, param_def in enumerate(param_defs):
        row_idx = i // num_columns
        col_idx = i % num_columns
        with cols_grid[row_idx][col_idx]:
            param_name = param_def.get("name")
            # Call the refactored helper and store the returned range
            param_range = _render_single_optimization_param(param_def)
            if param_range is not None:
                opt_params_ranges[param_name] = param_range

    ss.opt_params = opt_params_ranges


def _render_numeric_range_inputs(
    param_name: str, param_type: type, scaled_params: dict, value_format: str, messages: dict
) -> tuple[float, float, float]:
    """Render three number_input widgets for min, max, and step of a numeric range.

    Args:
        param_name (str): The name of the parameter.
        param_type (type): The type of the parameter (int or float).
        scaled_params (dict): A dictionary with scaled values for widget bounds and defaults.
        value_format (str): The format string for the number input widgets.
        messages (dict): A dictionary of UI text labels.

    Returns:
        tuple[float, float, float]: A tuple containing the user-inputted
                                    min, max, and step values.

    """
    col1, col2, col3 = st.columns(3)
    with col1:
        min_val = st.number_input(
            messages.get("param_display_min", "Min"),
            min_value=param_type(scaled_params["lowest"]),
            max_value=param_type(scaled_params["highest"]),
            value=param_type(scaled_params["min_default"]),
            format=value_format,
            key=f"opt_param_{param_name}_min",
        )
    with col2:
        max_val = st.number_input(
            messages.get("param_display_max", "Max"),
            min_value=param_type(scaled_params["lowest"]),
            max_value=param_type(scaled_params["highest"]),
            value=param_type(scaled_params["max_default"]),
            format=value_format,
            key=f"opt_param_{param_name}_max",
        )
    with col3:
        step_min_val = 0.01 if param_type is float else 1
        step_val = st.number_input(
            messages.get("param_display_step", "Step"),
            min_value=param_type(step_min_val),
            value=param_type(scaled_params["step_default"]),
            format=value_format,
            key=f"opt_param_{param_name}_step",
        )
    return min_val, max_val, step_val


def _render_single_optimization_param(param_def: dict) -> list | range | None:
    """Render UI widgets for a single optimization parameter and return its range.

    This function is a self-contained component that, given a parameter
    definition, renders the appropriate Streamlit widgets (e.g., multiselect
    for options, or number inputs for a numeric range). It then calculates
    and returns the specified range or list of options for optimization.

    Args:
        param_def (dict): The definition for a single parameter, containing
                          keys like 'name', 'type', 'default', 'options',
                          'lowest', 'highest', 'min', 'max', 'step'.

    Returns:
        list | range | None: A list of selected options for categorical
                              parameters, a `range` object for integers, a list
                              of floats for float parameters, or None if the
                              parameter type is not supported for optimization.

    """
    param_name = param_def.get("name")
    param_type = param_def.get("type")
    display_texts = MESSAGES.get("display_texts", {})
    messages = display_texts.get("messages", {})

    display_name = param_name.replace("_", " ").title()
    if param_name in ["sl_percent", "tp_percent"]:
        display_name = display_name.replace("Percent", "(%)")

    st.markdown(f"**{display_name}**")

    # --- Categorical Parameters (multiselect) ---
    if "options" in param_def:
        return st.multiselect(
            messages.get("select_options_multiselect", "Select Options"),
            param_def.get("options", []),
            default=[param_def.get("default")] if param_def.get("default") in param_def.get("options", []) else [],
            key=f"opt_param_{param_name}_options",
        )

    # --- Numeric Parameters (int, float) ---
    if param_type in (int, float):
        is_percentage = param_name in ["sl_percent", "tp_percent"]
        multiplier = 100.0 if is_percentage else 1.0
        value_format = "%d" if param_type is int else "%.2f"

        # Prepare scaled values for the widgets, fixing a potential bug in default value lookup
        scaled_params = {
            "lowest": param_def.get("lowest", 0) * multiplier,
            "highest": param_def.get("highest", 100) * multiplier,
            "min_default": param_def.get("min", param_def.get("lowest", 0)) * multiplier,
            "max_default": param_def.get("max", param_def.get("highest", 100)) * multiplier,
            "step_default": param_def.get("step", 1) * multiplier,
        }

        min_val, max_val, step_val = _render_numeric_range_inputs(
            param_name, param_type, scaled_params, value_format, messages
        )

        # Validate inputs
        if min_val > max_val:
            st.warning("Min value cannot be greater than Max value.")
            return []
        if step_val <= 0:
            st.warning("Step value must be positive.")
            return []

        # Un-scale values if they were percentages
        if is_percentage:
            min_val /= 100.0
            max_val /= 100.0
            step_val /= 100.0

        # Generate and return the final range or list
        if param_type is int:
            return range(int(min_val), int(max_val) + 1, int(step_val))

        if param_type is float:
            # Use np.linspace for robust float ranges
            num_steps = round((max_val - min_val) / step_val) + 1
            return np.linspace(min_val, max_val, num_steps, dtype=float).tolist()

    # --- Unsupported types ---
    st.warning(
        messages.get("param_type_not_supported_optimization", "Unsupported type for {param_name}").format(
            param_name=param_name
        )
    )
    return None
