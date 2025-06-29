# from typing import type

import numpy as np
import streamlit as st

from backtest_runner import start_backtest_process
from config import (
    # COMMISSION_PERCENT,
    # DATA_INTERVALS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    MESSAGES,
    # DISPLAY_TEXTS,
    # INITIAL_CAPITAL,
    MIN_DATE,
    streamlit_obj,
    # OPTIMIZATION_OBJECTIVES,
)
from optimizer_runner import start_optimization_process
from strategies.common_strategy import CommonStrategy
from utils import (
    calculate_optimization_combinations,
    parse_date,
    set_backtest_mode,
    set_optimization_mode,
)


# Show subheader
def show_subheader_according_to_mode() -> None:
    """Display a subheader in the UI based on the current mode (backtest or optimization).

    The subheader helps users identify whether they are in backtest or optimization mode.

    Returns:
        None

    """
    if "mode" not in st.session_state:
        st.session_state.mode = "backtest"  # default mode

    if st.session_state.mode == "backtest":
        st.subheader(MESSAGES["display_texts"]["mode_backtest"])
    else:
        st.subheader(MESSAGES["display_texts"]["mode_optimization"])


def make_sidebar() -> tuple[list[str], str, str, str, float, float]:
    """Render the sidebar UI for user input of global settings.

    Include asset selection, account details, data period, granularity, and mode selection.

    Returns:
        tuple: Contains tickers, start date, end date, data interval, initial capital, and commission percent.

    """
    # --- Global Inputs (defined before conditional UI for modes) ---

    # --- Global Inputs (defined before conditional UI for modes) ---
    # Asset and Account Details (moved here)
    tickers_input = (
        st.sidebar.text_area(MESSAGES["display_texts"]["ticker_input_label"], value="AAPL,MSFT", key="tickers")
        .replace(" ", "")
        .upper()
    )
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    col_capital, col_commission = st.sidebar.columns(2)
    with col_capital:
        initial_capital = st.number_input(
            MESSAGES["display_texts"]["initial_capital_label"],
            min_value=100.0,
            value=MESSAGES["general_settings"]["initial_capital"],
            step=100.0,
            key="initial_capital_main",
        )
    with col_commission:
        commission_percent = (
            st.number_input(
                MESSAGES["display_texts"]["commission_percent_label"],
                min_value=0.00,
                max_value=1.00,
                value=MESSAGES["general_settings"]["commission_percent"] * 100,
                step=0.01,
                format="%.2f",
                key="commission_percent_main",
            )
            / 100.0
        )

    st.sidebar.markdown("---")

    # Data Period and Granularity (moved here)
    st.sidebar.markdown(f"### {MESSAGES['display_texts']['data_period_granularity_header']}")
    col_start_date, col_end_date = st.sidebar.columns(2)
    with col_start_date:
        start_date_obj = st.date_input(
            MESSAGES["display_texts"]["start_date_label"],
            value=parse_date(DEFAULT_START_DATE),
            min_value=parse_date(MIN_DATE),
            format="DD/MM/YYYY",
            key="start_date",
        )
    with col_end_date:
        end_date_obj = st.date_input(
            MESSAGES["display_texts"]["end_date_label"],
            value=parse_date(DEFAULT_END_DATE),
            format="DD/MM/YYYY",
            key="end_date",
        )

    data_interval = st.sidebar.selectbox(
        MESSAGES["display_texts"]["data_granularity_label"],
        MESSAGES["general_settings"]["data_intervals"],
        index=MESSAGES["general_settings"]["data_intervals"].index("1d"),
        key="data_interval",
    )

    st.sidebar.markdown("---")

    # --- Sidebar for User Input (unchanged except for moved mode buttons) ---
    st.sidebar.header(MESSAGES["display_texts"]["sidebar_choose_mode"])

    col_mode1, col_mode2 = st.sidebar.columns(2)
    with col_mode1:
        if st.session_state.mode == "backtest":
            st.markdown(
                f"""<p class='mode-button active'>{MESSAGES["display_texts"]["button_backtest"]}</p>""",
                unsafe_allow_html=True,
            )
        else:
            st.button(
                MESSAGES["display_texts"]["button_backtest"],
                use_container_width=True,
                on_click=set_backtest_mode,
            )
    with col_mode2:
        if st.session_state.mode == "optimization":
            st.markdown(
                f"""<p class='mode-button active'>{MESSAGES["display_texts"]["button_optimization"]}</p>""",
                unsafe_allow_html=True,
            )
        else:
            st.button(
                MESSAGES["display_texts"]["button_optimization"],
                use_container_width=True,
                on_click=set_optimization_mode,
            )

    st.sidebar.markdown("---")

    return (
        tickers,
        start_date_obj.strftime("%Y-%m-%d"),
        end_date_obj.strftime("%Y-%m-%d"),
        data_interval,
        initial_capital,
        commission_percent,
    )


@st.fragment
def render_strategy_params(
    strategy_name: str,
    backtest_params_container: streamlit_obj,
) -> dict[str : int | float | str | bool]:
    """Dynamically renders widgets for the selected strategy's parameters.

    Renders widgets for all parameters (including SL/TP) using number_input and a column layout.

    Args:
        strategy_name (str): The name of the selected strategy.
        backtest_params_container (streamlit.delta_generator.DeltaGenerator): Streamlit container for parameter widgets.

    Returns:
        dict: A dictionary of the selected strategy's specific parameters (including SL/TP).

    """
    strat_class: type[CommonStrategy] = st.session_state.all_strategies[strategy_name]
    params = {}

    if issubclass(strat_class, CommonStrategy):
        param_defs: list[dict] = strat_class.PARAMS_INFO

        # Handle column layout for strategy parameters
        cols_per_row: int = 5  # Number of columns per row, modified to accommodate all parameters
        current_columns = []

        for i, param_def in enumerate(param_defs):
            if i % cols_per_row == 0:
                current_columns = st.columns(cols_per_row)

            with current_columns[i % cols_per_row]:
                param_name = param_def["name"]
                param_type = param_def["type"]
                default_value = param_def["default"]

                # More readable label for SL/TP
                display_name = param_name.replace("_", " ").title()
                if param_name in ["sl_percent", "tp_percent"]:
                    display_name = display_name.replace("Percent", "(%)")

                if "options" in param_def:  # For dropdown (e.g., moving average type)
                    selected_option = st.selectbox(
                        f"{display_name}",
                        param_def["options"],
                        index=param_def["options"].index(default_value),
                        key=f"param_{param_name}",
                    )
                    params[param_name] = selected_option
                elif param_type is int:
                    params[param_name] = st.number_input(
                        f"{display_name}",
                        min_value=param_def["lowest"],
                        max_value=param_def["highest"],
                        value=default_value,
                        step=param_def["step"],
                        key=f"param_{param_name}",
                    )
                elif param_type is float:
                    # Special format for SL/TP to show percentage
                    value_to_display = (
                        default_value * 100 if param_name in ["sl_percent", "tp_percent"] else default_value
                    )
                    min_val_display = (
                        param_def["lowest"] * 100 if param_name in ["sl_percent", "tp_percent"] else param_def["lowest"]
                    )
                    max_val_display = (
                        param_def["highest"] * 100
                        if param_name in ["sl_percent", "tp_percent"]
                        else param_def["highest"]
                    )
                    step_val_display = (
                        param_def["step"] * 100 if param_name in ["sl_percent", "tp_percent"] else param_def["step"]
                    )

                    params[param_name] = st.number_input(
                        f"{display_name}",
                        min_value=float(min_val_display),
                        max_value=float(max_val_display),
                        value=float(value_to_display),
                        step=float(step_val_display),
                        format="%.2f",  # Format for float, two decimals
                        key=f"param_{param_name}",
                    )
                    # Convert back to decimal if it was percentage
                    if param_name in ["sl_percent", "tp_percent"]:
                        params[param_name] /= 100.0
                else:
                    params[param_name] = st.text_input(
                        f"{display_name}",
                        value=str(default_value),
                        key=f"param_{param_name}_text",
                    )
                    try:
                        params[param_name] = param_type(params[param_name])
                    except ValueError:
                        st.warning(MESSAGES["display_texts"]["param_invalid_value"].format(param_name=param_name))
                        params[param_name] = default_value

    else:
        st.info(MESSAGES["display_texts"]["select_strategy_to_display_params"])

    return params


def make_body_backtesting_mode(
    tickers: list[str],
    start_date_yf: str,
    end_date_yf: str,
    data_interval: str,
    initial_capital: float,
    commission_percent: float,
) -> None:
    """Run the backtesting workflow for the selected tickers and strategy.

    Manage user input, run the backtest for each ticker, display results, and handle Monte Carlo simulation and export.

    Args:
        tickers (list): List of ticker symbols to backtest.
        start_date_yf (str): Start date for data download.
        end_date_yf (str): End date for data download.
        data_interval (str): Data granularity (e.g., '1d').
        initial_capital (float): Initial capital for the backtest.
        commission_percent (float): Commission percentage for trades.

    Returns:
        None

    """
    backtest_settings_container = st.container()
    st.subheader(MESSAGES["display_texts"]["strategy_params_subheader"])  # Specific heading for strategy parameters
    backtest_params_container = st.container()
    backtest_results_container = st.container()

    with backtest_settings_container:
        (selected_strategy_name, run_mc, mc_sampling_method, sims_length, num_sims) = render_backtest_settings(
            st.session_state.all_strategies, backtest_settings_container
        )

    # Strategy Parameters (dynamic and visible only in Backtest)
    with backtest_params_container:
        strategy_params: dict[str : int | float | str | bool] = render_strategy_params(
            selected_strategy_name, backtest_params_container
        )

    args_for_backtest = [
        tickers,
        start_date_yf,
        end_date_yf,
        data_interval,
        initial_capital,
        commission_percent,
        run_mc,
        mc_sampling_method,
        sims_length,
        num_sims,
        selected_strategy_name,
        strategy_params,
        backtest_results_container,
    ]

    with backtest_params_container:
        st.button(
            MESSAGES["display_texts"]["run_backtest_button"],
            key="run_backtest_button",
            on_click=start_backtest_process,
            args=args_for_backtest,
        )


def make_body_optimization_mode(
    tickers: list[str],
    start_date: str,
    end_date: str,
    data_interval: str,
    initial_capital: float,
    commission_percent: float,
) -> None:
    """Render the main UI and workflow for optimization mode.

    This function manages user input, optimization settings, and triggers the optimization process for the selected strategy.

    Args:
        tickers (list[str]): List of ticker symbols to optimize.
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
        data_interval (str): Data granularity (e.g., '1d').
        initial_capital (float): Initial capital for the optimization.
        commission_percent (float): Commission percentage for trades.

    Returns:
        None

    """
    opt_settings_container = st.container()
    opt_infos_container = st.container()
    opt_results_container = st.container()

    with opt_settings_container:
        # Mostra le importazioni per l'ottimizzazione
        (
            selected_strategy_name,
            objective_function_selection,
            optimization_method_selection,
            max_tries_sambo,
        ) = render_opt_settings()

        # Mostra le impostazioni per il Monte Carlo dell'ottimizzazione
        run_mc, promoted_combinations, mc_sampling_method, num_sims, sims_length = render_opt_mc_settings()

        # Mostra le impostazioni per la Walk Forward Optimization
        run_wfo, wfo_n_cycles, wfo_oos_ratio = render_opt_wfo_settings()

        strat_class: type[CommonStrategy] = st.session_state.all_strategies[selected_strategy_name]
        optimization_params_ranges: dict[str : list | range] = render_optimization_params(strat_class)

        (
            download_progress_placeholder,
            download_success_placeholder,
            run_progress_placeholder,
            run_success_placeholder,
            download_fail_placeholder,
            run_fail_placeholder,
        ) = create_opt_info_area(opt_infos_container)

        args_for_opt = [
            tickers,
            start_date,
            end_date,
            data_interval,
            initial_capital,
            commission_percent,
            objective_function_selection,
            optimization_method_selection,
            max_tries_sambo,
            run_mc,
            promoted_combinations,
            mc_sampling_method,
            num_sims,
            sims_length,
            strat_class,
            optimization_params_ranges,
            opt_results_container,
            run_wfo,
            wfo_n_cycles,
            wfo_oos_ratio,
            download_progress_placeholder,
            download_success_placeholder,
            run_progress_placeholder,
            run_success_placeholder,
            download_fail_placeholder,
            run_fail_placeholder,
        ]

        render_opt_button_and_pars_combs(optimization_params_ranges, args_for_opt)


def render_opt_button_and_pars_combs(
    optimization_params_ranges: dict[str, list | range],
    args_for_opt: list,
) -> None:
    """Render the optimization run button and display the number of parameter combinations.

    Args:
        optimization_params_ranges (dict): Dictionary of parameter ranges for optimization.
        args_for_opt (list): Arguments to pass to the optimization process.

    Returns:
        None

    """
    col_button, col_combs = st.columns([1, 3.8], vertical_alignment="center")
    with col_button:
        st.button(
            MESSAGES["display_texts"]["run_optimization_button"],
            icon="▶️",
            type="primary",
            use_container_width=True,
            key="run_optimization_button",
            on_click=start_optimization_process,
            args=args_for_opt,
        )

    with col_combs:
        # Display the number of combinations
        if optimization_params_ranges:
            num_combinations = calculate_optimization_combinations(optimization_params_ranges)
            st.info(f"""{MESSAGES["display_texts"]["messages"]["number_of_combinations_label"]}: {num_combinations}""")
        else:
            st.info(MESSAGES["display_texts"]["messages"]["select_strategy_to_set_opt_params"])
    # return exec_button


@st.fragment
def render_backtest_settings(
    all_strategies: dict[str : type[CommonStrategy]],
    backtest_settings: streamlit_obj,
) -> list[str, bool, str, int, int]:
    """Render the UI controls for selecting strategy and Monte Carlo settings in backtest mode.

    Allows the user to select a strategy, enable Monte Carlo simulation, and configure related parameters.

    Args:
        all_strategies (dict): Dictionary of available strategy classes.
        backtest_settings (streamlit.delta_generator.DeltaGenerator): Streamlit container for backtest settings UI.

    Returns:
        list: Selected strategy name, Monte Carlo enabled flag, sampling method, simulation length, and number of simulations.

    """
    (col_strat, col_run_mc, col_mc_sampling_method, col_num_sims, col_sims_length) = st.columns(
        5, vertical_alignment="bottom"
    )
    selected_strategy_name, run_mc, mc_sampling_method, sims_length, num_sims = (
        None,
        None,
        None,
        None,
        None,
    )

    with col_strat:
        # Strategy Selection in Backtest mode
        selected_strategy_name: str = st.selectbox(
            MESSAGES["display_texts"]["strategy_selection_label"],  # Label above
            list(all_strategies.keys()),
            key="selected_strategy",
        )

    with col_run_mc:
        run_mc: bool = st.checkbox("Run MonteCarlo (MC)?")

    if run_mc:
        with col_mc_sampling_method:
            mc_sampling_method: str = st.selectbox(
                "Sampling Method",
                options=["resampling_con_reimmissione", "permutazione"],
                help="Resampling con reimmissione (Bootstrap) è tipicamente preferibile per scenari futuri.",
            )

        with col_num_sims:
            num_sims: int = st.number_input(
                "# Simulations",
                value=100,  # Di default il doppio dei trade storici
                min_value=2,  # Idea: zero significa usa il numero di trade della serie originale
                max_value=100000,  # Un limite per evitare simulazioni troppo lunghe
            )

        if mc_sampling_method == "permutazione":
            sims_length = 0
        elif mc_sampling_method == "resampling_con_reimmissione":
            with col_sims_length:
                sims_length: int = st.number_input(
                    "# Trades per Simulation",
                    value=0,
                    min_value=0,
                    max_value=1000,  # Un limite per evitare simulazioni troppo lunghe
                    help="Se 0, il numero di trade è uguale a quello della serie originale",
                )

    return selected_strategy_name, run_mc, mc_sampling_method, sims_length, num_sims


@st.fragment
def render_opt_settings() -> tuple[str, str, str, int]:
    """Render the UI controls for selecting strategy, objective, and optimization method.

    Allows the user to select a strategy, objective function, optimization method, and configure SAMBO tries if applicable.

    Returns:
        tuple: Selected strategy name, objective function, optimization method, and max tries for SAMBO.

    """
    # --- Optimization Settings in the main area ---
    # Strategy, Objective, and Method on the same row
    col_strat, col_obj, col_method, col_max_tries = st.columns(4)

    all_strategies = st.session_state.all_strategies

    with col_strat:
        selected_strategy_name: str = st.selectbox(
            MESSAGES["display_texts"]["strategy_selection_label"],  # Label above
            list(all_strategies.keys()),
            key="selected_strategy",
        )
    with col_obj:
        objective_function_selection = st.selectbox(
            MESSAGES["display_texts"]["optimization_objective_label"],  # Label above
            list(MESSAGES["optimization_settings"]["objectives"].keys()),
            key="objective_function",
        )
    with col_method:
        optimization_method_selection = st.selectbox(
            MESSAGES["display_texts"]["optimization_method_label"],  # Label above
            ["Grid Search", "SAMBO"],
            key="optimization_method",
        )

    if optimization_method_selection == "SAMBO":
        with col_max_tries:
            max_tries_sambo = st.number_input(
                MESSAGES["display_texts"]["sambo_max_tries"],  # Label above
                min_value=1,
                max_value=5000,
                value=150,
                step=10,
                key="max_tries_SAMBO",
            )
    else:
        max_tries_sambo = None

    return (
        selected_strategy_name,
        objective_function_selection,
        optimization_method_selection,
        max_tries_sambo,
    )


@st.fragment
def render_opt_mc_settings() -> tuple[bool, int, str, int, int]:
    """Render the UI controls for Monte Carlo simulation settings in optimization mode.

    Allows the user to enable Monte Carlo simulation, set promoted combinations, sampling method, number of simulations, and trades per simulation.

    Returns:
        tuple: Monte Carlo enabled flag, promoted combinations, sampling method, number of simulations, and trades per simulation.

    """
    (
        col_enable_mc,
        col_promoted_combinations,
        col_mc_sampling_method,
        col_num_sims,
        col_sims_length,
    ) = st.columns(5, vertical_alignment="bottom")

    run_mc, promoted_combinations, mc_sampling_method, num_sims, sims_length = (
        None,
        None,
        None,
        None,
        None,
    )

    with col_enable_mc:
        run_mc: bool = st.checkbox("Run Monte Carlo?")

    if run_mc:
        with col_promoted_combinations:
            promoted_combinations: int = st.number_input(
                "# Promoted combinations", value=10, min_value=1, max_value=100
            )

        with col_mc_sampling_method:
            mc_sampling_method: str = st.selectbox(
                "Sampling Method",
                options=["resampling_con_reimmissione", "permutazione"],
                help="Resampling con reimmissione (Bootstrap) è tipicamente preferibile per scenari futuri.",
            )

        with col_num_sims:
            num_sims: int = st.number_input(
                "# Simulations",
                value=100,  # Di default il doppio dei trade storici
                min_value=2,  # Idea: zero significa usa il numero di trade della serie originale
                max_value=100000,  # Un limite per evitare simulazioni troppo lunghe
            )

        if mc_sampling_method == "permutazione":
            sims_length = 0
        elif mc_sampling_method == "resampling_con_reimmissione":
            with col_sims_length:
                sims_length: int = st.number_input(
                    "# Trades per Simulation",
                    value=0,
                    min_value=0,
                    max_value=1000,  # Un limite per evitare simulazioni troppo lunghe
                    help="Se 0, il numero di trade è uguale a quello della serie originale",
                )
    else:
        promoted_combinations, mc_sampling_method, num_sims, sims_length = (
            None,
            None,
            None,
            None,
        )

    return run_mc, promoted_combinations, mc_sampling_method, num_sims, sims_length


@st.fragment
def render_opt_wfo_settings() -> tuple[bool, int, float]:
    """Render the UI controls for Walk Forward Optimization (WFO) settings.

    Allows the user to enable WFO, set the number of cycles, and specify the out-of-sample ratio.

    Returns:
        tuple: WFO enabled flag, number of cycles, and out-of-sample ratio.

    """
    (col_enable_wfo, col_wfo_n_cycles, col_wfo_oos_ratio, _, _) = st.columns(5, vertical_alignment="bottom")

    with col_enable_wfo:
        run_wfo: bool = st.checkbox("Run WFO?")

    if run_wfo:
        with col_wfo_n_cycles:
            wfo_n_cycles: int = st.number_input("Cycles", value=10, min_value=1, max_value=50)

        with col_wfo_oos_ratio:
            wfo_oos_ratio: str = st.number_input(
                "Out-Of-Sample ratio",
                value=0.25,
                min_value=0.01,
                max_value=0.75,
                help="Percentuale della porzione Out Of Sample rispetto alla lunghezza totale del ciclo.",
            )

    else:
        wfo_n_cycles, wfo_oos_ratio = None, None

    return run_wfo, wfo_n_cycles, wfo_oos_ratio


def create_opt_info_area(opt_infos_container: st.delta_generator.DeltaGenerator) -> tuple:
    """Create and return placeholders for optimization info messages in the UI.

    Sets up columns and placeholders for progress, success, and failure messages during optimization.

    Args:
        opt_infos_container: The Streamlit container for optimization info messages.

    Returns:
        tuple: Placeholders for download progress, download success, run progress, run success, download failure, and run failure.

    """
    with opt_infos_container:
        # Placeholders for dynamic messages
        col_progress, col_success, col_failed = st.columns(3)
        with col_progress:
            download_progress_placeholder = st.empty()
            download_success_placeholder = st.empty()
        with col_success:
            run_progress_placeholder = st.empty()
            run_success_placeholder = st.empty()
        with col_failed:
            download_fail_placeholder = st.empty()
            run_fail_placeholder = st.empty()  # For backtest/optimization success/failure messages

    return (
        download_progress_placeholder,
        download_success_placeholder,
        run_progress_placeholder,
        run_success_placeholder,
        download_fail_placeholder,
        run_fail_placeholder,
    )


@st.fragment
def render_optimization_params(
    strategy_class: type[CommonStrategy],
) -> dict[str : int | float | str | bool]:
    """Dynamically renders widgets for the selected strategy's optimization parameters.

    Renders widgets for all optimizable parameters (including SL/TP) and collects their ranges.

    Args:
        strategy_class (type[CommonStrategy]): The selected strategy class.

    Returns:
        dict: A dictionary of parameter ranges for optimization.

    """
    st.subheader(
        MESSAGES["display_texts"]["optimization_params_subheader"]
    )  # Specific heading for optimization parameters
    opt_params_ranges = {}

    if not issubclass(strategy_class, CommonStrategy):
        st.info(MESSAGES["display_texts"]["messages"]["select_strategy_to_set_opt_params"])
        return opt_params_ranges

    param_defs: list[dict] = strategy_class.PARAMS_INFO

    num_columns = MESSAGES["general_settings"]["param_columns_opt_mode"]
    num_rows = len(param_defs) / num_columns
    num_rows = int(num_rows) if int(num_rows) == num_rows else int(num_rows) + 1
    cols = [st.columns(num_columns, border=True) for _ in range(num_rows)]

    for i, param_def in enumerate(param_defs):
        _render_single_optimization_param(i, num_columns, param_def, cols, opt_params_ranges)

    return opt_params_ranges


def _render_single_optimization_param(
    i: int,
    num_columns: int,
    param_def: dict,
    cols: list,
    opt_params_ranges: dict,
) -> None:
    """Render a single optimization parameter input widget in the UI.

    This function displays the appropriate Streamlit widget for a given parameter definition,
    collects user input, and updates the optimization parameter ranges dictionary.

    Args:
        i (int): The index of the parameter in the parameter definitions list.
        num_columns (int): The number of columns in the UI layout.
        param_def (dict): The parameter definition dictionary.
        cols (list): The list of Streamlit column objects for layout.
        opt_params_ranges (dict): The dictionary to update with the parameter's range or options.

    Returns:
        None

    """
    param_name = param_def["name"]
    param_type = param_def["type"]
    display_name = param_name.replace("_", " ").title()
    if param_name in ["sl_percent", "tp_percent"]:
        display_name = display_name.replace("Percent", "(%)")

    with cols[int(i / num_columns)][i % num_columns]:
        if "options" in param_def:  # For categorical/option parameters (e.g., moving average type)
            st.markdown(f"**{display_name}**")
            selected_options = st.multiselect(
                MESSAGES["display_texts"]["messages"]["select_options_multiselect"],
                param_def["options"],
                default=[param_def["default"]],
                key=f"opt_param_{param_name}_options",
            )
            opt_params_ranges[param_name] = selected_options
        elif param_type in (int, float):
            st.markdown(f"**{display_name}**")
            col1, col2, col3 = st.columns(3)

            value_format = "%d" if param_type is int else "%.2f"

            if param_type is int or (param_type is float and param_name not in ["sl_percent", "tp_percent"]):
                lowest_val_input = param_def["lowest"]
                highest_val_input = param_def["highest"]
                min_val_input = param_def["min"]
                max_val_input = param_def["max"]
                step_val_input = param_def["step"]
            elif param_name in ["sl_percent", "tp_percent"]:
                lowest_val_input = param_def["lowest"] * 100
                highest_val_input = param_def["highest"] * 100
                min_val_input = param_def["min"] * 100
                max_val_input = param_def["max"] * 100
                step_val_input = param_def["step"] * 100

            with col1:
                min_val = st.number_input(
                    MESSAGES["display_texts"]["messages"]["param_display_min"],
                    min_value=param_type(lowest_val_input),
                    max_value=param_type(highest_val_input),
                    value=param_type(min_val_input),
                    step=param_type(step_val_input),
                    format=value_format,
                    key=f"opt_param_{param_name}_min",
                )
            with col2:
                max_val = st.number_input(
                    MESSAGES["display_texts"]["messages"]["param_display_max"],
                    min_value=param_type(lowest_val_input),
                    max_value=param_type(highest_val_input),
                    value=param_type(max_val_input),
                    step=param_type(step_val_input),
                    format=value_format,
                    key=f"opt_param_{param_name}_max",
                )
            with col3:
                step_min_val = float(step_val_input) if step_val_input > 0 else (0.01 if param_type is float else 1)
                step_val = st.number_input(
                    MESSAGES["display_texts"]["messages"]["param_display_step"],
                    min_value=param_type(step_min_val),
                    value=param_type(step_val_input),
                    step=param_type(step_min_val),
                    format=value_format,
                    key=f"opt_param_{param_name}_step",
                )

            if param_name in ["sl_percent", "tp_percent"]:
                min_val /= 100.0
                max_val /= 100.0
                step_val /= 100.0

            if param_type is int:
                opt_params_ranges[param_name] = range(int(min_val), int(max_val) + int(step_val), int(step_val))
            else:
                opt_params_ranges[param_name] = sorted(
                    list(set([v for v in np.arange(min_val, max_val, step_val)] + [max_val]))
                )
        else:
            st.warning(
                MESSAGES["display_texts"]["messages"]["param_type_not_supported_optimization"].format(
                    param_name=param_name
                )
            )
