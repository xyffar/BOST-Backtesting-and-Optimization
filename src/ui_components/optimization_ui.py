import numpy as np
import streamlit as st

from src.calcs.optimizer_runner import start_optimization_process
from src.config.config import MESSAGES, ss
from src.ui_components.display_results import display_results
from src.utils.utils import _get_opt_button_label, calculate_optimization_combinations


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
