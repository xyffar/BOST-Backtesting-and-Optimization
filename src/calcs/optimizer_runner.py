# optimizer_runner.py
import itertools
import time
from collections.abc import Callable
from typing import Any

import backtesting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from backtesting import Backtest
import sambo
from sambo.plot import plot_convergence, plot_evaluations, plot_objective

from src.calcs.backtest_runner import get_benchmark_data, run_backtest
from src.config.config import MESSAGES, ss, streamlit_obj
from src.data_handlers.data_handler import get_ticker_data_and_infos
from src.data_handlers.excel_exporter import log_execution_data
from src.calcs.monte_carlo import run_montecarlos_for_best_combs
from strategies.common_strategy import CommonStrategy
from src.utils.utils import (
    OptimizationRecorder,
    add_benchmark_comparison,
    list_varying_params,
    record_all_optimizations,
    reset_ss_values_for_results,
)

# Alias di tipo per maggiore chiarezza e leggibilità
OptimizationParamsRanges = dict[str, range | tuple[float, float, float] | list[Any]]
BacktestOptimizationOutput = tuple[pd.DataFrame, Any] | None


def _initialize_bt_instance(
    data: pd.DataFrame,
) -> Backtest | None:
    """Initialize and return a Backtest instance.

    Configures a `backtesting.Backtest` instance with the provided data and
    global settings (strategy, capital, commission) from the session state.

    Args:
        data (pd.DataFrame): Financial data (OHLCV) for the backtest.

    Returns:
        Backtest | None: A configured Backtest instance, or None if the selected
                        strategy is not found in the session state.

    """
    strategy_name = ss.get("opt_strategy_wid")
    strategy_class = ss.get("all_strategies", {}).get(strategy_name)

    if not strategy_class:
        st.error(f"Strategy '{strategy_name}' not found. Cannot initialize backtest.")
        return None

    return Backtest(
        data,
        strategy_class,
        cash=ss.get("initial_capital_wid", 10000),
        commission=ss.get("commission_percent_wid", 0.2) / 100,
        exclusive_orders=True,  # Allows only one position at a time
    )


def _process_params_ranges(
    params_ranges: OptimizationParamsRanges,
) -> dict[str, list[Any]]:
    """Prepare parameter ranges for the backtesting.py optimizer.

    Converts UI-defined parameter ranges (e.g., `range` objects for integers,
    tuples for floats) into lists of values that the `backtesting.py`
    `optimize` method can consume.

    Args:
        params_ranges (OptimizationParamsRanges): A dictionary where keys are
            parameter names and values are either `range` objects, tuples of
            (min, max, step) for floats, or lists of categorical values.

    Returns:
        dict[str, list[Any]]: A dictionary of processed parameters, where each
                              value is a list of discrete values to test.

    """
    processed_params: dict[str, list[Any]] = {}
    messages = MESSAGES.get("display_texts", {}).get("optimizer_runner", {})

    for param, val_range in params_ranges.items():
        if isinstance(val_range, range):
            processed_params[param] = list(val_range)
        elif isinstance(val_range, tuple) and len(val_range) == 3:
            # Use np.linspace for robust float ranges
            min_val, max_val, step = val_range
            if step <= 0:
                st.warning(f"Step for parameter '{param}' must be positive. Using single value.")
                processed_params[param] = [min_val]
                continue
            num_steps = round((max_val - min_val) / step) + 1
            processed_params[param] = np.linspace(min_val, max_val, num_steps).tolist()
        elif isinstance(val_range, list):
            processed_params[param] = val_range
        else:
            st.warning(
                messages.get("unexpected_param_range_type", "Unexpected param range type for {param}: {type}.").format(
                    param=param, type=type(val_range).__name__
                )
            )
            # Treat as a single value if the type is unexpected
            processed_params[param] = [val_range]

    return processed_params


def _execute_optimization(
    bt: Backtest,
    processed_params_ranges: dict[str, list[Any] | Any],
    custom_constraint: Callable | None = None,
) -> BacktestOptimizationOutput:
    """Execute the core of the optimization with backtesting.py, handling different methods.

    Args:
        bt (Backtest): The Backtest instance.
        processed_params_ranges (dict[str, list[Any] | Any]): Parameters ready for optimization.
        custom_constraint (Callable | None): A custom constraint function, if provided.

    Returns:
        BacktestOptimizationOutput: A tuple containing the results DataFrame and optional
                                    SAMBO optimization data, or None if it fails.

    """
    # Get the strategy class from session state to find its constraint
    strategy_class = ss.get("all_strategies", {}).get(ss.get("opt_strategy_wid"))
    constraint_func = get_constraint(strategy_class, custom_constraint)

    # Safely get the optimization method argument from config
    messages = MESSAGES.get("display_texts", {}).get("optimizer_runner", {})
    method_arg = MESSAGES.get("optimization_settings", {}).get("methods", {}).get(ss.get("opt_method_wid"))

    if not method_arg:
        st.error(
            messages.get("optimization_method_not_supported", "Optimization method '{method}' not supported.").format(
                method=ss.get("opt_method_wid")
            )
        )
        return None

    try:
        with record_all_optimizations(bt) as recorder:
            result = bt.optimize(
                **processed_params_ranges,
                maximize=ss.get("opt_obj_func_wid"),
                method=method_arg,
                constraint=constraint_func,
                random_state=None,
                return_optimization=(method_arg == "sambo"),
                max_tries=ss.get("max_tries_SAMBO_wid"),
                return_heatmap=False,
            )

            # Safely extract SAMBO data if the method returned it
            sambo_data = result[1] if method_arg == "sambo" and isinstance(result, tuple) else None

            # Process the recorded results
            all_results_df = _manipulate_opt_results(processed_params_ranges, recorder, ss.get("opt_obj_func_wid"))

            if all_results_df is None or all_results_df.empty:
                return None

            return all_results_df, sambo_data
    except Exception as e:
        st.error(messages.get("optimization_error", "Error during optimization: {e}").format(e=e))
        return None


def _manipulate_opt_results(
    processed_params_ranges: dict[str, list[Any]],
    recorder: OptimizationRecorder,
    objective_func: str,
) -> pd.DataFrame | None:
    """Process and clean the raw optimization results from the recorder.

    This function takes the raw results, sets the parameter columns as the index,
    extracts trade returns, removes duplicate combinations, and sorts the
    results by the objective function.

    Args:
        processed_params_ranges (dict[str, list[Any]]): The dictionary of parameter ranges used.
        recorder (OptimizationRecorder): The recorder object containing the raw results.
        objective_func (str): The name of the objective function used for sorting.

    Returns:
        pd.DataFrame | None: A cleaned and sorted DataFrame of optimization results,
                              or None if no valid results were found.

    """
    all_results = recorder.get_dataframe()
    if all_results is None or all_results.empty:
        st.error(
            MESSAGES.get("display_texts", {})
            .get("optimizer_runner", {})
            .get("no_valid_results", "The optimizer returned no valid results.")
        )
        return None

    # Set the parameter columns as the index
    param_cols = [col for col in processed_params_ranges if col in all_results.columns]
    all_results = all_results.set_index(param_cols)

    # Select only the objective metrics and the trades column
    objective_cols = list(MESSAGES.get("optimization_settings", {}).get("objectives", {}).keys())
    all_results = all_results[[*objective_cols, "_trades"]]

    # Extract the list of trade returns from the nested DataFrame
    all_results["_trades"] = all_results["_trades"].apply(
        lambda nested_df: (
            nested_df["ReturnPct"].to_numpy()
            if isinstance(nested_df, pd.DataFrame) and "ReturnPct" in nested_df.columns
            else np.nan
        )
    )
    all_results.rename(columns={"_trades": "Trade_returns"}, inplace=True)

    # Remove duplicate parameter combinations, keeping the first result
    all_results = all_results.loc[~all_results.index.duplicated(keep="first")]

    # Sort by the objective function
    is_higher_better = MESSAGES.get("optimization_settings", {}).get("objectives", {}).get(objective_func, True)
    return all_results.sort_values(by=objective_func, ascending=not is_higher_better)


def get_constraint(strategy_class: backtesting.Strategy, custom_constraint: Callable | None) -> Callable | None:
    """Retrieve the optimization constraint function for a strategy.

    Returns the strategy's optimization_constraint if no custom constraint is provided, otherwise returns the custom constraint.

    Args:
        strategy_class: The strategy class to check for a constraint.
        custom_constraint: A custom constraint function, if provided.

    Returns:
        Callable[[pd.Series], bool] | None: The constraint function or None.

    """
    return getattr(strategy_class, "optimization_constraint", None) if custom_constraint is None else custom_constraint


def _plot_single_heatmap(
    results_df: pd.DataFrame,
    param1_col: str,
    param2_col: str,
    objective_display_name: str,
) -> plt.Figure | None:
    """Generate a single heatmap plot for two specific parameters.

    Aggregates objective values by taking the mean for duplicate combinations.
    """
    try:
        # Aggrega i valori dell'obiettivo per i due parametri di interesse
        # Questo risolve l'errore "Index contains duplicate entries"
        heatmap_data = results_df.groupby([param1_col, param2_col])[objective_display_name].mean().reset_index()

        heatmap_pivot = heatmap_data.pivot(index=param1_col, columns=param2_col, values=objective_display_name)

        # Dimensioni ulteriormente ridotte per permettere circa 4 grafici per riga
        fig, ax = plt.subplots()
        cax = ax.imshow(heatmap_pivot, cmap="viridis", origin="lower", aspect="auto")
        fig.colorbar(cax, label=objective_display_name)

        ax.set_xticks(np.arange(len(heatmap_pivot.columns)))
        ax.set_yticks(np.arange(len(heatmap_pivot.index)))

        # Formatta le etichette per i parametri numerici (arrotonda i float)
        x_labels = [round(val, 2) if isinstance(val, float) else val for val in heatmap_pivot.columns]
        y_labels = [round(val, 2) if isinstance(val, float) else val for val in heatmap_pivot.index]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        ax.set_xlabel(param2_col)
        ax.set_ylabel(param1_col)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(
            MESSAGES["display_texts"]["optimizer_runner"]["heatmap_title"].format(obj=objective_display_name)
            + f" ({param1_col} vs {param2_col})"
        )
        fig.tight_layout()
        return fig
    except Exception as e:
        st.error(
            MESSAGES["display_texts"]["optimizer_runner"]["heatmap_plot_error"].format(e=e)
            + f" (Parameters: {param1_col}, {param2_col})"
        )
        return None


def _plot_single_line_chart(results_df: pd.DataFrame, param_col: str, objective_display_name: str) -> plt.Figure | None:
    """Generate a line chart for a single parameter."""
    try:
        # Dimensioni ulteriormente ridotte per permettere circa 4 grafici per riga
        fig, ax = plt.subplots()
        ax.plot(results_df[param_col], results_df[objective_display_name], marker="o")
        ax.set_xlabel(param_col)
        ax.set_ylabel(objective_display_name)
        ax.set_title(
            MESSAGES["display_texts"]["optimizer_runner"]["single_param_plot_title"].format(
                obj=objective_display_name, param=param_col
            )
        )
        fig.tight_layout()
        return fig
    except Exception as e:
        st.error(
            MESSAGES["display_texts"]["optimizer_runner"]["heatmap_plot_error"].format(e=e)
            + f" (Parameter: {param_col})"
        )
        return None


def _generate_plots_for_grid_search_results(results_df: pd.DataFrame, objective_display_name: str, ticker: str) -> None:
    """Generate and store plots based on the number of varying parameters.

    This function analyzes the optimization results to determine how many
    parameters were varied. It then generates the appropriate plot:
    - 1 varying parameter: A line chart.
    - 2 varying parameters: A heatmap.
    - >2 varying parameters: A series of heatmaps for each pair combination.

    The generated plots are stored in the Streamlit session state (`ss`).

    Args:
        results_df (pd.DataFrame): The DataFrame of optimization results.
        objective_display_name (str): The name of the objective metric.
        ticker (str): The ticker symbol, used as a key to store the plots.

    """
    # Prepare a DataFrame with only the objective and parameter columns
    plot_df = results_df[objective_display_name].reset_index()

    # Identify which parameters have more than one unique value
    varying_params: list[tuple[int, str]] = list_varying_params(plot_df.drop(columns=objective_display_name))
    generated_plots: list[plt.Figure] = []

    if len(varying_params) == 1:
        param_name = varying_params[0][1]
        if fig := _plot_single_line_chart(plot_df, param_name, objective_display_name):
            generated_plots.append(fig)

    elif len(varying_params) == 2:
        param1_name = varying_params[0][1]
        param2_name = varying_params[1][1]
        if fig := _plot_single_heatmap(plot_df, param1_name, param2_name, objective_display_name):
            generated_plots.append(fig)

    elif len(varying_params) > 2:
        # Generate heatmaps for all combinations of two varying parameters
        param_names = [p[1] for p in varying_params]
        for param1_name, param2_name in itertools.combinations(param_names, 2):
            if fig := _plot_single_heatmap(plot_df, param1_name, param2_name, objective_display_name):
                generated_plots.append(fig)

    ss.opt_heatmaps[ticker] = generated_plots


def run_optimization(data: pd.DataFrame, custom_constraint: Callable, ticker: str, benchmark_comparison: float) -> None:
    """Run the full optimization workflow for a single ticker."""
    if data is None or data.empty:
        st.error(MESSAGES.get("display_texts", {}).get("optimizer_runner", {}).get("no_data_for_optimization", ""))
        return

    start_time = time.perf_counter()

    # 1. Initialize the Backtest instance
    bt = _initialize_bt_instance(data)
    if not bt:
        return

    # 2. Prepare parameter ranges for the optimizer
    processed_params = _process_params_ranges(ss.opt_params)

    # 3. Execute the core optimization
    optimization_output = _execute_optimization(bt, processed_params, custom_constraint)
    if not optimization_output:
        return

    all_comb_data, sambo_data = optimization_output

    # 4. Process results based on the optimization method
    if ss.opt_method_wid == "Grid Search":
        _generate_plots_for_grid_search_results(all_comb_data, ss.opt_obj_func_wid, ticker)
    elif ss.opt_method_wid == "SAMBO" and sambo_data:
        ss.opt_sambo_plots[ticker] = make_sambo_plots(all_comb_data, sambo_data)

    # 5. Store results in session state
    ss.opt_combs_ranking[ticker] = add_benchmark_comparison(all_comb_data, benchmark_comparison, ss.opt_obj_func_wid)
    ss.trade_returns[ticker] = all_comb_data["Trade_returns"]

    # 6. Log execution time
    end_time = time.perf_counter()
    pars_time_log = {
        "n_combs": len(all_comb_data),
        "periods": len(data),
        "opt_method": ss.opt_method_wid,
        "strategy": ss.opt_strategy_wid,
    }
    log_execution_data(start_time, end_time, action="Optimization", **pars_time_log)


def start_optimization_process(
    opt_infos_container: streamlit_obj,
    opt_results_container: streamlit_obj,
) -> None:
    """Initiate and manage the full optimization process.

    Orchestrate the entire optimization workflow. This function first checks if
    the user intends to clear previous results. If so, it resets relevant
    session state variables and clears UI containers. Otherwise, it performs
    input validation, fetches benchmark data, and iterates through each
    user-selected ticker. For each ticker, it downloads data, runs the
    optimization with the specified strategy and parameters, and optionally
    performs a Monte Carlo simulation on the best combinations.
    It displays progress in the UI throughout the process.

    All inputs for the optimization (tickers, strategy, parameters, etc.) are
    retrieved from the Streamlit session state (`ss`).

    Args:
        opt_infos_container (streamlit_obj): The Streamlit container
            designated for displaying informational messages, such as download
            and run progress.
        opt_results_container (streamlit_obj): The Streamlit container
            where the final optimization results (stats, plots, etc.) will be
            rendered.

    Returns:
        None: This function modifies the Streamlit UI and session state directly.

    Side Effects:
        - Resets session state variables related to optimization results if clearing.
        - Populates `ss.opt_combs_ranking` with ranked optimization results.
        - Populates `ss.opt_heatmaps` or `ss.opt_sambo_plots` with generated plots.
        - Populates `ss.opt_mc_results` with Monte Carlo simulation results if enabled.
        - Updates Streamlit UI placeholders with progress, success, and error messages.

    """
    if ss.opt_results_generated:
        reset_ss_values_for_results()
        _reset_info_res_containers(opt_infos_container, opt_results_container)
        return

    if check_incorrect_arguments_opt():
        return

    reset_ss_values_for_results()

    _reset_info_res_containers(opt_infos_container, opt_results_container)

    (
        download_progress_placeholder,
        download_success_placeholder,
        run_progress_placeholder,
        run_success_placeholder,
        download_fail_placeholder,
        run_fail_placeholder,
    ) = create_info_placeholders(opt_infos_container)  # For backtest/optimization success/failure messages

    benchmark_stats: backtesting._stats._Stats = get_benchmark_data(
        download_progress_placeholder,
        download_success_placeholder,
        ss.successful_downloads_tickers,
        ss.failed_downloads_tickers,
    )
    benchmark_comparison: float = benchmark_stats[ss.opt_obj_func_wid]

    with opt_results_container:
        progress_bar = st.progress(0)

    for i, ticker in enumerate(ss.tickers):
        data = get_ticker_data_and_infos(
            download_progress_placeholder,
            download_success_placeholder,
            download_fail_placeholder,
            i,
            ticker,
        )
        with opt_results_container:
            run_progress_placeholder = st.spinner(
                MESSAGES["display_texts"]["messages"]["running_optimization"].format(
                    ticker=ticker, current_idx=i + 1, total_tickers=len(ss.tickers)
                ),
                show_time=True,
            )

            with run_progress_placeholder:
                run_optimization(
                    data=data, custom_constraint=None, ticker=ticker, benchmark_comparison=benchmark_comparison
                )

                # Cominciamo col Monte Carlo, se richiesto!!
                run_montecarlos_for_best_combs(benchmark_stats, ticker)

                # cycles_summary, combs_stats = initiate_wfo(
                #     data,
                #     strat_class,
                #     optimization_params_ranges,
                #     objective_function_selection,
                #     optimization_method_selection,
                #     initial_capital,
                #     commission_percent,
                #     max_tries_sambo,
                #     run_wfo,
                #     wfo_n_cycles,
                #     wfo_oos_ratio,
                #     all_comb_data[all_comb_data.columns[:-1]],
                # )  # Passiamogli la lista delle combinazioni provate

                manage_opt_run_infos(
                    run_success_placeholder,
                    run_fail_placeholder,
                    ticker,
                )

            progress_bar.progress((i + 1) / len(ss.tickers))

    progress_bar.empty()

    ss.opt_results_generated = True


def _reset_info_res_containers(opt_infos_container: streamlit_obj, opt_results_container: streamlit_obj) -> None:
    """Clear the contents of the information and results containers in the UI.

    This function is used to remove any previously displayed messages, plots,
    or tables before a new optimization run starts or when clearing old results.

    Args:
        opt_infos_container (streamlit_obj): The Streamlit container for
            displaying informational messages (e.g., progress, success, errors).
        opt_results_container (streamlit_obj): The Streamlit container for
            displaying the main optimization results (e.g., tables, plots).

    """
    opt_infos_container.empty()
    opt_results_container.empty()


def create_info_placeholders(
    opt_infos_container: streamlit_obj,
) -> tuple[streamlit_obj, streamlit_obj, streamlit_obj, streamlit_obj, streamlit_obj, streamlit_obj]:
    """Create placeholders within a Streamlit container for displaying information messages during the optimization process.

    This function sets up a 3-column layout within the given container and creates empty placeholders for progress, success, and failure messages related to data downloads and optimization runs.

    Args:
        opt_infos_container (streamlit_obj): The Streamlit container where the placeholders will be created.

    Returns:
        tuple: A tuple of six Streamlit placeholder objects, in the order:
               (download_progress, download_success, run_progress, run_success, download_fail, run_fail).

    """
    with opt_infos_container:
        col_progress, col_success, col_failed = st.columns(3)

        # Placeholders for dynamic messages
        with col_progress:
            download_progress_placeholder = st.empty()
            download_success_placeholder = st.empty()
        with col_success:
            run_progress_placeholder = st.empty()
            run_success_placeholder = st.empty()
        with col_failed:
            download_fail_placeholder = st.empty()
            run_fail_placeholder = st.empty()

    return (
        download_progress_placeholder,
        download_success_placeholder,
        run_progress_placeholder,
        run_success_placeholder,
        download_fail_placeholder,
        run_fail_placeholder,
    )


def check_incorrect_arguments_opt() -> bool:
    """Check for incorrect or missing arguments required for the optimization process.

    Displays an error message using Streamlit if any required argument is missing or invalid.

    Returns:
        bool: True if any incorrect argument is found, False otherwise.

    """
    if not ss.tickers:
        st.error(MESSAGES["display_texts"]["messages"]["enter_ticker_error"])
        return True
    elif ss.opt_strategy_wid is None:
        st.error(MESSAGES["display_texts"]["messages"]["select_valid_strategy_error"])
        return True
    elif not ss.opt_params:
        st.error(MESSAGES["display_texts"]["messages"]["define_optimization_ranges_error"])
        return True
    else:
        return False


def manage_opt_run_infos(
    run_success_placeholder: streamlit_obj,
    run_fail_placeholder: streamlit_obj,
    ticker: str,
) -> None:
    """Update and display the status of optimization runs for a given ticker.

    This function updates the Streamlit session state lists for successful and failed
    optimization runs based on whether the ticker is present in the optimization
    results. It then displays a success or error message in the provided Streamlit
    placeholders, listing all tickers that have completed successfully or failed so far.

    Args:
        run_success_placeholder (streamlit_obj): Streamlit placeholder for displaying
            success messages related to optimization runs.
        run_fail_placeholder (streamlit_obj): Streamlit placeholder for displaying
            error messages related to failed optimization runs.
        ticker (str): The ticker symbol for which the optimization run was attempted.

    Returns:
        None: This function modifies the Streamlit UI and session state directly.

    Side Effects:
        - Appends the ticker to `st.session_state.successful_runs_tickers` or
          `st.session_state.failed_runs_tickers`.
        - Updates the UI with a message listing all successful or failed tickers.

    """
    if ticker in ss.opt_combs_ranking:
        st.session_state.successful_runs_tickers.append(ticker)
        run_success_placeholder.success(
            MESSAGES["display_texts"]["messages"]["execution_completed"]
            + ", ".join(st.session_state.successful_runs_tickers)
        )
    else:
        st.session_state.failed_runs_tickers.append(ticker)
        run_fail_placeholder.error(
            MESSAGES["display_texts"]["messages"]["execution_failed"] + ", ".join(st.session_state.failed_runs_tickers)
        )


def initiate_wfo(
    data: pd.DataFrame,
    strat_class: CommonStrategy,
    optimization_params_ranges: dict[str : range | list],
    objective_function: str,
    opt_method: str,
    initial_capital: float,
    commission_percent: float,
    max_tries_sambo: int,
    run_wfo: bool,
    n_cycles: int,
    oos_ratio: float,
    combinations_df: pd.DataFrame,
):
    if not run_wfo:
        return None, None

    param_names = list(combinations_df.columns)
    valid_combinations_set = set(combinations_df[param_names].itertuples(index=False, name=None))

    def dynamic_constraint(param_dict):
        # Estrai solo i parametri che ci interessano
        # param_values = tuple(kwargs.get(param) for param in param_names)
        return tuple(param_dict.values()) in valid_combinations_set

    params_to_test_in_wfo = {col: combinations_df[col].unique().tolist() for col in combinations_df.columns}

    # Crea e esegui l'ottimizzatore
    wfo = WalkForwardOptimizer(
        data=data,
        strategy_class=strat_class,
        objective_function=objective_function,
        opt_method=opt_method,
        param_grid=params_to_test_in_wfo,
        initial_capital=initial_capital,
        commission_percent=commission_percent,
        max_tries_sambo=max_tries_sambo,
        n_cycles=n_cycles,
        oos_ratio=oos_ratio,
        constraint_wfo=dynamic_constraint,
    )

    # Esegui l'ottimizzazione. Results_df restituisce un df in cui ogni riga è un cliclo e le colonne sono le date, i migliori parametri
    cycles_summary, combs_stats = wfo.run_wfo_optimization()

    return cycles_summary, combs_stats


class WalkForwardOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: CommonStrategy,
        param_grid: dict[str, range | list[float | int]],
        n_cycles: int,
        oos_ratio: float,
        objective_function: str,
        opt_method: str,
        initial_capital: float,
        commission_percent: float,
        max_tries_sambo: int,
        constraint_wfo: Callable,
    ):
        self.data = data.copy()
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.n_cycles = n_cycles
        self.oos_ratio = oos_ratio
        self.objective_func = objective_function
        self.opt_method = opt_method
        self.initial_capital = initial_capital
        self.comm_percent = commission_percent
        self.max_tries_sambo = max_tries_sambo
        self.results = []
        self.combs_stats = None
        self.optimal_params_history = []
        self.constraint_wfo = constraint_wfo

        # Validazione input
        self._validate_inputs()

    def _validate_inputs(self):
        """Valida gli input forniti"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel DataFrame: {missing_cols}")

        if not 0 < self.oos_ratio < 1:
            raise ValueError("test_ratio deve essere tra 0 e 1")

        # if self.n_cycles < 2:
        #     raise ValueError("n_cycles deve essere almeno 2")

    def _calculate_windows(self) -> list[tuple[int, int, int, int]]:
        """
        Calcola le finestre di training e test per ogni ciclo.

        Returns:
            Lista di tuple (train_start, train_end, test_start, test_end)
        """
        total_rows = len(self.data)

        # Calcola la dimensione di ogni ciclo
        cycle_size = int(total_rows // ((self.n_cycles - 1) * self.oos_ratio + 1))
        oos_size = int(cycle_size * self.oos_ratio)
        is_size = cycle_size - oos_size

        windows = []

        for i in range(self.n_cycles):
            if i == 0:
                is_start = 0
                is_end = is_size
                oos_start = is_end
                oos_end = min(oos_start + oos_size, total_rows)
            else:
                oos_start = oos_end
                oos_end = min(oos_start + oos_size, total_rows)
                is_end = oos_start
                is_start = max(is_end - is_size, 0)

            # Verifica che ci siano abbastanza dati
            if oos_end <= oos_start or is_end <= is_start:
                break

            windows.append((is_start, is_end, oos_start, oos_end))

        return windows

    def _generate_param_combinations(self) -> list[dict]:
        """Genera tutte le combinazioni possibili dei parametri."""
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        combinations.extend(dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values))
        return combinations

    def _optimize_parameters(self, train_data: pd.DataFrame) -> dict:
        """
        Ottimizza i parametri su un set di dati di training.

        Args:
            train_data: Dati per l'ottimizzazione

        Returns:
            Dizionario con i parametri ottimali
        """
        best_params = None
        best_metric = -np.inf

        param_combinations = self._generate_param_combinations()

        for params in param_combinations:
            try:
                # Simula il backtesting con i parametri correnti
                # Qui dovresti integrare con la tua libreria di backtesting preferita
                metric = self._evaluate_strategy(train_data, params)

                if metric > best_metric:
                    best_metric = metric
                    best_params = params.copy()

            except Exception as e:
                print(f"Errore con parametri {params}: {e}")
                continue

        return best_params or param_combinations[0]

    def _evaluate_strategy(self, data: pd.DataFrame, params: dict) -> float:
        """
        Valuta una strategia con parametri specifici.

        Args:
            data: Dati per la valutazione
            params: Parametri della strategia

        Returns:
            Metrica di performance (es. Sharpe Ratio, Return, etc.)
        """
        # PLACEHOLDER: Implementa qui la logica di backtesting
        # Esempio con una strategia semplice di media mobile

        if "ma_period" in params:
            data = data.copy()
            data["MA"] = data["Close"].rolling(window=params["ma_period"]).mean()
            data["Signal"] = np.where(data["Close"] > data["MA"], 1, -1)
            data["Returns"] = data["Close"].pct_change()
            data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]

            # Calcola metrica (es. Sharpe Ratio)
            if data["Strategy_Returns"].std() != 0:
                sharpe = data["Strategy_Returns"].mean() / data["Strategy_Returns"].std() * np.sqrt(252)
                return sharpe

        return 0.0

    def _test_strategy(self, test_data: pd.DataFrame, params: dict) -> dict:
        """
        Testa la strategia con parametri ottimali su dati out-of-sample.

        Args:
            test_data: Dati per il test
            params: Parametri ottimali

        Returns:
            Dizionario con le metriche di performance
        """
        try:
            # Implementa la logica di test
            test_data = test_data.copy()

            if "ma_period" in params:
                test_data["MA"] = test_data["Close"].rolling(window=params["ma_period"]).mean()
                test_data["Signal"] = np.where(test_data["Close"] > test_data["MA"], 1, -1)
                test_data["Returns"] = test_data["Close"].pct_change()
                test_data["Strategy_Returns"] = test_data["Signal"].shift(1) * test_data["Returns"]

                # Calcola metriche
                total_return = (1 + test_data["Strategy_Returns"]).prod() - 1
                volatility = test_data["Strategy_Returns"].std() * np.sqrt(252)
                sharpe = (
                    test_data["Strategy_Returns"].mean() / test_data["Strategy_Returns"].std() * np.sqrt(252)
                    if test_data["Strategy_Returns"].std() != 0
                    else 0
                )

                return {
                    "total_return": total_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": self._calculate_max_drawdown(test_data["Strategy_Returns"]),
                }
        except Exception as e:
            print(f"Errore nel test: {e}")

        return {
            "total_return": 0,
            "volatility": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calcola il massimo drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def run_wfo_optimization(self) -> pd.DataFrame:
        """Esegue la Walk Forward Optimization completa.

        Returns:
            DataFrame con i risultati di ogni ciclo

        """
        windows = self._calculate_windows()
        col_wfo_cycle, col_task = st.columns(2)
        with col_wfo_cycle:
            wfo_info = st.empty()
        with col_task:
            wfo_task = st.empty()

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            wfo_info.info(f"WFO - Ciclo {i + 1}/{len(windows)}")
            print(f"Training: righe {train_start} - {train_end} ({train_end - train_start} punti)")
            print(f"Test: righe {test_start} - {test_end} ({test_end - test_start} punti)")

            # Estrai dati di training e test
            train_data = self.data.iloc[train_start:train_end].copy()
            test_data = self.data.iloc[test_start:test_end].copy()

            # Ottimizza parametri sui dati di training
            wfo_task.info(f"Optimization of In Sample period {i + 1} in progress")
            combs_ranking, optimal_params, _, _, _, _, _ = run_optimization(
                train_data,
                self.strategy_class,
                self.param_grid,
                self.objective_func,
                "Grid Search",
                self.initial_capital,
                self.comm_percent,
                self.max_tries_sambo,
                self.constraint_wfo,
            )

            df_oos_results = pd.DataFrame()
            for index, row in combs_ranking[
                [col for col in combs_ranking.columns if col != self.objective_func]
            ].iterrows():
                param_comb = dict(row)
                # Testa sui dati out-of-sample
                wfo_task.info(f"Backtest of Out Of Sample period {i + 1}, combination {index + 1}/{len(combs_ranking)}")
                test_results, _, _, _, _ = run_backtest(
                    test_data,
                    self.strategy_class,
                    param_comb,
                    self.initial_capital,
                    self.comm_percent,
                    False,
                )
                oos_result = test_results[self.objective_func]
                row[f"{self.objective_func}_oos_cycle{i + 1}"] = oos_result
                df_oos_results = pd.concat([df_oos_results, row.to_frame().T], ignore_index=True)
            df_oos_results = df_oos_results.set_index(
                [col for col in df_oos_results.columns if col in self.param_grid.keys()]
            )

            # Salva risultati
            cycle_result = {
                "cycle": i + 1,
                "train_start": train_data.index[0].strftime("%d-%m-%Y"),
                "train_end": train_data.index[-1].strftime("%d-%m-%Y"),
                "test_start": test_data.index[0].strftime("%d-%m-%Y"),
                "test_end": test_data.index[-1].strftime("%d-%m-%Y"),
                "optimal_params": optimal_params,
                **test_results,
            }

            self.results.append(cycle_result)

            # Rendo i parametri parte dell'index (così posso fare l'outer join con gli altri cicli di WFO).
            # Poi rinomino la colonna obbiettivo altrimenti negli outer join, avrò colonne con los tesso nome.
            # Poi faccio l'outer join del dataframe con quello preesistente, se esiste
            # L'obbiettivo è avere un df in cui il multiindex indica i parametri e le colonne i valori della funzione obbiettivo nelle varie run
            combs_ranking = combs_ranking.set_index(
                [col for col in combs_ranking.columns if col != self.objective_func]
            )
            combs_ranking = combs_ranking.rename(
                columns={col: col + f"_is_cycle{i + 1}" for col in combs_ranking.columns if col == self.objective_func}
            )
            self.combs_stats = (
                pd.merge(
                    self.combs_stats,
                    combs_ranking,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
                if self.combs_stats is not None
                else combs_ranking
            )
            self.combs_stats = (
                pd.merge(
                    self.combs_stats,
                    df_oos_results,
                    how="outer",
                    left_index=True,
                    right_index=True,
                )
                if self.combs_stats is not None
                else combs_ranking
            )

            self.optimal_params_history.append(optimal_params)

        wfo_info.empty()
        wfo_task.empty()

        # Aggiungo un po' di statistiche per capire come si comportano mediamente i parametri
        columns_for_calcs = self.combs_stats[
            [
                col_name
                for col_name in self.combs_stats.columns  # prendiamo solo le colonne dei risultati dei cicli
                if col_name.rsplit("_is_cycle", 1)[0] == self.objective_func
            ]
        ]
        self.combs_stats["Avg"] = columns_for_calcs.mean(
            axis=1
        )  # mettiamo la media del KPI in ogni ciclo. Facciamo uguale per altre statistiche
        self.combs_stats["Std_Dev"] = columns_for_calcs.std(axis=1)
        self.combs_stats["Count"] = columns_for_calcs.count(axis=1)
        self.combs_stats["Max"] = columns_for_calcs.max(axis=1)
        self.combs_stats["Min"] = columns_for_calcs.min(axis=1)
        self.combs_stats["Ranks"] = self.combs_stats.rank(method="min", ascending=False).values.tolist()
        self.combs_stats = self.combs_stats.drop(columns=columns_for_calcs.columns)

        return pd.DataFrame(self.results), self.combs_stats

    def get_summary_statistics(self) -> dict:
        """Calcola statistiche riassuntive dell'ottimizzazione."""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        return {
            "total_cycles": len(self.results),
            "avg_return": df["total_return"].mean(),
            "avg_sharpe": df["sharpe_ratio"].mean(),
            "avg_volatility": df["volatility"].mean(),
            "avg_max_drawdown": df["max_drawdown"].mean(),
            "win_rate": (df["total_return"] > 0).sum() / len(df),
            "best_cycle": df.loc[df["sharpe_ratio"].idxmax()]["cycle"],
            "worst_cycle": df.loc[df["sharpe_ratio"].idxmin()]["cycle"],
        }

    def plot_wfo_timeline(self) -> None:
        """Plot the timeline of Walk Forward Optimization cycles.

        Visualizes the training and test periods for each WFO cycle using matplotlib.
        """
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter

            if not self.results:
                print("Nessun risultato da visualizzare. Esegui prima run_optimization()")
                return

            # Verifica se abbiamo una colonna Date
            has_dates = "Date" in self.data.columns
            if has_dates:
                dates = pd.to_datetime(self.data["Date"])
            else:
                # Crea date fittizie per la visualizzazione
                dates = pd.date_range("2020-01-01", periods=len(self.data), freq="D")

            # Crea figura con un solo subplot
            fig, ax1 = plt.subplots(figsize=(18, 10))  # Aumentata per avere più spazio

            # Colori per training e test
            train_color = "#2E8B57"  # Verde scuro
            test_color = "#DC143C"  # Rosso

            # Altezza delle barre (ridotta per lasciare spazio al testo)
            bar_height = 0.6

            # Disegna ogni ciclo
            for i, result in enumerate(self.results):
                y_pos = len(self.results) - i - 1  # Inverti l'ordine per avere il ciclo 1 in alto

                # Date di inizio e fine per training e test
                train_start_date = dates.iloc[result["train_start"]]
                train_end_date = dates.iloc[result["train_end"] - 1]
                test_start_date = dates.iloc[result["test_start"]]
                test_end_date = dates.iloc[result["test_end"] - 1]

                # Calcola durate
                train_duration = result["train_end"] - result["train_start"]
                test_duration = result["test_end"] - result["test_start"]

                # Barra di training (verde)
                train_width = (train_end_date - train_start_date).days
                train_rect = patches.Rectangle(
                    (train_start_date, y_pos - bar_height / 2),
                    pd.Timedelta(days=train_width),
                    bar_height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=train_color,
                    alpha=0.8,
                    label="Training" if i == 0 else "",
                )
                ax1.add_patch(train_rect)

                # Barra di test (rosso)
                test_width = (test_end_date - test_start_date).days
                test_rect = patches.Rectangle(
                    (test_start_date, y_pos - bar_height / 2),
                    pd.Timedelta(days=test_width),
                    bar_height,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=test_color,
                    alpha=0.8,
                    label="Test" if i == 0 else "",
                )
                ax1.add_patch(test_rect)

                # Etichetta del ciclo a sinistra
                ax1.text(
                    train_start_date - pd.Timedelta(days=30),
                    y_pos,
                    f"Ciclo {result['cycle']}",
                    ha="right",
                    va="center",
                    fontweight="bold",
                    fontsize=11,
                )

                # === TESTO SULLE BARRE ===

                # Data di inizio training (sopra la barra training, a sinistra)
                ax1.text(
                    train_start_date,
                    y_pos + bar_height / 2 + 0.08,
                    train_start_date.strftime("%d/%m/%y"),
                    ha="left",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    color="darkgreen",
                )

                # Data di fine training / inizio test (una sola data al punto di transizione)
                transition_date = train_end_date  # Fine training = inizio test
                ax1.text(
                    transition_date,
                    y_pos + bar_height / 2 + 0.08,
                    transition_date.strftime("%d/%m/%y"),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    color="black",
                )  # Nero per indicare la transizione

                # Data di fine test (sopra la barra test, a destra)
                ax1.text(
                    test_end_date,
                    y_pos + bar_height / 2 + 0.08,
                    test_end_date.strftime("%d/%m/%y"),
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    color="darkred",
                )

                # Durata training (al centro della barra training)
                train_center = train_start_date + pd.Timedelta(days=train_width / 2)
                ax1.text(
                    train_center,
                    y_pos,
                    f"{train_duration}d",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="darkgreen",
                        alpha=0.8,
                        edgecolor="none",
                    ),
                )

                # Durata test (al centro della barra test)
                test_center = test_start_date + pd.Timedelta(days=test_width / 2)
                ax1.text(
                    test_center,
                    y_pos,
                    f"{test_duration}d",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="darkred",
                        alpha=0.8,
                        edgecolor="none",
                    ),
                )

            # Configurazione degli assi del grafico (aumentato spazio verticale per le date)
            ax1.set_xlim(dates.min() - pd.Timedelta(days=50), dates.max() + pd.Timedelta(days=30))
            ax1.set_ylim(-0.7, len(self.results) - 0.3)  # Più spazio per le etichette

            # Formattazione delle date sull'asse x
            ax1.xaxis.set_major_formatter(DateFormatter("%m/%Y"))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # Rimuovi i tick dell'asse Y
            ax1.set_yticks([])

            # Calcola date di inizio e fine dell'intera WFO
            overall_start_date = dates.iloc[self.results[0]["train_start"]]
            overall_end_date = dates.iloc[self.results[-1]["test_end"] - 1]

            # Titolo del grafico
            ax1.set_title(
                f"Walk Forward Optimization Timeline\n"
                f"{overall_start_date.strftime('%d/%m/%Y')} - {overall_end_date.strftime('%d/%m/%Y')}\n"
                f"{len(self.results)} cicli - {self.oos_ratio * 100:.0f}% test ratio",
                fontsize=16,
                fontweight="bold",
                pad=25,
            )
            ax1.set_xlabel("Periodo Temporale", fontsize=14)

            # Legenda migliorata
            ax1.legend(loc="upper right", bbox_to_anchor=(1, 1), fontsize=12)

            # Griglia leggera
            ax1.grid(True, axis="x", alpha=0.3, linestyle="--")

            # Layout finale
            plt.tight_layout()
            plt.show()

            # # Stampa statistiche riassuntive più dettagliate
            # print("\n" + "="*90)
            # print("STATISTICHE WALK FORWARD OPTIMIZATION")
            # print("="*90)

            # total_cycles = len(self.results)
            # avg_train_duration = sum(r['train_end'] - r['train_start'] for r in self.results) / total_cycles
            # avg_test_duration = sum(r['test_end'] - r['test_start'] for r in self.results) / total_cycles

            # overall_start = self.results[0]['train_start']
            # overall_end = self.results[-1]['test_end']
            # total_period = overall_end - overall_start

            # print(f"📊 Totale cicli: {total_cycles}")
            # print(f"🏋️  Durata media training: {avg_train_duration:.1f} giorni")
            # print(f"🧪 Durata media test: {avg_test_duration:.1f} giorni")
            # print(f"📅 Periodo totale analizzato: {total_period} giorni")
            # print(f"🗓️  Range temporale: {overall_start_date.strftime('%d/%m/%Y')} - {overall_end_date.strftime('%d/%m/%Y')}")
            # print(f"📈 Rapporto out-of-sample: {self.oos_ratio*100:.1f}%")

            # # Statistiche dettagliate per ciclo
            # print(f"\n{'Ciclo':<8} {'Train Days':<12} {'Test Days':<12} {'Train Start':<12} {'Test End':<12}")
            # print("-" * 60)
            # for result in self.results:
            #     train_duration = result['train_end'] - result['train_start']
            #     test_duration = result['test_end'] - result['test_start']
            #     train_start = dates.iloc[result['train_start']].strftime('%d/%m/%y')
            #     test_end = dates.iloc[result['test_end']-1].strftime('%d/%m/%y')
            #     print(f"{result['cycle']:<8} {train_duration:<12} {test_duration:<12} {train_start:<12} {test_end:<12}")

            # # Calcola e mostra eventuali sovrapposizioni
            # overlaps = 0
            # for i in range(1, len(self.results)):
            #     if self.results[i]['train_start'] < self.results[i-1]['test_end']:
            #         overlaps += 1

            # if overlaps > 0:
            #     print(f"\n⚠️  Cicli con sovrapposizione temporale: {overlaps}")
            # else:
            #     print(f"\n✅ Nessuna sovrapposizione temporale tra cicli")

        except ImportError:
            print("Matplotlib non disponibile per la visualizzazione")

    def plot_results(self) -> None:
        """Visualizza i risultati dell'ottimizzazione."""
        try:
            import matplotlib.pyplot as plt

            if not self.results:
                print("Nessun risultato da visualizzare. Esegui prima run_optimization()")
                return

            df = pd.DataFrame(self.results)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Walk Forward Optimization Results", fontsize=16)

            # Return per ciclo
            axes[0, 0].bar(df["cycle"], df["total_return"])
            axes[0, 0].set_title("Return per Ciclo")
            axes[0, 0].set_xlabel("Ciclo")
            axes[0, 0].set_ylabel("Return")
            axes[0, 0].axhline(y=0, color="r", linestyle="--", alpha=0.5)

            # Sharpe Ratio per ciclo
            axes[0, 1].plot(df["cycle"], df["sharpe_ratio"], marker="o")
            axes[0, 1].set_title("Sharpe Ratio per Ciclo")
            axes[0, 1].set_xlabel("Ciclo")
            axes[0, 1].set_ylabel("Sharpe Ratio")
            axes[0, 1].axhline(y=0, color="r", linestyle="--", alpha=0.5)

            # Volatilità per ciclo
            axes[1, 0].plot(df["cycle"], df["volatility"], marker="s", color="orange")
            axes[1, 0].set_title("Volatilità per Ciclo")
            axes[1, 0].set_xlabel("Ciclo")
            axes[1, 0].set_ylabel("Volatilità")

            # Max Drawdown per ciclo
            axes[1, 1].bar(df["cycle"], df["max_drawdown"], color="red", alpha=0.7)
            axes[1, 1].set_title("Max Drawdown per Ciclo")
            axes[1, 1].set_xlabel("Ciclo")
            axes[1, 1].set_ylabel("Max Drawdown")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib non disponibile per la visualizzazione")


def make_sambo_plots(all_comb_data: pd.DataFrame, sambo_plots: object) -> None:
    """Display SAMBO optimization plots for the given parameter combinations and plot data.

    Args:
        all_comb_data (pd.DataFrame): DataFrame containing all optimization combinations.
        sambo_plots (object): SAMBO plot data object.

    Returns:
        None

    """
    varying_params: list[tuple[int, str]] = list_varying_params(all_comb_data.index.to_frame())
    varying_param_names = [p[1] for p in varying_params]
    varying_param_idx = [p[0] for p in varying_params]

    return [
        plot_objective(
            sambo_plots,
            names=varying_param_names,
            plot_dims=varying_param_idx,
            estimator="et",
        ),
        plot_evaluations(sambo_plots, names=varying_param_names, plot_dims=varying_param_idx),
        plot_convergence(sambo_plots),
    ]
