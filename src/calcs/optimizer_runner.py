# optimizer_runner.py
import itertools
import time
from collections.abc import Callable
from typing import Any

import backtesting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import streamlit as st
from backtesting import Backtest, Strategy
from matplotlib.figure import Figure
from sambo import OptimizeResult
from sambo.plot import plot_convergence, plot_evaluations, plot_objective

from src.calcs.backtest_runner import BenchmarkBacktestError, get_benchmark_data
from src.calcs.monte_carlo import get_montecarlo_stats
from src.config.config import MESSAGES, ss, streamlit_obj
from src.data_handlers.data_handler import DownloadedDataError, MissingColumnError, get_ticker_data_and_infos
from src.data_handlers.excel_exporter import log_execution_data
from src.utils.utils import (
    OptimizationRecorder,
    add_benchmark_comparison,
    list_varying_params,
    record_all_optimizations,
    reset_ss_values_for_optimization_results,
)
from strategies.common_strategy import CommonStrategy

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

    obj_func = ss.get("opt_obj_func_wid", "Return [%]")
    try:
        with record_all_optimizations(bt) as recorder:
            result = bt.optimize(
                **processed_params_ranges,
                maximize=obj_func,
                method=method_arg,
                constraint=constraint_func,
                random_state=None,
                return_optimization=(method_arg == "sambo"),
                max_tries=ss.get("max_tries_SAMBO_wid"),
                return_heatmap=False,
            )

            # Safely extract SAMBO data if the method returned it
            sambo_data: scipy.optimize.OptimizeResult = (
                result[1] if method_arg == "sambo" and isinstance(result, tuple) else None
            )

            # Process the recorded results
            all_results_df = _manipulate_opt_results(recorder, obj_func)

            if all_results_df is None or all_results_df.empty:
                return None

            return all_results_df, sambo_data
    except Exception as e:
        st.error(messages.get("optimization_error", "Error during optimization: {e}").format(e=e))
        return None


def _manipulate_opt_results(
    # processed_params_ranges: dict[str, list[Any]],
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
    # Step 1: Retrieve the raw optimization results from the recorder
    all_results = recorder.get_dataframe()

    # Check if the results are empty or None
    if all_results is None or all_results.empty:
        st.error(
            MESSAGES.get("display_texts", {})
            .get("optimizer_runner", {})
            .get("no_valid_results", "The optimizer returned no valid results.")
        )
        return None

    # Step 2: Extract trade returns from the raw results
    all_results["_trades"] = all_results["_trades"].map(
        lambda nested_df: (
            nested_df["ReturnPct"].to_numpy()
            if isinstance(nested_df, pd.DataFrame) and "ReturnPct" in nested_df.columns
            else np.nan
        )
    )

    # Step 3: Extract equity curve data from the raw results
    all_results["_equity_curve"] = all_results["_equity_curve"].map(
        lambda nested_df: pd.DataFrame(nested_df["Equity"].pct_change() + 1)
        if isinstance(nested_df, pd.DataFrame) and "Equity" in nested_df.columns
        else np.nan
    )

    # Step 4: Rename columns for better readability
    all_results.rename(columns={"_trades": "Trade_returns", "_equity_curve": "Equity_curve"}, inplace=True)

    # Step 5: Set "BT Stats" as the first level of the MultiIndex columns for the base stats
    all_available_stats: list = [stat["name"] for stat in MESSAGES["all_stats_properties"]]
    all_results.columns = pd.MultiIndex.from_tuples(
        [
            ("BT Stats", col) if col in [*all_available_stats, "Trade_returns", "Equity_curve"] else ("", col)
            for col in all_results.columns
        ]
    )

    # Step 6: Add strategy information to the results
    all_results[("Strategy", "Strat Name")] = all_results[("", "_strategy")].map(
        lambda x: getattr(x, "DISPLAY_NAME", x.__class__.__name__)
        if isinstance(x, (CommonStrategy, Strategy))
        else "Unknown Strategy"
    )
    all_results[("Strategy", "Param Names")] = all_results[("", "_strategy")].map(
        lambda x: tuple(x._params.keys()) if isinstance(x, (CommonStrategy, Strategy)) else ()
    )
    all_results[("Strategy", "Param Values")] = all_results[("", "_strategy")].map(
        lambda x: tuple(x._params.values()) if isinstance(x, (CommonStrategy, Strategy)) else ()
    )

    # Step 7: Remove unneeded columns
    all_results = all_results.drop(columns=[col for col in all_results.columns if col[0] == ""])

    # Step 8: Remove duplicate parameter combinations (based on strat related fields), keeping the first result
    all_results = all_results[
        ~all_results[[c for c in all_results.columns if c[0] == "Strategy"]].duplicated(keep="first")
    ]

    # Step 9: Add the objective function as an attribute for easy access
    all_results.attrs["objective_func"] = objective_func

    # Step 10: Sort the results by the objective function
    is_higher_better = MESSAGES.get("optimization_settings", {}).get("objectives", {}).get(objective_func, True)

    return all_results.sort_values(by=("BT Stats", objective_func), ascending=not is_higher_better).reset_index(
        drop=True
    )


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
) -> Figure | None:
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

        ax.set_xticklabels([str(label) for label in x_labels])
        ax.set_yticklabels([str(label) for label in y_labels])

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


def _plot_single_line_chart(results_df: pd.DataFrame, param_col: str, objective_display_name: str) -> Figure | None:
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
    # plot_df = results_df[objective_display_name].reset_index()
    plot_df = pd.DataFrame(
        results_df[("Strategy", "Param Values")].tolist(), columns=results_df[("Strategy", "Param Names")].iloc[0]
    )
    plot_df[objective_display_name] = results_df[("BT Stats", "Return [%]")]

    # Identify which parameters have more than one unique value
    varying_params: list[tuple[int, str]] = list_varying_params(results_df)
    generated_plots: list[Figure] = []

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


def run_optimization(
    data: pd.DataFrame, custom_constraint: Callable | None, ticker: str, benchmark_comparison: pd.Series
) -> None:
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
    ss.opt_master_results_table[ticker] = add_benchmark_comparison(
        all_comb_data,
        benchmark_comparison,
    )

    # 6. Convert timedelta columns to days (float) for better readability
    ss.opt_master_results_table[ticker] = ss.opt_master_results_table[ticker].apply(
        lambda col: col.dt.total_seconds() / (24 * 60 * 60) if pd.api.types.is_timedelta64_dtype(col) else col
    )

    # 7. Log execution time
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
    """Utilities to run strategy optimization workflows and related visualizations.

    This module orchestrates parameter optimization for backtesting.py strategies,
    supporting Grid Search and SAMBO methods. It prepares parameter grids from UI
    inputs, executes optimizations while recording all runs, cleans and ranks
    results, compares them against a benchmark, and optionally enriches outputs
    with Monte Carlo statistics. It also generates plots (line charts, heatmaps,
    and SAMBO diagnostics) and provides helpers for Streamlit UI messaging and
    lifecycle management.

    Key components:
    - _initialize_bt_instance: Build a Backtest from session settings.
    - _process_params_ranges: Normalize UI ranges to discrete test values.
    - _execute_optimization: Run optimizer with constraints and record results.
    - _manipulate_opt_results: Extract, deduplicate, and sort optimization stats.
    - Plot helpers for single- and multi-parameter results, and SAMBO plots.
    - start_optimization_process / run_optimization: End-to-end workflow per ticker,
    including data download, benchmark retrieval, plotting, MC, and logging.
    - WalkForwardOptimizer: Prototype for WFO windowing, evaluation, and summaries.

    Side effects:
    - Writes plots and tables to Streamlit session state (e.g., ss.opt_master_results_table,
    ss.opt_heatmaps, ss.opt_sambo_plots) and UI placeholders.
    - Logs execution timing to Excel via log_execution_data.
    """
    if ss.opt_results_generated:
        reset_ss_values_for_optimization_results()
        _reset_info_res_containers(opt_infos_container, opt_results_container)
        return

    if check_incorrect_arguments_opt():
        return

    reset_ss_values_for_optimization_results()

    _reset_info_res_containers(opt_infos_container, opt_results_container)

    (
        download_progress_placeholder,
        download_success_placeholder,
        run_progress_placeholder,
        run_success_placeholder,
        download_fail_placeholder,
        run_fail_placeholder,
    ) = create_info_placeholders(opt_infos_container)  # For backtest/optimization success/failure messages

    try:
        benchmark_stats: pd.Series = get_benchmark_data(
            start_date=ss.start_date_wid,
            end_date=ss.end_date_wid,
            interval=ss.data_interval_wid,
            initial_capital=ss.initial_capital_wid,
            commission_percent=ss.commission_percent_wid / 100,
            download_progress_placeholder=download_progress_placeholder,
            download_success_placeholder=download_success_placeholder,
            successful_downloads_tickers=ss.successful_downloads_tickers,
            failed_downloads_tickers=ss.failed_downloads_tickers,
        )
    except (DownloadedDataError, MissingColumnError, BenchmarkBacktestError) as e:
        st.warning(f"{e}")
        return

    with opt_results_container:
        progress_bar = st.progress(0)

    for i, ticker in enumerate(ss.tickers):
        try:
            data = get_ticker_data_and_infos(
                download_progress_placeholder,
                download_success_placeholder,
                download_fail_placeholder,
                i,
                ticker,
            )
        except Exception as e:
            st.warning(f"{e}")
            continue

        # Run optimization process for the current ticker.
        with opt_results_container:
            run_progress_placeholder = st.spinner(
                MESSAGES["display_texts"]["messages"]["running_optimization"].format(
                    ticker=ticker, current_idx=i + 1, total_tickers=len(ss.tickers)
                ),
                show_time=True,
            )

            with run_progress_placeholder:
                run_optimization(data=data, custom_constraint=None, ticker=ticker, benchmark_comparison=benchmark_stats)

                # Cominciamo col Monte Carlo, se richiesto!!
                # run_montecarlos_for_best_combs(benchmark_stats, ticker)
                if ss.get("opt_run_mc_wid"):
                    montecarlo_columns: pd.DataFrame = add_montecarlo_stats(
                        column_with_trade_returns=ss.opt_master_results_table[ticker][("BT Stats", "Trade_returns")],
                        initial_capital=ss.initial_capital_wid,
                        sampling_method=ss.opt_mc_sampling_method_wid,
                        num_sims=ss.opt_mc_n_sims_wid,
                        sim_length=ss.opt_mc_sim_length_wid,
                        benchmark=benchmark_stats,
                        data_interval=ss.data_interval_wid,
                    )
                    ss.opt_master_results_table[ticker][montecarlo_columns.columns] = montecarlo_columns

                # if ss["opt_run_wfo_wid"]:
                #     initiate_wfo(
                #         run_wfo=ss["opt_run_wfo_wid"],
                #         n_cycles=ss["opt_wfo_n_cycles_wid"],
                #         oos_ratio=ss["opt_wfo_oos_ratio_wid"],
                #         equity_lines=ss["equity_curves"].get(ticker),
                #     )

                # Gestiamo i risultati dell'ottimizzazione
                manage_opt_run_infos(
                    run_success_placeholder,
                    run_fail_placeholder,
                    ticker,
                )

            progress_bar.progress((i + 1) / len(ss.tickers))

    progress_bar.empty()

    ss.opt_results_generated = True


def add_montecarlo_stats(
    column_with_trade_returns: pd.Series,
    *,
    initial_capital: float,
    sampling_method: str,
    num_sims: int,
    sim_length: int,
    benchmark: pd.Series,
    data_interval: str = "1d",
) -> pd.DataFrame:
    """Perform Monte Carlo simulations on a column of trade returns.

    Parameters
    ----------
    column_with_trade_returns : pd.Series
        A column of trade returns to be used for the Monte Carlo simulations.
    initial_capital : float
        The initial capital.
    sampling_method : str
        The method for sampling trades ('permutazione' or 'resampling_con_reimmissione').
    num_sims : int
        The total number of simulations to run.
    sim_length : int
        The desired number of trades in each simulated path.
    benchmark : bt_stats
        The statistics from the benchmark backtest.
    data_interval : str, optional
        The data interval, by default "1d".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the Monte Carlo simulations.

    """
    return column_with_trade_returns.apply(
        get_montecarlo_stats,
        initial_capital=initial_capital,
        sampling_method=sampling_method,
        num_sims=num_sims,
        sim_length=sim_length,
        benchmark=benchmark,
        data_interval=data_interval,
    )


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
    if ticker in ss.opt_master_results_table:
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


def make_sambo_plots(all_comb_data: pd.DataFrame, sambo_plots: OptimizeResult) -> list[Figure]:
    """Display SAMBO optimization plots for the given parameter combinations and plot data.

    Args:
        all_comb_data (pd.DataFrame): DataFrame containing all optimization combinations.
        sambo_plots (object): SAMBO plot data object.

    Returns:
        None

    """
    varying_params: list[tuple[int, str]] = list_varying_params(
        all_comb_data[[("Strategy", "Param Values"), ("Strategy", "Param Names")]]
    )
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
