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

from backtest_runner import get_benchmark_data, run_backtest
from config import MESSAGES, streamlit_obj
from data_handler import get_ticker_data_and_infos
from display_results import display_results
from excel_exporter import log_execution_data
from monte_carlo import run_montecarlos_for_best_combs
from strategies.common_strategy import CommonStrategy
from utils import OptimizationRecorder, list_varying_params, record_all_optimizations

# Alias di tipo per maggiore chiarezza e leggibilità
OptimizationParamsRanges = dict[str, range | tuple[float, float, float] | list[Any]]
BacktestOptimizationOutput = pd.Series | tuple[pd.Series, pd.Series]


def _initialize_backtest_instance(
    data: pd.DataFrame,
    strategy_class: list[Backtest],  # Assumendo che strategy_class sia una classe ereditata da backtesting.Strategy
    initial_capital: float,
    commission_percent: float,
) -> Backtest:
    """Initialize and return a Backtest instance.

    Args:
        data (pd.DataFrame): Financial data (OHLCV) for the backtest.
        strategy_class (list[Backtest]): The strategy class to be tested.
        initial_capital (float): Initial capital.
        commission_percent (float): Commission percentage per trade.

    Returns:
        Backtest: A configured Backtest instance.

    """
    return Backtest(
        data,
        strategy_class,
        cash=initial_capital,
        commission=commission_percent,
        exclusive_orders=True,  # Consente una sola posizione alla volta
    )


# def _get_optimization_objective(objective_function_key: str) -> tuple[str, bool, str]:
#     """Retrieve the optimization objective function from the config.

#     The key is the metric name (and also the display name for the UI).
#     The value associated with the key is a boolean indicating whether to maximize (True) or minimize (False).

#     Args:
#         objective_function_key (str): Key of the objective function.

#     Returns:
#         tuple: (str, bool, str) - Objective function string, whether to maximize, display name.

#     Raises:
#         ValueError: If the objective function key is invalid or its format is incorrect.

#     """
#     maximize_bool: bool | None = MESSAGES["optimization_settings"]["objectives"].get(objective_function_key)

#     if maximize_bool is None:
#         raise ValueError(
#             MESSAGES["display_texts"]["optimizer_runner"]["invalid_objective_function"].format(
#                 key=objective_function_key
#             )
#             + MESSAGES["display_texts"]["optimizer_runner"]["check_config_format_simple"]
#         )

#     if not isinstance(maximize_bool, bool):
#         raise ValueError(
#             MESSAGES["display_texts"]["optimizer_runner"]["invalid_objective_function"].format(
#                 key=objective_function_key
#             )
#             + MESSAGES["display_texts"]["optimizer_runner"]["invalid_objective_value_type"].format(
#                 key=objective_function_key, type=type(maximize_bool).__name__
#             )
#         )

#     objective_func: str = objective_function_key
#     objective_display_name: str = objective_function_key

#     return objective_func, maximize_bool, objective_display_name


def _process_params_ranges(
    params_ranges: OptimizationParamsRanges,
) -> dict[str, list[Any] | Any]:
    """Prepare parameter ranges for backtesting.py.

    Converts 'range' objects and float tuples into lists/tuples suitable for optimization.

    Args:
        params_ranges (OptimizationParamsRanges): Dictionary of parameter ranges as defined by the UI.

    Returns:
        dict[str, list[Any] | Any]: Dictionary of processed parameters for backtesting.py.

    """
    processed_params: dict[str, list[Any] | Any] = {}
    for param, val_range in params_ranges.items():
        if isinstance(val_range, range):
            processed_params[param] = list(val_range)
        elif isinstance(val_range, tuple) and len(val_range) == 3:
            min_val, max_val, step_val = val_range
            # Aggiusta leggermente max_val per assicurare l'inclusione se lo step è piccolo
            # e la precisione del float causa problemi
            processed_params[param] = np.arange(min_val, max_val + step_val / 2, step_val).tolist()
        elif isinstance(val_range, list):
            processed_params[param] = val_range
        else:
            st.warning(
                MESSAGES["display_texts"]["optimizer_runner"]["unexpected_param_range_type"].format(
                    param=param, type=type(val_range).__name__
                )
            )
            processed_params[param] = [val_range]  # Tratta come un singolo valore se il tipo è inaspettato

    return processed_params


def _execute_optimization(
    bt: Backtest,
    processed_params_ranges: dict[str, list[Any] | Any],
    objective_func: str,
    method: str,
    strategy_class: list[Backtest],
    sambo_tries: int,
    custom_constraint: Callable | None = None,
) -> BacktestOptimizationOutput:
    """Execute the core of the optimization with backtesting.py, handling different methods.

    Args:
        bt (Backtest): The Backtest instance.
        processed_params_ranges (dict[str, list[Any] | Any]): Parameters ready for optimization.
        objective_func (str): The objective metric for optimization.
        method (str): The optimization method ("Grid Search" or "SAMBO").
        strategy_class (list[Backtest]): The strategy class to optimize.
        custom_constraint (Callable | None): A custom constraint function, if provided.
        sambo_tries (int): The number of tries for the SAMBO optimization method.

    Returns:
        BacktestOptimizationOutput: The raw output of the bt.optimize() function.

    Raises:
        ValueError: If the optimization method is not supported.

    """
    # Tenta di ottenere il vincolo di ottimizzazione dalla classe della strategia
    constraint_func = get_constraint(strategy_class, custom_constraint)
    method_arg = MESSAGES["optimization_settings"]["methods"][method]

    if method in MESSAGES["optimization_settings"]["methods"]:
        with record_all_optimizations(bt) as recorder:
            result = bt.optimize(
                **processed_params_ranges,
                maximize=objective_func,
                method=method_arg,
                constraint=constraint_func,
                random_state=None,  # Seed to ensure reproducible results
                return_optimization=(method_arg == "sambo"),  # Need to get results for SAMBO plots
                max_tries=sambo_tries,
                return_heatmap=False,  # we will get the heatmaps with the results of recorder
            )
            sambo_data = None if method_arg == "grid" else result[1]
            return _manipulate_opt_results(processed_params_ranges, recorder, objective_func), sambo_data
    else:
        raise ValueError(
            MESSAGES["display_texts"]["optimizer_runner"]["optimization_method_not_supported"].format(method=method)
        )

    # if method == "Grid Search":
    #     # Per Grid Search, backtesting.py restituisce (stats, heatmap_series)
    #     with record_all_optimizations(bt) as recorder:
    #         bt.optimize(
    #             **processed_params_ranges,
    #             maximize=objective_func,
    #             method="grid",
    #             constraint=constraint_func,
    #             return_heatmap=True,
    #         )
    #         return _manipulate_opt_results(processed_params_ranges, recorder, objective_func)

    # elif method == "SAMBO":
    #     # Per SAMBO, backtesting.py restituisce un singolo oggetto stats con i risultati dell'ottimizzazione allegati
    #     return bt.optimize(
    #         **processed_params_ranges,
    #         maximize=objective_func,
    #         method="sambo",
    #         constraint=constraint_func,
    #         random_state=42,  # Assicura la riproducibilità
    #         return_optimization=True,  # Questo restituisce l'oggetto ottimizzatore che contiene i metodi di plotting
    #         max_tries=sambo_tries,
    #         return_heatmap=True,
    #     )
    # else:
    #     raise ValueError(
    #         MESSAGES["display_texts"]["optimizer_runner"]["optimization_method_not_supported"].format(method=method)
    #     )


def _manipulate_opt_results(
    processed_params_ranges: dict[str, list[Any] | Any],
    recorder: OptimizationRecorder,
    objective_func: str,
) -> pd.DataFrame:
    all_results = recorder.get_dataframe()
    if len(all_results) == 0 or all_results is None:
        st.error("The optimizer returned no result!")
        return None
    all_results = all_results.set_index([col for col in processed_params_ranges if col in all_results.columns])
    all_results = all_results[[*list(MESSAGES["optimization_settings"]["objectives"].keys()), "_trades"]]
    all_results["_trades"] = all_results["_trades"].apply(
        lambda nested_df: (
            nested_df["ReturnPct"].values
            if isinstance(nested_df, pd.DataFrame) and "ReturnPct" in nested_df.columns
            else np.nan
        )
    )
    all_results.rename(columns={"_trades": "Trade_returns"}, inplace=True)
    all_results = all_results.drop_duplicates(
        subset=[col for col in all_results.columns if all_results[col].dtype != list], keep="first"
    )
    return all_results.sort_values(
        by=objective_func, ascending=not MESSAGES["optimization_settings"]["objectives"][objective_func]
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


def _generate_plots_for_grid_search_results(
    results_df: pd.DataFrame, objective_display_name: str, maximize: bool
) -> list[plt.Figure]:
    """Generate a list of plots (heatmap or line) based on the number of varying parameters.

    If there are more than 2 varying parameters, it generates heatmaps for each pair combination.
    """
    heatmap_df = results_df.copy()
    heatmap_df = heatmap_df[objective_display_name].reset_index()
    # Identifica i parametri che variano (hanno più di un valore unico)
    varying_param_names: list = list_varying_params(heatmap_df[heatmap_df.columns[:-1]])

    all_generated_plots: list[plt.Figure] = []

    if len(varying_param_names) == 2:
        if fig := _plot_single_heatmap(
            heatmap_df,
            varying_param_names[0][1],
            varying_param_names[1][1],
            objective_display_name,
        ):
            all_generated_plots.append(fig)
    elif len(varying_param_names) == 1:
        if fig := _plot_single_line_chart(heatmap_df, varying_param_names[0][1], objective_display_name):
            all_generated_plots.append(fig)
    elif len(varying_param_names) > 2:
        # Genera heatmaps per tutte le combinazioni di due parametri variabili
        for param1, param2 in itertools.combinations(varying_param_names, 2):
            if fig := _plot_single_heatmap(heatmap_df, param1[1], param2[1], objective_display_name):
                all_generated_plots.append(fig)
    return all_generated_plots


def _handle_grid_search_results(
    all_comb_data: tuple[pd.DataFrame, None],
    objective_display_name: str,
    maximize: bool,
) -> tuple[
    pd.DataFrame | None,
    dict[str, Any] | None,
    pd.Series | None,
    list[plt.Figure] | None,
]:
    comb_stats = all_comb_data[0][MESSAGES["optimization_settings"]["objectives"].keys()]

    # Genera i plot (heatmap/linea singola o multiple heatmap)
    heatmap_plots: list[plt.Figure] = _generate_plots_for_grid_search_results(
        comb_stats, objective_display_name, maximize
    )

    return all_comb_data[0], heatmap_plots


def _handle_sambo_results(
    optimization_output: tuple[pd.DataFrame, pd.DataFrame],
    objective_display_name: str,
    maximize: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None, pd.Series | None, dict[str, Any] | None]:
    # best_stats: dict[str, Any] = dict(optimization_output[0])

    # # Accedi ai migliori parametri direttamente dall'oggetto strategia allegato a stats
    # best_params: dict[str, Any] = optimization_output[0]._strategy._params

    # # Crea un semplice DataFrame per visualizzare il ranking delle combinazioni
    # results_df: pd.DataFrame = optimization_output[1].sort_values(ascending=not maximize).to_frame().reset_index()

    return optimization_output[0], optimization_output[1]


def run_optimization(
    data: pd.DataFrame,
    strategy_class: list[Backtest],
    params_ranges: OptimizationParamsRanges,
    objective_function_key: str,
    optimization_method: str,
    initial_capital: float,
    commission_percent: float,
    sambo_tries: int,
    custom_constraint: Callable,
) -> tuple[
    pd.DataFrame | None,
    dict[str, Any] | None,
    pd.Series | None,
    list[plt.Figure] | None,  # Modificato il tipo di ritorno per supportare più figure
    dict[str, Any] | None,
    str,
    str,
]:
    """Run the optimization process for a given strategy and parameter ranges.

    Executes the optimization using the specified method, processes results, and returns data for display and analysis.

    Args:
        data (pd.DataFrame): Financial data for the backtest.
        strategy_class (list[Backtest]): The strategy class to optimize.
        params_ranges (OptimizationParamsRanges): Parameter ranges for optimization.
        objective_function_key (str): The objective metric for optimization.
        optimization_method (str): The optimization method ("Grid Search" or "SAMBO").
        initial_capital (float): Initial capital for the backtest.
        commission_percent (float): Commission percentage per trade.
        sambo_tries (int): Number of tries for the SAMBO optimization method.
        custom_constraint (Callable): Custom constraint function for parameter combinations.

    Returns:
        tuple: Contains all combination data, heatmap plots, SAMBO plots, status string, and message string.

    """
    if data is None or data.empty:
        return (
            None,
            None,
            None,
            None,
            None,
            "failure",
            MESSAGES["display_texts"]["optimizer_runner"]["no_data_for_optimization"],
        )

    start_time = time.perf_counter()

    # 1. Inizializza l'istanza di Backtest
    bt: Backtest = _initialize_backtest_instance(data, strategy_class, initial_capital, commission_percent)

    # # 2. Recupera l'obiettivo di ottimizzazione
    # objective_func: str
    # maximize: bool
    # objective_display_name: str
    try:
        maximize = MESSAGES["optimization_settings"]["objectives"].get(objective_function_key)
    except ValueError as ve:
        return None, None, None, None, None, "failure", str(ve)

    # 3. Prepara i range dei parametri per backtesting.py
    processed_params: dict[str, list[Any] | Any] = _process_params_ranges(params_ranges)

    heatmap_plots: list[plt.Figure] | None = None  # Modificato: ora è una lista di figure
    sambo_plot_data: dict[str, Any] | None = None

    try:
        # 4. Esegui il core dell'ottimizzazione
        optimization_output: BacktestOptimizationOutput = _execute_optimization(
            bt,
            processed_params,
            objective_function_key,
            optimization_method,
            strategy_class,
            sambo_tries,
            custom_constraint,
        )  # Best stats, all combinations

        # Gestisce il caso in cui l'ottimizzazione non produce risultati validi
        if optimization_output is None:
            return (
                None,
                None,
                None,
                None,
                None,
                "failure",
                MESSAGES["display_texts"]["optimizer_runner"]["no_valid_results"],
            )

        # Controllo specifico per l'output di Grid Search (tuple di Series) o SAMBO (singola Series)
        if (
            isinstance(optimization_output, tuple) and (optimization_output[0] is None or optimization_output[0].empty)
        ) or (isinstance(optimization_output, pd.Series) and optimization_output.empty):
            return (
                None,
                None,
                None,
                None,
                None,
                "failure",
                MESSAGES["display_texts"]["optimizer_runner"]["no_valid_results"],
            )

        # 5. Estrai e formatta i risultati in base al metodo di ottimizzazione
        if optimization_method == "Grid Search":
            all_comb_data, heatmap_plots = _handle_grid_search_results(  # Assegna a heatmap_plots
                optimization_output, objective_function_key, maximize
            )
        elif optimization_method == "SAMBO":
            all_comb_data, sambo_plot_data = _handle_sambo_results(
                optimization_output, objective_function_key, maximize
            )

        end_time = time.perf_counter()
        pars_time_log = {
            "n_combs": len(all_comb_data),
            "periods": len(data),
            "opt_method": optimization_method,
            "strategy": strategy_class.DISPLAY_NAME,
        }
        log_execution_data(start_time, end_time, action="Optimization", **pars_time_log)

        # Controllo finale se i risultati sono stati estratti con successo
        if all_comb_data is None:
            return (
                None,
                None,
                None,
                None,
                None,
                "failure",
                MESSAGES["display_texts"]["optimizer_runner"]["no_valid_results"],
            )

        return (
            all_comb_data,
            heatmap_plots,
            sambo_plot_data,
            "success",
            MESSAGES["display_texts"]["optimizer_runner"]["optimization_success"],
        )

    except ValueError as ve:
        # Gestione specifica degli errori di valore (es. problemi di configurazione)
        return None, None, None, None, None, "failure", str(ve)
    except Exception as e:
        # Gestione generica degli errori per problemi inaspettati durante l'ottimizzazione
        return (
            None,
            None,
            None,
            None,
            None,
            "failure",
            MESSAGES["display_texts"]["optimizer_runner"]["optimization_error"].format(e=e),
        )


def start_optimization_process(
    tickers: list[str],
    start_date_yf: str,
    end_date_yf: str,
    data_interval: str,
    initial_capital: float,
    commission_percent: float,
    objective_function_selection: str,
    optimization_method_selection: str,
    max_tries_sambo: int,
    run_mc: bool,
    promoted_combinations: int,
    mc_sampling_method: str,
    num_sims: int,
    sims_length: int,
    strat_class: type[CommonStrategy],
    optimization_params_ranges: dict,
    results_container: streamlit_obj,
    run_wfo: bool,
    wfo_n_cycles: int,
    wfo_oos_ratio: float,
    download_progress_placeholder: streamlit_obj,
    download_success_placeholder: streamlit_obj,
    run_progress_placeholder: streamlit_obj,  # Not accessed, but kept for signature compatibility
    run_success_placeholder: streamlit_obj,
    download_fail_placeholder: streamlit_obj,
    run_fail_placeholder: streamlit_obj,
) -> None:
    """Start the optimization process for the selected tickers and strategy.

    Handles data loading, optimization execution, Monte Carlo simulation, Walk Forward Optimization, and result display for all tickers.

    Args:
        tickers: List of ticker symbols to optimize.
        start_date_yf: Start date for data download.
        end_date_yf: End date for data download.
        data_interval: Data granularity (e.g., '1d').
        initial_capital: Initial capital for the optimization.
        commission_percent: Commission percentage for trades.
        objective_function_selection: Selected objective function for optimization.
        optimization_method_selection: Selected optimization method.
        max_tries_sambo: Maximum tries for SAMBO optimization.
        run_mc: Whether to run Monte Carlo simulation.
        promoted_combinations: Number of promoted combinations for MC.
        mc_sampling_method: Monte Carlo sampling method.
        num_sims: Number of Monte Carlo simulations.
        sims_length: Number of trades per simulation.
        strat_class: The selected strategy class.
        optimization_params_ranges: Parameter ranges for optimization.
        results_container: Streamlit container for displaying results.
        run_wfo: Whether to run Walk Forward Optimization.
        wfo_n_cycles: Number of WFO cycles.
        wfo_oos_ratio: Out-of-sample ratio for WFO.
        download_progress_placeholder: Streamlit placeholder for download progress.
        download_success_placeholder: Streamlit placeholder for download success.
        run_progress_placeholder: Streamlit placeholder for run progress.
        run_success_placeholder: Streamlit placeholder for run success.
        download_fail_placeholder: Streamlit placeholder for download failure.
        run_fail_placeholder: Streamlit placeholder for run failure.

    Returns:
        None

    """
    if not tickers:
        st.error(MESSAGES["display_texts"]["messages"]["enter_ticker_error"])
    elif strat_class is None:
        st.error(MESSAGES["display_texts"]["messages"]["select_valid_strategy_error"])
    elif not optimization_params_ranges:
        st.error(MESSAGES["display_texts"]["messages"]["define_optimization_ranges_error"])
    else:
        all_ticker_results = {}

    # results_container = st.empty()
    # download_progress_placeholder = st.empty()
    # download_success_placeholder = st.empty()
    # run_progress_placeholder = st.empty()
    # run_success_placeholder = st.empty()
    # download_fail_placeholder = st.empty()
    # run_fail_placeholder = st.empty()

    download_progress_placeholder.container()
    download_success_placeholder.empty()
    download_success_placeholder.container()
    run_progress_placeholder.container()
    run_success_placeholder.container()
    download_fail_placeholder.container()
    run_fail_placeholder.container()

    with results_container:
        st.session_state.successful_downloads_tickers = []
        st.session_state.failed_downloads_tickers = []
        st.session_state.successful_runs_tickers = []
        st.session_state.failed_runs_tickers = []

        benchmark_stats = get_benchmark_data(
            start_date_yf,
            end_date_yf,
            data_interval,
            initial_capital,
            commission_percent,
            download_progress_placeholder,
            download_success_placeholder,
            st.session_state.successful_downloads_tickers,
            st.session_state.failed_downloads_tickers,
        )
        benchmark_comparison = benchmark_stats[objective_function_selection]

        progress_bar = st.progress(0)

        for i, ticker in enumerate(tickers):
            data = get_ticker_data_and_infos(
                tickers,
                start_date_yf,
                end_date_yf,
                data_interval,
                download_progress_placeholder,
                download_success_placeholder,
                download_fail_placeholder,
                i,
                ticker,
            )

            if data is not None:
                with st.spinner(
                    MESSAGES["display_texts"]["messages"]["running_optimization"].format(
                        ticker=ticker, current_idx=i + 1, total_tickers=len(tickers)
                    ),
                    show_time=True,
                ):
                    (all_comb_data, heatmap_plot, sambo_plots, run_status, run_msg) = run_optimization(
                        data,
                        strat_class,
                        optimization_params_ranges,
                        objective_function_selection,
                        optimization_method_selection,
                        initial_capital,
                        commission_percent,
                        max_tries_sambo,
                        None,
                    )

                    # Cominciamo col Monte Carlo, se richiesto!!
                    combs_with_mc_stats = run_montecarlos_for_best_combs(
                        initial_capital,
                        objective_function_selection,
                        run_mc,
                        promoted_combinations,
                        mc_sampling_method,
                        num_sims,
                        sims_length,
                        benchmark_stats,
                        ticker,
                        all_comb_data,
                    )
                    cycles_summary, combs_stats = initiate_wfo(
                        data,
                        strat_class,
                        optimization_params_ranges,
                        objective_function_selection,
                        optimization_method_selection,
                        initial_capital,
                        commission_percent,
                        max_tries_sambo,
                        run_wfo,
                        wfo_n_cycles,
                        wfo_oos_ratio,
                        all_comb_data[all_comb_data.columns[:-1]],
                    )  # Passiamogli la lista delle combinazioni provate

                manage_opt_run_infos(
                    all_ticker_results,
                    run_success_placeholder,
                    run_fail_placeholder,
                    ticker,
                    all_comb_data,
                    heatmap_plot,
                    sambo_plots,
                    run_status,
                    combs_with_mc_stats,
                )

            progress_bar.progress((i + 1) / len(tickers))

        display_results(
            ticker_results=all_ticker_results,
            benchmark_comparison=benchmark_comparison,
            is_optimization_mode=True,
            obj_func=objective_function_selection,
        )


def manage_opt_run_infos(
    all_ticker_results: dict,
    run_success_placeholder: streamlit_obj,
    run_fail_placeholder: streamlit_obj,
    ticker: str,
    optimization_results: object,
    heatmap_plot: object,
    sambo_plots: object,
    run_status: str,
    combs_with_mc_stats: object,
) -> None:
    """Manage and update the UI and results dictionary after each optimization run.

    Updates the success or failure placeholders and stores results for each ticker.

    Args:
        all_ticker_results (dict): Dictionary to store results for each ticker.
        run_success_placeholder: Streamlit placeholder for success messages.
        run_fail_placeholder: Streamlit placeholder for failure messages.
        ticker (str): The ticker symbol for the current run.
        optimization_results: Results of the optimization.
        heatmap_plot: Generated heatmap plot(s) for the run.
        sambo_plots: SAMBO-specific plots for the run.
        run_status (str): Status string ("success" or "failure").
        combs_with_mc_stats: Monte Carlo statistics for the best combinations.

    Returns:
        None

    """
    if run_status == "success":
        st.session_state.successful_runs_tickers.append(ticker)
        run_success_placeholder.success(
            MESSAGES["display_texts"]["messages"]["execution_completed"]
            + ", ".join(st.session_state.successful_runs_tickers)
        )
        if optimization_results is not None:
            all_ticker_results[ticker] = (
                optimization_results,
                heatmap_plot,
                sambo_plots,
                combs_with_mc_stats,
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
