# backtest_runner.py

import os
import time

import numpy as np  # Importa numpy per np.nan
import pandas as pd
import streamlit as st
from backtesting import Backtest

from src.calcs.monte_carlo import run_montecarlo
from src.config.config import MESSAGES, ss, streamlit_obj
from src.data_handlers.data_handler import download_data, get_sp500_data
from src.data_handlers.excel_exporter import log_execution_data
from src.ui_components.display_results import show_montecarlo_equity_lines, show_montecarlo_histogram
from src.utils.utils import reset_ss_values_for_results
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy
from strategies.common_strategy import CommonStrategy


def compare_with_benchmark(strategy_stats: pd.Series, benchmark_stats: pd.Series | None) -> pd.DataFrame:
    """Compare strategy performance metrics against a benchmark.

    Creates a DataFrame for a side-by-side comparison of key performance
    metrics from a strategy backtest against a benchmark backtest (e.g.,
    Buy & Hold on an index like SPY).

    The function uses a predefined list of metrics, extracts their values
    from both the strategy and benchmark statistics Series, rounds them, and
    formats them into a clean, readable table.

    Args:
        strategy_stats (pd.Series): The complete statistics Series from the
            strategy backtest, as returned by `backtesting.Backtest.run()`.
        benchmark_stats (pd.Series | None): The complete statistics Series from
            the benchmark backtest. If None, the benchmark column in the
            resulting DataFrame will be filled with NaN values.

    Returns:
        pd.DataFrame: A DataFrame comparing key metrics. The index consists of
            metric names (e.g., 'Return [%]', 'Sharpe Ratio'), and the
            columns are 'Strategia' and 'Benchmark (SPY)'. Numeric values
            are rounded to two decimal places.

    """
    metrics_to_display = [
        "Exposure Time [%]",
        "Equity Final [$]",  # Aggiunto simbolo valuta per chiarezza
        "Return [%]",
        "Return (Ann.) [%]",
        "Volatility (Ann.) [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max. Drawdown [%]",
        "Max. Drawdown Duration",
        "Avg. Drawdown Duration",
        "# Trades",
        "Win Rate [%]",
        "Avg. Trade Duration",
        "SQN",
    ]

    strategy_values = []
    benchmark_values = []

    for metric in metrics_to_display:
        strategy_value = strategy_stats.get(metric, np.nan)
        benchmark_value = benchmark_stats.get(metric, np.nan) if benchmark_stats is not None else np.nan

        # Arrotonda i valori numerici a due cifre decimali
        if isinstance(strategy_value, (int, float)) and not np.isnan(strategy_value) and not np.isinf(strategy_value):
            strategy_values.append(round(strategy_value, 2))
        else:
            strategy_values.append(strategy_value)

        if (
            isinstance(benchmark_value, (int, float))
            and not np.isnan(benchmark_value)
            and not np.isinf(benchmark_value)
        ):
            benchmark_values.append(round(benchmark_value, 2))
        else:
            benchmark_values.append(benchmark_value)

    comparison_df = pd.DataFrame(
        {
            "Metrica": metrics_to_display,
            "Strategia": strategy_values,
            "Benchmark (SPY)": benchmark_values,
        }
    )
    comparison_df.set_index("Metrica", inplace=True)
    if comparison_df is not None:
        styled_comparison_df = comparison_df.copy()
    for col in styled_comparison_df.select_dtypes(include=np.number).columns:
        styled_comparison_df[col] = styled_comparison_df[col].round(2)

    return styled_comparison_df


def run_backtest(
    ticker: str,
    data: pd.DataFrame,
    strategy_name: str,
    params: dict,
    initial_capital: float,
    commission_percent: float,
    is_plot_wanted: bool = True,
) -> None:
    """Execute a backtest for a single ticker using the specified strategy and parameters.

    This function initializes and runs a backtest using the `backtesting.py` library.
    It logs the execution time, generates an interactive plot of the results,
    and stores the backtest statistics, plot, and trade list in the Streamlit
    session state.

    Args:
        ticker (str): The ticker symbol for which the backtest is being run.
            Used as a key to store results in the session state.
        data (pd.DataFrame): The OHLCV data for the asset as a pandas DataFrame.
        strategy_name (str): The display name of the strategy to be backtested.
            This name is used to look up the strategy class in `ss.all_strategies`.
        params (dict): A dictionary of parameters to be passed to the strategy's `init` method.
        initial_capital (float): The initial cash amount for the backtest.
        commission_percent (float): The commission rate per trade, expressed as a decimal
            (e.g., 0.001 for 0.1%).
        is_plot_wanted (bool, optional): If True, an interactive plot of the backtest
            results is generated and stored. Defaults to True.

    Returns:
        None: This function does not return any value. It modifies the Streamlit
              session state (`ss`) directly with the backtest results.

    Side Effects:
        - Populates `ss.bt_stats[ticker]` with the backtest statistics (a pandas Series).
        - Populates `ss.backtest_trade_list[ticker]` with a DataFrame of executed trades.
        - If `is_plot_wanted` is True, populates `ss.backtest_interactive_plot[ticker]`
          with a Bokeh plot object.
        - Calls `log_execution_data` to log the performance of the backtest and plot generation.
        - Displays an error message in the Streamlit UI via `st.error` if the backtest fails.

    """
    if data is None or data.empty:
        st.error("Impossibile eseguire il backtest: dati non disponibili.")
        return

    strat_class: type[CommonStrategy] = st.session_state.all_strategies[strategy_name]
    bt = Backtest(
        data,
        strat_class,
        cash=initial_capital,
        commission=commission_percent,
        exclusive_orders=True,  # Consente una sola posizione alla volta
    )

    try:
        start_time = time.perf_counter()
        ss.bt_stats[ticker] = bt.run(**params)
        end_time = time.perf_counter()
        pars_time_log = {"periods": len(data), "strategy": strat_class.DISPLAY_NAME}
        log_execution_data(start_time, end_time, action="Backtest", **pars_time_log)

        if is_plot_wanted:
            start_time = time.perf_counter()
            ss.backtest_interactive_plot[ticker] = bt.plot(
                filename="TO BE DELETED.html", resample=False, open_browser=False
            )
            if os.path.exists("TO BE DELETED.html"):
                os.remove("TO BE DELETED.html")
            end_time = time.perf_counter()
            log_execution_data(start_time, end_time, action="Backtest_plot", **pars_time_log)

        ss.backtest_trade_list[ticker] = ss.bt_stats[ticker]._trades

        return

    except Exception as e:
        st.error(
            f"Errore durante l'esecuzione del backtest: {e}. Controlla i parametri della strategia o la logica interna."
        )
        return


def get_benchmark_data(
    download_progress_placeholder: streamlit_obj,
    download_success_placeholder: streamlit_obj,
    successful_downloads_tickers: list,
    failed_downloads_tickers: list,
) -> pd.Series | None:
    """Download S&P500 data and run a Buy & Hold backtest on it to serve as a benchmark.

    This function handles the data download process and the subsequent backtest
    execution for the benchmark.

        download_progress_placeholder (streamlit.delta_generator.DeltaGenerator):
            A Streamlit placeholder object to display download progress messages.
        download_success_placeholder (streamlit.delta_generator.DeltaGenerator):
            A Streamlit placeholder object to display download success messages.
        successful_downloads_tickers (list): A list to append the ticker of
                                             successfully downloaded data.
        failed_downloads_tickers (list): A list to append the ticker of
                                         failed data downloads.

    Returns:
        pd.Series or None: Statistics of the Buy & Hold benchmark backtest
                           (output of `Backtest.run()`). Returns None if data
                           download or backtest execution fails.

    """
    benchmark_raw_data = None  # Raw benchmark data
    benchmark_stats = None  # Complete benchmark buy&hold statistics

    # Download benchmark data
    download_progress_placeholder.info(
        MESSAGES["display_texts"]["messages"]["downloading_benchmark"].format(
            SP500_TICKER=MESSAGES["general_settings"]["sp500_ticker"]
        )
    )
    benchmark_raw_data, status, msg = get_sp500_data(ss.start_date_wid, ss.end_date_wid, ss.data_interval_wid)
    download_progress_placeholder.empty()  # Remove blue progress box

    if status != "success":
        failed_downloads_tickers.append(MESSAGES["general_settings"]["sp500_ticker"])
        st.warning(MESSAGES["display_texts"]["messages"]["benchmark_data_not_available"])
        return benchmark_stats

    successful_downloads_tickers.append(MESSAGES["general_settings"]["sp500_ticker"])
    download_success_placeholder.success(
        MESSAGES["display_texts"]["messages"]["download_success_benchmark"].format(
            SP500_TICKER=MESSAGES["general_settings"]["sp500_ticker"]
        )
    )

    # Run a backtest with BuyAndHold strategy on the benchmark
    if benchmark_raw_data is not None and not benchmark_raw_data.empty:
        try:
            bt_benchmark = Backtest(
                benchmark_raw_data,
                BuyAndHoldStrategy,  # Use the BuyAndHold strategy
                cash=ss.initial_capital_wid,
                commission=ss.commission_percent_wid / 100,  # Apply commissions to benchmark too for parity
                exclusive_orders=True,
            )
            benchmark_stats = bt_benchmark.run()
        except Exception as e:
            st.warning(
                MESSAGES["display_texts"]["messages"]["error_calculating_benchmark_stats"].format(
                    SP500_TICKER=MESSAGES["general_settings"]["sp500_ticker"], e=e
                )
            )
            benchmark_stats = None
    return benchmark_stats


def start_backtest_process(
    backtest_infos_container: streamlit_obj,
    backtest_results_container: streamlit_obj,
) -> None:
    """Initiate and manage the full backtesting process.

    Orchestrate the entire backtesting workflow. Start by resetting relevant
    session state variables and clearing UI containers. Then, fetch benchmark
    data and iterate through each user-selected ticker. For each ticker,
    download data, run the backtest with the specified strategy and parameters,
    and optionally perform a Monte Carlo simulation. Display progress in the UI
    throughout the process.

    All inputs for the backtest (tickers, strategy, parameters, etc.) are
    retrieved from the Streamlit session state (`ss`).

    Args:
        backtest_infos_container (streamlit_obj): The Streamlit container
            designated for displaying informational messages, such as download
            and run progress.
        backtest_results_container (streamlit_obj): The Streamlit container
            where the final backtest results (stats, plots, etc.) will be
            rendered.

    Returns:
        None: This function modifies the Streamlit UI and session state directly.

    """
    # for name in session_state_names:
    #     if session_state_names[name][1]:
    #         ss[name] = session_state_names[name][0]

    # Reset the result relatedsession states
    reset_ss_values_for_results()
    backtest_infos_container.empty()
    backtest_results_container.empty()

    with backtest_infos_container:
        col_progress, col_success, col_failed = st.columns(3)
        if not ss.tickers:
            st.error(MESSAGES["display_texts"]["messages"]["enter_ticker_error"])
            return
        elif ss.bt_strategy_wid is None:
            st.error(MESSAGES["display_texts"]["messages"]["select_valid_strategy_error"])
            return

        # Placeholders for dynamic messages
        with col_progress:
            download_progress_placeholder = st.empty()
            download_success_placeholder = st.empty()
        with col_success:
            run_progress_placeholder = st.empty()
            run_success_placeholder = st.empty()
        with col_failed:
            download_fail_placeholder = st.empty()
            run_fail_placeholder = st.empty()  # For backtest/optimization success/failure messages

    benchmark_stats = get_benchmark_data(
        download_progress_placeholder,
        download_success_placeholder,
        ss.successful_downloads_tickers,
        ss.failed_downloads_tickers,
    )

    with backtest_infos_container:
        progress_bar = st.progress(0)

    with backtest_results_container:
        for i, ticker in enumerate(ss.tickers):
            process_single_ticker(
                ticker,
                i,
                benchmark_stats,
                download_progress_placeholder,
                download_success_placeholder,
                download_fail_placeholder,
                run_progress_placeholder,
                run_success_placeholder,
                run_fail_placeholder,
                ss.successful_downloads_tickers,
                ss.failed_downloads_tickers,
                ss.successful_runs_tickers,
                ss.failed_runs_tickers,
                progress_bar,
            )
    ss.backtest_results_generated = True


def process_single_ticker(
    ticker: str,
    i: int,
    benchmark_stats: pd.Series | None,
    download_progress_placeholder: streamlit_obj,
    download_success_placeholder: streamlit_obj,
    download_fail_placeholder: streamlit_obj,
    run_progress_placeholder: streamlit_obj,
    run_success_placeholder: streamlit_obj,
    run_fail_placeholder: streamlit_obj,
    successful_downloads_tickers: list,
    failed_downloads_tickers: list,
    successful_runs_tickers: list,
    failed_runs_tickers: list,
    progress_bar: streamlit_obj,
) -> None:
    """Process a single ticker through the entire backtesting workflow.

    This function orchestrates the download, backtest, and optional Monte Carlo
    simulation for a single financial instrument (ticker). It manages UI updates
    for progress, success, and failure messages, and stores all generated
    results (statistics, plots, trade lists) in the Streamlit session state.

    The workflow includes:
    1.  Downloading historical data for the ticker.
    2.  Running the backtest with user-defined strategy and parameters.
    3.  Comparing the backtest results against a pre-calculated benchmark.
    4.  If enabled, performing a Monte Carlo simulation on the trade returns.
    5.  Displaying all relevant results and plots in the Streamlit UI.

    Args:
        ticker (str): The ticker symbol of the asset to process.
        i (int): The zero-based index of the current ticker in the list of all tickers,
            used for progress bar calculation.
        benchmark_stats (pd.Series | None): The statistics from the benchmark
            (e.g., Buy & Hold on SPY) backtest. If None, no comparison is made.
        download_progress_placeholder (streamlit_obj): Streamlit placeholder for
            displaying data download progress messages.
        download_success_placeholder (streamlit_obj): Streamlit placeholder for
            displaying data download success messages.
        download_fail_placeholder (streamlit_obj): Streamlit placeholder for
            displaying data download failure messages.
        run_progress_placeholder (streamlit_obj): Streamlit placeholder for
            displaying backtest execution progress messages.
        run_success_placeholder (streamlit_obj): Streamlit placeholder for
            displaying backtest execution success messages.
        run_fail_placeholder (streamlit_obj): Streamlit placeholder for
            displaying backtest execution failure messages.
        successful_downloads_tickers (list): A list to be updated with tickers
            that were successfully downloaded.
        failed_downloads_tickers (list): A list to be updated with tickers
            for which download failed.
        successful_runs_tickers (list): A list to be updated with tickers
            that were successfully backtested.
        failed_runs_tickers (list): A list to be updated with tickers for which
            the backtest failed.
        progress_bar (streamlit_obj): The main Streamlit progress bar to update
            after processing the ticker.

    """
    download_progress_placeholder.info(
        MESSAGES["display_texts"]["messages"]["downloading_ticker"].format(
            ticker=ticker, current_idx=i + 1, total_tickers=len(ss.tickers)
        )
    )

    data, status, _ = download_data(ticker, ss.start_date_wid, ss.end_date_wid, ss.data_interval_wid)

    download_progress_placeholder.empty()  # Remove blue progress box

    _update_download_status(
        status,
        ticker,
        successful_downloads_tickers,
        failed_downloads_tickers,
        download_success_placeholder,
        download_fail_placeholder,
    )

    if data is not None:
        run_progress_placeholder.info(MESSAGES["display_texts"]["messages"]["execution_in_progress"] + ticker)
        try:
            run_backtest(
                ticker,
                data,
                ss.bt_strategy_wid,
                ss.bt_params,
                ss.initial_capital_wid,
                ss.commission_percent_wid / 100,
                True,
            )
            run_status = "success"
        except Exception:
            run_status = "fail"
        run_progress_placeholder.empty()
    else:
        run_status = "fail"

    _update_run_status(
        run_status,
        ticker,
        successful_runs_tickers,
        failed_runs_tickers,
        run_success_placeholder,
        run_fail_placeholder,
    )

    if ticker not in ss.bt_stats.keys() or ss.bt_stats[ticker] is None:
        return

    if benchmark_stats is not None:
        ss.backtest_comp_with_benchmark_df[ticker] = compare_with_benchmark(ss.bt_stats[ticker], benchmark_stats)

    progress_bar.progress((i + 1) / len(ss.tickers))

    if (
        ss.run_mc_wid
        and data is not None
        and run_status == "success"
        and ss.backtest_trade_list[ticker] is not None
        and ss.bt_stats[ticker] is not None
    ):
        run_montecarlo(
            ticker=ticker,
            trades=(ss.backtest_trade_list[ticker]["ReturnPct"]),
            original_stats=ss.bt_stats[ticker],
            initial_capital=ss.initial_capital_wid,
            benchmark=benchmark_stats,
            sampling_method=ss.mc_sampling_method_wid,
            sim_length=ss.mc_sim_length_wid,
            num_sims=ss.mc_n_sims_wid,
        )
        if ticker in ss.backtest_mc_percentiles and ticker in ss.backtest_mc_probs_benchmark:
            # Grafico 1: Equity Lines Simulate
            show_montecarlo_equity_lines(
                ticker,
                ss.mc_pars,
                ss.matrice_equity_lines_simulati,
                ss.orig_current_equity_path,
                max_n_shown_lines=1000,
            )

            # Grafico 2: Istogramma della Distribuzione dei Drawdown Massimi
            show_montecarlo_histogram(
                ticker=ticker,
                metric=ss.mc_metrics_data["Max. Drawdown [%]"],
                title="Distribution of Simulated Max Drawdowns",
                x_label="Max. Drawdown [%]",
                perc_label="VaR Drawdown [%]",
                percentile=5,
                nickname="backtest_mc_var_plot",
            )

            # Grafico 3: Istogramma della Distribuzione del Capitale Finale
            show_montecarlo_histogram(
                ticker=ticker,
                metric=ss.mc_metrics_data["Return [%]"],
                title="Distribution of Return [%]",
                x_label="Return [%]",
                perc_label="Return [%]",
                percentile=5,
                nickname="backtest_mc_returns_plot",
            )

        else:
            st.warning(
                "The Monte Carlo statistics aren't available as the simulation returned "
                "None instead of dataframes. Check the outcome of the simulation"
            )


def _update_download_status(
    status: str,
    ticker: str,
    successful_downloads_tickers: list,
    failed_downloads_tickers: list,
    download_success_placeholder: streamlit_obj,
    download_fail_placeholder: streamlit_obj,
) -> None:
    if status == "success":
        successful_downloads_tickers.append(ticker)
        download_success_placeholder.success(
            MESSAGES["display_texts"]["messages"]["download_success_ticker"] + ", ".join(successful_downloads_tickers)
        )
    else:
        failed_downloads_tickers.append(ticker)
        download_fail_placeholder.error(
            MESSAGES["display_texts"]["messages"]["download_failed_ticker"] + ", ".join(failed_downloads_tickers)
        )


def _update_run_status(
    run_status: str,
    ticker: str,
    successful_runs_tickers: list,
    failed_runs_tickers: list,
    run_success_placeholder: streamlit_obj,
    run_fail_placeholder: streamlit_obj,
) -> None:
    if run_status != "success":
        failed_runs_tickers.append(ticker)
        run_fail_placeholder.error(
            MESSAGES["display_texts"]["messages"]["execution_failed"] + ", ".join(failed_runs_tickers)
        )
        return

    successful_runs_tickers.append(ticker)
    run_success_placeholder.success(
        MESSAGES["display_texts"]["messages"]["execution_completed"] + ", ".join(successful_runs_tickers)
    )
