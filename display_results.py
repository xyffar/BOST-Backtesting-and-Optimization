# display_results.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_bokeh
from sambo.plot import plot_convergence, plot_evaluations, plot_objective

from config import MESSAGES, ss
from utils import add_benchmark_comparison, list_varying_params


def display_results() -> None:
    """Display backtest or optimization results in separate tabs for each ticker.

    This function serves as the main entry point for presenting all analysis results
    within the Streamlit application, dynamically adjusting the display based on
    whether the application is in backtesting or optimization mode.

    Args:
        ticker_results (dict): Dictionary where keys are ticker symbols and values are
                               the results for that ticker.
                               In backtest mode, values are `(backtest_stats, equity_curve, trades_df, benchmark_comparison_df)`.
                               In optimization mode, values are `(all_comb_data, heatmap_plots, sambo_plots, MC_data)`.
        all_mc_statistics (dict[str : tuple[pd.DataFrame, pd.DataFrame, dict[str : int | str], np.ndarray, pd.DataFrame, pd.Series]], optional):
            Dictionary where keys are ticker symbols and values are tuples containing
            (df_percentiles, df_compare_benchmark, mc_pars, simulated_equity_lines, metrics_data, original_equity_path)
            from Monte Carlo simulations. Defaults to None. This argument is primarily used in backtesting mode.
        benchmark_comparison (float | int, optional):
            The total percentage return of the benchmark asset (e.g., S&P 500)
            for direct comparison. Defaults to None.
        is_optimization_mode (bool, optional):
            A boolean flag indicating if the application is currently in optimization mode.
            Defaults to False, implying backtest mode.
        obj_func (str, optional):
            The name of the objective function used during optimization.
            Relevant only if `is_optimization_mode` is True. Defaults to None.

    Returns:
        None: This function directly renders content to the Streamlit UI.

    """
    if not ss.backtest_results_generated and not ss.opt_results_generated:
        st.warning(MESSAGES["display_texts"]["messages"]["no_results_to_show"])
        return

    tickers_with_results = tuple((*ss.bt_stats.keys(), *ss.opt_combs_ranking.keys()))

    tabs = st.tabs(tickers_with_results)

    for i, ticker in enumerate(tickers_with_results):
        with tabs[i]:
            st.markdown(f"""### {MESSAGES["display_texts"]["messages"]["results_for_ticker"].format(ticker=ticker)}""")

            if ss.mode == "optimization":
                _display_optimization_results(
                    ticker_results[ticker],
                    benchmark_comparison,
                    obj_func,
                )
            elif ss.mode == "backtest":
                _display_backtest_results(ticker)
            else:
                st.error("The mode is neither backtesting or optimization!")


def _display_optimization_results(
    ticker_result: tuple,
    benchmark_comparison: float | int | None,
    obj_func: str | None,
) -> None:
    (all_comb_data, heatmap_plots, sambo_plots, mc_data) = ticker_result

    if all_comb_data is None:
        return

    all_comb_data = show_results(benchmark_comparison, obj_func, all_comb_data)

    if heatmap_plots:
        show_heatmaps(heatmap_plots)
    elif sambo_plots:
        show_sambo_plots(all_comb_data, sambo_plots)
    else:
        st.info(MESSAGES["display_texts"]["messages"]["heatmap_not_available"])

    show_mc_results(mc_data)


def show_results(
    benchmark_comparison: float | int | None,
    obj_func: str | None,
    all_comb_data: pd.DataFrame,
) -> pd.DataFrame:
    """Display the optimization results table with benchmark comparison.

    Args:
        benchmark_comparison (float | int | None): The benchmark value for comparison.
        obj_func (str | None): The objective function used for optimization.
        all_comb_data (pd.DataFrame): DataFrame containing all optimization combinations.

    Returns:
        pd.DataFrame: The DataFrame of all combinations (possibly filtered or modified).

    """
    st.markdown(f"""#### {MESSAGES["display_texts"]["messages"]["table_combinations_header"]}""")
    all_comb_data = all_comb_data[list(MESSAGES["optimization_settings"]["objectives"])]
    cambinations_ranking_with_benchmark = add_benchmark_comparison(all_comb_data, benchmark_comparison, obj_func)
    st.dataframe(cambinations_ranking_with_benchmark, use_container_width=True)
    return all_comb_data


def show_mc_results(mc_data: pd.DataFrame | None) -> None:
    """Display Monte Carlo analysis results if available.

    Args:
        mc_data (pd.DataFrame | None): Monte Carlo results data to display.

    Returns:
        None

    """
    if mc_data is not None:
        st.markdown("---")
        st.subheader("Analisi Monte Carlo")
        st.dataframe(mc_data)


def show_sambo_plots(all_comb_data: pd.DataFrame, sambo_plots: object) -> None:
    """Display SAMBO optimization plots for the given parameter combinations and plot data.

    Args:
        all_comb_data (pd.DataFrame): DataFrame containing all optimization combinations.
        sambo_plots (object): SAMBO plot data object.

    Returns:
        None

    """
    st.markdown(f"#### {MESSAGES['display_texts']['messages']['sambo_plots']}")
    varying_params: list[tuple[int, str]] = list_varying_params(all_comb_data.index.to_frame())
    varying_param_names = [p[1] for p in varying_params]
    varying_param_idx = [p[0] for p in varying_params]
    cols_sambos = st.columns(3, border=True)
    with cols_sambos[0]:
        st.pyplot(
            plot_objective(
                sambo_plots,
                names=varying_param_names,
                plot_dims=varying_param_idx,
                estimator="et",
            ),
            use_container_width=True,
        )
    with cols_sambos[1]:
        st.pyplot(
            plot_evaluations(sambo_plots, names=varying_param_names, plot_dims=varying_param_idx),
            use_container_width=True,
        )
    with cols_sambos[2]:
        st.pyplot(plot_convergence(sambo_plots), use_container_width=True)


def show_heatmaps(heatmap_plots: list) -> None:
    """Display optimization heatmap plots in a grid layout.

    Args:
        heatmap_plots (list): List of matplotlib Figure objects to display.

    Returns:
        None

    """
    st.markdown(f"#### {MESSAGES['display_texts']['messages']['optimization_heatmap']}")
    num_columns = 3
    cols = st.columns(num_columns)
    for i, fig in enumerate(heatmap_plots):
        with cols[i % num_columns]:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def _display_backtest_results(ticker: str) -> None:
    if ss.backtest_comp_with_benchmark_df[ticker] is not None:
        st.markdown(f"#### {MESSAGES['display_texts']['messages']['backtest_stats_benchmark_comparison']}")
        st.dataframe(ss.backtest_comp_with_benchmark_df[ticker], use_container_width=True)
    else:
        st.warning(MESSAGES["display_texts"]["messages"]["benchmark_comparison_not_available"])

    st.markdown(f"#### {MESSAGES['display_texts']['messages']['interactive_backtest_chart']}")
    streamlit_bokeh.streamlit_bokeh(
        ss.backtest_interactive_plot[ticker], use_container_width=False, theme="light_minimal"
    )

    st.markdown(f"#### {MESSAGES['display_texts']['messages']['list_of_trades']}")
    if ss.backtest_trade_list[ticker] is not None and not ss.backtest_trade_list[ticker].empty:
        st.dataframe(ss.backtest_trade_list[ticker], use_container_width=True)
    else:
        st.info(MESSAGES["display_texts"]["messages"]["no_trades_executed"])

    if ticker in ss.backtest_mc_percentiles.keys() and ss.backtest_mc_percentiles[ticker] is not None:
        show_montecarlo_results(ticker)


def show_montecarlo_results(ticker: str) -> None:
    """Display detailed Monte Carlo simulation results for a given ticker.

    This includes percentile dataframes, benchmark comparisons, and various
    visualizations like simulated equity lines and histograms of key metrics.

    Args:
        all_mc_statistics (dict): Dictionary containing all Monte Carlo
                                  statistics, keyed by ticker symbol. The value
                                  for each ticker is expected to be a tuple
                                  containing:
                                  - df_percentiles (pd.DataFrame): Percentile statistics.
                                  - df_compare_benchmark (pd.DataFrame): Benchmark comparison data.
                                  - mc_pars (dict): Monte Carlo simulation parameters.
                                  - simulated_equity_lines (np.ndarray): Array of simulated equity paths.
                                  - metrics_data (pd.DataFrame): Metrics from Monte Carlo simulations.
                                  - original_equity_path (pd.Series): The original equity curve.
        ticker (str): The ticker symbol for which to display Monte Carlo results.

    Returns:
        None: This function directly renders content to the Streamlit UI.

    """
    if ss.backtest_mc_percentiles[ticker] is not None and ss.backtest_mc_probs_benchmark[ticker] is not None:
        st.markdown("---")
        st.subheader("Monte Carlo Analysis")
        st.dataframe(
            ss.backtest_mc_percentiles[ticker].style.format(
                {s: "{:.2f}" for s in ss.backtest_mc_percentiles[ticker].columns}
            )
        )
        st.dataframe(ss.backtest_mc_probs_benchmark[ticker])
        cols = st.columns(3)
        with cols[0]:
            st.pyplot(ss.backtest_mc_equity_lines_plot[ticker])
        with cols[1]:
            st.pyplot(ss.backtest_mc_var_plot[ticker])
        with cols[2]:
            st.pyplot(ss.backtest_mc_returns_plot[ticker])


def show_montecarlo_histogram(
    ticker: str,
    metric: np.ndarray | pd.Series,
    title: str,
    x_label: str,
    perc_label: str,
    percentile: int | float,
    nickname: str,
) -> None:
    """Generate and displays a histogram for a given metric from Monte Carlo simulations.

    The histogram includes a vertical line indicating a specified percentile
    (e.g., VaR for drawdown or return percentile).

    Args:
        metric (pd.Series or np.array): The data series or array for which to
                                        generate the histogram.
        title (str): The title of the histogram plot.
        x_label (str): The label for the x-axis.
        perc_label (str): The label for the percentile line in the legend.
        percentile (int or float): The percentile to mark on the histogram (e.g., 5 for 5th percentile).

    Returns:
        None: This function directly renders the plot to the Streamlit UI.

    """
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(metric, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_title(title)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Frequency")
    ax2.axvline(
        np.percentile(metric, 5),
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"{perc_label} al {100 - percentile}%: {np.percentile(metric, percentile):.2f}",
    )
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    ss[nickname][ticker] = fig2
    # st.pyplot(fig2)


def show_montecarlo_equity_lines(
    ticker: str,
    mc_pars: dict,
    simulated_equity_lines: np.ndarray,
    original_equity_path: np.ndarray | pd.Series,
    max_n_shown_lines: int = 200,
) -> None:
    """Plot simulated Monte Carlo equity lines along with the original equity path.

    Highlights the paths with the maximum and minimum final capital.

    Args:
        mc_pars (dict): Dictionary containing Monte Carlo simulation parameters,
                        including '# Simulations'.
        simulated_equity_lines (np.ndarray): A 2D NumPy array where each row
                                             represents a simulated equity path.
        original_equity_path (pd.Series or np.array): The actual historical
                                                      equity curve.
        max_n_shown_lines (int, optional): The maximum number of simulated
                                           equity lines to display in the plot
                                           to prevent overcrowding. Defaults to 200.

    Returns:
        None: This function directly renders the plot to the Streamlit UI.

    """
    num_sims = mc_pars["# Simulations"]
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    # Plot di un sottoinsieme di linee simulate (per la massa)
    num_percorsi_da_mostrare = min(max_n_shown_lines, num_sims)
    for i in range(num_percorsi_da_mostrare):
        ax1.plot(simulated_equity_lines[i, :], lw=0.7, alpha=0.6)  # Colore più tenue per lo sfondo
        # Identifica la simulazione con il maggior capitale finale
    idx_max_equity = np.argmax(simulated_equity_lines[:, -1])
    max_equity_line = simulated_equity_lines[idx_max_equity, :]
    # Identifica la simulazione con il minor capitale finale
    idx_min_equity = np.argmin(simulated_equity_lines[:, -1])
    min_equity_line = simulated_equity_lines[idx_min_equity, :]
    # Plot delle linee specifiche sopra le altre
    ax1.plot(original_equity_path, color="black", linewidth=3, label="Original")
    ax1.plot(max_equity_line, color="green", linewidth=2.5, label="Max Final Capital")
    ax1.plot(min_equity_line, color="red", linewidth=2.5, label="Min Final Capital")

    ax1.set_title(f"Simulation of Monte Carlo Equity Lines ({num_sims} paths)")
    ax1.set_xlabel("Trade")
    ax1.set_ylabel("Capital")
    ax1.grid(True)
    ax1.legend()  # Aggiungi la legenda per le linee specifiche
    plt.tight_layout()
    ss.backtest_mc_equity_lines_plot[ticker] = fig1
    # st.pyplot(fig1)
