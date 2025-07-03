import time

import numpy as np
import pandas as pd

# import quantstats as qs
import streamlit as st

from config import MESSAGES, ss
from excel_exporter import log_execution_data


def run_montecarlo(
    ticker: str,
    trades: list[float],
    original_stats: pd.Series,
    initial_capital: float,
    benchmark: pd.Series,
    sampling_method: str,
    sim_length: int,
    num_sims: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str : int | str],
    np.ndarray,
    dict[str : np.array],
    list[float],
]:
    """Execute a Monte Carlo simulation for a trading strategy based on historical percentage trade returns.

    This function simulates various equity paths by resampling or permuting
    historical trade returns and calculates key performance metrics for each
    simulated path. It also provides percentile analysis and comparisons
    against a benchmark.

    Args:
        trades (list[float]): A list of historical percentage returns for each trade.
                              These are the raw percentage returns (e.g., 0.05 for 5%).
        original_stats (pd.Series): The statistics of the original backtest run
                                    (e.g., from `Backtest.run()`), used for calculating
                                    the original equity path and other initial metrics.
        initial_capital (float): The starting capital for each simulation.
        benchmark (pd.Series): Statistics of the benchmark's Buy & Hold performance
                               for comparison. Can be None if no benchmark is provided.
        sampling_method (str): The method used for sampling trade returns.
                               Expected values: 'resampling_con_reimmissione' (resampling with replacement)
                               or 'permutazione' (permutation).
        sim_length (int): The number of trades to simulate in each Monte Carlo path.
                          Typically, this is the number of trades in the original backtest.
        num_sims (int): The total number of Monte Carlo simulations to run.

    Returns:
        tuple: A tuple containing:
               - df_percentiles (pd.DataFrame): DataFrame showing percentiles of key metrics
                                                (e.g., Equity Final, Max Drawdown).
               - df_compare_benchmark (pd.DataFrame): DataFrame comparing Monte Carlo results
                                                       against the benchmark.
               - mc_pars (dict): Dictionary of Monte Carlo simulation parameters used.
               - simulated_equity_lines (np.ndarray): 2D NumPy array where each row is a
                                                      simulated equity path.
               - metrics_data (dict[str : np.array]): A dictionary where keys are metric names
                                                      and values are NumPy arrays of those
                                                      metrics across all simulations.
               - original_equity_path (list[float]): The calculated original equity curve.

    """
    if not trades:
        st.warning("Errore: La lista dei rendimenti percentuali dei trade non può essere vuota.")
        return None, None, None, None, None, None

    start_time = time.perf_counter()

    trade_returns_array = np.array(trades)
    num_trades = len(trade_returns_array)

    if sim_length < 0:
        st.warning("Errore: La lunghezza della simulazione deve essere maggiore di zero.")
        return None, None
    elif sampling_method == "resampling_con_reimmissione" and sim_length == 0:
        sim_length = num_trades
    if sampling_method == "permutazione":
        st.info(
            f"Info Monte Carlo: La lunghezza della simulazione non può essere maggiore "
            f"del numero di trade storici ({num_trades}) con il metodo 'permutazione'. "
            f"Non può essere neanche uguale poiché il capitale finale sarebbe lo stesso per "
            f"tutte le simulazioni. "
            f"Parametro impostato a {int(num_trades * 0.9)}"
        )
        sim_length = int(num_trades * 0.9)

    (
        all_total_returns,
        all_max_drawdowns,
        all_sharpe_ratios,
        all_sortino_ratios,
        all_sqn_values,
        matrice_equity_lines_simulati,
    ) = _run_montecarlo_simulations(
        trade_returns_array,
        initial_capital,
        sampling_method,
        sim_length,
        num_sims,
    )

    orig_current_equity_path = np.insert(initial_capital * (trade_returns_array + 1).cumprod(), 0, initial_capital)

    ss.mc_metrics_data = {
        "Return [%]": np.array(all_total_returns) * 100,
        "Max. Drawdown [%]": np.array(all_max_drawdowns),
        "Sharpe Ratio": np.array(all_sharpe_ratios),
        "Sortino Ratio": np.array(all_sortino_ratios),
        "SQN": np.array(all_sqn_values),
    }

    ss.backtest_mc_percentiles[ticker] = _build_metrics_percentiles_df(ss.mc_metrics_data, original_stats)
    ss.backtest_mc_probs_benchmark[ticker] = _build_benchmark_prob_df(ss.mc_metrics_data, benchmark)

    ss.mc_pars = {
        "# Simulations": num_sims,
        "Length Simulations": sim_length,
        "Sampling Method": sampling_method,
    }

    end_time = time.perf_counter()
    log_execution_data(start_time, end_time, action="Monte Carlo", **ss.mc_pars)

    ss.matrice_equity_lines_simulati = matrice_equity_lines_simulati
    ss.orig_current_equity_path = orig_current_equity_path


def _run_montecarlo_simulations(
    trade_returns_array: np.ndarray,
    initial_capital: float,
    sampling_method: str,
    sim_length: int,
    num_sims: int,
) -> tuple[list, list, list, list, list, np.ndarray]:
    """Highly optimized, vectorized Monte Carlo simulation for trade returns.

    Generates all sampled returns in a single operation and computes all metrics using numpy vectorization.

    """
    if sampling_method == "resampling_con_reimmissione":
        sampled_returns = np.random.choice(trade_returns_array, size=(num_sims, sim_length), replace=True)
    elif sampling_method == "permutazione":
        # For permutation, each row is a random permutation (without replacement) of the original array
        sampled_returns = np.array([np.random.permutation(trade_returns_array)[:sim_length] for _ in range(num_sims)])
    else:
        raise ValueError(
            "Errore: Metodo di campionamento non valido. Usa 'resampling_con_reimmissione' o 'permutazione'."
        )

    # Compute equity lines: shape (num_sims, sim_length+1)
    equity_lines = np.cumprod(sampled_returns + 1, axis=1) * initial_capital
    equity_lines = np.concatenate([np.full((num_sims, 1), initial_capital), equity_lines], axis=1)

    # Vectorized total returns
    all_total_returns = (equity_lines[:, -1] - initial_capital) / initial_capital

    # Vectorized max drawdown
    running_max = np.maximum.accumulate(equity_lines, axis=1)
    drawdowns = equity_lines / running_max - 1
    all_max_drawdowns = np.min(drawdowns, axis=1) * 100

    # Vectorized Sharpe ratio
    mean_returns = np.mean(sampled_returns, axis=1)
    std_returns = np.std(sampled_returns, axis=1)
    all_sharpe_ratios = np.where(std_returns == 0, 0.0, mean_returns / std_returns)

    # Vectorized Sortino ratio
    downside = np.sqrt(np.sum(np.where(sampled_returns < 0, sampled_returns**2, 0), axis=1) / sampled_returns.shape[1])
    all_sortino_ratios = np.where(downside == 0, 0.0, mean_returns / downside)

    # Vectorized SQN
    expectancy = mean_returns
    std_expectancy = std_returns
    all_sqn_values = np.where(
        std_expectancy == 0, 0.0, (expectancy / std_expectancy) * np.sqrt(sampled_returns.shape[1])
    )

    return (
        all_total_returns.tolist(),
        all_max_drawdowns.tolist(),
        all_sharpe_ratios.tolist(),
        all_sortino_ratios.tolist(),
        all_sqn_values.tolist(),
        equity_lines,
    )


def _build_metrics_percentiles_df(metrics_data: dict, original_stats: pd.Series) -> pd.DataFrame:
    percentiles = [50, 80, 90, 95, 99]
    metrics_df: pd.DataFrame = pd.DataFrame(index=["Originale"] + [f"Livello {p}%" for p in percentiles])

    for key in metrics_data:
        metrics_df.loc["Originale", key] = original_stats[key]

    for metric_name in metrics_data:
        for p in percentiles:
            metrics_df.loc[f"Livello {p}%", metric_name] = np.percentile(metrics_data[metric_name], 100 - p)
    return metrics_df


def _build_benchmark_prob_df(metrics_data: dict, benchmark: pd.Series) -> pd.DataFrame:
    prob_data = {
        key: {
            "Benchmark": f"{benchmark[key]:.2f}",
            "Prob. >= Benchmark": f"{np.mean(metrics_data[key] >= benchmark[key]) * 100:.2f}%",
        }
        for key in metrics_data
    }
    return pd.DataFrame.from_dict(prob_data, orient="index")


def run_montecarlos_for_best_combs(
    initial_capital: float,
    objective_function_selection: str,
    run_mc: bool,
    promoted_combinations: int,
    mc_sampling_method: str,
    num_sims: int,
    sims_length: int,
    benchmark_stats: pd.Series,
    ticker: str,
    all_comb_data: pd.DataFrame,
) -> pd.DataFrame | None:
    """Execute Monte Carlo simulations for a specified number of "best" combinations.

    This function iterates through the top-performing strategy combinations
    (identified by `objective_function_selection` and `promoted_combinations`),
    extracts their trade returns, runs a Monte Carlo simulation for each,
    and collects the relevant statistics (original and a specified percentile like 95th)
    into a combined DataFrame.

    Args:
        initial_capital (float): The initial capital for the Monte Carlo simulations.
        commission_percent (float): Commission percentage to apply (though typically
                                    its impact on trade returns is handled during backtesting).
        objective_function_selection (str): The name of the objective function used
                                            to rank the combinations (e.g., 'Sharpe Ratio').
        run_mc (bool): A flag indicating whether to proceed with Monte Carlo simulations.
                       If False, the function returns None.
        promoted_combinations (int): The number of top-ranked combinations for which to run MC.
        mc_sampling_method (str): The sampling method for Monte Carlo ('resampling_con_reimmissione' or 'permutazione').
        num_sims (int): The number of simulations to run for each combination.
        sims_length (int): The length (number of trades) for each simulation.
        strat_class: The strategy class used for the backtest (currently not directly used within this function's
                     Monte Carlo logic but might be a remnant from previous `run_backtest` calls in the loop).
        benchmark_stats: Benchmark statistics for comparison within Monte Carlo.
        ticker (str): The ticker symbol associated with the data.
        data (pd.DataFrame): Historical market data (OHLCV) on which the strategies were backtested.
                             (Currently not directly used for MC, but part of context).
        all_comb_data (pd.DataFrame): DataFrame containing all optimization combinations
                                      and their backtest results, including a column named
                                      'Trade_returns' which is a list of trade returns for each combination.

    Returns:
        pd.DataFrame or None: A DataFrame containing the parameters of the best combinations
                              along with their Monte Carlo statistics (original and a specified percentile).
                              Returns None if `run_mc` is False, if `all_comb_data` is empty,
                              or if the 'Trade_returns' column is missing or empty for a combination.

    """
    combs_with_mc_stats = None  # Inizializziamo il df per il MC nel caso poi non venga riempito
    if run_mc:
        obj_column_name: str = objective_function_selection  # Ottengo nome dell'obiettivo
        best_combs = all_comb_data.sort_values(
            by=obj_column_name, ascending=not MESSAGES["optimization_settings"]["objectives"][obj_column_name]
        )  # Ordino secondo l'obiettivo
        best_combs = best_combs.head(promoted_combinations)  # Prendo solo le prime combinazioni
        combs_with_mc_stats = pd.DataFrame()
        mc_run_placeholder = st.empty()
        for n in range(
            min(promoted_combinations, len(best_combs))
        ):  # itera per ogni combinazione fino ad arrivare al numero stabilito
            mc_run_placeholder.info(f"Montecarlo running for {ticker} ({n + 1}/{promoted_combinations})")
            current_comb = best_combs.iloc[n]
            strategy_params = dict(
                zip(best_combs.index.names, best_combs.index[n], strict=False)
            )  # Prendo una combinazione alla volta
            trades_df = all_comb_data["Trade_returns"].iloc[n]
            if trades_df is not None:  # Se non ci sono trades, il MC non si può fare
                all_mc_statistics = run_montecarlo(
                    trades=list(trades_df),  # Faccio il Monte Carlo della combinazione
                    original_stats=current_comb,
                    initial_capital=initial_capital,
                    benchmark=benchmark_stats,
                    sampling_method=mc_sampling_method,
                    sim_length=sims_length,
                    num_sims=num_sims,
                )

                # Dalle statistiche Monte Carlo mi prendo solo quelle reali della strategia e
                # quelle al 95 percentile
                mc_req_stats = all_mc_statistics[0][
                    all_mc_statistics[0].index.astype(str).str.contains("Originale")
                    | all_mc_statistics[0].index.astype(str).str.contains("95")
                ]
                suffix_map = {"Originale": "O", "Livello 95%": "p95"}
                mc_req_stats = pd.DataFrame(
                    {  # Trasformo due righe in un'unica
                        f"{idx[1]}-{suffix_map[idx[0]]}": [val] for idx, val in mc_req_stats.stack().items()
                    },
                    index=[0],
                )
            else:
                mc_req_stats = None
                # Unisco la riga dei parametri a quella delle statistiche
            comb_with_mc = pd.concat([pd.DataFrame([strategy_params]), mc_req_stats], axis=1)
            combs_with_mc_stats = pd.concat([combs_with_mc_stats, comb_with_mc], ignore_index=True)
        mc_run_placeholder.empty()
    return combs_with_mc_stats
