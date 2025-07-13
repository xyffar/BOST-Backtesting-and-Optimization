import time

import numpy as np
import pandas as pd

# import quantstats as qs
import streamlit as st

from config import MESSAGES, ss
from excel_exporter import log_execution_data


def _adjust_simulation_length(sampling_method: str, sim_length: int, num_trades: int) -> int:
    """Adjust the simulation length based on the sampling method and historical data.

    Args:
        sampling_method (str): The chosen sampling method.
        sim_length (int): The user-defined simulation length.
        num_trades (int): The number of trades in the original backtest.

    Returns:
        int: The adjusted simulation length to be used in the simulation.

    """
    messages = MESSAGES.get("display_texts", {}).get("messages", {})
    if sim_length < 0:
        st.warning(messages.get("mc_negative_sim_length_warning", "Simulation length must be non-negative."))
        return num_trades

    if sampling_method == "resampling_con_reimmissione":
        # If 0, use the original number of trades. Otherwise, use the specified length.
        return sim_length if sim_length > 0 else num_trades

    if sampling_method == "permutazione":
        # For permutation, length cannot be >= num_trades, as final equity would be identical.
        # A common practice is to use a slightly smaller length if the user's choice is invalid.
        if sim_length >= num_trades:
            adjusted_length = int(num_trades * 0.9)
            st.info(
                messages.get(
                    "mc_permutation_length_info",
                    "For 'permutation' sampling, simulation length cannot be >= number of historical trades ({num_trades}). "
                    "Adjusting length to {adjusted_length}.",
                ).format(num_trades=num_trades, adjusted_length=adjusted_length)
            )
            return adjusted_length
        return sim_length

    return sim_length


def run_montecarlo(
    ticker: str,
    trades: pd.Series,
    original_stats: pd.Series,
    initial_capital: float,
    benchmark: pd.Series | None,
    sampling_method: str,
    sim_length: int,
    num_sims: int,
) -> None:
    """Orchestrate and execute the Monte Carlo simulation workflow.

    This function validates inputs, runs the simulations, processes the results
    into statistical dataframes, logs the execution, and stores all generated
    data and plots in the Streamlit session state for later display.

    Args:
        ticker (str): The ticker symbol for which the simulation is run.
        trades (pd.Series): A Series of percentage returns for each trade.
        original_stats (pd.Series): The statistics from the original backtest run.
        initial_capital (float): The starting capital for the simulations.
        benchmark (pd.Series | None): The statistics from the benchmark backtest.
        sampling_method (str): The method for sampling trades ('permutazione' or 'resampling_con_reimmissione').
        sim_length (int): The desired number of trades in each simulated path.
        num_sims (int): The total number of simulations to run.

    """
    # 1. --- Input Validation ---
    messages = MESSAGES.get("display_texts", {}).get("messages", {})
    if trades is None or trades.empty:
        st.warning(
            messages.get(
                "mc_empty_trades_error", "Monte Carlo analysis cannot be run: The list of trade returns is empty."
            )
        )
        return

    start_time = time.perf_counter()

    # 2. --- Prepare Simulation Parameters ---
    num_trades = len(trades)
    adjusted_sim_length = _adjust_simulation_length(sampling_method, sim_length, num_trades)
    trade_returns_array = trades.to_numpy()

    # 3. --- Run Simulations ---
    sim_results = _run_montecarlo_simulations(
        trade_returns_array, initial_capital, sampling_method, adjusted_sim_length, num_sims
    )

    if sim_results is None:
        st.warning(f"Monte Carlo simulation for {ticker} did not produce results.")
        return

    all_total_returns, all_max_drawdowns, all_sharpe_ratios, all_sortino_ratios, all_sqn_values, equity_lines = (
        sim_results
    )

    # 4. --- Process and Store Results ---
    ss.mc_metrics_data = {
        "Return [%]": np.array(all_total_returns) * 100,
        "Max. Drawdown [%]": np.array(all_max_drawdowns),
        "Sharpe Ratio": np.array(all_sharpe_ratios),
        "Sortino Ratio": np.array(all_sortino_ratios),
        "SQN": np.array(all_sqn_values),
    }

    ss.backtest_mc_percentiles[ticker] = _build_metrics_percentiles_df(ss.mc_metrics_data, original_stats)
    if benchmark is not None:
        ss.backtest_mc_probs_benchmark[ticker] = _build_benchmark_prob_df(ss.mc_metrics_data, benchmark)

    # 5. --- Logging ---
    ss.mc_pars = {
        "# Simulations": num_sims,
        "Length Simulations": adjusted_sim_length,
        "Sampling Method": sampling_method,
    }
    end_time = time.perf_counter()
    log_execution_data(start_time, end_time, action="Monte Carlo", **ss.mc_pars)

    # 6. --- Store Artifacts for Plotting ---
    ss.matrice_equity_lines_simulati = equity_lines
    ss.orig_current_equity_path = np.insert(initial_capital * (trades + 1).cumprod(), 0, initial_capital)


def _run_montecarlo_simulations(
    trade_returns_array: np.ndarray,
    initial_capital: float,
    sampling_method: str,
    sim_length: int,
    num_sims: int,
) -> tuple[list, list, list, list, list, np.ndarray]:
    """Run a highly optimized, vectorized Monte Carlo simulation.

    This function generates all simulated trade paths in a single vectorized
    operation, then computes all performance metrics (e.g., return, drawdown,
    Sharpe ratio) using numpy vectorization for maximum efficiency.

    """
    # --- 1. Generate Sampled Trade Returns ---
    if sampling_method == "permutazione":
        # Permutation: create `num_sims` random permutations of the original trades.
        # This is vectorized by tiling the original array and permuting each row.
        rng = np.random.default_rng()
        tiled_returns = np.tile(trade_returns_array, (num_sims, 1))
        permuted_returns = rng.permuted(tiled_returns, axis=1)
        sampled_returns = permuted_returns[:, :sim_length]
    elif sampling_method == "resampling_con_reimmissione":
        # Bootstrap resampling: sample with replacement.
        sampled_returns = np.random.choice(trade_returns_array, size=(num_sims, sim_length), replace=True)
    else:
        messages = MESSAGES.get("display_texts", {}).get("messages", {})
        raise ValueError(
            messages.get("mc_invalid_sampling_method", "Invalid sampling method: '{method}'.").format(
                method=sampling_method
            )
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
    """Costruisce un DataFrame con le statistiche percentili delle metriche Monte Carlo.

    Confronta le statistiche del backtest originale con vari livelli di percentile
    (es. 50°, 80°, 95°) derivati dai risultati della simulazione Monte Carlo.

    Args:
        metrics_data (dict): Un dizionario dove le chiavi sono i nomi delle metriche
                             e i valori sono array numpy con i risultati della simulazione.
        original_stats (pd.Series): Le statistiche del backtest originale.

    Returns:
        pd.DataFrame: Un DataFrame con le metriche come colonne e i livelli
                      originale/percentile come indice.

    """
    percentiles_to_calc = [50, 80, 90, 95, 99]
    index_names = ["Originale"] + [f"Livello {p}%" for p in percentiles_to_calc]

    data_for_df = {}
    for metric_name, sim_results in metrics_data.items():
        original_value = original_stats.get(metric_name, np.nan)
        # Per un livello di confidenza del 95%, si usa il 5° percentile (100 - 95).
        percentile_values = [np.percentile(sim_results, 100 - p) for p in percentiles_to_calc]
        data_for_df[metric_name] = [original_value, *percentile_values]

    return pd.DataFrame(data_for_df, index=index_names)


def _build_benchmark_prob_df(metrics_data: dict, benchmark: pd.Series) -> pd.DataFrame:
    """Costruisce un DataFrame che mostra la probabilità di superare un benchmark.

    Per ogni metrica nei risultati della simulazione Monte Carlo, calcola la
    percentuale di simulazioni in cui il valore della metrica è stato
    maggiore o uguale al valore della stessa metrica nel benchmark.

    Args:
        metrics_data (dict): Un dizionario dove le chiavi sono i nomi delle metriche
                             e i valori sono array numpy con i risultati della simulazione.
        benchmark (pd.Series): Le statistiche del backtest di benchmark.

    Returns:
        pd.DataFrame: Un DataFrame con le metriche come indice e le colonne
                      'Benchmark' e 'Prob. >= Benchmark'.

    """
    prob_data = {
        key: {
            "Benchmark": f"{benchmark[key]:.2f}",
            "Prob. >= Benchmark": f"{np.mean(metrics_data[key] >= benchmark[key]) * 100:.2f}%",
        }
        for key in metrics_data
    }
    return pd.DataFrame.from_dict(prob_data, orient="index")


def _process_single_mc_combination(
    current_comb: pd.Series,
    strategy_params: dict,
    trades_for_this_comb: list | None,
    benchmark_stats: pd.Series,
    ticker: str,
) -> dict:
    """Run Monte Carlo simulation for a single parameter combination and process its results.

    Args:
        current_comb (pd.Series): The row from the results DataFrame representing the current combination.
        strategy_params (dict): A dictionary of the strategy parameters for this combination.
        trades_for_this_comb (list | None): The list of trade returns for this combination.
        benchmark_stats (pd.Series): The benchmark statistics for comparison.
        ticker (str): The ticker symbol.

    Returns:
        dict: A flat dictionary combining the strategy parameters with the processed Monte Carlo statistics.

    """
    if trades_for_this_comb is None:
        # No trades, so no MC stats, just return the params
        return strategy_params

    # Run the main MC simulation for this combination
    run_montecarlo(
        ticker=ticker,
        trades=pd.Series(trades_for_this_comb),
        initial_capital=ss.initial_capital_wid,
        sampling_method=ss.opt_mc_sampling_method_wid,
        num_sims=ss.opt_mc_n_sims_wid,
        sim_length=ss.opt_mc_sim_length_wid,
        original_stats=current_comb,
        benchmark=benchmark_stats,
    )

    # Process the results: pivot the percentile stats into a flat structure
    mc_percentiles = ss.backtest_mc_percentiles.get(ticker)
    if mc_percentiles is None or mc_percentiles.empty:
        return strategy_params

    # Filter for 'Original' and '95%' confidence level stats
    mc_req_stats = mc_percentiles[mc_percentiles.index.astype(str).str.contains("Originale|95")]

    if mc_req_stats.empty:
        return strategy_params

    suffix_map = {"Originale": "O", "Livello 95%": "p95"}

    # This creates a flat dictionary from the two-row DataFrame
    flat_stats = {
        f"{metric_name}-{suffix_map[level]}": value
        for level, row in mc_req_stats.iterrows()
        for metric_name, value in row.items()
        if level in suffix_map
    }

    # Combine strategy params with the flattened stats
    return strategy_params | flat_stats


def run_montecarlos_for_best_combs(
    benchmark_stats: pd.Series,
    ticker: str,
) -> None:
    """Run Monte Carlo simulations for the best performing parameter combinations.

    This function selects the top combinations from an optimization run,
    executes a Monte Carlo simulation for each, and stores the aggregated
    results in the session state.

    Args:
        benchmark_stats (pd.Series): The statistics of the benchmark for comparison.
        ticker (str): The ticker symbol for which to run the simulations.

    """
    if not ss.get("opt_run_mc_wid"):
        return

    # 1. Select best combinations from optimization results
    obj_col = ss.get("opt_obj_func_wid")
    ranking_df = ss.get("opt_combs_ranking", {}).get(ticker)

    if not obj_col or ranking_df is None or ranking_df.empty:
        return

    is_higher_better = MESSAGES.get("optimization_settings", {}).get("objectives", {}).get(obj_col, True)
    sorted_combs = ranking_df.sort_values(by=obj_col, ascending=not is_higher_better)
    num_to_promote = ss.get("mc_promoted_combs_wid", 10)
    best_combs = sorted_combs.head(num_to_promote)

    # 2. Run MC for each combination and collect results
    mc_run_placeholder = st.empty()
    all_results_list = []
    num_best_combs = len(best_combs)
    messages = MESSAGES.get("display_texts", {}).get("messages", {})

    for i, (idx, current_comb) in enumerate(best_combs.iterrows()):
        mc_run_placeholder.info(
            messages.get("mc_running_for_comb", "Running Monte Carlo for {ticker} (comb. {i}/{total})").format(
                ticker=ticker, i=i + 1, total=num_best_combs
            )
        )

        strategy_params = dict(zip(best_combs.index.names, idx, strict=False))
        trades = ss.get("trade_returns", {}).get(ticker, {}).get(idx)

        processed_result = _process_single_mc_combination(
            current_comb, strategy_params, trades, benchmark_stats, ticker
        )
        all_results_list.append(processed_result)

    mc_run_placeholder.empty()

    # 3. Create final DataFrame and store in session state
    if not all_results_list:
        ss.opt_mc_results[ticker] = pd.DataFrame()
        return

    final_mc_df = pd.DataFrame(all_results_list).set_index(list(best_combs.index.names))
    ss.opt_mc_results[ticker] = final_mc_df
