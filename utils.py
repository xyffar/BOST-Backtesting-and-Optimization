# utils.py

import datetime as dt
import importlib  # .util
import inspect
import math  # Import math for floor function
import os  # Import the os module
import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import backtesting
import numpy as np  # Import numpy for rounding
import pandas as pd
import streamlit as st

from strategies.common_strategy import CommonStrategy  # Importa la strategia base

# Import necessary for handling Bokeh plots via streamlit-bokeh-events
# from streamlit_bokeh import streamlit_bokeh


def format_date(date_obj: dt.date) -> str:
    """Format a datetime.date object into a "dd/mm/yyyy" string."""
    return date_obj.strftime("%d/%m/%Y")


def parse_date(date_string: str) -> dt.date:
    """Parse a date string in "dd/mm/yyyy" format into a datetime.date object."""
    return dt.datetime.strptime(date_string, "%d/%m/%Y").date()  # Expected format dd/mm/yyyy


# New functions for mode switching and subheader display
def set_backtest_mode() -> None:
    """Set the application mode to backtest.

    Updates the Streamlit session state to indicate backtest mode.
    """
    st.session_state.mode = "backtest"


def set_optimization_mode() -> None:
    """Set the application mode to optimization.

    Updates the Streamlit session state to indicate optimization mode.
    """
    st.session_state.mode = "optimization"


def calculate_optimization_combinations(opt_params_ranges: dict[str:Any]) -> int:
    """Calculate the total number of parameter combinations for optimization.

    Args:
        opt_params_ranges (dict): A dictionary of parameter ranges for optimization.

    Returns:
        int: The total number of unique parameter combinations.

    """
    total_combinations = 1
    for param_name, param_range in opt_params_ranges.items():
        if isinstance(param_range, range):  # Integer ranges (from render_optimization_params)
            num_steps = len(param_range)
        elif isinstance(param_range, tuple) and len(param_range) == 3:  # Float ranges (min, max, step)
            min_val, max_val, step_val = param_range
            if step_val == 0:
                # Handle edge case to prevent division by zero, treat as 1 combination if range is single point
                num_steps = 1
            else:
                # Calculate steps for float range, ensure it's at least 1 if min == max
                num_steps = (
                    math.floor(round((max_val - min_val) / step_val, 5)) + 1
                )  # Round to avoid float precision issues
                if min_val == max_val:
                    num_steps = 1
        elif isinstance(param_range, list):  # Categorical options (list of values)
            num_steps = len(param_range)
        else:
            # Should not happen if render_optimization_params is consistent
            st.warning(f"Unexpected parameter range type for {param_name}: {type(param_range)}")
            num_steps = 1  # Default to 1 to avoid breaking calculation

        total_combinations *= num_steps
    return int(total_combinations)


def load_strategies() -> dict[str, type[CommonStrategy]]:
    """Find and dynamically load all strategy classes inheriting from CommonStrategy.

    Searches Python files in the same directory starting with 'strategy_'. Returns a dictionary mapping each strategy's DISPLAY_NAME to its class.

    """
    from config import MESSAGES

    strategies: dict[str, type[CommonStrategy]] = {}
    current_dir: str = (
        os.path.dirname(__file__) + "/" + MESSAGES["general_settings"]["folder_strategies"]
    )  # Directory corrente del file

    for filename in os.listdir(current_dir):
        if filename.startswith("strategy_") and filename.endswith(".py") and filename != "common_strategy.py":
            module_name: str = (
                MESSAGES["general_settings"]["folder_strategies"] + "." + filename[:-3]
            )  # Rimuovi '.py' e metti la cartella
            try:
                # Importa il modulo dinamicamente
                module: Any = importlib.import_module(module_name)

                # Itera su tutti i membri del modulo
                for name, obj in inspect.getmembers(module):
                    # Controlla se l'oggetto è una classe, non è la classe BaseStrategy stessa,
                    # e se eredita da BaseStrategy.
                    if inspect.isclass(obj) and obj != CommonStrategy and issubclass(obj, CommonStrategy):
                        if hasattr(obj, "DISPLAY_NAME") and isinstance(obj.DISPLAY_NAME, str):
                            strategies[obj.DISPLAY_NAME] = obj
                        else:
                            print(
                                f"Avviso: La strategia '{name}' nel modulo '{module_name}' non ha un attributo 'DISPLAY_NAME' valido. Verrà ignorata."
                            )
            except Exception as e:
                print(f"Errore durante il caricamento della strategia dal file '{filename}': {e}")

    # Ordina le strategie per nome visualizzato per un menu a tendina pulito
    sorted_strategies: dict[str, type[CommonStrategy]] = dict(sorted(strategies.items()))
    return sorted_strategies


def list_varying_params(all_combs: pd.DataFrame) -> list[tuple[int, str]]:
    """Return the names of parameters that varied across optimization combinations.

    Takes the DataFrame of all optimization combinations and returns the names of parameters with at least two unique values.

    Returns:
        list: A list of parameter names.

    """
    varying_param_columns = []
    varying_param_columns.extend(
        (idx, col) for idx, col in enumerate(all_combs.columns) if all_combs[col].nunique() > 1
    )
    return varying_param_columns


def add_benchmark_comparison(
    optimization_heatmap_df: pd.DataFrame,
    benchmark_comparison: float | int,
    obj_func: str | None = None,
) -> pd.DataFrame:
    """Add benchmark comparison columns to the optimization heatmap DataFrame.

    Adds columns for the benchmark value and the percentage variation from the benchmark for the selected objective function.

    Args:
        optimization_heatmap_df (pd.DataFrame): DataFrame with optimization results.
        benchmark_comparison (float | int): The benchmark value to compare against.
        obj_func (str, optional): The name of the objective function column.

    Returns:
        pd.DataFrame: DataFrame with added 'Benchmark' and 'Var. [%]' columns.

    """
    df_with_benchmark = optimization_heatmap_df.copy()
    df_with_benchmark["Benchmark"] = benchmark_comparison
    df_with_benchmark["Var. [%]"] = (
        100 * (df_with_benchmark[obj_func] - df_with_benchmark["Benchmark"]) / df_with_benchmark["Benchmark"]
    )
    return df_with_benchmark


def calculate_total_returns(equity_line: np.ndarray, initial_capital: float) -> float:
    """Calculate the total percentage return of the equity line.

    Args:
        equity_line (np.ndarray): Series of capital over time.
        initial_capital (float): Initial capital.

    Returns:
        float: Total percentage return (e.g., 0.10 for 10%).

    """
    if initial_capital == 0:
        return 0.0
    return (equity_line[-1] - initial_capital) / initial_capital


def calculate_max_drawdown(equity_line: list) -> float:
    """Calculate the maximum percentage drawdown of the equity line.

    Args:
        equity_line (np.ndarray): Series of capital over time.

    Returns:
        float: Maximum percentage drawdown (negative, e.g., -0.05 for -5%).

    """
    return (pd.Series(equity_line) / pd.Series(equity_line).expanding(min_periods=0).max()).min() - 1


def calculate_sharpe_ratio(trade_returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio for a series of per-trade returns.

    Args:
        trade_returns (np.ndarray): Percentage returns for each trade.
        risk_free_rate (float): Risk-free rate per trade period (default 0).

    Returns:
        float: Sharpe Ratio.

    """
    if len(trade_returns) < 2:
        return 0.0  # Requires at least 2 trades for standard deviation
    mean_return = np.mean(trade_returns)
    std_return = np.std(trade_returns)
    return 0.0 if std_return == 0 else (mean_return - risk_free_rate) / std_return


def calculate_sortino_ratio(trade_returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sortino Ratio for a series of per-trade returns.

    Args:
        trade_returns (np.ndarray): Percentage returns for each trade.
        risk_free_rate (float): Risk-free rate (default 0).

    Returns:
        float: Sortino Ratio.

    """
    downside = np.sqrt((trade_returns[trade_returns < 0] ** 2).sum() / len(trade_returns))

    return trade_returns.mean() / downside


def calculate_sqn(trade_returns: np.ndarray) -> float:
    """Calculate the System Quality Number (SQN) by Van Tharp.

    Args:
        trade_returns (np.ndarray): Percentage returns for each trade.

    Returns:
        float: SQN.

    """
    if len(trade_returns) == 0:
        return 0.0
    expectancy = np.mean(trade_returns)
    std_expectancy = np.std(trade_returns)
    if std_expectancy == 0:
        return 0.0
    return (expectancy / std_expectancy) * np.sqrt(len(trade_returns))


# def round_multiindex_float_levels(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
#     """
#     Arrotonda i valori float nei livelli di un DataFrame MultiIndex alla
#     cifra decimale specificata per ridurre la frammentarietà.

#     Args:
#         df (pd.DataFrame): Il DataFrame con un MultiIndex.
#         decimals (int): Il numero di cifre decimali a cui arrotondare.

#     Returns:
#         pd.DataFrame: Un nuovo DataFrame con il MultiIndex modificato,
#                       dove i livelli float sono stati arrotondati.
#                       Restituisce una copia del DataFrame originale se non ha un MultiIndex.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("L'input deve essere un'istanza di pandas.DataFrame.")

#     if not isinstance(df.index, pd.MultiIndex):
#         print(
#             "Avviso: Il DataFrame non ha un MultiIndex. Restituisco il DataFrame originale."
#         )
#         return df.copy()

#     # Ottieni i nomi dei livelli dell'indice originale
#     original_level_names = df.copy().index.names

#     # Ottieni le matrici (array) sottostanti per ciascun livello del MultiIndex
#     # e prepara per la modifica
#     modified_levels_data = []

#     # df.index.levels restituisce una FrozenList di Index/CategoricalIndex.
#     # Convertirli in array per manipolare i valori direttamente.
#     for level_index_data in df.index.levels:
#         level_array = (
#             level_index_data.to_numpy()
#         )  # Converte il livello in un array NumPy

#         # Controlla se il dtype dell'array è numerico float
#         if np.issubdtype(level_array.dtype, np.floating):
#             # Arrotonda i valori float del livello
#             rounded_array = np.round(level_array, decimals=decimals)
#             modified_levels_data.append(rounded_array)
#             print(
#                 f"Livello '{original_level_names[len(modified_levels_data)-1]}' (tipo float) arrotondato a {decimals} decimali."
#             )
#         else:
#             # Se il livello non è float, lo aggiunge così com'è
#             modified_levels_data.append(level_array)
#             print(
#                 f"Livello '{original_level_names[len(modified_levels_data)-1]}' (tipo non float) lasciato invariato."
#             )

#     # Ricostruisci un nuovo MultiIndex con i dati modificati
#     new_multiindex = pd.MultiIndex.from_arrays(
#         modified_levels_data, names=original_level_names
#     )

#     # Crea un nuovo DataFrame con il MultiIndex arrotondato
#     # Utilizza .copy() per assicurarsi di non modificare il DataFrame originale in-place
#     df_rounded = df.copy()
#     df_rounded.index = new_multiindex

#     return df_rounded


class OptimizationRecorder:
    """Thread-safe recorder for optimization results.

    Collects and stores results from optimization runs, allowing retrieval as a DataFrame and clearing of results.
    """

    def __init__(self) -> None:
        """Initialize the OptimizationRecorder with an empty results list and a threading lock."""
        self.results = []
        self.lock = threading.Lock()

    def record(self, stats: pd.Series, params: dict) -> None:
        """Record results in a thread-safe manner.

        Args:
            stats: The statistics object with a to_dict() method.
            params: The parameters used for the run.

        """
        with self.lock:
            result_dict = stats.to_dict()
            result_dict.update(params)
            self.results.append(result_dict)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the recorded results as a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all recorded results.

        """
        return pd.DataFrame(self.results)

    def clear(self) -> None:
        """Clear all recorded results."""
        self.results = []


@contextmanager
def record_all_optimizations(
    backtest_instance: backtesting.Backtest,
) -> OptimizationRecorder:  # type: ignore
    """Context manager to record all optimization results for a backtest instance.

    Temporarily replaces the run method of the backtest instance to record results for each run, then restores the original method.

    Args:
        backtest_instance (backtesting.Backtest): The backtest instance whose run method will be wrapped.

    Returns:
        OptimizationRecorder: Recorder object containing all recorded results.

    """
    recorder = OptimizationRecorder()
    original_run: Callable = backtest_instance.run

    def recording_run(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
        try:
            stats = original_run(*args, **kwargs)
            recorder.record(stats, kwargs)
            return stats
        except Exception as e:
            return f"Exception while recording result of backtesting run: {e}"

    # Temporarily replace the run method
    backtest_instance.run = recording_run

    try:
        yield recorder
    finally:
        # Always restore the original run method
        backtest_instance.run = original_run
