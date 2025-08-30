from typing import Any

import numpy as np


# Funzione per calcolare l'Equity Finale
def calculate_equity_final(
    returns_2d_array: np.ndarray, *, initial_equity: float = 100.0, **kwargs: dict[str, Any]
) -> np.ndarray:
    """Calculate the final equity value after applying returns to an initial equity.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        initial_equity (float, optional): The starting equity value. Defaults to 100.0.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of final equity values for each path in the returns array.

    """
    cumulative_returns = np.cumprod(1 + returns_2d_array, axis=1)
    return initial_equity * cumulative_returns[-1, :]


# Funzione per calcolare l'Equity Peak
def calculate_equity_peak(
    returns_2d_array: np.ndarray, *, initial_equity: float = 100.0, **kwargs: dict[str, Any]
) -> np.ndarray:
    """Calculate the equity peak value after applying returns to an initial equity.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        initial_equity (float, optional): The starting equity value. Defaults to 100.0.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of peak equity values for each path in the returns array.

    """
    cumulative_returns = np.cumprod(1 + returns_2d_array, axis=1)
    equity_curve = initial_equity * cumulative_returns
    return np.max(equity_curve, axis=1)


# Funzione per calcolare il Ritorno Totale
def calculate_total_return(returns_2d_array: np.ndarray, **kwargs: dict[str, Any]) -> np.ndarray:
    """Calculate the total percentage return for each series of returns.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of total percentage returns for each path in the returns array.

    """
    return (np.prod(1 + returns_2d_array, axis=1) - 1) * 100


# Funzione per calcolare il Ritorno Annualizzato
# def calculate_annualized_return(returns_2d_array: np.ndarray, annualization_factor: int = 252) -> np.ndarray:
#     """Calculate the annualized percentage return for each series of returns.

#     Args:
#         returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
#         annualization_factor (int, optional): The number of periods in a year. Defaults to 252.

#     Returns:
#         np.ndarray: An array of annualized percentage returns for each path in the returns array.

#     """
#     num_periods = returns_2d_array.shape[0]
#     total_return_factor = np.prod(1 + returns_2d_array, axis=1)
#     annualized_return = (total_return_factor ** (annualization_factor / num_periods) - 1) * 100
#     return annualized_return


# Funzione per calcolare la Volatilità Annualizzata
def calculate_annualized_volatility(
    returns_2d_array: np.ndarray, *, annualization_factor: int = 252, **kwargs: dict[str, Any]
) -> np.ndarray:
    """Calculate the annualized volatility for each series of returns.

    The annualized volatility is calculated as the standard deviation of the daily returns,
    multiplied by the square root of the annualization factor, and then multiplied by 100
    to convert to a percentage.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of annualized volatilities for each path in the returns array.

    """
    period_volatility = np.std(returns_2d_array, axis=1)
    return period_volatility * np.sqrt(annualization_factor) * 100


# Funzione per calcolare il CAGR (Compound Annual Growth Rate)
def calculate_cagr(
    returns_2d_array: np.ndarray, *, annualization_factor: int = 252, **kwargs: dict[str, Any]
) -> np.ndarray:
    """Calculate the Compound Annual Growth Rate (CAGR) for each series of returns.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of CAGRs for each path in the returns array.

    """
    num_periods = returns_2d_array.shape[0]
    total_return_factor = np.prod(1 + returns_2d_array, axis=1)
    return (total_return_factor ** (annualization_factor / num_periods) - 1) * 100


# Funzione per calcolare lo Sharpe Ratio
def calculate_sharpe_ratio(
    returns_2d_array: np.ndarray,
    *,
    risk_free_rate: float,
    annualize: bool = False,
    annualization_factor: int | None = None,
    **kwargs: dict[str, Any],
) -> np.ndarray:
    """Calculate the Sharpe Ratio for each series of returns.

    The Sharpe Ratio is a measure of risk-adjusted return. It is calculated as the
    mean excess return (above the risk-free rate) divided by the standard deviation
    of the returns.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        risk_free_rate (float): The risk-free rate of return (same period as the returns).
        annualize (bool, optional): Whether to annualize the Sharpe Ratio. Defaults to False.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of Sharpe Ratios for each path in the returns array.

    """
    excess_returns = returns_2d_array - risk_free_rate
    mean_excess_returns = np.mean(excess_returns, axis=1)
    std_dev_returns = np.std(returns_2d_array, axis=1)

    if annualize and annualization_factor is not None:
        sharpe_ratio = (mean_excess_returns * annualization_factor) / (std_dev_returns * np.sqrt(annualization_factor))
    elif not annualize:
        sharpe_ratio = mean_excess_returns / std_dev_returns
    else:
        raise ValueError("If annualize is True, annualization_factor must be provided.")

    sharpe_ratio = np.nan_to_num(sharpe_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    return sharpe_ratio


# Funzione per calcolare il Max. Drawdown
def calculate_max_drawdown(
    returns_2d_array: np.ndarray, *, initial_equity: float = 100.0, **kwargs: dict[str, Any]
) -> np.ndarray:
    """Calculate the maximum drawdown for each series of returns.

    The maximum drawdown is a measure of risk that measures the maximum loss
    from peak to trough. It is calculated as the minimum cumulative return
    divided by the maximum cumulative return.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        initial_equity (float, optional): The starting equity value. Defaults to 100.0.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of maximum drawdowns for each path in the returns array.

    """
    cumulative_returns = np.cumprod(1 + returns_2d_array, axis=1)
    equity_curve = initial_equity * cumulative_returns

    rolling_max = np.maximum.accumulate(equity_curve, axis=1)

    drawdown = (equity_curve - rolling_max) / rolling_max

    return np.min(drawdown, axis=1) * 100


# Funzione per calcolare il Calmar Ratio
def calculate_calmar_ratio(
    returns_2d_array: np.ndarray,
    *,
    annualization_factor: int = 252,
    initial_equity: float = 100.0,
    **kwargs: dict[str, Any],
) -> np.ndarray:
    """Calculate the Calmar Ratio for each series of returns.

    The Calmar Ratio is a measure of risk-adjusted return. It is calculated as the
    annualized return divided by the maximum drawdown.

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        initial_equity (float, optional): The starting equity value. Defaults to 100.0.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of Calmar Ratios for each path in the returns array.

    """
    annualized_return = calculate_cagr(returns_2d_array, annualization_factor=annualization_factor)
    # np.prod(1 + returns_2d_array, axis=1) ** (annualization_factor / returns_2d_array.shape[0]) - 1

    # cumulative_returns = np.cumprod(1 + returns_2d_array, axis=1)
    # equity_curve = initial_equity * cumulative_returns
    # rolling_max = np.maximum.accumulate(equity_curve, axis=1)
    # drawdown = (equity_curve - rolling_max) / rolling_max
    # max_drawdown_factor = np.min(drawdown, axis=1)
    max_drawdown_factor = (
        calculate_max_drawdown(returns_2d_array, initial_equity=initial_equity) / 100.0
    )  # Convert to factor
    return np.where(
        max_drawdown_factor != 0,
        annualized_return / np.abs(max_drawdown_factor),
        0.0,
    )


# Funzione per calcolare il Sortino Ratio
def calculate_sortino_ratio(
    returns_2d_array: np.ndarray,
    *,
    risk_free_rate: float,
    annualize: bool = False,
    annualization_factor: int = 252,
    minimum_acceptable_return: float = 0.0,
    **kwargs: dict[str, Any],
) -> np.ndarray:
    """Calculate the Sortino Ratio for each series of returns.

    The Sortino Ratio is a measure of risk-adjusted return. It is calculated as the
    mean excess return (above the risk-free rate) divided by the standard deviation
    of the downside returns (returns below the minimum acceptable return).

    Args:
        returns_2d_array (np.ndarray): A 2D array of returns, where each column represents a different path.
        risk_free_rate (float): The risk-free rate of return.
        annualize (bool, optional): Whether to annualize the Sortino Ratio. Defaults to False.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        minimum_acceptable_return (float, optional): The minimum acceptable return. Defaults to 0.0.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        np.ndarray: An array of Sortino Ratios for each path in the returns array.

    """
    excess_returns = returns_2d_array - risk_free_rate

    downside_returns = np.where(
        returns_2d_array < minimum_acceptable_return, returns_2d_array - minimum_acceptable_return, 0
    )
    downside_deviation = np.std(downside_returns, axis=1)

    mean_excess_returns_annualized = np.mean(excess_returns, axis=1) * (annualization_factor if annualize else 1)

    return np.where(
        downside_deviation != 0,
        mean_excess_returns_annualized / (downside_deviation * (np.sqrt(annualization_factor) if annualize else 1)),
        0.0,
    )


# Funzione per calcolare Alpha e Beta
def calculate_alpha_beta(
    returns_2d_array: np.ndarray,
    *,
    risk_free_rate: float,
    annualization_factor: int = 252,
    benchmark_returns_2d_array: np.ndarray,
    **kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the alpha and beta for each series of returns against a benchmark.

    Alpha represents the excess return of an investment relative to the return of a benchmark index.
    Beta measures the volatility, or systematic risk, of an investment in comparison to the market as a whole.

    Args:
        returns_2d_array (np.ndarray): A 2D array of asset returns, where each column represents a different asset.
        risk_free_rate (float): The risk-free rate of return.
        annualization_factor (int, optional): The number of periods in a year. Defaults to 252.
        benchmark_returns_2d_array (np.ndarray, optional): A 2D array of benchmark returns. If None, defaults to a 10% annualized return.
        **kwargs (dict[str, Any]): Additional keyword arguments.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays representing the alphas and betas for each asset compared to the benchmark.

    """
    if benchmark_returns_2d_array is None:
        benchmark_returns_2d_array = np.full_like(
            returns_2d_array, (1.1 ** (1 / annualization_factor)) - 1
        )  # Default benchmark returns (10% annualized)

    if returns_2d_array.shape[0] != benchmark_returns_2d_array.shape[0]:
        raise ValueError(
            "Gli array di rendimenti e benchmark devono avere la stessa lunghezza (stesso numero di periodi)."
        )

    excess_returns = returns_2d_array - risk_free_rate
    excess_benchmark_returns = benchmark_returns_2d_array - risk_free_rate

    num_assets = returns_2d_array.shape[1]
    alphas = np.zeros(num_assets)
    betas = np.zeros(num_assets)

    for i in range(num_assets):
        y = excess_returns[:, i]
        x = excess_benchmark_returns[:, i]

        valid_indices = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_indices]
        y_clean = y[valid_indices]

        if len(x_clean) < 2:
            betas[i] = np.nan
            alphas[i] = np.nan
            continue

        coefficients = np.polyfit(x_clean, y_clean, 1)

        betas[i] = coefficients[0]
        alpha_period = coefficients[1]  # Like daily

        alphas[i] = alpha_period * annualization_factor * 100

    return alphas, betas


FUNCTION_MAP = {
    "calculate_equity_final": calculate_equity_final,
    "calculate_equity_peak": calculate_equity_peak,
    "calculate_max_drawdown": calculate_max_drawdown,
    "calculate_sharpe_ratio": calculate_sharpe_ratio,
    "calculate_alpha_beta": calculate_alpha_beta,
    "calculate_sortino_ratio": calculate_sortino_ratio,
    "calculate_calmar_ratio": calculate_calmar_ratio,
    "calculate_cagr": calculate_cagr,
    "calculate_annualized_volatility": calculate_annualized_volatility,
    "calculate_total_return": calculate_total_return,
}
