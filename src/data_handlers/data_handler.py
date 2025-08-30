# data_handler.py

import pandas as pd
import yfinance as yf

from src.config.config import MESSAGES, ss, streamlit_obj


class DownloadedDataError(Exception):
    """Custom exception indicating an error with downloaded data."""

    def __init__(self, message: str) -> None:
        """Initialize the DownloadedDataError with a descriptive error message."""
        super().__init__(message)


class MissingColumnError(Exception):
    """Custom exception indicating that required columns are missing in the data."""

    def __init__(self, message: str) -> None:
        """Initialize the MissingColumnInData with a descriptive error message."""
        super().__init__(message)


def download_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Download historical data for a given ticker using yfinance.

    Args:
        ticker (str): The asset ticker symbol (e.g., "AAPL").
        start_date (str): Start date for the data (format 'YYYY-MM-DD').
        end_date (str): End date for the data (format 'YYYY-MM-DD').
        interval (str): Data interval (e.g., "1d", "1h").

    Returns:
        tuple: (pd.DataFrame, str, str) - (Data, Status, Message).
               Status can be "success" or "failure".
               Message includes the ticker and the result.

    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )  # Disable yfinance's internal progress

    if data is None or data.empty:
        raise DownloadedDataError(MESSAGES["display_texts"]["data_handler"]["no_data_found"].format(ticker=ticker))

    # If columns are a MultiIndex (common with yfinance when downloading
    # multiple items or specific fields),
    # remove the upper or lower level to flatten it.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)

    # Now that columns are a simple Index, capitalize them
    data.columns = data.columns.str.capitalize()

    # Ensure required columns are present
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in data.columns for col in required_cols):
        raise MissingColumnError(
            MESSAGES["display_texts"]["data_handler"]["required_columns_missing"].format(ticker=ticker)
        )

    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    return data


def get_sp500_data(start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Download historical S&P 500 benchmark data using yfinance.

    This function specifically targets the S&P 500 (SPY ticker) to provide
    a benchmark for strategy performance comparison.

    Args:
        start_date (str): Start date for data download in 'YYYY-MM-DD' format.
        end_date (str): End date for data download in 'YYYY-MM-DD' format.
        interval (str): Interval for data (e.g., "1d", "1h").

    Returns:
        tuple: (pd.DataFrame, str, str) - (Data, Status, Message).
               Status can be "success" or "failure".
               Message indicates the result of the download.

    """
    # try:
    data = yf.download(
        MESSAGES["general_settings"]["sp500_ticker"],
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        multi_level_index=False,
        auto_adjust=True,
    )  # Disable yfinance's internal progress

    if data is None or data.empty:
        raise DownloadedDataError(
            MESSAGES["display_texts"]["data_handler"]["no_benchmark_data_found"].format(
                SP500_TICKER=MESSAGES["general_settings"]["sp500_ticker"]
            )
        )

    # If columns are a MultiIndex, flatten it
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)

    # Now that columns are a simple Index, capitalize them
    data.columns = data.columns.str.capitalize()
    # Ensure required columns are present
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in data.columns for col in required_cols):
        raise MissingColumnError(
            MESSAGES["display_texts"]["data_handler"]["required_columns_missing"].format(ticker="Benchmark")
        )

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

    # except Exception as e:
    #     return (
    #         None,
    #         "failure",
    #         MESSAGES["display_texts"]["data_handler"]["benchmark_download_error"].format(
    #             SP500_TICKER=MESSAGES["general_settings"]["sp500_ticker"], e=e
    #         ),
    #     )


def calculate_benchmark_return(benchmark_data: pd.DataFrame) -> float:
    """Calculate the total percentage return of the benchmark data.

    The return is calculated from the 'Close' prices of the first and last
    available data points in the benchmark DataFrame.

    Args:
        benchmark_data (pd.DataFrame): DataFrame containing benchmark historical data,
                                       expected to have a 'Close' column.

    Returns:
        float: The total percentage return of the benchmark. Returns 0.0 if data is empty,
               'Close' column is missing, or the starting price is zero.

    """
    if benchmark_data is None or benchmark_data.empty:
        return 0.0

    if "Close" not in benchmark_data.columns:
        return 0.0

    start_price = benchmark_data["Close"].iloc[0]
    end_price = benchmark_data["Close"].iloc[-1]

    if start_price == 0:
        return 0.0

    return ((end_price - start_price) / start_price) * 100


def get_ticker_data_and_infos(
    download_progress_placeholder: streamlit_obj,
    download_success_placeholder: streamlit_obj,
    download_fail_placeholder: streamlit_obj,
    i: int,
    ticker: str,
) -> pd.DataFrame:
    """Download data for a specific ticker and manage Streamlit UI elements to display download progress, success, or failure messages.

    Updates Streamlit session state with successful and failed downloads.

    Args:
        tickers (list[str]): A list of all ticker symbols for which data is being processed.
                             Used to display overall progress (e.g., "1/5 tickers").
        start_date_yf (str): Start date for data download in 'YYYY-MM-DD' format.
        end_date_yf (str): End date for data download in 'YYYY-MM-DD' format.
        data_interval (str): Interval for data (e.g., '1d', '1h').
        download_progress_placeholder (streamlit.delta_generator.DeltaGenerator):
            Streamlit placeholder element used to display a temporary progress message
            during data download. This placeholder is emptied upon completion.
        download_success_placeholder (streamlit.delta_generator.DeltaGenerator):
            Streamlit placeholder element used to display a success message after
            a ticker's data has been successfully downloaded.
        download_fail_placeholder (streamlit.delta_generator.DeltaGenerator):
            Streamlit placeholder element used to display a warning/error message
            if a ticker's data fails to download.
        i (int): The zero-based index of the current `ticker` in the `tickers` list.
                 Used for progress display.
        ticker (str): The current ticker symbol for which to download data.

    Returns:
        pd.DataFrame | None: The downloaded data as a pandas DataFrame if successful, None otherwise.

    """
    download_progress_placeholder.info(
        MESSAGES["display_texts"]["messages"]["downloading_ticker"].format(
            ticker=ticker, current_idx=i + 1, total_tickers=len(ss.tickers)
        )
    )
    try:
        data = download_data(ticker, ss.start_date_wid, ss.end_date_wid, ss.data_interval_wid)
    except (DownloadedDataError, MissingColumnError) as e:
        ss.failed_downloads_tickers.append(ticker)
        download_fail_placeholder.error(
            MESSAGES["display_texts"]["messages"]["download_failed_ticker"] + ", ".join(ss.failed_downloads_tickers)
        )
        raise e
    else:
        ss.successful_downloads_tickers.append(ticker)
        download_success_placeholder.success(
            MESSAGES["display_texts"]["messages"]["download_success_ticker"]
            + ", ".join(ss.successful_downloads_tickers)
        )
        return data

    finally:
        download_progress_placeholder.empty()  # Remove blue progress box
