# strategies/strategy_bollinger_bands.py

from collections.abc import Callable
from typing import Any, ClassVar

import pandas as pd
import pandas_ta as ta

from strategies.common_strategy import CommonStrategy


class BollingerBandsStrategy(CommonStrategy):
    """Strategy based on Bollinger Bands.

    Bollinger Bands are a volatility indicator that plots two standard deviations
    above and below a moving average (MA) of prices. They are used to identify
    potential price movements and to anticipate potential breakouts.

    This strategy buys when the closing price crosses below the lower Bollinger Band
    and sells when the closing price crosses above the upper Bollinger Band.
    """

    # Parameters specific to this strategy
    bb_period: int = 20
    bb_dev: float = 2.0

    # Name for display
    DISPLAY_NAME = "Bollinger's Bands"

    optimization_constraint: Callable[[pd.Series], bool] | None = None  # No specific constraint for this strategy

    # Definition of parameters for the UI and optimization
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "bb_period",
            "type": int,
            "default": 20,
            "lowest": 5,
            "highest": 50,
            "min": 10,
            "max": 40,
            "step": 1,
        },
        {
            "name": "bb_dev",
            "type": float,
            "default": 2.0,
            "lowest": 1.0,
            "highest": 3.0,
            "min": 1.5,
            "max": 2.5,
            "step": 0.1,
        },
        # SL/TP parameters
        {
            "name": "sl_pct",
            "type": float,
            "default": 0.05,
            "lowest": 0.00,
            "highest": 0.99,
            "min": 0.05,
            "max": 0.05,
            "step": 0.005,
        },
        {
            "name": "tp_pct",
            "type": float,
            "default": 0.00,
            "lowest": 0.00,
            "highest": 2.00,
            "min": 0.00,
            "max": 0.00,
            "step": 0.01,
        },
    ]

    def init(self) -> None:
        """Initialize the strategy by calculating Bollinger Bands."""
        close_series: pd.Series = pd.Series(self.data.Close)
        # Calculate Bollinger Bands using pandas_ta
        (
            self.bb_lower,
            self.bb_middle,
            self.bb_upper,
            self.bb_bandwidth,
            self.bb_percent,
        ) = self.I(
            ta.bbands,
            close_series,
            length=self.bb_period,
            std=self.bb_dev,
            append=False,  # Do not append to the original dataframe, backtesting.py handles it
        )

    def next(self) -> None:
        """Logic to buy and sell based on Bollinger Bands signals."""
        # Buy signal: Close price crosses below the lower Bollinger Band from the previous tick.
        buy_signal = self.data.Close[-1] > self.bb_lower[-1] and self.data.Close[-2] <= self.bb_lower[-2]

        # Sell signal: Close price crosses above the upper Bollinger Band or closes above the middle Bollinger Band after being above.
        close_signal = (self.data.Close[-1] < self.bb_upper[-1] and self.data.Close[-2] >= self.bb_upper[-2]) or (
            self.data.Close[-1] < self.bb_middle[-1] and self.data.Close[-2] >= self.bb_middle[-2]
        )

        # Trading logic:
        # If there is no open position and there is a buy signal, open a long position.
        if not self.position and buy_signal:
            self._buy_long()

        # If there is an open position and there is a close signal, close the position.
        elif self.position and close_signal:
            self._close_position()
