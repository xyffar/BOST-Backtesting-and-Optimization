# strategies/strategy_rsi.py

from collections.abc import Callable
from typing import Any, ClassVar

import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


class RSIStrategy(CommonStrategy):
    """RSI strategy that implements a trading logic based on the RSI (Relative Strength Index).

    This strategy allows only long positions.
    """

    # Parameters specific to this strategy
    rsi_period: int = 14  # RSI period
    oversold_level: int = 30  # Oversold level
    overbought_level: int = 70  # Overbought level

    # Name for visualization
    DISPLAY_NAME: ClassVar[str] = "RSI"

    # Optimization constraint: oversold_level must be less than overbought_level
    optimization_constraint: ClassVar[Callable[[pd.Series], bool] | None] = (
        lambda s: s.oversold_level < s.overbought_level
    )

    # Definition of parameters for the UI and optimization
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "rsi_period",
            "type": int,
            "default": 14,
            "lowest": 2,
            "highest": 50,
            "min": 5,
            "max": 30,
            "step": 1,
        },
        {
            "name": "oversold_level",
            "type": int,
            "default": 30,
            "lowest": 10,
            "highest": 45,
            "min": 20,
            "max": 40,
            "step": 1,
        },
        {
            "name": "overbought_level",
            "type": int,
            "default": 70,
            "lowest": 55,
            "highest": 90,
            "min": 60,
            "max": 80,
            "step": 1,
        },
        # Stop loss and take profit parameters
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
        """Initialize the strategy by calculating the RSI."""
        close_series: pd.Series = pd.Series(self.data.Close)
        self.rsi: pd.Series = self.I(ta.rsi, close_series, self.rsi_period)

    def next(self) -> None:
        """Implement the trading logic based on the RSI.

        Only long positions are allowed.
        """
        # Buy signal: RSI crosses above the oversold level
        buy_signal: bool = crossover(self.rsi, self.oversold_level)

        # Close signal: RSI crosses below the overbought level or generally falls from an overbought level.
        close_signal: bool = crossover(self.overbought_level, self.rsi)

        # Trading logic:
        # If no position is open and there is a buy signal, open a long position.
        if not self.position:
            if buy_signal:
                self._buy_long()  # Use the helper method from the base class
        # If a position is open and there is a close signal, close the position.
        elif close_signal:
            self._close_position()  # Use the helper method from the base class
