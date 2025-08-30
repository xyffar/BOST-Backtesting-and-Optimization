# strategies/strategy_breakout.py

from collections.abc import Callable  # Added List, Dict, Any
from typing import Any, ClassVar

import pandas as pd

from strategies.common_strategy import CommonStrategy  # Importa la base strategy


class BreakoutStrategy(CommonStrategy):
    """Breakout strategy based on dynamic support and resistance levels."""

    # Strategy parameters
    lookback_period: int = 20  # Number of bars to consider for calculating support and resistance levels

    # Name for display
    DISPLAY_NAME = "Breakout Supports/Resistances"

    # Optimization constraint: lookback_period should be greater than 1
    optimization_constraint: Callable[[pd.Series], bool] | None = lambda s: s.lookback_period > 1

    # Definition of parameters for UI and optimization
    # This list overrides PARAMS_INFO from CommonStrategy
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "lookback_period",
            "type": int,
            "default": 20,
            "lowest": 2,
            "highest": 200,
            "min": 2,
            "max": 50,
            "step": 1,
        },
        # Stop Loss and Take Profit parameters
        # You can keep default values from CommonStrategy or customize them here
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
        """Initialize the strategy. Pre-calculate dynamic support and resistance levels on the previous 'lookback_period' bars."""
        # The maximum mobile period serves as a dynamic resistance level.
        # We use self.I to register the indicator that will be updated automatically at each tick.
        self.resistance_level = self.I(
            lambda data: pd.Series(data).shift(1).rolling(int(self.lookback_period)).max(),
            self.data.High,
        )

        # The minimum mobile period serves as a dynamic support level.
        self.support_level = self.I(
            lambda data: pd.Series(data).shift(1).rolling(int(self.lookback_period)).min(),
            self.data.Low,
        )

    def next(self) -> None:
        """Implement trading logic for each new data bar."""
        # Ensure we have enough data to calculate support and resistance levels
        if len(self.data.Close) < self.lookback_period:
            return

        current_close = self.data.Close[-1]

        # Access the pre-calculated support and resistance levels for the current bar
        current_resistance = self.resistance_level[-1]
        current_support = self.support_level[-1]

        # Buy logic:
        # If the current closing price exceeds the dynamic resistance level
        # And we don't have an open position (for a breakout).
        if current_close > current_resistance and not self.position:
            self._buy_long()

        # Sell/close position logic:
        # If we have an open long position
        # And the current closing price falls below the dynamic support level
        # (indicating a breakout fall or a trend reversal).
        elif self.position and current_close < current_support:
            self._close_position()
