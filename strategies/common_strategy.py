# strategies/common_strategy.py

from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
)

import pandas as pd
from backtesting import Strategy


# This class serves as a base for all strategies, implementing common logic
# for managing long positions, stop loss, and take profit.
class CommonStrategy(Strategy):
    """Implement base logic for trading strategies: manage long positions, stop loss, and take profit.

    Subclass this for specific strategies.
    """

    # Default parameters for Stop Loss and Take Profit. Will be overridden by the UI.
    sl_pct: float = 0.05
    tp_pct: float = 0.10

    DISPLAY_NAME: ClassVar[str] = "Common Strategy"

    # Attribute to define strategy-specific optimization constraints.
    # Must be a callable that accepts a 'stats' object and returns True/False.
    # If None, no additional constraints will be applied beyond intrinsic ones.
    optimization_constraint: Callable[[pd.Series], bool] | None = None

    # Class variable to define strategy parameters.
    # Should be overridden in child strategies for their specific parameters,
    # and should also include sl_pct and tp_pct for optimization.
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "sl_pct",
            "type": float,
            "default": 0.05,
            "lowest": 0.00,
            "highest": 0.99,
            "min": 0.005,
            "max": 0.1,
            "step": 0.005,
        },  # Optimizable
        {
            "name": "tp_pct",
            "type": float,
            "default": 0.10,
            "lowest": 0.00,
            "highest": 2.00,
            "min": 0.00,
            "max": 0.5,
            "step": 0.01,
        },  # Optimizable
    ]

    def init(self) -> None:
        """Initialize the strategy. Override this method in subclasses to implement custom logic."""
        # Questo è un placeholder. Le classi derivate devono implementare la propria logica.
        pass

    def next(self) -> None:
        """Execute logic for each new bar. Override this method in subclasses to implement custom logic."""
        # Questo è un placeholder. Le classi derivate devono implementare la propria logica.
        pass

    def _buy_long(self) -> None:
        """Open a long position if no position is currently open, setting stop loss and take profit."""
        if not self.position:  # Entra solo se non c'è una posizione aperta
            # current_close: float = self.data.Close[-1]
            sl_price: float = self.data.Close[-1] * (1 - self.sl_pct) if self.sl_pct != 0 else None
            tp_price: float = self.data.Close[-1] * (1 + self.tp_pct) if self.tp_pct != 0 else None
            self.buy(sl=sl_price, tp=tp_price)
            # print(f"LONG - Entry: {current_close:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")

    def _close_position(self) -> None:
        """Close the current open position, if any."""
        if self.position:
            self.position.close()
