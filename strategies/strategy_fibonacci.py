# strategies/strategy_fibonacci.py
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting.backtesting import Strategy

# from backtesting.lib import crossover


class FibonacciStrategy(Strategy):
    """Vectorized trading strategy based on Fibonacci retracements. This version executes ONLY LONG (buy) trades.

    The strategy seeks to enter positions when the price retraces
    to a key Fibonacci level (e.g., 0.5 or 0.618) within an uptrend,
    and shows a continuation signal. All calculation and signal generation
    logic is performed vectorially in the `init()` method.
    The `next()` method manages explicit buy and close logic.
    """

    # Strategy parameters, optimizable via Backtest.optimize()
    ma_period = 50  # Period for the Moving Average (to identify main trend)
    ma_type: Literal["SMA", "EMA", "WMA", "RMA"] = "SMA"  # Tipo di media mobile (SMA, EMA, WMA, RMA)
    fib_lookback = 20  # Lookback period to identify swing high/low for Fibonacci
    retracement_level = 0.618  # Fibonacci retracement level for entry (e.g., 0.382, 0.5, 0.618, 0.786)
    sl_factor = 0.01  # Percentage factor for Stop Loss (e.g., 1% below swing low)
    tp_factor = 0.02  # Percentage factor for Take Profit (e.g., 2% above swing high)
    tolerance_pct = 0.001  # Percentage tolerance for touching the Fibonacci level (0.1%)

    # Nome per visualizzazione
    DISPLAY_NAME = "Fibonacci Strategy"

    # Vincolo di ottimizzazione: n1 deve essere minore di n2
    optimization_constraint: Callable[[pd.Series], bool] | None = None

    # Definizione dei parametri per la UI e l'ottimizzazione
    # Questa lista sovrascrive PARAMS_INFO da CommonStrategy
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "ma_period",
            "type": int,
            "default": 50,
            "lowest": 2,
            "highest": 250,
            "min": 20,
            "max": 100,
            "step": 1,
        },
        {
            "name": "ma_type",
            "type": str,
            "default": "SMA",
            "options": ["SMA", "EMA", "WMA", "RMA"],
        },
        {
            "name": "fib_lookback",
            "type": int,
            "default": 20,
            "lowest": 2,
            "highest": 100,
            "min": 10,
            "max": 30,
            "step": 1,
        },
        {
            "name": "retracement_level",
            "type": float,
            "default": 0.618,
            "options": [0.236, 0.382, 0.5, 0.618, 0.786],
        },
        {
            "name": "tolerance_pct",
            "type": float,
            "default": 0.1,
            "lowest": 0.0,
            "highest": 1.0,
            "min": 0.0,
            "max": 0.5,
            "step": 0.1,
        },
        {
            "name": "sl_factor",
            "type": float,
            "default": 1.0,
            "lowest": 0.00,
            "highest": 15.0,
            "min": 0.5,
            "max": 5.0,
            "step": 0.50,
        },
        {
            "name": "tp_factor",
            "type": float,
            "default": 3.00,
            "lowest": 0.00,
            "highest": 100.00,
            "min": 0.00,
            "max": 5.00,
            "step": 0.50,
        },
    ]

    def init(self) -> None:
        """Initialize indicators and calculate all trading signals vectorially.

        This method is called once at the start of the backtest.
        """
        super().init()
        # open = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # Calculate Simple Moving Average for trend confirmation
        if self.ma_type not in ["SMA", "EMA", "WMA", "RMA"]:
            raise ValueError(f"Tipo di media mobile non valido: {self.ma_type}")

        # Seleziona la funzione della media mobile in base al tipo
        ma_func: Callable = getattr(ta, self.ma_type.lower())  # Usa getattr per ottenere la funzione dinamica

        # Calcola le medie mobili
        self.ma = self.I(ma_func, close, length=self.ma_period)

        # Calculate rolling highs and lows for the entire dataset.
        # Use .shift(1) to ensure these swings are based on *previous* data,
        # avoiding look-ahead bias.
        self.swing_highs_shifted = self.I(
            lambda x: x.rolling(self.fib_lookback).max().shift(1), high, name="SwingHighs", overlay=True
        )
        self.swing_lows_shifted = self.I(
            lambda x: x.rolling(self.fib_lookback).min().shift(1), low, name="SwingLows", overlay=True
        )

        # Calculate the difference between swing high and low.
        # Replace 0 with NaN to avoid division by zero in flat ranges.
        fib_diff = self.swing_highs_shifted - self.swing_lows_shifted
        fib_diff[fib_diff == 0] = np.nan

        # Calculate Fibonacci levels for the entire dataset vectorially.
        self.long_fib_level = self.I(
            lambda sh, fd: sh - (fd * self.retracement_level),
            self.swing_highs_shifted,
            fib_diff,
            name="FibonacciLevel",
            overlay=True,
        )

        # Define tolerance bands around Fibonacci levels (vector Series).
        long_tolerance_low_series = self.long_fib_level * (1 - self.tolerance_pct / 100)
        long_tolerance_high_series = self.long_fib_level * (1 + self.tolerance_pct / 100)

        # Determine main trend direction (boolean Series).
        is_uptrend = self.data.Close > self.ma

        # Condition for touching the Fibonacci level with tolerance (boolean Series).
        # Check if the current candle (from its low to its high) has crossed the tolerance zone.
        long_fib_touch = (self.data.Low <= long_tolerance_high_series) & (self.data.High >= long_tolerance_low_series)

        # Bullish candle (boolean Series).
        bullish_candle = self.data.Close > self.data.Open

        # Bounce confirmation (candle close above the Fibonacci level) (boolean Series).
        long_bounce_confirm = self.data.Close > self.long_fib_level

        # Calculate final buy signals as boolean Series.
        # These will be used in next()
        self.buy_signals_series = self.I(
            lambda a, b, c, d: a & b & c & d,
            is_uptrend,
            long_fib_touch,
            bullish_candle,
            long_bounce_confirm,
            name="EntrySignal",
            overlay=False,
        )

        # No sell signals in a long-only strategy
        # self.sell_signals_series = pd.Series(False, index=self.data.index)

        # Pre-calculate Stop Loss and Take Profit prices for each potential signal.
        # self.sl_prices_series = pd.Series(index=range(len(self.data)), dtype=float)
        # self.tp_prices_series = pd.Series(index=range(len(self.data)), dtype=float)

        # Apply SL/TP only where there are valid buy signals
        if self.sl_factor > 0:
            self.sl_prices_series = self.I(
                lambda sl_shifted, lo: min(sl_shifted, lo) * (1 - self.sl_factor / 100),
                self.swing_lows_shifted,
                self.data.Low,
                name="SLLong",
                overlay=True,
            )
        else:
            self.sl_prices_series = self.I(
                lambda s: pd.Series([None] * len(s), dtype=object), self.swing_lows_shifted, name="SLLong", overlay=True
            )

        if self.tp_factor > 0:
            self.tp_prices_series = self.I(
                lambda sh_shifted, hi: max(sh_shifted, hi) * (1 + self.tp_factor / 100),
                self.swing_highs_shifted,
                self.data.High,
                name="TPLong",
                overlay=True,
            )
        else:
            self.tp_prices_series = self.I(
                lambda s: pd.Series([None] * len(s), dtype=object),
                self.swing_highs_shifted,
                name="TPLong",
                overlay=True,
            )

        # Add a vectorized exit signal: close the position if price crosses the MA
        # This signal applies only to long positions.
        # self.exit_signals_series = crossover(close, self.ma) | crossover(self.ma, close)
        self.exit_signals_series = self.I(
            lambda cl, ma: (pd.Series(ma) > pd.Series(cl)).astype(int).diff().fillna(0).replace(-1, 0),
            close,
            self.ma,
            name="ExitSignals",
            overlay=False,
        )

    def next(self) -> None:
        """Execute trading logic for each new bar.

        Entry and exit logic is managed explicitly here,
        based on the signals pre-calculated in init().
        """
        # Exit logic for open positions
        if self.position:
            # If there is an exit signal on the current bar, close the position
            if self.exit_signals_series[-1]:
                self.position.close()
                return

        # Entry logic (only if there are no open positions)
        if not self.position:
            # If there is a buy signal on the current bar
            if self.buy_signals_series[-1]:
                # Place the buy order with pre-calculated SL and TP for this bar.
                self.buy(sl=self.sl_prices_series[-1], tp=self.tp_prices_series[-1])
