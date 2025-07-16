# strategies/strategy_ma_crossover.py

from collections.abc import Callable  # Added List, Dict, Any
from typing import Any, ClassVar, Literal

import pandas as pd
import pandas_ta as ta
from backtesting.lib import crossover

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia Crossover di Medie Mobili
class MACrossoverStrategy(CommonStrategy):
    """A trading strategy based on the crossover of two Moving Averages (MA).

    This strategy extends `CommonStrategy` and implements a simple MA crossover
    system. It generates a buy signal when a faster moving average (n1) crosses
    above a slower moving average (n2), and closes the position when the faster MA
    crosses below the slower MA. It supports various types of moving averages
    (SMA, EMA, WMA, RMA) through the `pandas_ta` library.

    Parameters
    ----------
        n1 (int): The period for the faster Moving Average.
        n2 (int): The period for the slower Moving Average.
        ma_type (Literal["SMA", "EMA", "WMA", "RMA"]): The type of Moving Average to use.
                                                       Defaults to "SMA".

    Attributes
    ----------
        DISPLAY_NAME (ClassVar[str]): A human-readable name for the strategy, "Crossover
            Medie Mobili".
        optimization_constraint (Callable[[pd.Series], bool] | None): A lambda function
            that enforces `n1 < n2` during optimization.
        PARAMS_INFO (ClassVar[list[dict[str, Any]]]): A list of dictionaries defining
            the parameters for UI input and optimization, including their types,
            default values, and ranges. This overrides the base class's `PARAMS_INFO`.

    """

    # Parametri di default specifici per questa strategia (per backtesting.py)
    n1: int = 10
    n2: int = 20
    ma_type: Literal["SMA", "EMA", "WMA", "RMA"] = "SMA"  # Tipo di media mobile (SMA, EMA, WMA, RMA)

    # Nome per visualizzazione
    DISPLAY_NAME = "Crossover Medie Mobili"

    # Vincolo di ottimizzazione: n1 deve essere minore di n2
    optimization_constraint: Callable[[pd.Series], bool] | None = lambda s: s.n1 < s.n2

    # Definizione dei parametri per la UI e l'ottimizzazione
    # Questa lista sovrascrive PARAMS_INFO da CommonStrategy
    PARAMS_INFO: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "n1",
            "type": int,
            "default": 10,
            "lowest": 2,
            "highest": 100,
            "min": 5,
            "max": 50,
            "step": 1,
        },
        {
            "name": "n2",
            "type": int,
            "default": 20,
            "lowest": 3,
            "highest": 100,
            "min": 10,
            "max": 100,
            "step": 1,
        },
        {
            "name": "ma_type",
            "type": str,
            "default": "SMA",
            "options": ["SMA", "EMA", "WMA", "RMA"],
        },
        # SL/TP parameters - Puoi lasciare i valori di default da CommonStrategy o personalizzarli
        {
            "name": "sl_percent",
            "type": float,
            "default": 0.05,
            "lowest": 0.00,
            "highest": 0.99,
            "min": 0.05,
            "max": 0.05,
            "step": 0.005,
        },
        {
            "name": "tp_percent",
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
        """Inizializza la strategia calcolando le medie mobili."""
        # Trasforma in serie
        close_series: pd.Series = pd.Series(self.data.Close)

        # Assicura che il tipo di MA sia valido
        if self.ma_type not in ["SMA", "EMA", "WMA", "RMA"]:
            raise ValueError(f"Tipo di media mobile non valido: {self.ma_type}")

        # Seleziona la funzione della media mobile in base al tipo
        ma_func: Callable = getattr(ta, self.ma_type.lower())  # Usa getattr per ottenere la funzione dinamica

        # Calcola le medie mobili
        self.ma1 = self.I(ma_func, close_series, self.n1)
        self.ma2 = self.I(ma_func, close_series, self.n2)

    def next(self) -> None:
        """Implementa la logica di trading per il crossover delle medie mobili."""
        # Segnale di acquisto: MA più veloce (ma1) attraversa al di sopra MA più lenta (ma2)
        buy_signal: bool = crossover(self.ma1, self.ma2)

        # Segnale di chiusura/ribassista: MA più veloce (ma1) attraversa
        # al di sotto MA più lenta (ma2)
        # Questo funge da condizione di uscita oltre a SL/TP.
        close_signal: bool = crossover(self.ma2, self.ma1)

        # Logica di entrata:
        # Se non c'è una posizione aperta e c'è un segnale di acquisto, apri una posizione LONG.
        if not self.position:
            if buy_signal:
                # La logica di acquisto con SL/TP è gestita dal metodo _buy_long di CommonStrategy
                self._buy_long()
        # Logica di uscita:
        # Se c'è una posizione aperta e c'è un segnale di chiusura, chiudi la posizione.
        elif close_signal:
            self._close_position()
