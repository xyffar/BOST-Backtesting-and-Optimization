# strategies/strategy_rsi.py

from typing import Any, Callable, Dict, List, Optional  # Added List, Dict

import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia basata sull'RSI
class RSIStrategy(CommonStrategy):
    # Parametri specifici per questa strategia
    rsi_period: int = 14
    oversold_level: int = 30
    overbought_level: int = 70

    # Nome per visualizzazione
    DISPLAY_NAME = "RSI"

    # Vincolo di ottimizzazione: oversold_level deve essere minore di overbought_level
    optimization_constraint: Optional[Callable[[pd.Series], bool]] = (
        lambda s: s.oversold_level < s.overbought_level
    )

    # Definizione dei parametri per la UI e l'ottimizzazione
    PARAMS_INFO: List[Dict[str, Any]] = [
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
        # SL/TP parameters
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
        """
        Inizializza la strategia calcolando l'RSI.
        """
        CloseSeries: pd.Series = pd.Series(self.data.Close)
        self.rsi = self.I(ta.rsi, CloseSeries, self.rsi_period)

    def next(self) -> None:
        """
        Implementa la logica di trading basata sull'RSI.
        Permette solo posizioni LONG.
        """
        # Segnale di acquisto: RSI attraversa al di sopra il livello di oversold
        buy_signal: bool = crossover(self.rsi, self.oversold_level)

        # Segnale di chiusura/ribassista: RSI attraversa al di sotto il livello di overbought
        # o in generale scende da un livello di overbought.
        close_signal: bool = crossover(self.overbought_level, self.rsi)

        # Logica di entrata:
        # Se non c'è una posizione aperta e c'è un segnale di acquisto, apri una posizione LONG.
        if not self.position:
            if buy_signal:
                self._buy_long()  # Usa il metodo helper dalla classe base
        # Logica di uscita:
        # Se c'è una posizione aperta e c'è un segnale di chiusura, chiudi la posizione.
        elif close_signal:
            self._close_position()  # Usa il metodo helper dalla classe base
