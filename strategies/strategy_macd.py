# strategies/strategy_macd.py

from typing import Any, Callable, Dict, List, Optional  # Added List, Dict

import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia basata sul MACD
class MACDStrategy(CommonStrategy):
    # Parametri specifici per questa strategia
    n_fast: int = 12
    n_slow: int = 26
    n_signal: int = 9

    # Nome per visualizzazione
    DISPLAY_NAME = "MACD"

    optimization_constraint: Optional[Callable[[pd.Series], bool]] = (
        lambda s: s.n_fast < s.n_slow
    )

    # Definizione dei parametri per la UI e l'ottimizzazione
    PARAMS_INFO: List[Dict[str, Any]] = [
        {
            "name": "n_fast",
            "type": int,
            "default": 12,
            "lowest": 2,
            "highest": 50,
            "min": 5,
            "max": 30,
            "step": 1,
        },
        {
            "name": "n_slow",
            "type": int,
            "default": 26,
            "lowest": 5,
            "highest": 100,
            "min": 15,
            "max": 60,
            "step": 1,
        },
        {
            "name": "n_signal",
            "type": int,
            "default": 9,
            "lowest": 1,
            "highest": 30,
            "min": 5,
            "max": 20,
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
        Inizializza la strategia calcolando il MACD.
        """
        CloseSeries: pd.Series = pd.Series(self.data.Close)
        # Calcola MACD, MACD_H (histogram) e MACD_S (signal line) usando pandas_ta
        # pandas_ta restituisce una tupla o DataFrame con le colonne nominate, es. MACD_12_26_9, MACDH_12_26_9, MACDS_12_26_9
        self.macd, _, self.signal = self.I(
            ta.macd,
            CloseSeries,
            fast=self.n_fast,
            slow=self.n_slow,
            signal=self.n_signal,
            append=False,
        )

    def next(self) -> None:
        """
        Implementa la logica di trading basata sul MACD.
        Permette solo posizioni LONG.
        """
        # Segnale di acquisto: La linea MACD attraversa al di sopra la linea del segnale
        buy_signal: bool = crossover(self.macd, self.signal)

        # Segnale di chiusura/ribassista: La linea MACD attraversa al di sotto la linea del segnale
        close_signal: bool = crossover(self.signal, self.macd)

        # Logica di entrata:
        # Se non c'è una posizione aperta e c'è un segnale di acquisto, apri una posizione LONG.
        if not self.position:
            if buy_signal:
                self._buy_long()  # Usa il metodo helper dalla classe base
        # Logica di uscita:
        # Se c'è una posizione aperta e c'è un segnale di chiusura, chiudi la posizione.
        elif close_signal:
            self._close_position()  # Usa il metodo helper dalla classe base
