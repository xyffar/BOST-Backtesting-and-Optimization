# strategies/strategy_bollinger_bands.py

from typing import Any, Callable, Dict, List, Optional  # Added List, Dict

import pandas as pd
import pandas_ta as ta
from backtesting import Strategy

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia basata sulle Bande di Bollinger
class BollingerBandsStrategy(CommonStrategy):
    # Parametri di default specifici per questa strategia
    bb_period: int = 20
    bb_dev: float = 2.0

    # Nome per visualizzazione
    DISPLAY_NAME = "Bollinger's Bands"

    optimization_constraint: Optional[Callable[[pd.Series], bool]] = (
        None  # No specific constraint for this strategy
    )

    # Definizione dei parametri per la UI e l'ottimizzazione
    PARAMS_INFO: List[Dict[str, Any]] = [
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
        Inizializza la strategia calcolando le Bande di Bollinger.
        """
        CloseSeries: pd.Series = pd.Series(self.data.Close)
        # Calcola le Bande di Bollinger usando pandas_ta
        # bb_df contiene Lower Band, Middle Band, Upper Band (di solito 'BBL', 'BBM', 'BBU')
        # pandas_ta restituisce un DataFrame con colonne nominate, es. BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        (
            self.bb_lower,
            self.bb_middle,
            self.bb_upper,
            self.bb_bandwidth,
            self.bb_percent,
        ) = self.I(
            ta.bbands,
            CloseSeries,
            length=self.bb_period,
            std=self.bb_dev,
            append=False,  # Non appendere al dataframe originale, backtesting.py lo gestisce
        )

    def next(self) -> None:
        """
        Implementa la logica di trading per le Bande di Bollinger.
        Permette solo posizioni LONG.
        """
        # Segnale di acquisto: Il prezzo di chiusura attraversa al di sotto la banda inferiore dal tick precedente.
        # Condizione semplificata: Close attraversa da sotto la banda inferiore
        buy_signal: bool = (
            self.data.Close[-1] > self.bb_lower[-1]
            and self.data.Close[-2] <= self.bb_lower[-2]
        )

        # Segnale di chiusura/ribassista: Il prezzo di chiusura attraversa al di sopra la banda superiore
        # o chiude al di sotto della banda centrale dopo essere stato sopra.
        close_signal: bool = (
            self.data.Close[-1] < self.bb_upper[-1]
            and self.data.Close[-2] >= self.bb_upper[-2]
        ) or (
            self.data.Close[-1] < self.bb_middle[-1]
            and self.data.Close[-2] >= self.bb_middle[-2]
        )

        # Logica di entrata:
        # Se non c'è una posizione aperta e c'è un segnale di acquisto, apri una posizione LONG.
        if not self.position:
            if buy_signal:
                self._buy_long()  # Usa il metodo helper dalla classe base
        # Logica di uscita:
        # Se c'è una posizione aperta e c'è un segnale di chiusura, chiudi la posizione.
        elif close_signal:
            self._close_position()  # Usa il metodo helper dalla classe base
