# strategies/strategy_rsi.py

from typing import Any, Callable, Dict, List, Optional  # Added List, Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover, resample_apply

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia basata sull'RSI
class RsiMultiTimeframeStrategy(CommonStrategy):
    # Parametri specifici per questa strategia
    rsi_p_low_timeframe: int = 14  # Periodo RSI per il timeframe di base
    rsi_p_high_timeframe: int = 21  # Periodo RSI per il timeframe superiore
    oversold_level: int = 30
    overbought_level: int = 70
    # Frequenza del timeframe superiore (es. 'W-FRI' per settimanale, 'M' per mensile)
    # Assicurati che i tuoi dati di input siano sufficientemente granulari per questa risample.
    higher_timeframe = "W-FRI"  # Settimanale, chiusura di venerdì
    # Soglia per l'RSI ad alto timeframe per considerare il momento bullish o bearish
    higher_timeframe_threshold: int = 50
    # Compro quando l'RSI fa crossover rialzista rispetto l'oversold o l'overbought?
    when_to_buy = "Oversold"
    # Nome per visualizzazione
    DISPLAY_NAME = "RSI Multi Timeframe"
    # Vincolo di ottimizzazione: oversold_level deve essere minore di overbought_level
    optimization_constraint: Optional[Callable[[pd.Series], bool]] = (
        lambda s: s.oversold_level < s.overbought_level
    )
    # Definizione dei parametri per la UI e l'ottimizzazione
    PARAMS_INFO: List[Dict[str, Any]] = [
        {
            "name": "rsi_p_low_timeframe",
            "type": int,
            "default": 14,
            "lowest": 2,
            "highest": 50,
            "min": 5,
            "max": 30,
            "step": 1,
        },
        {
            "name": "rsi_p_high_timeframe",
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
        {
            "name": "higher_timeframe_threshold",
            "type": int,
            "default": 50,
            "lowest": 10,
            "highest": 90,
            "min": 40,
            "max": 60,
            "step": 1,
        },
        {
            "name": "when_to_buy",
            "type": str,
            "default": "Oversold",
            "options": ["Oversold", "Overbought"],
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
        self.rsi_base_t = self.I(ta.rsi, CloseSeries, self.rsi_p_low_timeframe)
        self.rsi_sup_t = resample_apply(
            "W-FRI", ta.rsi, self.data.Close, self.rsi_p_high_timeframe
        )
        match self.when_to_buy:
            case "Oversold":
                self.crossover_with_oversold = True
            case "Overbought":
                self.crossover_with_oversold = False
            case _:
                raise ValueError("Value is different from Oversold or Overbought!!")

    def next(self) -> None:
        """
        Viene chiamato ad ogni nuova barra di dati. Implementa la logica di trading.
        """
        # Controlla se ci sono abbastanza dati per entrambi gli indicatori
        # e che i valori RSI non siano NaN (TA-Lib produce NaN per le prime barre)
        if np.isnan(self.rsi_base_t[-1]) or np.isnan(self.rsi_sup_t[-1]):
            return

        # Condizione per il trend rialzista del timeframe superiore: RSI lento sopra 50
        # (Un RSI superiore a 50 indica un momentum rialzista)
        trend_bullish = self.rsi_sup_t[-1] > self.higher_timeframe_threshold

        # Condizione per il trend ribassista del timeframe superiore: RSI lento sotto 50
        # (Un RSI inferiore a 50 indica un momentum ribassista)
        trend_bearish = self.rsi_sup_t[-1] < self.higher_timeframe_threshold

        # Segnale di acquisto:
        # 1. RSI veloce è ipervenduto (sotto lower_bound) E incrocia al rialzo la soglia
        # 2. Il trend del timeframe superiore è rialzista
        # 3. Non abbiamo già una posizione aperta
        if (
            crossover(
                self.rsi_base_t,
                (
                    self.oversold_level
                    if self.crossover_with_oversold
                    else self.overbought_level
                ),
            )
            and trend_bullish
            and not self.position
        ):
            self._buy_long()

        # Segnale di vendita/chiusura:
        # 1. RSI veloce è ipercomprato (sopra upper_bound) E incrocia al ribasso la soglia
        # 2. Il trend del timeframe superiore è ribassista (o non più rialzista)
        # 3. Abbiamo una posizione long aperta
        elif (
            crossover(
                (
                    self.overbought_level
                    if self.crossover_with_oversold
                    else self.oversold_level
                ),
                self.rsi_base_t,
            )
            and trend_bearish
            and self.position.is_long
        ):
            self._close_position()  # Usa il metodo helper dalla classe base                self._buy_long() # Usa il metodo helper dalla classe base
