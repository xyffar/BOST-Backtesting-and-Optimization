# strategies/strategy_breakout.py

from typing import Any, Callable, Dict, List, Literal, Optional  # Added List, Dict, Any

import pandas as pd
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


# La strategia Crossover di Medie Mobili
class BreakoutStrategy(CommonStrategy):
    """
    Strategia di Breakout basata sui massimi/minimi di N periodi come Supporti e Resistenze dinamici.
    Acquista quando il prezzo di chiusura supera il livello di Resistenza (massimo più alto degli ultimi 'lookback_period' barre).
    Vende (o chiude la posizione) quando il prezzo di chiusura scende sotto il livello di Supporto (minimo più basso
    degli ultimi 'lookback_period' barre).
    """

    # Parametri della strategia
    lookback_period = 20  # Numero di barre da considerare per calcolare i livelli di Supporto e Resistenza

    # Nome per visualizzazione
    DISPLAY_NAME = "Breakout supporti/resistenze"

    # Vincolo di ottimizzazione: n1 deve essere minore di n2
    optimization_constraint: Optional[Callable[[pd.Series], bool]] = (
        lambda s: s.lookback_period > 1
    )

    # Definizione dei parametri per la UI e l'ottimizzazione
    # Questa lista sovrascrive PARAMS_INFO da CommonStrategy
    PARAMS_INFO: List[Dict[str, Any]] = [
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
        # SL/TP parameters - Puoi lasciare i valori di default da CommonStrategy o personalizzarli qui
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
        Inizializza la strategia. Pre-calcola i livelli di Resistenza e Supporto dinamici
        sui 'lookback_period' barre precedenti.
        """
        # Il massimo mobile del periodo funge da Resistenza dinamica.
        # Utilizziamo self.I per registrare l'indicatore che verrà aggiornato automaticamente ad ogni tick.
        self.resistance_level = self.I(
            lambda data: pd.Series(data)
            .shift(1)
            .rolling(int(self.lookback_period))
            .max(),
            self.data.High,
        )

        # Il minimo mobile del periodo funge da Supporto dinamico.
        self.support_level = self.I(
            lambda data: pd.Series(data)
            .shift(1)
            .rolling(int(self.lookback_period))
            .min(),
            self.data.Low,
        )

    def next(self) -> None:
        """
        Implementa la logica di trading per ogni nuova barra di dati.
        """
        # Assicurati di avere abbastanza dati per il calcolo dei livelli di Supporto/Resistenza
        if len(self.data.Close) < self.lookback_period:
            return

        current_close = self.data.Close[-1]

        # Accedi ai valori pre-calcolati dei livelli per la barra corrente
        current_resistance = self.resistance_level[-1]
        current_support = self.support_level[-1]

        # Logica di acquisto:
        # Se il prezzo di chiusura corrente supera il livello di Resistenza dinamica
        # E non abbiamo una posizione aperta (per un breakout rialzista).
        if current_close > current_resistance and not self.position:
            self._buy_long()

        # Logica di vendita/chiusura della posizione:
        # Se abbiamo una posizione long aperta
        # E il prezzo di chiusura corrente scende al di sotto del livello di Supporto dinamico
        # (indicando un breakout ribassista o un'inversione di tendenza).
        elif self.position and current_close < current_support:
            self._close_position()
