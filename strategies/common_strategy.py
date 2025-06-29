# strategies/common_strategy.py

from typing import (
    Any,  # Added List, Dict for type hinting PARAMS_INFO
    Callable,
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy


# Questa classe funge da base per tutte le strategie, implementando la logica comune
# per la gestione delle posizioni lunghe, stop loss e take profit.
class CommonStrategy(Strategy):
    # Parametri di default per Stop Loss e Take Profit. Saranno sovrascritti dall'UI.
    sl_percent: float = 0.05
    tp_percent: float = 0.10

    # Attributo per definire i vincoli di ottimizzazione specifici della strategia.
    # Deve essere una funzione callable che accetta un oggetto 'stats' e restituisce True/False.
    # Se None, nessun vincolo aggiuntivo verrà applicato oltre a quelli intrinseci.
    optimization_constraint: Optional[Callable[[pd.Series], bool]] = None

    # Nuova variabile di classe per definire i parametri della strategia
    # Questo verrà sovrascritto nelle strategie figlie per i loro parametri specifici
    # e includerà anche sl_percent e tp_percent per l'ottimizzazione.
    PARAMS_INFO: List[Dict[str, Any]] = [
        {
            "name": "sl_percent",
            "type": float,
            "default": 0.05,
            "lowest": 0.00,
            "highest": 0.99,
            "min": 0.005,
            "max": 0.1,
            "step": 0.005,
        },  # Ottimizzabile
        {
            "name": "tp_percent",
            "type": float,
            "default": 0.10,
            "lowest": 0.00,
            "highest": 2.00,
            "min": 0.00,
            "max": 0.5,
            "step": 0.01,
        },  # Ottimizzabile
    ]
    # Nota: I valori predefiniti di 'lowest', 'highest', 'min', 'max' e 'step' per sl_percent e tp_percent
    # sono stati inseriti qui come valori ragionevoli. Potrai modificarli nelle singole strategie
    # se desideri range di ottimizzazione diversi per SL/TP per ogni strategia.

    def init(self) -> None:
        """
        Inizializza la strategia. Questo metodo è chiamato una volta all'inizio del backtest.
        Calcola gli indicatori tecnici necessari.
        """
        # Questo è un placeholder. Le classi derivate devono implementare la propria logica.
        pass

    def next(self) -> None:
        """
        Logica di trading eseguita ad ogni tick (barra) dei dati.
        Implementa la gestione delle posizioni solo LONG, con SL/TP.
        """
        # Questo è un placeholder. Le classi derivate devono implementare la propria logica.
        pass

    # Metodi helper comuni per la gestione delle posizioni (se necessario)
    def _buy_long(self) -> None:
        """
        Esegue un ordine di acquisto LONG con Stop Loss e Take Profit.
        """
        if not self.position:  # Entra solo se non c'è una posizione aperta
            current_close: float = self.data.Close[-1]
            sl_price: float = (
                self.data.Close[-1] * (1 - self.sl_percent)
                if self.sl_percent != 0
                else None
            )
            tp_price: float = (
                self.data.Close[-1] * (1 + self.tp_percent)
                if self.tp_percent != 0
                else None
            )
            self.buy(sl=sl_price, tp=tp_price)
            # print(f"LONG - Entry: {current_close:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")

    def _close_position(self) -> None:
        """
        Chiude qualsiasi posizione aperta.
        """
        if self.position:
            self.position.close()
