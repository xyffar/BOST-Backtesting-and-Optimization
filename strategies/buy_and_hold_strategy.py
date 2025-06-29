# strategies/buy_and_hold_strategy.py

from typing import (
    Any,
    Callable,  # Added List, Dict, Any for type hinting
    Dict,
    List,
    Optional,
)

import pandas as pd  # Needed for type hinting pd.Series in optimization_constraint
from backtesting import Strategy

from strategies.common_strategy import CommonStrategy  # Importa la strategia base


class BuyAndHoldStrategy(CommonStrategy):  # Eredita da CommonStrategy
    """
    Una strategia di esempio che implementa semplicemente un approccio Buy & Hold.
    Compra alla prima opportunità disponibile e mantiene la posizione fino alla fine del backtest.
    """

    # Nome per visualizzazione
    DISPLAY_NAME = "Buy & Hold"

    # optimization_constraint: Optional[Callable[[pd.Series], bool]] = None # Già in CommonStrategy, può essere omesso se non sovrascritto.

    # Questa strategia eredita PARAMS_INFO da CommonStrategy,
    # che include sl_percent e tp_percent.
    # Non ha parametri specifici aggiuntivi da ottimizzare per la sua logica.
    # Quindi non è necessario definire una nuova PARAMS_INFO qui,
    # a meno che non si vogliano range di SL/TP diversi.

    def init(self) -> None:
        # Non sono necessari indicatori per una strategia Buy & Hold.
        pass

    def next(self) -> None:
        # Entra nel mercato solo se non c'è una posizione aperta.
        # Questo garantisce un singolo acquisto all'inizio dei dati disponibili.
        if not self.position:
            self.buy()
