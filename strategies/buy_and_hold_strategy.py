# strategies/buy_and_hold_strategy.py

from strategies.common_strategy import CommonStrategy  # Import base strategy


class BuyAndHoldStrategy(CommonStrategy):
    """Example strategy implementing a simple Buy & Hold approach.

    Buys at the first available opportunity and holds the position until the end of the backtest.
    """

    DISPLAY_NAME = "Buy & Hold"

    def init(self) -> None:
        """Initialize the strategy.

        No indicators are required for Buy & Hold.
        """
        pass

    def next(self) -> None:
        """Execute the strategy logic at each bar/tick.

        Enters the market only if there is no open position,
        ensuring a single buy at the start of the available data.
        """
        # Entra nel mercato solo se non c'è una posizione aperta.
        # Questo garantisce un singolo acquisto all'inizio dei dati disponibili.
        if not self.position:
            self.buy()
