"""Moving average crossover strategy implementation."""

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS


class MovingAverageCrossover(BaseStrategy):
    """A strategy that uses moving averages to generate buy and sell signals."""

    def __init__(
        self,
        sma_candles: int = 5,
        lma_candles: int = 180,
    ):
        """Initialize the moving average crossover strategy.

        Args:
            sma_candles: The number of candles to use for the short moving average.
            lma_candles: The number of candles to use for the long moving average.
        """

        super().__init__()
        self.sma_candles = sma_candles
        self.lma_candles = lma_candles

        if sma_candles >= lma_candles:
            raise ValueError(
                f"Short moving average ({sma_candles}) must be less than long moving average ({lma_candles})."
            )

        # TODO: Respect CANDLE_UNIT.
        if lma_candles > BACKTESTING_MAX_LOOKBACK_DAYS * 6 * 0.5:
            raise ValueError(
                f"This strategy relies on too many lookback candles ({lma_candles}) for meaningful evaluation to be done. "
                f"Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on market data.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """

        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()

        sma = data_sorted["Close"].rolling(self.sma_candles).mean()
        lma = data_sorted["Close"].rolling(self.lma_candles).mean()

        scaling_factor = 10.0
        score = (sma - lma) * scaling_factor / data_sorted["Close"]

        return pd.Series(
            data=score.fillna(0).clip(-1, 1).values,
            index=pd.to_datetime(data_sorted["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        return self.generate_historical_scores(data, symbol).iloc[-1]
