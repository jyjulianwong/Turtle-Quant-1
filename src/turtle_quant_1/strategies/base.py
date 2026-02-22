"""Base classes for trading strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class Signal(BaseModel):
    """Trading signal with comprehensive information.

    Attributes:
        strategies: List of strategy names that contributed to this signal.
        action: The trading action (BUY, HOLD, or SELL).
        score: The final aggregated score between -1.0 and +1.0.
    """

    strategies: List[str] = Field(
        ...,
        min_length=1,
        description="List of strategy names that contributed to this signal",
    )
    scores: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping strategy names to their scores",
    )
    weights: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping strategy names to their weights",
    )
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="The final aggregated score between -1.0 and +1.0",
    )  # pyrefly: ignore[no-matching-overload]
    action: SignalAction = Field(
        ..., description="The trading action, i.e. BUY, HOLD, or SELL"
    )
    take_profit_value: Optional[float] = Field(
        default=None,
        description="The take profit value, i.e. the price at which to sell",
    )
    stop_loss_value: Optional[float] = Field(
        default=None,
        description="The stop loss value, i.e. the price at which to sell",
    )


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self):
        """Initialize the strategy."""
        pass

    @abstractmethod
    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on market data.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            Score between -1.0 (strong sell) and +1.0 (strong buy).
            0.0 represents hold/neutral.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate the latest score for prediction for a symbol based on market data.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            Score between -1.0 (strong sell) and +1.0 (strong buy).
            0.0 represents hold/neutral.
        """
        raise NotImplementedError()

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that the input data has the required format.

        Args:
            data: DataFrame to validate.

        Raises:
            ValueError: If data format is invalid.
        """
        required_columns = ["datetime", "Open", "High", "Low", "Close", "Volume"]

        if data.empty:
            raise ValueError("Data cannot be empty")

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < 2:
            raise ValueError("Need at least 2 data points to generate signals")


class BaseStrategyEngine(ABC):
    """Base class for strategy engines that aggregate multiple strategies."""

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Dict[str, float] = {},
        buy_unit_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        """Initialize the strategy engine.

        Args:
            strategies: List of strategies to aggregate.
            weights: Optional dict mapping strategy class names to weights.
                    If empty, equal weights are used for all strategies.
        """
        if not strategies:
            raise ValueError("At least one strategy is required")

        self.strategies = strategies

        # Handle weights as dict mapping strategy class names to weights
        strategy_names = [type(strategy).__name__ for strategy in strategies]

        if not weights:
            # Equal weights for all strategies
            equal_weight = 1.0 / len(strategies)
            self.weights = {name: equal_weight for name in strategy_names}
        else:
            # Check if all strategies have weights specified
            missing_strategies = set(strategy_names) - set(weights.keys())
            if missing_strategies:
                raise ValueError(
                    f"Missing weights for strategies: {missing_strategies}"
                )

            # Check for extra weights
            extra_weights = set(weights.keys()) - set(strategy_names)
            if extra_weights:
                raise ValueError(
                    f"Weights specified for unknown strategies: {extra_weights}"
                )

            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                self.weights = {
                    name: weight / total_weight for name, weight in weights.items()
                }
            else:
                self.weights = weights.copy()

        if not (-1.0 <= sell_threshold <= buy_unit_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: -1.0 <= sell_threshold <= buy_unit_threshold <= 1.0"
            )

        self.buy_unit_threshold = buy_unit_threshold
        self.sell_threshold = sell_threshold

    @abstractmethod
    def aggregate_scores(self, data: pd.DataFrame, symbol: str) -> float:
        """Aggregate scores from all strategies using weighted average.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Aggregated score between -1.0 and +1.0.
        """
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """Generate a trading signal by aggregating all strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Signal object containing action, score, and contributing strategies.
        """
        pass

    def get_breakdown(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get individual scores from each strategy for analysis.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Dictionary mapping strategy names to their scores.
        """
        breakdown = {}
        for strategy in self.strategies:
            score = strategy.generate_prediction_score(data, symbol)
            breakdown[type(strategy).__name__] = max(-1.0, min(1.0, score))
        return breakdown
