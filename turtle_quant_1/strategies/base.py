"""Base classes for trading strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.preprocessing import normalize
import numpy as np


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


class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, name: str):
        """Initialize the strategy.

        Args:
            name: Name of the strategy.
        """
        self.name = name

    @abstractmethod
    def generate_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a trading score for a symbol based on market data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            Score between -1.0 (strong sell) and +1.0 (strong buy).
            0.0 represents hold/neutral.
        """
        pass

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
        weights: Optional[List[float]] = None,
        buy_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        """Initialize the strategy engine.

        Args:
            strategies: List of strategies to aggregate.
            weights: Optional list of weights for each strategy.
                    If None, equal weights are used.
        """
        if not strategies:
            raise ValueError("At least one strategy is required")

        self.strategies = strategies

        if weights is None:
            # Equal weights
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("Number of weights must match number of strategies")
            if abs(sum(weights) - 1.0) > 1e-6:
                # Convert to 2D array because sklearn expects 2D input
                weights_array = np.array(weights).reshape(1, -1)
                # Apply L1 normalization
                normalized_array = normalize(weights_array, norm="l1")
                # Flatten back to 1D
                weights = normalized_array.flatten().tolist()

            self.weights = weights

        if not (-1.0 <= sell_threshold <= buy_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy: -1.0 <= sell_threshold <= buy_threshold <= 1.0"
            )

        self.buy_threshold = buy_threshold
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
            score = strategy.generate_score(data, symbol)
            breakdown[strategy.name] = max(-1.0, min(1.0, score))
        return breakdown
