"""Base classes for trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class Signal:
    """Trading signal with comprehensive information.

    Attributes:
        strategies: List of strategy names that contributed to this signal.
        action: The trading action (BUY, HOLD, or SELL).
        score: The final aggregated score between -1.0 and +1.0.
    """

    strategies: List[str]
    action: SignalAction
    score: float

    def __post_init__(self):
        """Validate signal data after initialization."""
        if not isinstance(self.strategies, list):
            raise ValueError("strategies must be a list")
        if not self.strategies:
            raise ValueError("strategies list cannot be empty")
        if not isinstance(self.action, SignalAction):
            raise ValueError("action must be a SignalAction enum value")
        if not isinstance(self.score, (int, float)):
            raise ValueError("score must be a number")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("score must be between -1.0 and +1.0")

    @property
    def action_value(self) -> str:
        """Get the string value of the action for backward compatibility."""
        return self.action.value

    def __str__(self) -> str:
        """String representation of the signal."""
        strategies_str = ", ".join(self.strategies)
        return f"Signal(action={self.action.value}, score={self.score:.3f}, strategies=[{strategies_str}])"

    def __repr__(self) -> str:
        """Detailed string representation of the signal."""
        return f"Signal(strategies={self.strategies}, action={self.action}, score={self.score})"


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
        self, strategies: List[BaseStrategy], weights: Optional[List[float]] = None
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
                raise ValueError("Weights must sum to 1.0")
            self.weights = weights

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
