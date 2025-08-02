"""Strategy engine for aggregating multiple strategies and generating trading signals."""

import importlib
import inspect
import pkgutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from turtle_quant_1.strategies import mean_reversion, momentum
from turtle_quant_1.strategies.base import (
    BaseStrategy,
    BaseStrategyEngine,
    Signal,
    SignalAction,
)


class StrategyEngine(BaseStrategyEngine):
    """Strategy engine that aggregates multiple strategies to generate trading signals.

    The engine combines multiple strategy scores using weighted averages and converts
    the final aggregated score to actionable trading signals (BUY, HOLD, SELL).
    """

    # Class-level thread pool executor
    _executor = ThreadPoolExecutor(max_workers=4)
    _executor_lock = Lock()

    @classmethod
    def _get_strategy_types(cls) -> Dict[str, Type[BaseStrategy]]:
        """Automatically discover all strategy classes in the strategies package."""
        strategy_classes = {}

        # List of modules to search in
        strategy_modules = [mean_reversion, momentum]

        for package in strategy_modules:
            # Walk through all modules in the package
            for _, module_name, _ in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                # Import the module
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseStrategy
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseStrategy)
                        and obj != BaseStrategy
                    ):
                        strategy_classes[name] = obj

        return strategy_classes

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StrategyEngine":
        """Create a strategy engine from a configuration dictionary."""
        # Discover strategy classes on module import
        strategy_types = cls._get_strategy_types()

        strategies = []
        for class_name, params in config["strategies"].items():
            if class_name not in strategy_types:
                raise ValueError(f"Unknown strategy class: {class_name}")
            strategy_class = strategy_types[class_name]
            strategy: BaseStrategy = strategy_class(**params)
            strategies.append(strategy)

        return cls(
            strategies=strategies,
            weights=config["weights"] if "weights" in config else [],
            buy_unit_threshold=config["buy_unit_threshold"]
            if "buy_unit_threshold" in config
            else 0.3,
            sell_threshold=config["sell_threshold"]
            if "sell_threshold" in config
            else -0.3,
        )

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: List[float] = [],
        buy_unit_threshold: float = 0.3,
        sell_threshold: float = -0.3,
    ):
        """Initialize the strategy engine.

        Args:
            strategies: List of strategies to aggregate.
            weights: Optional list of weights for each strategy. If None, equal weights are used.
            buy_unit_threshold: Minimum aggregated score to generate a BUY signal (default: 0.3).
            sell_threshold: Maximum aggregated score to generate a SELL signal (default: -0.3).
        """
        super().__init__(strategies, weights, buy_unit_threshold, sell_threshold)
        self._scores = {}
        self._scores_lock = Lock()  # Add thread-safe lock for scores dictionary

    def get_signal_confidence(self, data: pd.DataFrame, symbol: str) -> float:
        """Post-analysis metric. Calculate confidence level for the generated signal.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Confidence level between 0.0 (low confidence) and 1.0 (high confidence).
        """
        aggregated_score = self.aggregate_scores(data, symbol)

        # Calculate distance from neutral (0) as a proxy for confidence
        # The further from 0, the higher the confidence
        confidence = abs(aggregated_score)

        return min(confidence, 1.0)

    def get_signal_agreement(self, data: pd.DataFrame, symbol: str) -> float:
        """Post-analysis metric. Calculate agreement level between strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Agreement level between 0.0 (complete disagreement) and 1.0 (perfect agreement).
        """
        strategy_scores = self.get_breakdown(data, symbol)
        scores = list(strategy_scores.values())

        if len(scores) <= 1:
            return 1.0  # Perfect agreement with only one strategy

        # Calculate variance of scores as a measure of disagreement
        import numpy as np

        score_variance = np.var(scores)

        # Convert variance to agreement (lower variance = higher agreement)
        # Max possible variance for scores in [-1, 1] is 1.0 (when scores are -1 and +1)
        max_variance = 1.0
        agreement = 1.0 - min(
            score_variance / max_variance, 1.0
        )  # pyrefly: ignore[no-matching-overload]

        return agreement

    def _run_strategy(
        self, strategy: BaseStrategy, weight: float, data: pd.DataFrame, symbol: str
    ) -> Tuple[str, float, float]:
        """Process a single strategy and return its score.

        Args:
            strategy: Strategy instance to process
            weight: Weight for this strategy
            data: DataFrame with OHLCV data
            symbol: The symbol being analyzed

        Returns:
            Tuple of (strategy name, raw score, weighted score)
        """
        strategy_name = type(strategy).__name__
        score = strategy.generate_prediction_score(data, symbol)
        # Ensure score is within bounds
        score = max(-1.0, min(1.0, score))
        weighted_score = score * weight
        return strategy_name, score, weighted_score

    def aggregate_scores(self, data: pd.DataFrame, symbol: str) -> float:
        """Aggregate scores from all strategies using weighted average with parallel processing.

        This method is thread-safe and can be called from multiple threads simultaneously.
        It uses a shared thread pool executor at the class level for efficient resource management.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Aggregated score between -1.0 and +1.0.
        """
        total_score = 0.0

        # Submit tasks to the shared executor
        with self._executor_lock:  # Ensure thread-safe submission of tasks
            future_to_strategy = {
                self._executor.submit(
                    self._run_strategy, strategy, weight, data, symbol
                ): strategy
                for strategy, weight in zip(self.strategies, self.weights)
            }

        # Process completed tasks
        for future in as_completed(future_to_strategy):
            try:
                strategy_name, score, weighted_score = future.result()
                # Thread-safe update of scores dictionary
                with self._scores_lock:
                    self._scores[strategy_name] = score
                total_score += weighted_score
            except Exception as e:
                # Log the error but continue processing other strategies
                print(f"Error processing strategy: {e}")

        return total_score

    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """Generate a trading signal by aggregating all strategies.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Signal object containing action, score, and contributing strategies.
        """
        # Get aggregated score from all strategies
        aggregated_score = self.aggregate_scores(data, symbol)

        # Convert score to trading signal based on thresholds
        if aggregated_score >= self.buy_unit_threshold:
            action = SignalAction.BUY
        elif aggregated_score <= self.sell_threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        # Get strategy names, scores, and weights
        strategy_names = [type(strategy).__name__ for strategy in self.strategies]
        strategy_scores = self._scores
        strategy_weights = {
            type(strategy).__name__: weight
            for strategy, weight in zip(self.strategies, self.weights)
        }

        # Create and return Signal object
        return Signal(
            strategies=strategy_names,
            scores=strategy_scores,
            weights=strategy_weights,
            action=action,
            score=aggregated_score,
        )

    @classmethod
    def shutdown(cls):
        """Shutdown the thread pool executor.

        This should be called when the application is shutting down to ensure proper cleanup.
        """
        cls._executor.shutdown(wait=True)
