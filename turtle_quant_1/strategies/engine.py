"""Strategy engine for aggregating multiple strategies and generating trading signals."""

import importlib
import inspect
import pkgutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Type

import pandas as pd

from turtle_quant_1.strategies import mean_reversion, momentum
from turtle_quant_1.strategies.base import (
    BaseStrategy,
    BaseStrategyEngine,
    Signal,
    SignalAction,
)


def _run_strategy_process(
    strategy_type: Type[BaseStrategy],
    strategy_params: Dict[str, Any],
    weight: float,
    data: pd.DataFrame,
    symbol: str,
) -> Tuple[str, float, float]:
    """Process a single strategy in a separate process and return its score.

    This function is defined at module level to ensure it's pickle-able for multiprocessing.

    Args:
        strategy_type: Strategy class to instantiate
        strategy_params: Parameters to initialize the strategy
        weight: Weight for this strategy
        data: DataFrame with OHLCV data
        symbol: The symbol being analyzed

    Returns:
        Tuple of (strategy name, raw score, weighted score)
    """
    # Instantiate strategy in the worker process
    strategy = strategy_type(**strategy_params)
    strategy_name = strategy_type.__name__
    score = strategy.generate_prediction_score(data, symbol)
    # Ensure score is within bounds
    score = max(-1.0, min(1.0, score))
    weighted_score = score * weight
    return strategy_name, score, weighted_score


class StrategyEngine(BaseStrategyEngine):
    """Strategy engine that aggregates multiple strategies to generate trading signals.

    The engine combines multiple strategy scores using weighted averages and converts
    the final aggregated score to actionable trading signals (BUY, HOLD, SELL).

    This implementation uses multiprocessing for better CPU utilization.
    """

    # Class-level process pool executor
    _executor = ProcessPoolExecutor(max_workers=4)

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

    @classmethod
    def shutdown(cls):
        """Shutdown the process pool executor.

        This should be called when the application is shutting down to ensure proper cleanup.
        """
        cls._executor.shutdown(wait=True)

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

        # Store strategy types and their initialization parameters for multiprocessing
        self._strategy_init_info = []
        for strategy in strategies:
            strategy_type = type(strategy)
            # Extract initialization parameters from the strategy instance
            strategy_params = strategy.__dict__.copy()
            self._strategy_init_info.append((strategy_type, strategy_params))

        self._last_scores = {}

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

    def aggregate_scores(self, data: pd.DataFrame, symbol: str) -> float:
        """Aggregate scores from all strategies using weighted average with parallel processing.

        This method uses multiprocessing for better CPU utilization when calculating
        strategy scores. Each strategy runs in a separate process.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Aggregated score between -1.0 and +1.0.
        """
        total_score = 0.0
        strategy_scores = {}

        # Submit tasks to the process pool
        future_to_strategy = {}
        for (strategy_type, strategy_params), weight in zip(
            self._strategy_init_info, self.weights
        ):
            future = self._executor.submit(
                _run_strategy_process,
                strategy_type,
                strategy_params,
                weight,
                data,
                symbol,
            )
            future_to_strategy[future] = strategy_type.__name__

        # Process completed tasks
        for future in as_completed(future_to_strategy):
            try:
                strategy_name, score, weighted_score = future.result()
                strategy_scores[strategy_name] = score
                total_score += weighted_score
            except Exception as e:
                # Log the error but continue processing other strategies
                print(f"Error processing strategy {future_to_strategy[future]}: {e}")

        # Store scores for use in other methods
        self._last_scores = strategy_scores

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
        strategy_names = [info[0].__name__ for info in self._strategy_init_info]
        strategy_scores = self._last_scores
        strategy_weights = {
            info[0].__name__: weight
            for info, weight in zip(self._strategy_init_info, self.weights)
        }

        # Create and return Signal object
        return Signal(
            strategies=strategy_names,
            scores=strategy_scores,
            weights=strategy_weights,
            action=action,
            score=aggregated_score,
        )

    def get_breakdown(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get individual strategy scores breakdown.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Dictionary mapping strategy names to their individual scores.
        """
        # Run aggregate_scores to populate _last_scores if needed
        self.aggregate_scores(data, symbol)
        return self._last_scores
