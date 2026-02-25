"""Strategy engine for aggregating multiple strategies and generating trading signals."""

import atexit
import importlib
import inspect
import logging
import pkgutil
import signal
import threading
import weakref
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

from turtle_quant_1.config import CANDLE_UNIT, MAX_WORKERS
from turtle_quant_1.strategies import candlesticks, mean_reversion, momentum
from turtle_quant_1.strategies.base import (
    BaseStrategy,
    BaseStrategyEngine,
    Signal,
    SignalAction,
)
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.helpers import calc_atr_value

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    if np.isnan(score):
        raise ValueError(f"Score is NaN for strategy {strategy_name}")
    weighted_score = score * weight
    return strategy_name, score, weighted_score


class _ExecutorManager:
    """Manages ProcessPoolExecutor instances to avoid resource conflicts."""

    def __init__(self):
        self._executors = weakref.WeakSet()
        self._lock = threading.Lock()

        # Register cleanup on process exit
        atexit.register(self._cleanup_executors)

        # Register signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (ValueError, OSError) as e:
            # Signal handling might not be available in all contexts (e.g., threads)
            logger.error(f"Error registering signal handler: {e}")

    def create_executor(self, n_workers: int = MAX_WORKERS) -> ProcessPoolExecutor:
        """Create a new ProcessPoolExecutor instance."""
        with self._lock:
            executor = ProcessPoolExecutor(max_workers=n_workers)
            self._executors.add(executor)
            return executor

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._cleanup_executors()

    def _cleanup_executors(self):
        """Cleanup all managed executors."""
        try:
            with self._lock:
                # Create a list to avoid modifying WeakSet during iteration
                executors_to_cleanup = list(self._executors)
                for executor in executors_to_cleanup:
                    try:
                        # Force shutdown without waiting to avoid hanging during interpreter shutdown
                        executor.shutdown(wait=False)
                    except Exception as e:
                        logger.error(f"Error shutting down executor: {e}")
        except Exception as e:
            # Ignore any errors during global cleanup
            logger.error(f"Error during global cleanup: {e}")


# Global executor manager
_executor_manager = _ExecutorManager()


class StrategyEngine(BaseStrategyEngine):
    """Strategy engine that aggregates multiple strategies to generate trading signals.

    The engine combines multiple strategy scores using weighted averages and converts
    the final aggregated score to actionable trading signals (BUY, HOLD, SELL).

    This implementation uses instance-level multiprocessing for thread-safety and
    proper resource management in multi-process environments.
    """

    @classmethod
    def _get_strategy_types(cls) -> Dict[str, Type[BaseStrategy]]:
        """Automatically discover all strategy classes in the strategies package."""
        strategy_classes = {}

        # List of modules to search in
        strategy_modules = [candlesticks, mean_reversion, momentum]

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
            weights=config.get("weights", {}),
            buy_unit_threshold=config.get("buy_unit_threshold", 0.3),
            sell_threshold=config.get("sell_threshold", -0.3),
        )

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Dict[str, float] = {},
        buy_unit_threshold: float = 0.3,
        sell_threshold: float = -0.3,
        n_workers: int = MAX_WORKERS,
    ):
        """Initialize the strategy engine.

        Args:
            strategies: List of strategies to aggregate.
            weights: Optional dict mapping strategy class names to weights. If empty, equal weights are used.
            buy_unit_threshold: Minimum aggregated score to generate a BUY signal (default: 0.3).
            sell_threshold: Maximum aggregated score to generate a SELL signal (default: -0.3).
            n_workers: Maximum number of worker processes for parallel execution (default: MAX_WORKERS).
        """
        super().__init__(strategies, weights, buy_unit_threshold, sell_threshold)

        # Store strategy types and their initialization parameters for multiprocessing
        self._strategy_init_info = []
        for strategy in strategies:
            strategy_type = type(strategy)

            # Get __init__ parameter names (excluding 'self')
            init_sig = inspect.signature(strategy.__init__)
            param_names = [p for p in init_sig.parameters if p != "self"]

            # Extract only those parameters from the instance
            strategy_params = {
                name: getattr(strategy, name, None) for name in param_names
            }

            self._strategy_init_info.append((strategy_type, strategy_params))

        self._last_scores = {}

        # Create instance-level executor for process safety
        self._n_workers = n_workers
        self._executor = None
        self._executor_lock = threading.Lock()

        if self._n_workers == 0:
            logger.info("Running strategies sequentially...")
        else:
            logger.info(
                f"Running strategies in parallel with {self._n_workers} workers..."
            )

        atexit.register(self._cleanup_executor)

    def _cleanup_executor(self):
        """Clean up the executor instance."""
        if self._executor is not None:
            with self._executor_lock:
                if self._executor is not None:
                    try:
                        self._executor.shutdown(wait=False)  # Don't wait during cleanup
                    except Exception as e:
                        logger.error(f"Error shutting down executor: {e}")
                    finally:
                        self._executor = None

    def _get_executor(self) -> ProcessPoolExecutor:
        """Get or create the ProcessPoolExecutor for this instance."""
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:  # Double-check locking
                    self._executor = _executor_manager.create_executor(
                        self._n_workers
                    )  # pyrefly: ignore[bad-assignment]
        return self._executor  # pyrefly: ignore[bad-return]

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
        agreement = 1.0 - min(float(score_variance) / max_variance, 1.0)

        return agreement

    def aggregate_scores(self, data: pd.DataFrame, symbol: str) -> float:
        """Aggregate scores from all strategies using weighted average with parallel processing.

        This method uses instance-level multiprocessing for better CPU utilization and
        process safety when multiple StrategyEngine instances are used simultaneously.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Aggregated score between -1.0 and +1.0.
        """
        total_score = 0.0
        strategy_scores = {}

        if self._n_workers == 0:
            for strategy_type, strategy_params in self._strategy_init_info:
                strategy_name = strategy_type.__name__
                weight = self.weights[strategy_name]
                strategy_name, score, weighted_score = _run_strategy_process(
                    strategy_type, strategy_params, weight, data, symbol
                )
                strategy_scores[strategy_name] = score
                total_score += weighted_score
        else:
            # Get the executor for this instance
            executor = self._get_executor()

            # Submit tasks to the process pool
            future_to_strategy = {}
            for strategy_type, strategy_params in self._strategy_init_info:
                strategy_name = strategy_type.__name__
                weight = self.weights[strategy_name]
                future = executor.submit(
                    _run_strategy_process,
                    strategy_type,
                    strategy_params,
                    weight,
                    data,
                    symbol,
                )
                future_to_strategy[future] = strategy_name

            # Process completed tasks
            for future in as_completed(future_to_strategy):
                try:
                    strategy_name, score, weighted_score = future.result()
                    strategy_scores[strategy_name] = score
                    total_score += weighted_score
                except Exception as e:
                    # Log the error but continue processing other strategies
                    logger.error(
                        f"Error processing strategy {future_to_strategy[future]}: {e}"
                    )

        # Store scores for use in other methods
        self._last_scores = strategy_scores

        return total_score

    def calc_take_profit_value(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate the take profit value for a given symbol.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.
        """
        raise NotImplementedError(
            "Take profit value calculation is not implemented yet"
        )

    def calc_stop_loss_value(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate the stop loss value for a given symbol.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.
        """
        curr_price = data.iloc[-1]["Close"]
        k = 3  # Multiplier
        atr = calc_atr_value(
            data=data,
            lookback=convert_units(3, "MONTH", CANDLE_UNIT),
            return_log_space=False,
        )
        return curr_price - k * atr

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

        # Create and return Signal object
        return Signal(
            strategies=[info[0].__name__ for info in self._strategy_init_info],
            scores=self._last_scores,
            weights=self.weights,
            action=action,
            score=aggregated_score,
            stop_loss_value=self.calc_stop_loss_value(data, symbol)
            if action == SignalAction.BUY
            else None,
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
