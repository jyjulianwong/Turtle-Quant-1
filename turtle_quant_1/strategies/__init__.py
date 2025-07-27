"""The strategies module for implementing trading strategies and signal generation.

This module provides a flexible framework for implementing trading strategies that analyze
market data and generate trading signals. The main components are:

1. BaseStrategy: Abstract base class for implementing individual trading strategies
2. BaseStrategyEngine: Abstract base class for strategy aggregation engines
3. LinearRegressionStrategy: Simple linear regression-based strategy
4. StrategyEngine: Concrete implementation that aggregates multiple strategies
5. Signal: Enum for trading signals (BUY, HOLD, SELL)

Usage Example:
    ```python
    from turtle_quant_1.strategies import LinearRegressionStrategy, StrategyEngine, Signal, SignalAction
    from turtle_quant_1.data_processing import DataProcessor

    # Create a strategy
    strategy = LinearRegressionStrategy(lookback_periods=20)

    # Create an engine with the strategy
    engine = StrategyEngine(
        strategies=[strategy],
        buy_threshold=0.3,
        sell_threshold=-0.3
    )

    # Get market data
    processor = DataProcessor()
    # ... (load data)

    # Generate enhanced signal
    signal = engine.generate_signal(data, "AAPL")

    # Access signal information
    print(f"Action: {signal.action.value}")  # BUY, HOLD, or SELL
    print(f"Score: {signal.score}")         # -1.0 to +1.0
    print(f"Strategies: {signal.strategies}") # List of strategy names

    # Check signal type
    if signal.action == SignalAction.BUY:
        print("Execute buy order")

    # Get detailed analysis
    analysis = engine.analyze_symbol(data, "AAPL")
    ```

The strategies module integrates with the data_processing layer to consume OHLCV market data
and produces trading signals that can be used by the execution layer.
"""

from turtle_quant_1.strategies.base import (
    BaseStrategy,
    BaseStrategyEngine,
    Signal,
    SignalAction,
)
from turtle_quant_1.strategies.engine import StrategyEngine
from turtle_quant_1.strategies.linear_regression_strategy import (
    LinearRegressionStrategy,
)

__all__ = [
    "BaseStrategy",
    "BaseStrategyEngine",
    "Signal",
    "SignalAction",
    "StrategyEngine",
    "LinearRegressionStrategy",
]
