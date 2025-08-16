"""Main entry point for backtesting.

This script contains custom backtesting test cases to verify the backtesting engine.
For implementation details, see runner.py.
"""

import logging
from typing import List

from turtle_quant_1.backtesting.runner import (
    BacktestingTestCase,
    BacktestingTestRunner,
)
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.engine import StrategyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_cases() -> List[BacktestingTestCase]:
    """Create predefined test cases for backtesting."""
    test_cases = []

    # Test Case 1: Standard backtesting with moderate capital
    # pyrefly: ignore
    strategies = [strategy_type() for strategy_type in BaseStrategy.__subclasses__()]
    logger.info(
        f"Found {len(strategies)} strategies: {[type(strategy).__name__ for strategy in strategies]}"
    )

    weights = {}

    strategy_engine = StrategyEngine(
        strategies=strategies,
        weights=weights,
        buy_unit_threshold=0.2,
        sell_threshold=-0.2,
    )

    test_cases.append(
        BacktestingTestCase(
            name="default",
            description="Standard backtesting with $10,000 initial capital",
            strategy_engine=strategy_engine,
            initial_capital=10000.0,
        )
    )

    return test_cases


if __name__ == "__main__":
    """Main function to run all backtesting tests."""
    # Create test cases
    test_cases = create_test_cases()

    # Create and run test runner
    runner = BacktestingTestRunner(test_cases)
    results = runner.run_tests()

    logger.info("All backtesting tests completed!")
