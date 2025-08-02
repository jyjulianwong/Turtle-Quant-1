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
from turtle_quant_1.strategies.engine import StrategyEngine
from turtle_quant_1.strategies.momentum import (
    LinearRegression,
    MovingAverageCrossover,
    RelativeStrengthIndex,
)
from turtle_quant_1.strategies.mean_reversion import BollingerBand

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_cases() -> List[BacktestingTestCase]:
    """Create predefined test cases for backtesting."""
    test_cases = []

    linear_regression = LinearRegression()
    moving_average_crossover = MovingAverageCrossover()
    relative_strength_index = RelativeStrengthIndex()
    bollinger_band = BollingerBand()

    # Test Case 1: Zero capital test
    strategy_engine_aggressive = StrategyEngine(
        strategies=[
            linear_regression,
            moving_average_crossover,
            relative_strength_index,
            bollinger_band,
        ],
        buy_unit_threshold=0.2,
        sell_threshold=-0.2,
    )

    test_cases.append(
        BacktestingTestCase(
            name="zero_capital",
            description="Test with $0 starting capital to verify no BUY orders are executed",
            strategy_engine=strategy_engine_aggressive,
            initial_capital=0.0,
            expected_results={
                "total_transactions": 0
            },  # Expect no transactions with $0
        )
    )

    # Test Case 2: Standard backtesting with moderate capital
    strategy_engine = StrategyEngine(
        strategies=[
            linear_regression,
            moving_average_crossover,
            relative_strength_index,
            bollinger_band,
        ],
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
