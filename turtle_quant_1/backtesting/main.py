"""Main entry point for backtesting."""

import logging
from typing import List

from turtle_quant_1.backtesting.test_runner import (
    BacktestingTestCase,
    BacktestingTestRunner,
)
from turtle_quant_1.strategies.engine import StrategyEngine
from turtle_quant_1.strategies.momentum.linear_regression_strategy import (
    LinearRegressionStrategy,
)
from turtle_quant_1.strategies.momentum.moving_average_crossover_strategy import (
    MovingAverageCrossoverStrategy,
)
from turtle_quant_1.strategies.momentum.relative_strength_index_strategy import (
    RelativeStrengthIndexStrategy,
)
from turtle_quant_1.strategies.mean_reversion.bollinger_band_strategy import (
    BollingerBandStrategy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_cases() -> List[BacktestingTestCase]:
    """Create predefined test cases for backtesting."""
    test_cases = []

    linear_regression_strategy = LinearRegressionStrategy()
    moving_average_crossover_strategy = MovingAverageCrossoverStrategy()
    relative_strength_index_strategy = RelativeStrengthIndexStrategy()
    bollinger_band_strategy = BollingerBandStrategy()

    # Test Case 1: Zero capital test
    strategy_engine_aggressive = StrategyEngine(
        strategies=[
            linear_regression_strategy,
            moving_average_crossover_strategy,
            relative_strength_index_strategy,
            bollinger_band_strategy,
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
        buy_threshold=0.2,
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
            linear_regression_strategy,
            moving_average_crossover_strategy,
            relative_strength_index_strategy,
            bollinger_band_strategy,
        ],
        weights=[1.0, 1.0, 1.0, 1.0],
        buy_threshold=0.2,
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
    results = runner.run_all_tests()

    logger.info("All backtesting tests completed!")
