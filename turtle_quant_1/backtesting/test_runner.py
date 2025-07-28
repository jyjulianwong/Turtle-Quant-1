"""Test runner for backtesting engine."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from turtle_quant_1.config import (
    BACKTESTING_SYMBOLS,
    MAX_HISTORY_DAYS,
    BACKTESTING_MAX_LOOKBACK_DAYS,
    BACKTESTING_MAX_LOOKFORWARD_DAYS,
)
from turtle_quant_1.backtesting import BacktestingEngine
from turtle_quant_1.strategies.engine import StrategyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestingTestCase:
    """Configuration for a backtesting test case."""

    name: str
    description: str
    strategy_engine: StrategyEngine
    initial_capital: float
    symbols: List[str] = BACKTESTING_SYMBOLS
    max_history_days: int = MAX_HISTORY_DAYS
    max_lookback_days: int = BACKTESTING_MAX_LOOKBACK_DAYS
    max_lookforward_days: int = BACKTESTING_MAX_LOOKFORWARD_DAYS
    expected_results: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default values after initialization."""
        pass


class BacktestingTestRunner:
    """Runner for executing and reporting backtesting test cases."""

    def __init__(self, test_cases: List[BacktestingTestCase]):
        self.test_cases = test_cases
        self.results = []

    def _run_single_test(self, test_case: BacktestingTestCase) -> Dict[str, Any]:
        """Run a single test case."""
        # Create backtesting engine
        backtesting_engine = BacktestingEngine(
            strategy_engine=test_case.strategy_engine,
            symbols=test_case.symbols,
            initial_capital=test_case.initial_capital,
            max_history_days=test_case.max_history_days,
            max_lookback_days=test_case.max_lookback_days,
            max_lookforward_days=test_case.max_lookforward_days,
        )

        # Run the backtest
        results = backtesting_engine.run_backtest()

        # Print results for this test case
        self._print_test_results(test_case, results)

        return results

    def _validate_test_results(
        self, test_case: BacktestingTestCase, results: Dict[str, Any]
    ):
        """Validate results against expected outcomes."""
        expectations = test_case.expected_results or {}

        for key, expected_value in expectations.items():
            if key in results:
                actual_value = results[key]
                if actual_value != expected_value:
                    logger.warning(
                        f"Test case '{test_case.name}': Expected {key}={expected_value}, "
                        f"but got {actual_value}"
                    )

    def _print_test_results(
        self, test_case: BacktestingTestCase, results: Dict[str, Any]
    ):
        """Print results for a single test case."""
        print("\n" + "=" * 70)
        print(f"TEST CASE: {test_case.name}")
        print(f"Description: {test_case.description}")
        print("=" * 70)

        # Core metrics
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: ${results['total_return_dollars']:,.2f}")
        print(f"Return Percentage: {results['total_return_percent']:.2f}%")

        # Simulation details
        print(
            f"Simulation Period: {results['simulation_start'].strftime('%Y-%m-%d')} to {results['simulation_end'].strftime('%Y-%m-%d')}"
        )
        print(f"Symbols Traded: {results['symbols_traded']}")
        print(f"Total Signals Generated: {results['total_signals']:,}")
        print(f"Total Transactions Executed: {results['total_transactions']:,}")
        print(f"Final Cash: ${results['final_cash']:,.2f}")
        print(f"Final Holdings: {results['final_holdings']}")

        # Transaction analysis
        self._print_transaction_analysis(results)

        print("=" * 70)

    def _print_transaction_analysis(self, results: Dict[str, Any]):
        """Print detailed transaction analysis."""
        transaction_history = results["transaction_history"]

        if transaction_history.empty:
            print("\nNo transactions were executed during the simulation.")
            return

        # Basic transaction stats
        buy_transactions = transaction_history[transaction_history["action"] == "BUY"]
        sell_transactions = transaction_history[transaction_history["action"] == "SELL"]

        print("\nTransaction Summary:")
        print(f"  Total Buy Orders: {len(buy_transactions)}")
        print(f"  Total Sell Orders: {len(sell_transactions)}")
        print(
            f"  Total Volume Traded: ${transaction_history['total_value'].sum():,.2f}"
        )

        if not buy_transactions.empty:
            print(f"  Average Buy Price: ${buy_transactions['price'].mean():.2f}")
        if not sell_transactions.empty:
            print(f"  Average Sell Price: ${sell_transactions['price'].mean():.2f}")

        # Show sample transactions
        if len(transaction_history) > 0:
            print(f"\nFirst {min(5, len(transaction_history))} Transactions:")
            print(transaction_history.head(5).to_string(index=False))

            if len(transaction_history) > 5:
                print(f"\nLast {min(5, len(transaction_history))} Transactions:")
                print(transaction_history.tail(5).to_string(index=False))

    def _print_summary(self):
        """Print overall test summary."""
        passed_tests = [r for r in self.results if r["status"] == "PASSED"]
        failed_tests = [r for r in self.results if r["status"] == "FAILED"]

        print("\n" + "=" * 70)
        print("TEST EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total Test Cases: {len(self.test_cases)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")

        if failed_tests:
            print("\nFailed Tests:")
            for result in failed_tests:
                print(f"  - {result['test_case'].name}: {result['error']}")

        if passed_tests:
            print("\nTest Results Summary:")
            for result in passed_tests:
                test_case = result["test_case"]
                print(
                    f"  {test_case.name}:\t${result['total_return_dollars']:,.2f}\treturn ({result['total_return_percent']:.2f}%)"
                )

        print("=" * 70)

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all test cases and return results."""
        logger.info(f"Running {len(self.test_cases)} backtesting test cases...")

        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(
                f"Running test case {i}/{len(self.test_cases)}: {test_case.name}"
            )

            try:
                result = self._run_single_test(test_case)
                result["test_case"] = test_case
                result["status"] = "PASSED"
                self.results.append(result)

                # Validate expected outcomes if provided
                if test_case.expected_results:
                    self._validate_test_results(test_case, result)

            except Exception as e:
                logger.error(f"Test case '{test_case.name}' failed: {e}")
                self.results.append(
                    {"test_case": test_case, "status": "FAILED", "error": str(e)}
                )
                continue

        self._print_summary()
        return self.results
