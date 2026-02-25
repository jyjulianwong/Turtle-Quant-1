"""Test runner for backtesting engine."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from turtle_quant_1.backtesting.engine import BacktestingEngine
from turtle_quant_1.backtesting.models import TestCaseResponse, TestCaseResults
from turtle_quant_1.config import (
    BACKTESTING_MAX_LOOKBACK_DAYS,
    BACKTESTING_MAX_LOOKFORWARD_DAYS,
    BACKTESTING_SYMBOLS,
    MAX_HISTORY_DAYS,
)
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
    symbols: List[str] = field(default_factory=lambda: BACKTESTING_SYMBOLS)
    max_history_days: int = field(default_factory=lambda: MAX_HISTORY_DAYS)
    max_lookback_days: int = field(
        default_factory=lambda: BACKTESTING_MAX_LOOKBACK_DAYS
    )
    max_lookforward_days: int = field(
        default_factory=lambda: BACKTESTING_MAX_LOOKFORWARD_DAYS
    )
    expected_results: Optional[Dict[str, Any]] = None


class BacktestingTestRunner:
    """Runner for executing and reporting backtesting test cases."""

    def __init__(self, test_cases: List[BacktestingTestCase]):
        self.test_cases = test_cases
        self.results: List[TestCaseResponse] = []

    def _run_single_test(self, test_case: BacktestingTestCase) -> TestCaseResults:
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
        self, test_case: BacktestingTestCase, results: TestCaseResults
    ) -> bool:
        """Validate results against expected outcomes.

        Returns:
            True if all validations pass, False otherwise.
        """
        expectations = test_case.expected_results or {}
        all_valid = True

        for key, expected_value in expectations.items():
            if hasattr(results, key):
                actual_value = getattr(results, key)
                if actual_value != expected_value:
                    logger.warning(
                        f"Test case '{test_case.name}': Expected {key}={expected_value}, "
                        f"but got {actual_value}"
                    )
                    all_valid = False
            else:
                logger.warning(
                    f"Test case '{test_case.name}': Expected field '{key}' not found in results"
                )
                all_valid = False

        return all_valid

    def _print_test_results(
        self, test_case: BacktestingTestCase, results: TestCaseResults
    ):
        """Print results for a single test case."""
        print("\n" + "=" * 70)
        print(f"TEST CASE: {test_case.name}")
        print(f"Description: {test_case.description}")
        print("=" * 70)

        # Core metrics
        print(f"Initial Capital: ${results.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results.final_portfolio_value:,.2f}")
        print(f"Total Return: ${results.total_return_dollars:,.2f}")
        print(f"Return Percentage: {results.total_return_percent:.2f}%")

        # Simulation details
        print(
            f"Simulation Period: {results.simulation_start.strftime('%Y-%m-%d')} to {results.simulation_end.strftime('%Y-%m-%d')}"
        )
        print(f"Symbols Traded: {results.symbols_traded}")
        print(f"Total Signals Generated: {results.total_signals:,}")
        print(f"Total Transactions Executed: {results.total_transactions:,}")
        print(f"Final Cash: ${results.final_cash:,.2f}")
        print(f"Final Holdings: {results.final_holdings}")

        # Transaction analysis
        self._print_transaction_analysis(results)

        # Use the convenient summary properties
        print(f"\nPortfolio Summary: {results.portfolio_summary_str}")
        print(f"Return Summary: {results.return_summary_str}")
        print(f"Transaction Summary: {results.transaction_summary_str}")

        print("=" * 70)

    def _print_transaction_analysis(self, results: TestCaseResults):
        """Print detailed transaction analysis."""
        transaction_history = results.transaction_history

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
        passed_tests = [r for r in self.results if r.is_successful]
        failed_tests = [r for r in self.results if r.is_failed]

        print("\n" + "=" * 70)
        print("TEST EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total Test Cases: {len(self.test_cases)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")

        if failed_tests:
            print("\nFailed Tests:")
            for result in failed_tests:
                print(
                    f"  - {result.test_case_name}: {result.error_type} - {result.error_message}"
                )

        if passed_tests:
            print("\nTest Results Summary:")
            for result in passed_tests:
                if result.results:
                    execution_time = (
                        f" ({result.execution_time_seconds:.2f}s)"
                        if result.execution_time_seconds
                        else ""
                    )
                    print(
                        f"  {result.test_case_name:<30} ${result.results.total_return_dollars:>12,.2f}  return ({result.results.total_return_percent:>7.2f}%){execution_time}"
                    )

        # Print test summary using the new summary method
        summary = self.get_run_summary()
        print("\nOverall Test Statistics:")
        print(f"  Success Rate: {summary['success_rate']:.1f}%")
        print(f"  Total Execution Time: {summary['total_execution_time_seconds']:.2f}s")
        print(
            f"  Average Execution Time: {summary['average_execution_time_seconds']:.2f}s per test"
        )

        print("=" * 70)

    def run_tests(self) -> List[TestCaseResponse]:
        """Run all test cases and return results."""
        logger.info(f"Running {len(self.test_cases)} backtesting test cases...")

        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(
                f"Running test case {i}/{len(self.test_cases)}: {test_case.name}"
            )

            try:
                start_time = time.time()
                backtesting_results = self._run_single_test(test_case)
                execution_time = time.time() - start_time

                # Validate expected outcomes if provided
                validation_passed = True
                if test_case.expected_results:
                    validation_passed = self._validate_test_results(
                        test_case, backtesting_results
                    )

                test_result = TestCaseResponse.from_success(
                    test_case_name=test_case.name,
                    test_case_description=test_case.description,
                    results=backtesting_results,
                    execution_time=execution_time,
                )
                self.results.append(test_result)

                if not validation_passed:
                    logger.warning(
                        f"Test case '{test_case.name}' passed execution but failed validation"
                    )

            except Exception as e:
                execution_time = (
                    time.time() - start_time  # pyrefly: ignore[unbound-name]
                    if "start_time" in locals()
                    else None
                )
                logger.error(f"Test case '{test_case.name}' failed: {e}")

                test_result = TestCaseResponse.from_failure(
                    test_case_name=test_case.name,
                    test_case_description=test_case.description,
                    error=e,
                    execution_time=execution_time,
                )
                self.results.append(test_result)
                continue

        self._print_summary()
        return self.results

    def get_successful_responses(self) -> List[TestCaseResponse]:
        """Get successful test case responses."""
        return [result for result in self.results if result.is_successful]

    def get_failed_responses(self) -> List[TestCaseResponse]:
        """Get information about failed test cases."""
        return [result for result in self.results if result.is_failed]

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of all test results."""
        successful = self.get_successful_responses()
        failed = self.get_failed_responses()

        total_execution_time = sum(
            r.execution_time_seconds or 0
            for r in self.results
            if r.execution_time_seconds is not None
        )

        return {
            "total_tests": len(self.results),
            "successful_tests": len(successful),
            "failed_tests": len(failed),
            "success_rate": len(successful) / len(self.results) * 100
            if self.results
            else 0,
            "total_execution_time_seconds": total_execution_time,
            "average_execution_time_seconds": total_execution_time / len(self.results)
            if self.results
            else 0,
        }
