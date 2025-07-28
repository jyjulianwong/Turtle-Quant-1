"""Example usage of the backtesting engine."""

import logging

from turtle_quant_1.config import (
    BACKTESTING_SYMBOLS,
    MAX_HISTORY_DAYS,
    BACKTESTING_MAX_LOOKBACK_DAYS,
    BACKTESTING_MAX_LOOKFORWARD_DAYS,
)
from turtle_quant_1.backtesting import BacktestingEngine
from turtle_quant_1.strategies.engine import StrategyEngine
from turtle_quant_1.strategies.linear_regression_strategy import (
    LinearRegressionStrategy,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_backtesting_default_example():
    """Run a complete backtesting example."""
    logger.info("Starting backtesting example...")

    # Create strategies
    linear_strategy = LinearRegressionStrategy(
        lookback_periods=50, name="LinearRegression_50"
    )

    # Create strategy engine
    strategy_engine = StrategyEngine(
        strategies=[linear_strategy],
        weights=[1.0],  # Single strategy gets full weight
        buy_threshold=0.2,  # Lower threshold for more trading
        sell_threshold=-0.2,  # Higher threshold for more trading
    )

    # Create backtesting engine
    # Start with $10,000 initial capital for demonstration
    backtesting_engine = BacktestingEngine(
        strategy_engine=strategy_engine,
        symbols=BACKTESTING_SYMBOLS,  # Using configured symbols
        initial_capital=10000.0,  # $10,000 starting capital
        max_history_days=MAX_HISTORY_DAYS,  # Load 3 years of data (36 months)
        max_lookback_days=BACKTESTING_MAX_LOOKBACK_DAYS,  # Use 1 year for strategy signals (12 months)
        max_lookforward_days=BACKTESTING_MAX_LOOKFORWARD_DAYS,  # Simulate 1 year of trading (12 months)
    )

    # Run the backtest
    try:
        results = backtesting_engine.run_backtest()

        # Print results
        print("\n" + "=" * 60)
        print("BACKTESTING RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: ${results['total_return_dollars']:,.2f}")
        print(f"Return Percentage: {results['total_return_percent']:.2f}%")
        print(
            f"Simulation Period: {results['simulation_start'].strftime('%Y-%m-%d')} to {results['simulation_end'].strftime('%Y-%m-%d')}"
        )
        print(f"Symbols Traded: {results['symbols_traded']}")
        print(f"Total Signals Generated: {results['total_signals_generated']:,}")
        print(
            f"Total Transactions Executed: {results['total_transactions_executed']:,}"
        )
        print(f"Final Cash: ${results['final_cash']:,.2f}")
        print(f"Final Holdings: {results['final_holdings']}")

        # Show transaction history
        transaction_history = results["transaction_history"]
        if not transaction_history.empty:
            print("\nFirst 10 Transactions:")
            print(transaction_history.head(10).to_string(index=False))

            print("\nLast 10 Transactions:")
            print(transaction_history.tail(10).to_string(index=False))

            # Transaction summary
            buy_transactions = transaction_history[
                transaction_history["action"] == "BUY"
            ]
            sell_transactions = transaction_history[
                transaction_history["action"] == "SELL"
            ]

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
        else:
            print("\nNo transactions were executed during the simulation.")

        print("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        raise


def run_backtesting_zero_capital_example():
    """Run backtesting example starting with zero capital."""
    logger.info("Starting zero-capital backtesting example...")

    # Create a simple strategy
    linear_strategy = LinearRegressionStrategy(
        lookback_periods=20, name="LinearRegression_20"
    )

    # Create strategy engine with more aggressive thresholds
    strategy_engine = StrategyEngine(
        strategies=[linear_strategy],
        weights=[1.0],
        buy_threshold=0.1,  # Very low threshold
        sell_threshold=-0.1,  # Very high threshold
    )

    # Create backtesting engine starting with $0
    backtesting_engine = BacktestingEngine(
        strategy_engine=strategy_engine,
        symbols=BACKTESTING_SYMBOLS,
        initial_capital=0.0,  # Start with $0
        max_history_days=MAX_HISTORY_DAYS,  # Load 2 years of data (24 months)
        max_lookback_days=BACKTESTING_MAX_LOOKBACK_DAYS,  # Use 1 year for strategy signals (12 months)
        max_lookforward_days=BACKTESTING_MAX_LOOKFORWARD_DAYS,  # 6 months simulation
    )

    # Run the backtest
    try:
        results = backtesting_engine.run_backtest()

        print("\n" + "=" * 60)
        print("ZERO-CAPITAL BACKTESTING RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: ${results['total_return_dollars']:,.2f}")
        print(f"Signals Generated: {results['total_signals_generated']:,}")
        print(f"Transactions Executed: {results['total_transactions_executed']:,}")
        print(
            "NOTE: With $0 starting capital, no BUY orders should have been executed."
        )
        print("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Zero-capital backtesting failed: {e}")
        raise


if __name__ == "__main__":
    # Run the main example with initial capital
    main_results = run_backtesting_default_example()

    # Run the zero-capital example
    zero_results = run_backtesting_zero_capital_example()

    print("\nExample completed successfully!")
    print(f"Default example return: ${main_results['total_return_dollars']:,.2f}")
    print(f"Zero-capital example return: ${zero_results['total_return_dollars']:,.2f}")
