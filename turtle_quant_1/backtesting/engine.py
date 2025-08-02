"""Backtesting engine for strategy evaluation."""

import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import pytz
import quantstats as qs

from turtle_quant_1.backtesting.portfolio import Portfolio
from turtle_quant_1.backtesting.models import TestCaseResults, PortfolioSummary
from turtle_quant_1.config import (
    BACKTESTING_MAX_LOOKBACK_DAYS,
    BACKTESTING_MAX_LOOKFORWARD_DAYS,
    BACKTESTING_SYMBOLS,
    HOST_TIMEZONE,
    MAX_HISTORY_DAYS,
)
from turtle_quant_1.data_processing.processor import DataProcessor
from turtle_quant_1.strategies.base import BaseStrategyEngine, SignalAction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLOBAL_DATA_CACHE: dict[str, pd.DataFrame] = {}
GLOBAL_DATA_CACHE_LOCK = threading.Lock()


class BacktestingEngine:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        strategy_engine: BaseStrategyEngine,
        symbols: list[str] = BACKTESTING_SYMBOLS,
        initial_capital: float = 0.0,
        max_history_days: int = MAX_HISTORY_DAYS,
        max_lookback_days: int = BACKTESTING_MAX_LOOKBACK_DAYS,
        max_lookforward_days: int = BACKTESTING_MAX_LOOKFORWARD_DAYS,
    ):
        """Initialize the backtesting engine.

        Args:
            strategy_engine: The strategy engine to use for generating signals.
            symbols: List of symbols to backtest. If None, uses BACKTESTING_SYMBOLS.
            initial_capital: Starting capital for the portfolio (default: 0.0).
            max_history_days: Maximum days of history to download.
            max_lookback_days: Days of data to use for strategy signals.
            max_lookforward_days: Days of data to simulate trading on.
        """
        self.strategy_engine = strategy_engine
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_history_days = max_history_days
        self.max_lookback_days = max_lookback_days
        self.max_lookforward_days = max_lookforward_days

        self.portfolio = Portfolio(initial_capital)

        # Initialize data management components
        self.data_processor = DataProcessor(symbols=self.symbols)
        self.data_cache: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            with GLOBAL_DATA_CACHE_LOCK:
                if symbol not in GLOBAL_DATA_CACHE:
                    GLOBAL_DATA_CACHE[symbol] = self._load_data_for_symbol(
                        symbol, impute_data=True
                    )
                self.data_cache[symbol] = GLOBAL_DATA_CACHE[symbol]

        # For quantstats metrics
        self.portfolio_returns: List[Tuple[datetime, float]] = []

        logger.info(f"Initialized backtesting engine for symbols: {self.symbols}")

    def _load_data_for_symbol(
        self, symbol: str, impute_data: bool = True
    ) -> pd.DataFrame:
        """Load data for a symbol using DataMaintainer and DataProcessor.

        Args:
            symbol: Symbol to load data for.
            impute_data: Whether to impute data before loading.

        Returns:
            DataFrame with OHLCV data.
        """
        logger.info(f"Loading data for {symbol}...")

        # Calculate date ranges
        end_date = datetime.now().astimezone(pytz.timezone(HOST_TIMEZONE))
        start_date = end_date - timedelta(days=self.max_history_days)

        try:
            # Load the data using DataProcessor
            data = self.data_processor.load_data(
                symbol, start_date, end_date, impute_data
            )

            if not data.empty:
                logger.info(f"Loaded {len(data)} records for {symbol}")
                # Ensure datetime column is datetime type
                data["datetime"] = pd.to_datetime(data["datetime"])
                return data.sort_values("datetime")
            else:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame(
                    columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
                )

        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

    def _get_lookback_data(
        self, symbol: str, current_index: int, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get lookback data for strategy analysis.

        Args:
            symbol: Symbol to get data for.
            current_index: Current index in the data.
            data: Full data DataFrame.

        Returns:
            DataFrame with lookback data for strategy analysis.
        """
        if current_index < 1:
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Calculate lookback period in data points
        # For hourly data, we want roughly lookback_months * 30 * 24 hours
        lookback_hours = self.max_lookback_days * 24
        start_index = max(0, current_index - lookback_hours)

        return data.iloc[start_index : current_index + 1].copy()

    def _get_simulation_time_range(
        self, symbols: List[str]
    ) -> Tuple[datetime, datetime]:
        """Find the overall time range for simulation across all symbols.

        Args:
            symbols: List of symbols to check.

        Returns:
            Tuple of (start_time, end_time) for the simulation range. Timezone-aware.
        """
        start_times = []
        end_times = []

        for symbol in symbols:
            data = self.data_cache[symbol]
            if not data.empty:
                start_times.append(data["datetime"].min())
                end_times.append(data["datetime"].max())

        if not start_times:
            raise ValueError("No data found for any symbols")

        # Use the latest start time and earliest end time to ensure data availability
        simulation_start = max(start_times)  # Latest start time
        simulation_end = min(end_times)  # Earliest end time

        return simulation_start, simulation_end

    def _generate_simulation_ticks(
        self, start_time: datetime, end_time: datetime
    ) -> List[datetime]:
        """Generate simulation ticks from start_time to end_time.

        Args:
            start_time: Start of simulation period. Timezone-aware.
            end_time: End of simulation period. Timezone-aware.

        Returns:
            List of datetime objects representing simulation ticks. Timezone-aware.
        """
        # Start from the beginning of the start date (00:00:00)
        tick_start = start_time.replace(hour=0, minute=0, second=0)

        # End at the end of the end date (23:59:59)
        tick_end = end_time.replace(hour=23, minute=59, second=59)

        ticks = []
        curr_tick = tick_start

        while curr_tick <= tick_end:
            # TODO: Respect CANDLE_UNIT.
            # NOTE: Simulate the schedule of the production host environment.
            ticks.append(curr_tick.replace(hour=9, minute=0, second=0))
            ticks.append(curr_tick.replace(hour=12, minute=0, second=0))
            ticks.append(curr_tick.replace(hour=15, minute=0, second=0))
            ticks.append(curr_tick.replace(hour=18, minute=0, second=0))
            curr_tick += timedelta(days=1)

        return ticks

    def _execute_signal(
        self,
        symbol: str,
        signal_action: SignalAction,
        price: float,
        timestamp: datetime,
    ) -> bool:
        """Execute a trading signal.

        Args:
            symbol: Symbol to trade.
            signal_action: The signal action (BUY, SELL, HOLD).
            price: Current price.
            timestamp: Current timestamp. Timezone-aware.

        Returns:
            True if a transaction was executed, False otherwise.
        """
        if signal_action == SignalAction.BUY:
            # Buy 100 dollars worth of the symbol if we have enough cash
            units = 100.0 / price
            success = self.portfolio.buy(symbol, units, price, timestamp)
            if success:
                logger.debug(f"BUY: {symbol} at ${price:.2f} on {timestamp}")
            else:
                logger.debug(
                    f"BUY FAILED: Insufficient funds for {symbol} at ${price:.2f}"
                )
            return success

        elif signal_action == SignalAction.SELL:
            # Sell all holdings of this symbol
            success = self.portfolio.sell_holdings(symbol, price, timestamp)
            if success:
                logger.debug(f"SELL ALL: {symbol} at ${price:.2f} on {timestamp}")
            else:
                logger.debug(f"SELL FAILED: No holdings for {symbol}")
            return success

        # HOLD - No action
        return False

    def run_backtest(self) -> TestCaseResults:
        """Run the backtesting simulation.

        Returns:
            BacktestingResults with backtesting results.
        """
        logger.info("Starting backtesting simulation...")

        # Find the overall time range across all symbols
        history_start, history_end = self._get_simulation_time_range(self.symbols)

        # Split into lookback and simulation periods
        lookback_end = history_start + timedelta(days=self.max_lookback_days)
        simulation_start = lookback_end
        simulation_end = min(
            history_end,
            simulation_start + timedelta(days=self.max_lookforward_days),
        )

        logger.info(f"Simulation period: {simulation_start} to {simulation_end}")

        # Generate simulation ticks for the entire simulation period
        simulation_ticks = self._generate_simulation_ticks(
            simulation_start, simulation_end
        )

        logger.info(
            f"Generated {len(simulation_ticks)} simulation ticks for simulation"
        )

        if len(simulation_ticks) < 2:
            raise ValueError("Not enough simulation ticks for meaningful backtesting")

        # Run the simulation using simulation ticks
        total_signals = 0
        total_transactions = 0

        # Get current prices for all symbols at this timestamp
        current_prices = {}

        for i, timestamp in enumerate(simulation_ticks):
            # Generate signals for each symbol
            for symbol in self.symbols:
                # Get full historical data up to this point for strategy analysis
                full_data = self.data_cache[symbol].copy()
                current_data_mask = full_data["datetime"] <= timestamp
                current_full_data = full_data[current_data_mask]

                if len(current_full_data) < 2:
                    continue  # Not enough data for strategy

                # Get lookback data for strategy
                current_index = len(current_full_data) - 1
                lookback_data = self._get_lookback_data(
                    symbol, current_index, current_full_data
                )

                if len(lookback_data) < 2:
                    continue  # Not enough lookback data

                # Get current price (use the closest available price)
                current_row = current_full_data.iloc[-1]
                current_price = current_row["Close"]
                current_prices[symbol] = current_price

                # Generate signal
                signal = self.strategy_engine.generate_signal(lookback_data, symbol)
                total_signals += 1

                # Execute signal
                transaction_executed = self._execute_signal(
                    symbol, signal.action, current_price, timestamp
                )
                if transaction_executed:
                    total_transactions += 1

            # Record portfolio value for returns calculation (once per timestamp)
            if current_prices:  # Only record if we have price data
                portfolio_value = self.portfolio.get_portfolio_value(current_prices)
                self.portfolio_returns.append((timestamp, portfolio_value))

            # Log progress periodically
            if i % 100 == 0:  # Every 100 simulation ticks
                logger.info(
                    f"Progress: {i}/{len(simulation_ticks)} ({100 * i / len(simulation_ticks):.1f}%) | "
                    f"Portfolio Value: ${portfolio_value:.2f}"
                )

        # Calculate final results
        final_prices = {}
        for symbol in self.symbols:
            data = self.data_cache[symbol]
            if not data.empty:
                # Get the last available price for each symbol
                final_data = data[data["datetime"] <= simulation_end]
                if not final_data.empty:
                    final_prices[symbol] = final_data.iloc[-1]["Close"]

        final_portfolio_value = self.portfolio.get_portfolio_value(final_prices)
        total_return = self.portfolio.get_return_dollars(final_prices)

        # Create portfolio summary and results directly
        portfolio_summary = self.portfolio.get_summary(final_prices)

        results = TestCaseResults(
            initial_capital=self.initial_capital,
            final_portfolio_value=final_portfolio_value,
            final_holdings=self.portfolio.get_current_holdings(),
            final_cash=self.portfolio.cash,
            total_return_dollars=total_return,
            total_return_percent=self.portfolio.get_return_percent(final_prices),
            portfolio_summary=PortfolioSummary(**portfolio_summary),
            total_signals=total_signals,
            total_transactions=total_transactions,
            total_simulation_ticks=len(simulation_ticks),
            symbols_traded=self.symbols,
            transaction_history=self.portfolio.get_transaction_history(),
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            metrics=self.get_metrics(benchmark="SPY"),  # TODO: Hard-coded.
        )

        logger.info(
            f"Backtesting complete. Total return: ${total_return:.2f} "
            f"({self.portfolio.get_return_percent(final_prices):.2f}%) | "
            f"Signals generated: {total_signals} | Transactions: {total_transactions}"
        )

        return results

    def get_metrics(self, benchmark: str = "SPY") -> Dict:
        """Calculate metrics for the portfolio performance.

        NOTE: This is currently a wrapper around quantstats metrics.
        TODO: Implement our own metrics.

        Args:
            benchmark: Benchmark symbol to compare against (default: SPY)

        Returns:
            Dictionary containing quantstats metrics
        """
        if len(self.portfolio_returns) < 2:
            logger.warning("Not enough portfolio returns data for metrics")
            logger.warning("Run the backtest first before calculating metrics")
            return {}

        try:
            # Convert portfolio values to returns
            df_returns = pd.DataFrame(
                self.portfolio_returns, columns=["date", "portfolio_value"]
            )

            # Remove duplicate timestamps if any
            df_returns = df_returns.drop_duplicates(subset=["date"], keep="last")

            # Sort by date to ensure proper chronological order
            df_returns = df_returns.sort_values("date")

            # Set index and calculate returns
            df_returns = df_returns.set_index("date")
            df_returns["returns"] = df_returns["portfolio_value"].pct_change().dropna()

            if df_returns["returns"].empty or len(df_returns["returns"]) < 2:
                logger.warning("Insufficient returns data for quantstats metrics")
                return {}

            # Calculate quantstats metrics
            metrics = qs.reports.metrics(
                returns=df_returns["returns"],
                benchmark=benchmark,
                display=False,
            )
            if metrics is None:
                raise ValueError("Quantstats metrics returned None")

            return (
                metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
            )  # pyrefly: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Failed to calculate quantstats metrics: {e}")
            return {}
