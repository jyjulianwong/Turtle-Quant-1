"""Backtesting engine for strategy evaluation."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from turtle_quant_1.backtesting.portfolio import Portfolio
from turtle_quant_1.config import (
    BACKTESTING_MAX_LOOKBACK_MONTHS,
    BACKTESTING_MAX_LOOKFORWARD_MONTHS,
    BACKTESTING_SYMBOLS,
    MAX_HISTORY_MONTHS,
)
from turtle_quant_1.data_processing.processor import DataProcessor
from turtle_quant_1.strategies.base import BaseStrategyEngine, SignalAction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        strategy_engine: BaseStrategyEngine,
        symbols: list[str] = BACKTESTING_SYMBOLS,
        initial_capital: float = 0.0,
        max_history_months: int = MAX_HISTORY_MONTHS,
        max_lookback_months: int = BACKTESTING_MAX_LOOKBACK_MONTHS,
        max_lookforward_months: int = BACKTESTING_MAX_LOOKFORWARD_MONTHS,
    ):
        """Initialize the backtesting engine.

        Args:
            strategy_engine: The strategy engine to use for generating signals.
            symbols: List of symbols to backtest. If None, uses BACKTESTING_SYMBOLS.
            initial_capital: Starting capital for the portfolio (default: 0.0).
            max_history_months: Maximum months of history to download.
            max_lookback_months: Months of data to use for strategy signals.
            max_lookforward_months: Months of data to simulate trading on.
        """
        self.strategy_engine = strategy_engine
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.max_history_months = max_history_months
        self.max_lookback_months = max_lookback_months
        self.max_lookforward_months = max_lookforward_months

        self.portfolio = Portfolio(initial_capital)

        # Initialize data management components
        self.data_processor = DataProcessor(symbols=self.symbols)
        self.data_cache: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            self.data_cache[symbol] = self._load_data_for_symbol(
                symbol, impute_data=True
            )

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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.max_history_months * 30)

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
        lookback_hours = self.max_lookback_months * 30 * 24
        start_index = max(0, current_index - lookback_hours)

        return data.iloc[start_index : current_index + 1].copy()

    def _find_common_time_range(self, symbols: List[str]) -> Tuple[datetime, datetime]:
        """Find the common time range across all symbols.

        Args:
            symbols: List of symbols to check.

        Returns:
            Tuple of (start_time, end_time) for the common range.
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

        common_start = max(start_times)  # Latest start time
        common_end = min(end_times)  # Earliest end time

        return common_start, common_end

    def _find_common_timestamps(
        self, simulation_data: Dict[str, pd.DataFrame]
    ) -> List[datetime]:
        """Find common timestamps across all symbols for synchronized trading.

        Args:
            simulation_data: Dictionary of symbol -> simulation data.

        Returns:
            List of common timestamps sorted chronologically.
        """
        if not simulation_data:
            return []

        # Get timestamps for each symbol
        timestamp_sets = []
        for symbol, data in simulation_data.items():
            if not data.empty:
                timestamps = set(data["datetime"].tolist())
                timestamp_sets.append(timestamps)

        if not timestamp_sets:
            return []

        # Find intersection of all timestamp sets
        common_timestamps = timestamp_sets[0]
        for ts_set in timestamp_sets[1:]:
            common_timestamps = common_timestamps.intersection(ts_set)

        # Convert to sorted list
        return sorted(list(common_timestamps))

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
            timestamp: Current timestamp.

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
            success = self.portfolio.sell_all(symbol, price, timestamp)
            if success:
                logger.debug(f"SELL ALL: {symbol} at ${price:.2f} on {timestamp}")
            else:
                logger.debug(f"SELL FAILED: No holdings for {symbol}")
            return success

        # HOLD - No action
        return False

    def run_backtest(self) -> Dict:
        """Run the backtesting simulation.

        Returns:
            Dictionary with backtesting results.
        """
        logger.info("Starting backtesting simulation...")

        # Find the common time range across all symbols
        common_start, common_end = self._find_common_time_range(self.symbols)

        # Split into lookback and lookforward periods
        lookback_end = common_start + timedelta(days=self.max_lookback_months * 30)
        simulation_start = lookback_end
        simulation_end = min(
            common_end,
            simulation_start + timedelta(days=self.max_lookforward_months * 30),
        )

        logger.info(f"Simulation period: {simulation_start} to {simulation_end}")

        # Get simulation data for each symbol
        simulation_data = {}
        for symbol in self.symbols:
            full_data = self.data_cache[symbol]
            sim_mask = (full_data["datetime"] >= simulation_start) & (
                full_data["datetime"] <= simulation_end
            )
            simulation_data[symbol] = full_data[sim_mask].reset_index(drop=True)

        # Find common timestamps across all symbols for synchronized trading
        common_timestamps = self._find_common_timestamps(simulation_data)

        logger.info(f"Found {len(common_timestamps)} common timestamps for simulation")

        if len(common_timestamps) < 2:
            raise ValueError("Not enough common timestamps for meaningful backtesting")

        # Run the simulation
        total_signals = 0
        total_transactions = 0

        for i, timestamp in enumerate(common_timestamps):
            # Get current prices for all symbols
            current_prices = {}

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

                # Get current price
                current_row = current_full_data.iloc[-1]
                current_price = current_row["Close"]
                current_prices[symbol] = current_price

                # Generate signal
                try:
                    signal = self.strategy_engine.generate_signal(lookback_data, symbol)
                    total_signals += 1

                    # Execute signal
                    transaction_executed = self._execute_signal(
                        symbol, signal.action, current_price, timestamp
                    )
                    if transaction_executed:
                        total_transactions += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to generate signal for {symbol} at {timestamp}: {e}"
                    )

            # Log progress periodically
            if i % 100 == 0:
                portfolio_value = self.portfolio.get_portfolio_value(current_prices)
                logger.info(
                    f"Progress: {i}/{len(common_timestamps)} ({100 * i / len(common_timestamps):.1f}%) | "
                    f"Portfolio: ${portfolio_value:.2f}"
                )

        # Calculate final results
        final_prices = {}
        for symbol in self.symbols:
            final_data = simulation_data[symbol]
            if not final_data.empty:
                final_prices[symbol] = final_data.iloc[-1]["Close"]

        final_portfolio_value = self.portfolio.get_portfolio_value(final_prices)
        total_return = self.portfolio.get_total_return(final_prices)

        results = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_return_dollars": total_return,
            "total_return_percent": self.portfolio.get_return_percentage(final_prices),
            "total_signals_generated": total_signals,
            "total_transactions_executed": total_transactions,
            "simulation_start": simulation_start,
            "simulation_end": simulation_end,
            "symbols_traded": self.symbols,
            "final_holdings": self.portfolio.get_current_holdings(),
            "final_cash": self.portfolio.cash,
            "transaction_history": self.portfolio.get_transaction_history(),
            "portfolio_summary": self.portfolio.get_summary(final_prices),
        }

        logger.info(
            f"Backtesting complete. Total return: ${total_return:.2f} "
            f"({self.portfolio.get_return_percentage(final_prices):.2f}%)"
        )

        return results
