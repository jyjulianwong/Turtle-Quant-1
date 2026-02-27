"""Trading engine for executing trading signals."""

import logging
from datetime import datetime

import pandas as pd

from turtle_quant_1.backtesting.portfolio import Portfolio
from turtle_quant_1.config import CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategyEngine, Signal, SignalAction
from turtle_quant_1.strategies.helpers.candle_units import convert_units

logger = logging.getLogger(__name__)


class TradingEngine:
    """Engine that processes ticks and executes trading signals."""

    def __init__(
        self,
        strategy_engine: BaseStrategyEngine,
        portfolio: Portfolio,
        max_lookback_days: int,
    ):
        """Initialise the trading engine.

        Args:
            strategy_engine: The strategy engine used to generate signals.
            portfolio: The portfolio to trade against.
            max_lookback_days: Days of history fed to the strategy on each tick.
        """
        self.strategy_engine = strategy_engine
        self.portfolio = portfolio
        self.max_lookback_days = max_lookback_days

        self.signal_history_list: list[dict] = []
        self.total_signals: int = 0
        self.total_transactions: int = 0

    def _get_lookback_data(
        self, symbol: str, current_index: int, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Slice *data* to the lookback window ending at *current_index*.

        Args:
            symbol: Symbol the data belongs to (unused here, kept for symmetry).
            current_index: The row index of the current candle in *data*.
            data: Full historical DataFrame for the symbol.

        Returns:
            DataFrame slice covering the lookback window.
        """
        if current_index < 1:
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        lookback_units = convert_units(self.max_lookback_days, "1D", CANDLE_UNIT)
        start_index = max(0, current_index - lookback_units)
        return data.iloc[start_index : current_index + 1]

    def _handle_signal(
        self,
        symbol: str,
        signal: Signal,
        price: float,
        timestamp: datetime,
    ) -> bool:
        """Execute a trading signal against the portfolio.

        Args:
            symbol: Symbol to trade.
            signal: The signal to execute.
            price: Current price.
            timestamp: Current timestamp. Timezone-aware.

        Returns:
            True if a transaction was executed, False otherwise.
        """
        if signal.action == SignalAction.BUY:
            units = 100.0 / price
            success = self.portfolio.buy(
                symbol=symbol,
                quantity=units,
                take_profit_value=signal.take_profit_value
                if signal.take_profit_value is not None
                else float("inf"),
                stop_loss_value=signal.stop_loss_value
                if signal.stop_loss_value is not None
                else 0.0,
                price=price,
                timestamp=timestamp,
            )
            if success:
                logger.debug(f"BUY: {symbol} at ${price:.2f} on {timestamp}")
            else:
                logger.debug(
                    f"BUY FAILED: Insufficient funds for {symbol} at ${price:.2f}"
                )
            return success

        elif signal.action == SignalAction.SELL:
            success = self.portfolio.sell_holdings(symbol, price, timestamp)
            if success:
                logger.debug(f"SELL ALL: {symbol} at ${price:.2f} on {timestamp}")
            else:
                logger.debug(f"SELL ALL FAILED: No holdings for {symbol}")
            return success

        # HOLD – no action
        return False

    def handle_tick(
        self,
        symbol: str,
        curr_data: pd.DataFrame,
        timestamp: datetime,
    ) -> None:
        """Process a single tick for *symbol*.

        Args:
            symbol: Symbol being processed.
            curr_data: All candles available up to (and including) *timestamp*.
            timestamp: Current tick timestamp. Timezone-aware.

        Returns:
            The current close price when the tick was processed successfully,
            or None when there was insufficient data to act.
        """
        curr_index = len(curr_data) - 1
        lookback_data = self._get_lookback_data(symbol, curr_index, curr_data)

        if len(lookback_data) < 2:
            return None

        curr_price = lookback_data.iloc[-1]["Close"]

        signal = self.strategy_engine.generate_signal(lookback_data, symbol)
        self.total_signals += 1

        transaction_executed = self._handle_signal(
            symbol, signal, curr_price, timestamp
        )
        if transaction_executed:
            self.total_transactions += 1

        self.portfolio.check_take_profit_triggers({symbol: curr_price})
        self.portfolio.check_stop_loss_triggers({symbol: curr_price})

        self.signal_history_list.append(
            {
                "datetime": timestamp,
                "symbol": symbol,
                "action": signal.action.value,
                "score": signal.score,
                "strategies": signal.strategies,
                "executed": transaction_executed,
            }
        )
