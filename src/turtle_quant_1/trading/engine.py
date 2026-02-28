"""Trading engine for executing trading signals."""

import logging
from datetime import datetime

import pandas as pd

from turtle_quant_1.backtesting.portfolio import Portfolio
from turtle_quant_1.config import CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategyEngine, Signal, SignalAction
from turtle_quant_1.strategies.helpers.candle_units import CandleUnit, convert_units

logger = logging.getLogger(__name__)


class TradingEngine:
    """Trading engine that processes ticks and executes trading signals.

    Nomenclature:
    - Signal: A single trading signal that is independent of time.
    - Tick: A single trigger of the trading engine, typically defined by when the
        host environment gets spooled up and triggered.
    - Event: A single candle at the interval of `CANDLE_UNIT`. This will be at a smaller
        interval than a tick, i.e. between every tick, multiple events will have passed.
    - "curr..." variables: The variable refers to the latest value at the time of the
        **tick**, not at the event.
    """

    def __init__(
        self,
        strategy_engine: BaseStrategyEngine,
        portfolio: Portfolio,
        max_lookback_days: int,
        tick_interval: CandleUnit = "4H",
        event_interval: CandleUnit = "30M",
    ):
        """Initialise the trading engine.

        Args:
            strategy_engine: The strategy engine used to generate signals.
            portfolio: The portfolio to trade against.
            max_lookback_days: Days of history fed to the strategy on each tick.
            tick_interval: Duration of one tick window as a candle-unit string
                (e.g. "4H"). Used together with CANDLE_UNIT to determine how
                many candle events are replayed per tick.
            event_interval: Duration of one event window as a candle-unit string
                (e.g. "1H"). Used together with CANDLE_UNIT to determine how
                many candle events are replayed per event.
        """
        self.strategy_engine = strategy_engine
        self.portfolio = portfolio
        self.max_lookback_days = max_lookback_days
        self.tick_interval = tick_interval
        self.event_interval = event_interval

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
        logger.debug(f"Getting lookback data for {symbol} at index {current_index}...")

        if current_index < 1:
            logger.debug(
                f"Not enough data to get lookback data for {symbol} at index {current_index}"
            )
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
                take_profit_value=(
                    signal.take_profit_value
                    if signal.take_profit_value is not None
                    else float("inf")
                ),
                stop_loss_value=(
                    signal.stop_loss_value
                    if signal.stop_loss_value is not None
                    else 0.0
                ),
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

    def _handle_event(
        self,
        symbol: str,
        curr_data: pd.DataFrame,
        event_index: int,
    ) -> None:
        """Process a single candle event for *symbol*.

        Args:
            symbol: Symbol being processed.
            event_index: Positional index of the current candle in *curr_data*.
            curr_data: All candles available up to the current tick timestamp.
        """
        lookback_data = self._get_lookback_data(
            symbol=symbol, current_index=event_index, data=curr_data
        )
        if len(lookback_data) < 2:
            # There are not enough data points to generate a signal.
            return

        # NOTE: Use the latest price at the time of the tick (not the event),
        # since the buy/sell price will be executed on the tick itself, i.e. delayed.
        curr_price = curr_data.iloc[-1]["Close"]
        # NOTE: Use the timestamp of the event, since the signal is based on the event.
        event_timestamp = lookback_data.iloc[event_index]["datetime"]

        signal = self.strategy_engine.generate_signal(data=lookback_data, symbol=symbol)
        self.total_signals += 1

        transaction_executed = self._handle_signal(
            symbol=symbol, signal=signal, price=curr_price, timestamp=event_timestamp
        )
        if transaction_executed:
            self.total_transactions += 1

        self.portfolio.check_take_profit_triggers({symbol: curr_price})
        self.portfolio.check_stop_loss_triggers({symbol: curr_price})

        self.signal_history_list.append(
            {
                "datetime": event_timestamp,
                "symbol": symbol,
                "action": signal.action.value,
                "score": signal.score,
                "strategies": signal.strategies,
                "executed": transaction_executed,
            }
        )

    def handle_tick(
        self,
        symbol: str,
        curr_data: pd.DataFrame,
    ) -> None:
        """Process a tick for *symbol* by replaying every event in the tick window.

        Rather than acting on the latest candle alone, iterates over every candle
        within the most recent tick window of *curr_data*, simulating an
        event-driven system where each arriving candle is assessed individually.
        The window size is determined by *tick_interval* set at construction time.

        Args:
            symbol: Symbol being processed.
            curr_data: All candles available up to the current tick timestamp.
        """
        # Get the number of events between the last tick and the current tick.
        n_units_per_event = convert_units(1, self.event_interval, CANDLE_UNIT)
        n_events_per_tick = convert_units(1, self.tick_interval, self.event_interval)
        n_units_per_tick = n_events_per_tick * n_units_per_event
        last_tick_index = max(0, len(curr_data) - n_units_per_tick)

        for i in range(last_tick_index, len(curr_data), n_units_per_event):
            self._handle_event(symbol=symbol, curr_data=curr_data, event_index=i)
