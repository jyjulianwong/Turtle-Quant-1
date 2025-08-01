"""Portfolio management for backtesting."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from turtle_quant_1.backtesting.models import PortfolioTransaction


class Portfolio:
    """Portfolio class to track holdings, capital, and transactions during backtesting."""

    def __init__(self, initial_capital: float = 0.0):
        """Initialize the portfolio.

        Args:
            initial_capital: Starting capital in dollars (default: 0.0).
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, float] = {}  # symbol -> quantity
        self.transactions: List[PortfolioTransaction] = []

    def get_current_holdings(self) -> Dict[str, float]:
        """Get current holdings.

        Returns:
            Dictionary of symbol -> quantity held.
        """
        return self.holdings.copy()

    def get_quantity_held(self, symbol: str) -> float:
        """Get quantity held for a specific symbol.

        Args:
            symbol: The symbol to check.

        Returns:
            Quantity held (0.0 if not held).
        """
        return self.holdings.get(symbol, 0.0)

    def buy(
        self, symbol: str, quantity: float, price: float, timestamp: datetime
    ) -> bool:
        """Execute a buy order.

        Args:
            symbol: Symbol to buy.
            quantity: Quantity to buy.
            price: Price per unit.
            timestamp: Time of transaction.

        Returns:
            True if transaction was successful, False if insufficient funds.
        """
        total_cost = quantity * price

        if total_cost > self.cash:
            return False  # Insufficient funds

        # Execute transaction
        self.cash -= total_cost
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + quantity

        # Record transaction
        transaction = PortfolioTransaction(
            timestamp=timestamp,
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            total_value=total_cost,
        )
        self.transactions.append(transaction)

        return True

    def sell(
        self, symbol: str, quantity: float, price: float, timestamp: datetime
    ) -> bool:
        """Execute a sell order.

        Args:
            symbol: Symbol to sell.
            quantity: Quantity to sell.
            price: Price per unit.
            timestamp: Time of transaction.

        Returns:
            True if transaction was successful, False if insufficient holdings.
        """
        current_holdings = self.holdings.get(symbol, 0.0)

        if quantity > current_holdings:
            return False  # Insufficient holdings

        # Execute transaction
        total_proceeds = quantity * price
        self.cash += total_proceeds
        self.holdings[symbol] = current_holdings - quantity

        # Clean up zero holdings
        if self.holdings[symbol] == 0.0:
            del self.holdings[symbol]

        # Record transaction
        transaction = PortfolioTransaction(
            timestamp=timestamp,
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            total_value=total_proceeds,
        )
        self.transactions.append(transaction)

        return True

    def sell_holdings(self, symbol: str, price: float, timestamp: datetime) -> bool:
        """Sell all holdings of a symbol.

        Args:
            symbol: Symbol to sell.
            price: Price per unit.
            timestamp: Time of transaction.

        Returns:
            True if transaction was successful, False if no holdings.
        """
        quantity = self.holdings.get(symbol, 0.0)

        if quantity == 0.0:
            return False  # No holdings to sell

        return self.sell(symbol, quantity, price, timestamp)

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + holdings).

        Args:
            current_prices: Dictionary of symbol -> current price.

        Returns:
            Total portfolio value in dollars.
        """
        holdings_value = 0.0

        for symbol, quantity in self.holdings.items():
            if symbol in current_prices:
                holdings_value += quantity * current_prices[symbol]

        return self.cash + holdings_value

    def get_return_dollars(self, current_prices: Dict[str, float]) -> float:
        """Calculate total return in dollars.

        Args:
            current_prices: Dictionary of symbol -> current price.

        Returns:
            Total return in dollars (final value - initial capital).
        """
        final_value = self.get_portfolio_value(current_prices)
        return final_value - self.initial_capital

    def get_return_percent(self, current_prices: Dict[str, float]) -> float:
        """Calculate total return as percentage.

        Args:
            current_prices: Dictionary of symbol -> current price.

        Returns:
            Total return as percentage. Returns 0.0 if initial capital was 0.
        """
        if self.initial_capital == 0.0:
            return 0.0

        total_return = self.get_return_dollars(current_prices)
        return (total_return / self.initial_capital) * 100.0

    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame.

        Returns:
            DataFrame with transaction history.
        """
        if not self.transactions:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "action",
                    "quantity",
                    "price",
                    "total_value",
                ]
            )

        data = []
        for txn in self.transactions:
            data.append(
                {
                    "timestamp": txn.timestamp,
                    "symbol": txn.symbol,
                    "action": txn.action,
                    "quantity": txn.quantity,
                    "price": txn.price,
                    "total_value": txn.total_value,
                }
            )

        return pd.DataFrame(data)

    def get_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get portfolio summary.

        Args:
            current_prices: Dictionary of symbol -> current price for valuation.

        Returns:
            Dictionary with portfolio summary statistics.
        """
        summary = {
            "initial_capital": self.initial_capital,
            "current_cash": self.cash,
            "current_holdings": self.holdings.copy(),
            "total_transactions": len(self.transactions),
        }

        if current_prices:
            summary.update(
                {
                    "portfolio_value": self.get_portfolio_value(current_prices),
                    "total_return_dollars": self.get_return_dollars(current_prices),
                    "total_return_percent": self.get_return_percent(current_prices),
                }
            )

        return summary
