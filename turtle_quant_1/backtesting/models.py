"""Pydantic models for backtesting results."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class PortfolioTransaction(BaseModel):
    """Pydantic model for a single transaction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime
    symbol: str
    action: str = Field(..., description="Transaction action: 'BUY' or 'SELL'")
    quantity: float
    price: float
    total_value: float


class PortfolioSummary(BaseModel):
    """Pydantic model for portfolio summary."""

    initial_capital: float
    current_cash: float
    current_holdings: Dict[str, float] = Field(
        ..., description="Symbol to quantity mapping"
    )
    total_transactions: int
    portfolio_value: Optional[float] = None
    total_return_dollars: Optional[float] = None
    total_return_percent: Optional[float] = None


class TestCaseResults(BaseModel):
    """Pydantic model for backtesting results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Portfolio data
    initial_capital: float
    final_portfolio_value: float
    final_holdings: Dict[str, float] = Field(
        ..., description="Symbol to quantity mapping"
    )
    final_cash: float
    total_return_dollars: float
    total_return_percent: float
    portfolio_summary: PortfolioSummary

    # Transaction data
    total_signals: int
    total_transactions: int
    total_simulation_ticks: int
    symbols_traded: List[str]
    transaction_history: pd.DataFrame = Field(
        ..., description="DataFrame with transaction history"
    )

    # Time period
    simulation_start: datetime
    simulation_end: datetime

    # Evaluation metrics
    metrics: Dict[str, Any]

    @property
    def portfolio_summary_str(self) -> str:
        """Get a formatted summary of the portfolio."""
        return (
            f"Initial Capital: ${self.initial_capital:.2f} | "
            f"Final Portfolio Value: ${self.final_portfolio_value:.2f} | "
            f"Final Cash: ${self.final_cash:.2f}"
        )

    @property
    def return_summary_str(self) -> str:
        """Get a formatted summary of returns."""
        return (
            f"Total Return: ${self.total_return_dollars:.2f} "
            f"({self.total_return_percent:.2f}%)"
        )

    @property
    def transaction_summary_str(self) -> str:
        """Get a formatted summary of transaction activity."""
        return (
            f"Signals: {self.total_signals} | "
            f"Transactions: {self.total_transactions} | "
            f"Simulation Ticks: {self.total_simulation_ticks}"
        )


class TestCaseStatus(str, Enum):
    """Enumeration for test case execution status."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TestCaseResponse(BaseModel):
    """Pydantic model for backtesting test case response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Test case information
    test_case_name: str = Field(..., description="Name of the test case")
    test_case_description: str = Field(..., description="Description of the test case")
    status: TestCaseStatus = Field(..., description="Execution status of the test case")

    # Results (only present for successful test cases)
    results: Optional[TestCaseResults] = Field(
        None, description="Backtesting results if successful"
    )

    # Error information (only present for failed test cases)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Type of error if failed")

    # Execution metadata
    execution_time_seconds: Optional[float] = Field(
        None, description="Time taken to execute the test case"
    )

    @classmethod
    def from_success(
        cls,
        test_case_name: str,
        test_case_description: str,
        results: TestCaseResults,
        execution_time: Optional[float] = None,
    ) -> "TestCaseResponse":
        """Create a successful test case response."""
        return cls(  # pyrefly: ignore[missing-argument]
            test_case_name=test_case_name,
            test_case_description=test_case_description,
            status=TestCaseStatus.PASSED,
            results=results,
            execution_time_seconds=execution_time,
        )

    @classmethod
    def from_failure(
        cls,
        test_case_name: str,
        test_case_description: str,
        error: Exception,
        execution_time: Optional[float] = None,
    ) -> "TestCaseResponse":
        """Create a failed test case response."""
        return cls(  # pyrefly: ignore[missing-argument]
            test_case_name=test_case_name,
            test_case_description=test_case_description,
            status=TestCaseStatus.FAILED,
            error_message=str(error),
            error_type=type(error).__name__,
            execution_time_seconds=execution_time,
        )

    @property
    def is_successful(self) -> bool:
        """Check if the test case was successful."""
        return self.status == TestCaseStatus.PASSED

    @property
    def is_failed(self) -> bool:
        """Check if the test case failed."""
        return self.status == TestCaseStatus.FAILED

    @property
    def summary_str(self) -> str:
        """Get a formatted summary of the test case result."""
        if self.is_successful and self.results:
            return (
                f"{self.test_case_name}: {self.status.value} | "
                f"Return: {self.results.return_summary_str} | "
                f"Time: {self.execution_time_seconds:.2f}s"
                if self.execution_time_seconds
                else f"Return: {self.results.return_summary_str}"
            )
        elif self.is_failed:
            return (
                f"{self.test_case_name}: {self.status.value} | "
                f"Error: {self.error_type} - {self.error_message} | "
                f"Time: {self.execution_time_seconds:.2f}s"
                if self.execution_time_seconds
                else f"Error: {self.error_type} - {self.error_message}"
            )
        else:
            return f"{self.test_case_name}: {self.status.value}"
