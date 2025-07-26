"""Unit tests for data fetchers."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from turtle_quant_1.data_processing.alpha_vantage_fetcher import AlphaVantageDataFetcher
from turtle_quant_1.data_processing.yfinance_fetcher import YFinanceDataFetcher


@pytest.fixture
def symbols():
    """Fixture for test symbols."""
    return ["AAPL", "GOOGL"]


@pytest.fixture
def dates():
    """Fixture for test dates."""
    return {"start": datetime(2024, 1, 1), "end": datetime(2024, 1, 2)}


@pytest.fixture
def yfinance_fetcher(symbols):
    """Fixture for YFinanceDataFetcher instance."""
    return YFinanceDataFetcher(symbols)


@pytest.fixture
def alpha_vantage_fetcher(symbols):
    """Fixture for AlphaVantageDataFetcher instance."""
    return AlphaVantageDataFetcher(symbols)


class TestYFinanceDataFetcher:
    """Test cases for YFinanceDataFetcher."""

    @patch("turtle_quant_1.data_processing.yfinance_fetcher.yf.Ticker")
    def test_fetch_hourly_ohlcv_success(self, mock_ticker_class, symbols, dates):
        """Test successful hourly OHLCV data fetching."""
        # Mock ticker instance and its history method
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Create fetcher after mocking
        yfinance_fetcher = YFinanceDataFetcher(symbols)

        # Mock data from yfinance (has datetime index, no datetime column)
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range(dates["start"], periods=2, freq="h"),
        )

        mock_ticker.history.return_value = mock_data

        # Test
        result = yfinance_fetcher.fetch_hourly_ohlcv(
            "AAPL",
            dates["start"],
            dates["end"],
        )

        # Expected result should have datetime column
        expected_data = mock_data.reset_index().rename(columns={"index": "datetime"})
        expected_data = expected_data[
            ["datetime", "Open", "High", "Low", "Close", "Volume"]
        ]

        # Assertions
        mock_ticker.history.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_data)

    @patch("turtle_quant_1.data_processing.yfinance_fetcher.yf.Ticker")
    def test_fetch_hourly_ohlcv_empty_data(self, mock_ticker_class, symbols, dates):
        """Test handling of empty data response."""
        # Mock ticker instance and its history method
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        # Create fetcher after mocking
        yfinance_fetcher = YFinanceDataFetcher(symbols)

        # Test
        result = yfinance_fetcher.fetch_hourly_ohlcv(
            "AAPL",
            dates["start"],
            dates["end"],
        )

        # Assertions
        assert result.empty


class TestAlphaVantageDataFetcher:
    """Test cases for AlphaVantageDataFetcher."""

    @patch("turtle_quant_1.data_processing.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_hourly_ohlcv_success(self, mock_timeseries_class, symbols, dates):
        """Test successful hourly OHLCV data fetching."""
        # Mock TimeSeries instance
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts

        # Create fetcher after mocking
        alpha_vantage_fetcher = AlphaVantageDataFetcher(symbols)

        # Mock data from alpha_vantage (returns tuple of data, metadata)
        mock_data = pd.DataFrame(
            {
                "1. open": [150.0, 151.0],
                "2. high": [152.0, 153.0],
                "3. low": [149.0, 150.0],
                "4. close": [151.0, 152.0],
                "5. volume": [1000000, 1100000],
            },
            index=pd.date_range(dates["start"], periods=2, freq="h"),
        )

        mock_metadata = {
            "Meta Data": {
                "Information": "Intraday (60min) open, high, low, close prices and volume"
            }
        }
        mock_ts.get_intraday.return_value = (mock_data, mock_metadata)

        # Test
        result = alpha_vantage_fetcher.fetch_hourly_ohlcv(
            "AAPL",
            dates["start"],
            dates["end"],
        )

        # Assertions
        assert not result.empty
        assert len(result) == 2
        assert all(
            col in result.columns
            for col in ["datetime", "Open", "High", "Low", "Close", "Volume"]
        )

    @patch("turtle_quant_1.data_processing.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_hourly_ohlcv_api_error(self, mock_timeseries_class, symbols, dates):
        """Test handling of API error response."""
        # Mock TimeSeries instance to raise an exception
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_intraday.side_effect = Exception("API error")

        # Create fetcher after mocking
        alpha_vantage_fetcher = AlphaVantageDataFetcher(symbols)

        # Test and assert
        with pytest.raises(Exception):
            alpha_vantage_fetcher.fetch_hourly_ohlcv(
                "AAPL",
                dates["start"],
                dates["end"],
            )
