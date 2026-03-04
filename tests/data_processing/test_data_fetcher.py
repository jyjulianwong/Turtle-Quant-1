"""Unit tests for data fetchers."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher import (
    AlphaVantageDataFetcher,
)
from turtle_quant_1.data_processing.adapters.yfinance_fetcher import YFinanceDataFetcher


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

    @patch("turtle_quant_1.data_processing.adapters.yfinance_fetcher.yf.Ticker")
    def test_fetch_ohlcv_success(self, mock_ticker_class, symbols, dates):
        """Test successful OHLCV data fetching at the default 5M frequency."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

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
            index=pd.date_range(dates["start"], periods=2, freq="5min"),
        )

        mock_ticker.history.return_value = mock_data

        result = yfinance_fetcher.fetch_ohlcv(
            "AAPL",
            dates["start"],
            dates["end"],
        )

        # Expected result should have datetime column
        expected_data = mock_data.reset_index().rename(columns={"index": "datetime"})
        expected_data = expected_data[
            ["datetime", "Open", "High", "Low", "Close", "Volume"]
        ]

        mock_ticker.history.assert_called_once()
        pd.testing.assert_frame_equal(result, expected_data)

    @patch("turtle_quant_1.data_processing.adapters.yfinance_fetcher.yf.Ticker")
    def test_fetch_ohlcv_empty_data(self, mock_ticker_class, symbols, dates):
        """Test handling of empty data response."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        yfinance_fetcher = YFinanceDataFetcher(symbols)

        result = yfinance_fetcher.fetch_ohlcv(
            "AAPL",
            dates["start"],
            dates["end"],
        )

        assert result.empty


def _make_av_intraday_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a mock Alpha Vantage intraday DataFrame with AV-style column names."""
    n = len(index)
    return pd.DataFrame(
        {
            "1. open": [150.0 + i for i in range(n)],
            "2. high": [152.0 + i for i in range(n)],
            "3. low": [149.0 + i for i in range(n)],
            "4. close": [151.0 + i for i in range(n)],
            "5. volume": [1_000_000 + i * 100_000 for i in range(n)],
        },
        index=index,
    )


def _make_av_daily_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a mock Alpha Vantage daily DataFrame with AV-style column names."""
    n = len(index)
    return pd.DataFrame(
        {
            "1. open": [150.0 + i for i in range(n)],
            "2. high": [152.0 + i for i in range(n)],
            "3. low": [149.0 + i for i in range(n)],
            "4. close": [151.0 + i for i in range(n)],
            "5. volume": [1_000_000 + i * 100_000 for i in range(n)],
        },
        index=index,
    )


_EXPECTED_COLUMNS = ["datetime", "Open", "High", "Low", "Close", "Volume"]


class TestAlphaVantageDataFetcher:
    """Test cases for AlphaVantageDataFetcher."""

    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_5m_success(self, mock_timeseries_class, symbols, dates):
        """Test successful OHLCV data fetching at the default 5M frequency."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_intraday.return_value = (
            _make_av_intraday_df(pd.date_range(dates["start"], periods=2, freq="5min")),
            {},
        )

        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"], freq="5M"
        )

        mock_ts.get_intraday.assert_called_once_with(
            symbol="AAPL", interval="5min", outputsize="full"
        )
        assert not result.empty
        assert len(result) == 2
        assert list(result.columns) == _EXPECTED_COLUMNS

    @pytest.mark.parametrize(
        "freq,expected_interval",
        [
            ("15M", "15min"),
            ("30M", "30min"),
            ("1H", "60min"),
        ],
    )
    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_intraday_uses_correct_interval(
        self, mock_timeseries_class, freq, expected_interval, symbols, dates
    ):
        """Test that each intraday CandleUnit maps to the right AV interval string."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_intraday.return_value = (
            _make_av_intraday_df(
                pd.date_range(dates["start"], periods=1, freq=expected_interval)
            ),
            {},
        )

        AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"], freq=freq
        )

        mock_ts.get_intraday.assert_called_once_with(
            symbol="AAPL", interval=expected_interval, outputsize="full"
        )

    @pytest.mark.parametrize("freq", ["2H", "4H"])
    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_resampled(self, mock_timeseries_class, freq, symbols):
        """Test that 2H/4H data is fetched at 60min and resampled correctly."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        # Use UTC-aligned bars so pandas resample bucketing is deterministic:
        # 4 bars at 08:00, 09:00, 10:00, 11:00 UTC → 2×2H or 1×4H buckets.
        bar_start = datetime(2024, 9, 4, 8, 0, tzinfo=pytz.UTC)
        mock_ts.get_intraday.return_value = (
            _make_av_intraday_df(pd.date_range(bar_start, periods=4, freq="60min")),
            {},
        )

        fetch_start = bar_start
        fetch_end = bar_start + pd.Timedelta(hours=4)
        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", fetch_start, fetch_end, freq=freq
        )

        mock_ts.get_intraday.assert_called_once_with(
            symbol="AAPL", interval="60min", outputsize="full"
        )
        assert list(result.columns) == _EXPECTED_COLUMNS
        # 4 one-hour bars should collapse into 2 two-hour (or 1 four-hour) bars
        expected_rows = 2 if freq == "2H" else 1
        assert len(result) == expected_rows

    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_daily(self, mock_timeseries_class, symbols, dates):
        """Test that 1D frequency calls get_daily instead of get_intraday."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_daily.return_value = (
            _make_av_daily_df(pd.date_range("2024-09-04", periods=3, freq="D")),
            {},
        )

        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"], freq="1D"
        )

        mock_ts.get_daily.assert_called_once_with(symbol="AAPL", outputsize="full")
        mock_ts.get_intraday.assert_not_called()
        assert not result.empty
        assert list(result.columns) == _EXPECTED_COLUMNS

    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_weekly(self, mock_timeseries_class, symbols, dates):
        """Test that 1W frequency calls get_weekly instead of get_intraday."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_weekly.return_value = (
            _make_av_daily_df(pd.date_range("2024-09-01", periods=3, freq="W")),
            {},
        )

        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"], freq="1W"
        )

        mock_ts.get_weekly.assert_called_once_with(symbol="AAPL")
        mock_ts.get_intraday.assert_not_called()
        assert list(result.columns) == _EXPECTED_COLUMNS

    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_empty_data(self, mock_timeseries_class, symbols, dates):
        """Test handling of an empty response from Alpha Vantage."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_intraday.return_value = (pd.DataFrame(), {})

        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"]
        )

        assert result.empty

    @patch("turtle_quant_1.data_processing.adapters.alpha_vantage_fetcher.TimeSeries")
    def test_fetch_ohlcv_api_error_returns_empty_df(
        self, mock_timeseries_class, symbols, dates
    ):
        """Test that API errors are caught and an empty DataFrame is returned."""
        mock_ts = MagicMock()
        mock_timeseries_class.return_value = mock_ts
        mock_ts.get_intraday.side_effect = Exception("API error")

        result = AlphaVantageDataFetcher(symbols).fetch_ohlcv(
            "AAPL", dates["start"], dates["end"]
        )

        assert result.empty
        assert list(result.columns) == _EXPECTED_COLUMNS
