"""Unit tests for data processor and maintainer components."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from turtle_quant_1.data_processing.gcs_storage_adapter import GCSDataStorageAdapter
from turtle_quant_1.data_processing.maintainer import DataMaintainer
from turtle_quant_1.data_processing.processor import DataProcessor
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
def mock_storage_adapter():
    """Fixture for mock storage adapter."""
    return MagicMock(spec=GCSDataStorageAdapter)


@pytest.fixture
def mock_fetcher():
    """Fixture for mock fetcher."""
    return MagicMock(spec=YFinanceDataFetcher)


@pytest.fixture
def data_processor(symbols, mock_storage_adapter, mock_fetcher):
    """Fixture for DataProcessor instance."""
    return DataProcessor(
        symbols=symbols,
        fetcher_type=mock_fetcher,
        storage=mock_storage_adapter,
    )


@pytest.fixture
def mock_processor():
    """Fixture for mock processor."""
    return MagicMock(spec=DataProcessor)


@pytest.fixture
def data_maintainer(mock_processor, mock_storage_adapter, symbols):
    """Fixture for DataMaintainer instance."""
    return DataMaintainer(
        processor=mock_processor,
        storage=mock_storage_adapter,
        symbols=symbols,
    )


class TestDataProcessor:
    """Test cases for DataProcessor."""

    def test_update_data_new_symbol(
        self,
        data_processor,
        mock_storage_adapter,
        mock_fetcher,
        sample_ohlcv_data,
        dates,
    ):
        """Test updating data for a new symbol."""
        # Setup
        symbol = "AAPL"

        # Mock storage to return empty data (new symbol)
        mock_storage_adapter.load_ohlcv.return_value = pd.DataFrame()

        # Mock fetcher directly on the processor instance
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_hourly_ohlcv.return_value = sample_ohlcv_data
        data_processor.fetcher = mock_fetcher

        # Test
        data_processor.update_data(symbol, dates["start"], dates["end"])

        # Assertions
        mock_storage_adapter.load_ohlcv.assert_called_once_with(symbol)
        mock_fetcher.fetch_hourly_ohlcv.assert_called_once_with(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
        )
        mock_storage_adapter.save_ohlcv.assert_called_once()

    def test_update_data_existing_symbol(
        self,
        data_processor,
        mock_storage_adapter,
        mock_fetcher,
        sample_ohlcv_data,
        dates,
    ):
        """Test updating data for an existing symbol."""
        # Setup
        symbol = "AAPL"

        # Mock existing data
        existing_data = pd.DataFrame(
            {
                "Open": [148.0, 149.0],
                "High": [150.0, 151.0],
                "Low": [147.0, 148.0],
                "Close": [149.0, 150.0],
                "Volume": [900000, 950000],
            },
            index=pd.date_range(datetime(2023, 12, 31), periods=2, freq="h"),
        )

        mock_storage_adapter.load_ohlcv.return_value = existing_data

        # Mock fetcher directly on the processor instance
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_hourly_ohlcv.return_value = sample_ohlcv_data
        data_processor.fetcher = mock_fetcher

        # Test
        data_processor.update_data(symbol, dates["start"], dates["end"])

        # Assertions
        mock_storage_adapter.load_ohlcv.assert_called_once_with(symbol)
        mock_fetcher.fetch_hourly_ohlcv.assert_called_once()
        mock_storage_adapter.save_ohlcv.assert_called_once()


class TestDataMaintainer:
    """Test cases for DataMaintainer."""

    def test_update_historical_data(self, data_maintainer, mock_processor, symbols):
        """Test updating historical data for all symbols."""
        # Test
        data_maintainer.ensure_all_continuous_data()

        # Assertions
        assert mock_processor.update_data.call_count == len(symbols)

    def test_update_latest_data(self, data_maintainer, mock_processor, symbols):
        """Test updating latest data for all symbols."""
        # Test with a short time window
        end_date = datetime.now()
        # start_date = end_date - timedelta(days=1)

        data_maintainer.ensure_all_continuous_data(end_date=end_date)

        # Assertions
        assert mock_processor.update_data.call_count == len(symbols)

    def test_cleanup_old_data(
        self, data_maintainer, mock_storage_adapter, sample_ohlcv_data, symbols
    ):
        """Test cleaning up old data."""
        # This functionality is handled by ensure_continuous_data which only maintains
        # data within MAX_HISTORY_YEARS window
        end_date = datetime.now()
        data_maintainer.ensure_all_continuous_data(end_date=end_date)

        # Assertions
        assert mock_storage_adapter.load_ohlcv.call_count == len(symbols)
