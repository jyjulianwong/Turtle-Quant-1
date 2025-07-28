"""Unit tests for data processor component."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from turtle_quant_1.data_processing.base import BaseDataMaintainer
from turtle_quant_1.data_processing.gcs_storage_adapter import GCSDataStorageAdapter
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
def mock_maintainer():
    """Fixture for mock maintainer."""
    return MagicMock(spec=BaseDataMaintainer)


@pytest.fixture
def sample_ohlcv_data():
    """Fixture for sample OHLCV data."""
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0],
            "High": [152.0, 153.0],
            "Low": [149.0, 150.0],
            "Close": [151.0, 152.0],
            "Volume": [1000000, 1050000],
        },
        index=pd.date_range(datetime(2024, 1, 1), periods=2, freq="h"),
    )


@pytest.fixture
def data_processor(symbols, mock_storage_adapter, mock_fetcher, mock_maintainer):
    """Fixture for DataProcessor instance."""
    return DataProcessor(
        symbols=symbols,
        storage=mock_storage_adapter,
        fetcher=mock_fetcher,
        maintainer=mock_maintainer,
    )


class TestDataProcessor:
    """Test cases for DataProcessor."""

    def test_load_data_from_storage(
        self,
        data_processor,
        mock_storage_adapter,
        sample_ohlcv_data,
        dates,
    ):
        """Test loading data from storage when data exists."""
        # Setup
        symbol = "AAPL"
        mock_storage_adapter.load_data.return_value = sample_ohlcv_data

        # Test
        result = data_processor.load_data(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
            impute_data=False,
        )

        # Assertions
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)
        mock_storage_adapter.load_data.assert_called_once_with(
            symbol=symbol,
        )
        assert not mock_storage_adapter.save_data.called
        assert symbol in data_processor.data_cache

    def test_load_data_from_fetcher(
        self,
        data_processor,
        mock_storage_adapter,
        mock_fetcher,
        sample_ohlcv_data,
        dates,
    ):
        """Test loading data from fetcher when storage is empty."""
        # Setup
        symbol = "AAPL"
        mock_storage_adapter.load_ohlcv.return_value = pd.DataFrame()
        mock_fetcher.fetch_hourly_ohlcv.return_value = sample_ohlcv_data

        # Test
        result = data_processor.load_data(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
            impute_data=False,
        )

        # Assertions
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)
        mock_storage_adapter.load_data.assert_called_once_with(
            symbol=symbol,
        )
        mock_fetcher.fetch_hourly_ohlcv.assert_called_once_with(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
        )
        mock_storage_adapter.save_data.assert_called_once_with(
            symbol=symbol,
            data=sample_ohlcv_data,
        )
        assert symbol in data_processor.data_cache

    def test_load_data_from_cache(
        self,
        data_processor,
        mock_storage_adapter,
        mock_fetcher,
        sample_ohlcv_data,
        dates,
    ):
        """Test loading data from cache when available."""
        # Setup
        symbol = "AAPL"
        data_processor.data_cache[symbol] = sample_ohlcv_data

        # Test
        result = data_processor.load_data(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
            impute_data=False,
        )

        # Assertions
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)
        assert not mock_storage_adapter.load_data.called
        assert not mock_fetcher.fetch_hourly_ohlcv.called
        assert not mock_storage_adapter.save_data.called

    def test_load_data_with_imputation(
        self,
        data_processor,
        mock_storage_adapter,
        mock_maintainer,
        sample_ohlcv_data,
        dates,
    ):
        """Test loading data with imputation enabled."""
        # Setup
        symbol = "AAPL"
        mock_storage_adapter.load_data.return_value = sample_ohlcv_data

        # Create imputed data with an extra row
        imputed_data = sample_ohlcv_data.copy()
        imputed_data.loc[datetime(2024, 1, 1, 2)] = [
            153.0,
            154.0,
            152.0,
            153.0,
            1100000,
        ]

        mock_maintainer.impute_data.return_value = imputed_data

        # Test
        result = data_processor.load_data(
            symbol=symbol,
            start_date=dates["start"],
            end_date=dates["end"],
            impute_data=True,
        )

        # Assertions
        pd.testing.assert_frame_equal(result, imputed_data)
        mock_maintainer.impute_data.assert_called_once_with(
            symbol,
            sample_ohlcv_data,
            dates["end"],
        )
        mock_storage_adapter.save_data.assert_called_once_with(
            symbol=symbol,
            data=imputed_data,
        )
        assert symbol in data_processor.data_cache

    def test_save_data(
        self,
        data_processor,
        mock_storage_adapter,
        sample_ohlcv_data,
    ):
        """Test saving data explicitly."""
        # Setup
        symbol = "AAPL"

        # Test
        data_processor.save_data(symbol=symbol, data=sample_ohlcv_data)

        # Assertions
        mock_storage_adapter.save_data.assert_called_once_with(
            symbol=symbol,
            data=sample_ohlcv_data,
        )
