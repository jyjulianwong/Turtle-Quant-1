"""Unit tests for data storage components."""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from turtle_quant_1.data_processing.adapters.gcs_storage_adapter import (
    GCSDataStorageAdapter,
)


@pytest.fixture
def mock_storage_adapter():
    """Fixture for mock storage adapter."""
    with patch("google.cloud.storage.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket

        storage = GCSDataStorageAdapter()
        storage.client = mock_client
        storage.bucket = mock_bucket
        return storage


@pytest.fixture
def symbol():
    """Fixture for test symbol."""
    return "AAPL"


class TestGCSDataStorage:
    """Test cases for GCSDataStorage."""

    def test_save_data_success(self, mock_storage_adapter, symbol, sample_ohlcv_data):
        """Test successful data saving to GCS."""
        # Mock blob
        mock_blob = MagicMock()
        mock_storage_adapter.bucket.blob.return_value = mock_blob

        # Test
        mock_storage_adapter.save_data(symbol, sample_ohlcv_data)

        # Assertions
        mock_storage_adapter.bucket.blob.assert_called_once()
        mock_blob.upload_from_file.assert_called_once()

    def test_load_data_success(self, mock_storage_adapter, symbol, sample_ohlcv_data):
        """Test successful data loading from GCS."""
        # Mock blob
        mock_blob = MagicMock()
        mock_storage_adapter.bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        # Create a proper parquet buffer for the mock
        buffer = io.BytesIO()
        sample_ohlcv_data.to_parquet(buffer)
        buffer.seek(0)

        def mock_download_to_file(file_obj):
            file_obj.write(buffer.getvalue())
            file_obj.seek(0)

        mock_blob.download_to_file.side_effect = mock_download_to_file

        # Test
        result = mock_storage_adapter.load_data(symbol)

        # Assertions
        mock_storage_adapter.bucket.blob.assert_called_once()
        mock_blob.download_to_file.assert_called_once()
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)

    def test_load_data_not_found(self, mock_storage_adapter, symbol):
        """Test handling of non-existent data."""
        # Mock blob
        mock_blob = MagicMock()
        mock_storage_adapter.bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        # Test
        result = mock_storage_adapter.load_data(symbol)

        # Assertions
        assert result.empty
        assert list(result.columns) == [
            "datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

    def test_delete_data_success(self, mock_storage_adapter, symbol):
        """Test successful data deletion from GCS."""
        # Mock blob
        mock_blob = MagicMock()
        mock_storage_adapter.bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        # Test
        mock_storage_adapter.delete_data(symbol)

        # Assertions
        mock_storage_adapter.bucket.blob.assert_called_once()
        mock_blob.delete.assert_called_once()

    def test_delete_data_not_found(self, mock_storage_adapter, symbol):
        """Test deletion of non-existent data."""
        # Mock blob
        mock_blob = MagicMock()
        mock_storage_adapter.bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        # Test
        mock_storage_adapter.delete_data(symbol)

        # Assertions
        mock_storage_adapter.bucket.blob.assert_called_once()
        mock_blob.delete.assert_not_called()
