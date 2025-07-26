"""Test configuration and fixtures."""

from datetime import datetime

import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Create a sample OHLCV DataFrame for testing."""
    data = pd.DataFrame(
        {
            "Open": [150.0, 151.0],
            "High": [152.0, 153.0],
            "Low": [149.0, 150.0],
            "Close": [151.0, 152.0],
            "Volume": [1000000, 1100000],
        },
        index=pd.date_range(datetime(2024, 1, 1), periods=2, freq="h"),
    )

    # Reset index to make datetime a column (matches expected format)
    data = data.reset_index()
    data = data.rename(columns={"index": "datetime"})
    return data[["datetime", "Open", "High", "Low", "Close", "Volume"]]


@pytest.fixture
def mock_gcs_credentials(monkeypatch):
    """Mock GCS credentials for testing."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "mock_credentials.json")


@pytest.fixture
def mock_alpha_vantage_key(monkeypatch):
    """Mock Alpha Vantage API key for testing."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "mock_api_key")
