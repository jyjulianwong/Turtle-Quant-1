"""Test configuration and fixtures."""

import logging
import os
from datetime import datetime

import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_sessionstart(session):
    workerinput = getattr(session.config, "workerinput", None)
    if workerinput is None:
        logger.info("Running on Master Process, or not running in parallel at all...")
        os.environ["TURTLEQUANT1_ENV"] = "d"
        os.environ["TURTLEQUANT1_GCLOUD_REGION"] = "us-east1"
        os.environ["TURTLEQUANT1_GCLOUD_PROJECT_ID"] = "jyw-turtlequant1-p"
        os.environ["TURTLEQUANT1_GCLOUD_STB_DATA_NAME"] = (
            "jyw-turtlequant1-p-stb-usea1-data"
        )
        os.environ["TURTLEQUANT1_ALPHA_VANTAGE_API_KEY"] = "0"
        os.environ["TURTLEQUANT1_MAX_WORKERS"] = "0"
    else:
        logger.info(f"Running on Worker: {workerinput['workerid']}...")


def pytest_sessionfinish(session, exitstatus):
    workerinput = getattr(session.config, "workerinput", None)
    if workerinput is None:
        logger.info("Exiting the Master Process, or running serially...")
    else:
        logger.info(f"Exiting Worker: {workerinput['workerid']}...")


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
