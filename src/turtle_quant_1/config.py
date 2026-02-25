"""The configuration module for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment configuration
ENV = os.getenv("TURTLEQUANT1_ENV", "p").lower()
if ENV not in ["d", "p"]:
    raise ValueError(
        "TURTLEQUANT1_ENV must be either 'd' (development) or 'p' (production)"
    )

# API keys
ALPHA_VANTAGE_API_KEY = os.getenv("TURTLEQUANT1_ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("TURTLEQUANT1_ALPHA_VANTAGE_API_KEY environment variable not set")

# Infrastructure
GCLOUD_REGION = os.getenv("TURTLEQUANT1_GCLOUD_REGION")
if not GCLOUD_REGION:
    raise ValueError("TURTLEQUANT1_GCLOUD_REGION environment variable not set")

GCLOUD_PROJECT_ID = os.getenv("TURTLEQUANT1_GCLOUD_PROJECT_ID")
if not GCLOUD_PROJECT_ID:
    raise ValueError("TURTLEQUANT1_GCLOUD_PROJECT_ID environment variable not set")

GCLOUD_STB_DATA_NAME = os.getenv("TURTLEQUANT1_GCLOUD_STB_DATA_NAME")
if not GCLOUD_STB_DATA_NAME:
    raise ValueError("TURTLEQUANT1_GCLOUD_STB_DATA_NAME environment variable not set")

# Path shortcuts
PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent

# Performance
# The maximum number of workers to use for parallel processing
# NOTE: Recommended: Number of CPU cores - 1
MAX_WORKERS = int(os.getenv("TURTLEQUANT1_MAX_WORKERS", 0))

# Constants
# The base unit for each data point
CANDLE_UNIT = "HOUR"
# The maximum amount of data that is ever stored in the database for each symbol
# Make sure the lookback and lookforward periods are less than this value
# The history limit for YFinance is 730 days
MAX_HISTORY_DAYS = 700
# In case of missing data, the maximum number of gaps to impute
# This is to prevent hitting API limits when fetching data to fill gaps
MAX_CANDLE_GAPS_TO_FILL = 100

# Mapping of GCP regions to their corresponding PYTZ timezone
_gcloud_region_to_pytz_dict = {
    "us-central1": "America/Chicago",
    "us-east1": "America/New_York",
    "us-east4": "America/New_York",
    "us-west1": "America/Los_Angeles",
    "us-west2": "America/Los_Angeles",
    "us-west3": "America/Denver",
    "us-west4": "America/Los_Angeles",
    "northamerica-northeast1": "America/Toronto",
    "southamerica-east1": "America/Sao_Paulo",
    "europe-central2": "Europe/Warsaw",
    "europe-north1": "Europe/Helsinki",
    "europe-west1": "Europe/Brussels",
    "europe-west2": "Europe/London",
    "europe-west3": "Europe/Frankfurt",
    "europe-west4": "Europe/Amsterdam",
    "asia-east1": "Asia/Taipei",
    "asia-east2": "Asia/Hong_Kong",
    "asia-northeast1": "Asia/Tokyo",
    "asia-northeast2": "Asia/Osaka",
    "asia-south1": "Asia/Kolkata",
    "asia-southeast1": "Asia/Singapore",
    "asia-southeast2": "Asia/Jakarta",
    "australia-southeast1": "Australia/Sydney",
    "me-west1": "Asia/Dubai",
    "me-central1": "Asia/Riyadh",
    "africa-south1": "Africa/Johannesburg",
}
HOST_TIMEZONE = "Europe/London"  # Default to London
if GCLOUD_REGION and ENV == "p":
    HOST_TIMEZONE = _gcloud_region_to_pytz_dict[GCLOUD_REGION]

# Timezones are necessary to ensure all timestamps are timezone-aware during conversion steps
MARKET_HOURS = {
    "NYSE": {
        "opening": "09:30",
        "closing": "15:30",  # 16:00
        "timezone": "America/New_York",
    },
    "XECB": {
        "opening": "08:00",
        "closing": "16:00",  # 16:30
        "timezone": "Europe/Paris",
    },
    "ECB": {
        "opening": "08:00",
        "closing": "16:00",  # 16:30
        "timezone": "Europe/Paris",
    },
    "LSE": {
        "opening": "08:00",
        "closing": "16:00",  # 16:30
        "timezone": "Europe/London",
    },
}

# Mapping of symbols to their corresponding market
SYMBOL_MARKETS = {
    # US index
    "SPY": "NYSE",
    # Gold
    "GLD": "NYSE",
    # Energy / Oil
    "XOM": "NYSE",
    "CVX": "NYSE",
    # Healthcare
    "JNJ": "NYSE",
    # Defence
    "BA": "NYSE",
    "LMT": "NYSE",
    # Technology
    "MSFT": "NYSE",
    "GOOG": "NYSE",
}

# Backtesting constants
# The maximum lookback period for backtesting
BACKTESTING_MAX_LOOKBACK_DAYS = 365
# The maximum lookforward period for backtesting
# (700 - 365)
BACKTESTING_MAX_LOOKFORWARD_DAYS = 335
# The symbols to use for backtesting
BACKTESTING_SYMBOLS = [
    "SPY",
    "GLD",
    "XOM",
    "CVX",
    "JNJ",
    "BA",
    "LMT",
    "MSFT",
    "GOOG",
]

# Live mode constants
# The symbols to use for live mode
LIVE_SYMBOLS = [
    "SPY",
    "GLD",
    "XOM",
    "CVX",
    "JNJ",
    "BA",
    "LMT",
    "MSFT",
    "GOOG",
]
