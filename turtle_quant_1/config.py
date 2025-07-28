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

# API Keys
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

PROJECT_ROOT = Path(__file__).parent.parent
PACKAGE_ROOT = Path(__file__).parent

CANDLE_UNIT = "HOUR"
MAX_HISTORY_DAYS = 700  # The history limit for YFinance is 730 days
MAX_CANDLE_GAPS_TO_FILL = 100

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
        "closing": "15:30",
        "timezone": "America/New_York",
    },
    "XECB": {
        "opening": "08:00",
        "closing": "16:30",
        "timezone": "Europe/Paris",
    },
    "ECB": {
        "opening": "08:00",
        "closing": "16:30",
        "timezone": "Europe/Paris",
    },
    "LSE": {
        "opening": "08:00",
        "closing": "16:30",
        "timezone": "Europe/London",
    },
}

SYMBOL_MARKETS = {
    "MSFT": "NYSE",
    "GOOG": "NYSE",
}

BACKTESTING_MAX_LOOKBACK_DAYS = 365
BACKTESTING_MAX_LOOKFORWARD_DAYS = 335  # (700 - 365)
BACKTESTING_SYMBOLS = ["MSFT", "GOOG"]

LIVE_SYMBOLS = ["MSFT", "GOOG"]
