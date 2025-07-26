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
MAX_HISTORY_YEARS = 5

SYMBOLS = ["AAPL", "MSFT", "GOOG"]
