"""The data processing module that contains data ingestion, pre-processing and storage handlers.

The data processing layer has the following responsibilities:
- Connecting to various data sources and APIs
- Downloading data from sources for a given time period, e.g. historical bootstrapping
- Downloading the latest data from sources and appending it to historical data
- Pre-processing the data into a unified format for the strategy engine
- Deleting outdated data from the data store
"""

from turtle_quant_1.data_processing.alpha_vantage_fetcher import AlphaVantageDataFetcher
from turtle_quant_1.data_processing.base import BaseDataFetcher, BaseDataStorageAdapter
from turtle_quant_1.data_processing.gcs_storage_adapter import GCSDataStorageAdapter
from turtle_quant_1.data_processing.maintainer import DataMaintainer
from turtle_quant_1.data_processing.processor import DataProcessor
from turtle_quant_1.data_processing.yfinance_fetcher import YFinanceDataFetcher

__all__ = [
    "AlphaVantageDataFetcher",
    "BaseDataFetcher",
    "BaseDataStorageAdapter",
    "DataMaintainer",
    "DataProcessor",
    "GCSDataStorageAdapter",
    "YFinanceDataFetcher",
]
