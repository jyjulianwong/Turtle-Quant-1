"""The data processing module that contains data ingestion, pre-processing and storage handlers.

The data processing layer has the following responsibilities:
- Connecting to various data sources and APIs
- Downloading data from sources for a given time period, e.g. historical bootstrapping
- Downloading the latest data from sources and appending it to historical data
- Pre-processing the data into a unified format for the strategy engine
- Deleting outdated data from the data store
"""
