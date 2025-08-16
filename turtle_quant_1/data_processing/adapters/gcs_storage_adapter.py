"""Google Cloud Storage implementation for data storage."""

import io
from datetime import datetime
from typing import Optional

import pandas as pd
from google.cloud import storage

from turtle_quant_1.config import CANDLE_UNIT, GCLOUD_PROJECT_ID, GCLOUD_STB_DATA_NAME
from turtle_quant_1.data_processing.base import BaseDataStorageAdapter


class GCSDataStorageAdapter(BaseDataStorageAdapter):
    """Google Cloud Storage implementation for data storage."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        """Initialize the GCS data storage.

        Args:
            bucket_name: Name of the GCS bucket. If None, reads from GCS_BUCKET_NAME env var.
            project_id: GCP project ID. If None, reads from GCP_PROJECT_ID env var.
        """
        self.project_id = project_id or GCLOUD_PROJECT_ID
        self.bucket_name = bucket_name or GCLOUD_STB_DATA_NAME

        self.client = storage.Client(project=self.project_id)
        self.bucket = self.client.bucket(self.bucket_name)

    def _get_blob_name(self, symbol: str) -> str:
        """Get the blob name for a symbol's data.

        Args:
            symbol: The symbol to get the blob name for.

        Returns:
            The blob name in the format: ohlcv/{symbol}/hourly.parquet
        """
        unit_file_name_dict = {
            "HOUR": "hourly.parquet",
            "DAY": "daily.parquet",
            "WEEK": "weekly.parquet",
            "MONTH": "monthly.parquet",
        }

        return f"ohlcv/{symbol}/{unit_file_name_dict[CANDLE_UNIT]}"

    def load_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from GCS.

        Args:
            symbol: The symbol to load data for.
            start_date: Start date for the data. If None, returns all data.
            end_date: End date for the data. If None, returns all data.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        blob_name = self._get_blob_name(symbol)
        blob = self.bucket.blob(blob_name)

        # Check if blob exists
        if not blob.exists():
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Download and read parquet file
        with io.BytesIO() as buffer:
            blob.download_to_file(buffer)
            buffer.seek(0)
            df = pd.read_parquet(buffer)

        # Filter by date range if provided
        if start_date is not None and end_date is not None and not df.empty:
            mask = (df["datetime"] >= start_date) & (df["datetime"] <= end_date)
            return df[mask]

        return df

    def save_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """Save OHLCV data to GCS.

        Args:
            symbol: The symbol the data belongs to.
            data: DataFrame with OHLCV data.
        """
        blob_name = self._get_blob_name(symbol)
        blob = self.bucket.blob(blob_name)

        # Convert DataFrame to parquet and upload
        with io.BytesIO() as buffer:
            data.to_parquet(buffer)
            buffer.seek(0)
            blob.upload_from_file(buffer, content_type="application/octet-stream")

    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load data from GCS (alias for load_ohlcv).

        Args:
            symbol: The symbol to load data for.

        Returns:
            DataFrame with data.
        """
        return self.load_ohlcv(symbol)

    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to GCS (alias for save_ohlcv).

        Args:
            symbol: The symbol the data belongs to.
            data: DataFrame with data.
        """
        self.save_ohlcv(symbol, data)

    def delete_data(self, symbol: str) -> None:
        """Delete data from GCS.

        Args:
            symbol: The symbol to delete data for.
        """
        blob_name = self._get_blob_name(symbol)
        blob = self.bucket.blob(blob_name)

        if blob.exists():
            blob.delete()
