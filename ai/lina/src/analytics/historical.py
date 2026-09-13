"""
Historical Analytics.

Provides daily, weekly, and monthly aggregations of crowd data.
Works with CSV or JSON historical count data.

NOTE: All results labeled as SYNTHETIC DEMO DATA when generated from synthetic sources.
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HistoricalAnalytics:
    """
    Computes daily/weekly/monthly analytics from historical crowd count records.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None

    def load_csv(self, filepath: str) -> bool:
        """
        Load historical data from a CSV file.

        Expected columns: timestamp, people_count, [queue_length], [zone_id]

        Args:
            filepath: Path to CSV file.

        Returns:
            True if loaded successfully.
        """
        try:
            df = pd.read_csv(filepath)
            required = {"timestamp", "people_count"}
            if not required.issubset(df.columns):
                logger.error("Historical CSV missing required columns: %s", required - set(df.columns))
                return False
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self._df = df
            logger.info("Loaded %d historical records from %s", len(df), filepath)
            return True
        except Exception as exc:
            logger.error("Failed to load historical CSV: %s", exc)
            return False

    def load_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Load from an in-memory DataFrame.

        Args:
            df: DataFrame with at least 'timestamp' and 'people_count' columns.

        Returns:
            True if valid.
        """
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            self._df = df
            logger.info("Loaded %d records from in-memory DataFrame", len(df))
            return True
        except Exception as exc:
            logger.error("Failed to load DataFrame: %s", exc)
            return False

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Return the loaded DataFrame."""
        return self._df

    def daily_summary(self, date_str: Optional[str] = None) -> dict:
        """
        Compute daily analytics for a given date or for all available dates.

        Args:
            date_str: Optional date filter string (e.g., "2026-09-13").

        Returns:
            Dict of daily summary metrics.
        """
        df = self._get_df("daily_summary")
        if df is None:
            return {}

        df["date"] = df["timestamp"].dt.date

        if date_str:
            try:
                target_date = pd.to_datetime(date_str).date()
                df = df[df["date"] == target_date]
            except ValueError:
                logger.warning("Invalid date_str for daily_summary: %s", date_str)

        if df.empty:
            return {}

        df = df.copy()  # Avoid SettingWithCopyWarning
        daily = df.groupby("date").agg(
            total_visitors=("people_count", "sum"),
            avg_visitors=("people_count", "mean"),
            max_visitors=("people_count", "max"),
            min_visitors=("people_count", "min"),
        ).reset_index()

        # Add peak hour per day
        df["hour"] = df["timestamp"].dt.hour
        hourly_max = df.groupby(["date", "hour"])["people_count"].mean()
        peak_hours = hourly_max.groupby(level=0).idxmax().apply(lambda x: x[1] if x else None)

        result = []
        for _, row in daily.iterrows():
            date = row["date"]
            peak_hour = peak_hours.get(date, None)
            result.append({
                "date": str(date),
                "total_visitors": int(row["total_visitors"]),
                "avg_visitors": round(float(row["avg_visitors"]), 1),
                "max_visitors": int(row["max_visitors"]),
                "min_visitors": int(row["min_visitors"]),
                "peak_hour": f"{peak_hour:02d}:00" if peak_hour is not None else "N/A",
            })

        return {"period": "daily", "records": result}

    def weekly_summary(self) -> dict:
        """
        Compute weekly aggregated analytics.

        Returns:
            Dict of weekly summary metrics.
        """
        df = self._get_df("weekly_summary")
        if df is None:
            return {}

        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["year"] = df["timestamp"].dt.year

        weekly = df.groupby(["year", "week"]).agg(
            total_visitors=("people_count", "sum"),
            avg_daily_visitors=("people_count", "mean"),
            peak_count=("people_count", "max"),
        ).reset_index()

        result = []
        for _, row in weekly.iterrows():
            result.append({
                "year": int(row["year"]),
                "week": int(row["week"]),
                "total_visitors": int(row["total_visitors"]),
                "avg_visitors_per_interval": round(float(row["avg_daily_visitors"]), 1),
                "peak_count": int(row["peak_count"]),
            })

        return {"period": "weekly", "records": result}

    def monthly_summary(self) -> dict:
        """
        Compute monthly aggregated analytics.

        Returns:
            Dict of monthly summary metrics.
        """
        df = self._get_df("monthly_summary")
        if df is None:
            return {}

        df["month"] = df["timestamp"].dt.month
        df["year"] = df["timestamp"].dt.year
        df["day_name"] = df["timestamp"].dt.day_name()

        monthly = df.groupby(["year", "month"]).agg(
            total_visitors=("people_count", "sum"),
            avg_visitors=("people_count", "mean"),
            peak_count=("people_count", "max"),
        ).reset_index()

        # Busiest day of week per month
        busiest_day = df.groupby(["year", "month", "day_name"])["people_count"].mean()
        busiest_per_month = busiest_day.groupby(level=[0, 1]).idxmax().apply(
            lambda x: x[2] if isinstance(x, tuple) and len(x) > 2 else "N/A"
        )

        result = []
        for _, row in monthly.iterrows():
            key = (int(row["year"]), int(row["month"]))
            busiest = busiest_per_month.get(key, "N/A")
            result.append({
                "year": int(row["year"]),
                "month": int(row["month"]),
                "total_visitors": int(row["total_visitors"]),
                "avg_visitors_per_interval": round(float(row["avg_visitors"]), 1),
                "peak_count": int(row["peak_count"]),
                "busiest_day_of_week": str(busiest),
            })

        return {"period": "monthly", "records": result}

    def get_busiest_zones(self) -> dict:
        """
        Identify busiest and least busy zones if zone_id column is present.

        Returns:
            Dict with busiest_zone and least_busy_zone, or empty dict if not available.
        """
        df = self._get_df("get_busiest_zones")
        if df is None or "zone_id" not in df.columns:
            return {}

        zone_avg = df.groupby("zone_id")["people_count"].mean()
        return {
            "busiest_zone": str(zone_avg.idxmax()),
            "least_busy_zone": str(zone_avg.idxmin()),
            "zone_averages": {str(k): round(float(v), 1) for k, v in zone_avg.items()},
        }

    def _get_df(self, caller: str) -> Optional[pd.DataFrame]:
        """Check data is loaded."""
        if self._df is None:
            logger.warning("%s called but no data loaded. Use load_csv() or load_dataframe() first.", caller)
            return None
        return self._df
