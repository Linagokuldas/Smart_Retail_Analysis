"""
Peak Hours Analyzer.

Analyzes historical crowd count data to identify:
- Busiest hours of the day
- Busiest days of the week
- Traffic level for the current hour

Uses aggregated time-series data — NOT individual frame detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAFFIC_HIGH = "HIGH"
TRAFFIC_MEDIUM = "MEDIUM"
TRAFFIC_LOW = "LOW"


class PeakHoursAnalyzer:
    """
    Computes peak hour metrics from historical or real-time crowd count data.
    """

    def __init__(self, thresholds: dict):
        """
        Args:
            thresholds: Loaded thresholds.json dict. Reads from "peak_hours" sub-key.
        """
        ph_cfg = thresholds.get("peak_hours", {})
        self.high_pct = float(ph_cfg.get("high_traffic_percentile", 75))
        self.very_high_pct = float(ph_cfg.get("very_high_traffic_percentile", 90))

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Compute peak hour analytics from a time-series DataFrame.

        Args:
            df: DataFrame with at least columns:
                'timestamp' (datetime-parseable string) and 'people_count' (int).

        Returns:
            Dict with peak hour analytics.
        """
        if df.empty:
            logger.warning("PeakHoursAnalyzer: empty DataFrame provided")
            return self._empty_result()

        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_name"] = df["timestamp"].dt.day_name()
            df["date"] = df["timestamp"].dt.date
        except Exception as exc:
            logger.error("Failed to parse timestamps in PeakHoursAnalyzer: %s", exc)
            return self._empty_result()

        # Hourly aggregation
        hourly = df.groupby("hour")["people_count"].mean()
        peak_hour_num = int(hourly.idxmax())
        min_hour_num = int(hourly.idxmin())

        # Daily aggregation
        daily = df.groupby("day_name")["people_count"].mean()
        busiest_day = str(daily.idxmax())

        # Percentile thresholds for traffic level
        p_high = np.percentile(df["people_count"], self.high_pct)
        p_very_high = np.percentile(df["people_count"], self.very_high_pct)

        hourly_summary = {
            f"{h:02d}:00-{(h+1):02d}:00": round(float(v), 1)
            for h, v in hourly.items()
        }

        return {
            "peak_hour": f"{peak_hour_num:02d}:00-{(peak_hour_num+1):02d}:00",
            "peak_hour_avg_people": round(float(hourly[peak_hour_num]), 1),
            "min_traffic_hour": f"{min_hour_num:02d}:00-{(min_hour_num+1):02d}:00",
            "min_hour_avg_people": round(float(hourly[min_hour_num]), 1),
            "busiest_day": busiest_day,
            "overall_avg_people": round(float(df["people_count"].mean()), 1),
            "overall_max_people": int(df["people_count"].max()),
            "hourly_averages": hourly_summary,
            "traffic_thresholds": {
                "high": round(float(p_high), 1),
                "very_high": round(float(p_very_high), 1),
            },
        }

    def classify_current(self, current_count: int, historical_df: pd.DataFrame) -> str:
        """
        Classify the current crowd count against historical percentiles.

        Args:
            current_count: Number of people currently in the store.
            historical_df: Historical crowd count DataFrame.

        Returns:
            "LOW", "MEDIUM", or "HIGH" traffic level string.
        """
        if historical_df.empty:
            return TRAFFIC_MEDIUM

        counts = historical_df["people_count"].values
        p_high = np.percentile(counts, self.high_pct)
        p_very_high = np.percentile(counts, self.very_high_pct)

        if current_count >= p_very_high:
            return TRAFFIC_HIGH
        elif current_count >= p_high:
            return TRAFFIC_MEDIUM
        return TRAFFIC_LOW

    def _empty_result(self) -> dict:
        return {
            "peak_hour": "N/A",
            "peak_hour_avg_people": 0,
            "min_traffic_hour": "N/A",
            "min_hour_avg_people": 0,
            "busiest_day": "N/A",
            "overall_avg_people": 0,
            "overall_max_people": 0,
            "hourly_averages": {},
            "traffic_thresholds": {"high": 0, "very_high": 0},
        }
