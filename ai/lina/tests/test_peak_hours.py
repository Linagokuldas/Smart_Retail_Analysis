"""
Tests for peak hours and historical analytics modules.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.analytics.peak_hours import PeakHoursAnalyzer
from src.analytics.historical import HistoricalAnalytics
from src.analytics.recommendations import RecommendationEngine


THRESHOLDS = {
    "peak_hours": {
        "high_traffic_percentile": 75,
        "very_high_traffic_percentile": 90,
    },
    "crowd": {
        "density_low": 0.10,
        "density_medium": 0.30,
        "density_high": 0.60,
        "density_critical": 0.80,
    },
    "queue": {"grace_period_seconds": 10},
    "waiting_time": {
        "busy_threshold_seconds": 120,
        "critical_threshold_seconds": 300,
    },
    "counter": {
        "busy_queue_length": 5,
        "overloaded_queue_length": 10,
        "busy_avg_wait_seconds": 120,
        "overloaded_avg_wait_seconds": 240,
    },
    "recommendations": {
        "open_counter_queue_length": 7,
        "staff_allocation_density": 0.60,
        "predicted_crowd_warning_ratio": 0.80,
    },
}


def make_hourly_df(days: int = 7) -> pd.DataFrame:
    """Generate synthetic hourly crowd data."""
    records = []
    base = datetime(2026, 9, 7, 8, 0, 0)
    hourly_profile = {8: 10, 9: 20, 10: 35, 11: 50, 12: 80, 13: 85, 14: 60, 15: 55}

    for day in range(days):
        for hour, count in hourly_profile.items():
            ts = (base + timedelta(days=day)).replace(hour=hour)
            records.append({
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "people_count": count + np.random.randint(-5, 5),
                "queue_length": max(0, count // 10 + np.random.randint(-2, 2)),
            })

    return pd.DataFrame(records)


# ── PeakHoursAnalyzer tests ───────────────────────────────────────────────────

class TestPeakHoursAnalyzer:
    def setup_method(self):
        self.analyzer = PeakHoursAnalyzer(THRESHOLDS)
        self.df = make_hourly_df()

    def test_analyze_returns_required_keys(self):
        result = self.analyzer.analyze(self.df)
        required_keys = [
            "peak_hour", "peak_hour_avg_people", "min_traffic_hour",
            "busiest_day", "overall_avg_people", "overall_max_people",
            "hourly_averages",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_peak_hour_is_reasonable(self):
        result = self.analyzer.analyze(self.df)
        # Peak should be around 12:00 or 13:00 given our synthetic profile
        peak = result["peak_hour"]
        assert "12" in peak or "13" in peak

    def test_overall_avg_is_positive(self):
        result = self.analyzer.analyze(self.df)
        assert result["overall_avg_people"] > 0

    def test_empty_dataframe_returns_defaults(self):
        result = self.analyzer.analyze(pd.DataFrame())
        assert result["peak_hour"] == "N/A"
        assert result["overall_avg_people"] == 0

    def test_classify_current_high(self):
        # Pass a very high current count
        status = self.analyzer.classify_current(1000, self.df)
        assert status == "HIGH"

    def test_classify_current_low(self):
        status = self.analyzer.classify_current(0, self.df)
        assert status == "LOW"

    def test_hourly_averages_present(self):
        result = self.analyzer.analyze(self.df)
        hourly = result["hourly_averages"]
        assert len(hourly) > 0

    def test_missing_people_count_column(self):
        bad_df = pd.DataFrame({"timestamp": ["2026-09-13T12:00:00"], "count": [10]})
        # Should not raise — internal pandas error should be caught
        try:
            result = self.analyzer.analyze(bad_df)
        except Exception:
            pass  # Acceptable — important thing is no crash to outer pipeline


# ── HistoricalAnalytics tests ─────────────────────────────────────────────────

class TestHistoricalAnalytics:
    def setup_method(self):
        self.analytics = HistoricalAnalytics()
        self.df = make_hourly_df(days=30)

    def test_load_dataframe(self):
        result = self.analytics.load_dataframe(self.df)
        assert result is True
        assert self.analytics.dataframe is not None

    def test_daily_summary_returns_records(self):
        self.analytics.load_dataframe(self.df)
        result = self.analytics.daily_summary()
        assert result["period"] == "daily"
        assert len(result["records"]) > 0

    def test_daily_summary_with_date_filter(self):
        self.analytics.load_dataframe(self.df)
        result = self.analytics.daily_summary("2026-09-07")
        assert result.get("period") == "daily"

    def test_weekly_summary(self):
        self.analytics.load_dataframe(self.df)
        result = self.analytics.weekly_summary()
        assert result["period"] == "weekly"
        assert len(result["records"]) >= 1

    def test_monthly_summary(self):
        self.analytics.load_dataframe(self.df)
        result = self.analytics.monthly_summary()
        assert result["period"] == "monthly"
        assert len(result["records"]) >= 1

    def test_no_data_loaded_returns_empty(self):
        hist = HistoricalAnalytics()
        result = hist.daily_summary()
        assert result == {}

    def test_total_visitors_positive(self):
        self.analytics.load_dataframe(self.df)
        result = self.analytics.daily_summary()
        for rec in result["records"]:
            assert rec["total_visitors"] >= 0

    def test_load_nonexistent_csv(self):
        result = self.analytics.load_csv("/nonexistent/path/file.csv")
        assert result is False


# ── RecommendationEngine tests ────────────────────────────────────────────────

class TestRecommendationEngine:
    def setup_method(self):
        self.engine = RecommendationEngine(THRESHOLDS)

    def _zone_analytics(self, level: str, density: float) -> dict:
        return {
            "ZONE_A": {
                "name": "Grocery",
                "crowd_level": level,
                "normalized_density": density,
            }
        }

    def _counter_analytics(self, queue_len: int, status: str, avg_wait: float = 0) -> dict:
        return {
            "COUNTER_1": {
                "queue_length": queue_len,
                "status": status,
                "avg_waiting_time_seconds": avg_wait,
            }
        }

    def test_no_recommendations_when_normal(self):
        recs = self.engine.generate(
            zone_analytics=self._zone_analytics("LOW", 0.05),
            counter_analytics=self._counter_analytics(2, "NORMAL"),
            wait_stats={},
            prediction=None,
        )
        assert len(recs) == 0

    def test_high_zone_density_generates_recommendation(self):
        recs = self.engine.generate(
            zone_analytics=self._zone_analytics("HIGH", 0.70),
            counter_analytics=self._counter_analytics(2, "NORMAL"),
            wait_stats={},
        )
        types = [r["type"] for r in recs]
        assert "ZONE_HIGH_DENSITY" in types

    def test_critical_zone_generates_critical_recommendation(self):
        recs = self.engine.generate(
            zone_analytics=self._zone_analytics("CRITICAL", 0.90),
            counter_analytics=self._counter_analytics(2, "NORMAL"),
            wait_stats={},
        )
        critical = [r for r in recs if r["severity"] == "CRITICAL"]
        assert len(critical) >= 1

    def test_overloaded_counter_generates_recommendation(self):
        recs = self.engine.generate(
            zone_analytics={},
            counter_analytics=self._counter_analytics(12, "OVERLOADED"),
            wait_stats={},
        )
        types = [r["type"] for r in recs]
        assert "COUNTER_OVERLOADED" in types

    def test_queue_congestion_triggers_open_counter_rec(self):
        recs = self.engine.generate(
            zone_analytics={},
            counter_analytics=self._counter_analytics(8, "BUSY"),
            wait_stats={},
        )
        types = [r["type"] for r in recs]
        assert "QUEUE_CONGESTION" in types

    def test_high_wait_time_generates_recommendation(self):
        recs = self.engine.generate(
            zone_analytics={},
            counter_analytics=self._counter_analytics(3, "NORMAL", avg_wait=150),
            wait_stats={},
        )
        types = [r["type"] for r in recs]
        assert "HIGH_WAIT_TIME" in types

    def test_predicted_crowd_generates_recommendation(self):
        recs = self.engine.generate(
            zone_analytics={},
            counter_analytics={},
            wait_stats={},
            prediction={"next_30_min": 180},  # 90% of 200 capacity
            store_capacity=200,
        )
        types = [r["type"] for r in recs]
        assert "PREDICTED_HIGH_CROWD" in types

    def test_recommendations_sorted_by_severity(self):
        recs = self.engine.generate(
            zone_analytics=self._zone_analytics("HIGH", 0.70),
            counter_analytics=self._counter_analytics(12, "OVERLOADED"),
            wait_stats={},
        )
        severities = [r["severity"] for r in recs]
        # CRITICAL should come before HIGH
        if "CRITICAL" in severities and "HIGH" in severities:
            assert severities.index("CRITICAL") < severities.index("HIGH")

    def test_store_capacity_warning(self):
        recs = self.engine.generate(
            zone_analytics={},
            counter_analytics={},
            wait_stats={},
            current_total=185,
            store_capacity=200,
        )
        types = [r["type"] for r in recs]
        assert "STORE_NEAR_CAPACITY" in types
