"""
Tests for queue analytics modules:
- QueueDetector
- WaitingTimeTracker
- CounterAnalytics
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pytest

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.queue.queue_detector import QueueDetector
from src.queue.waiting_time import WaitingTimeTracker
from src.queue.counter_analytics import CounterAnalytics


# ── Fixtures ─────────────────────────────────────────────────────────────────

ZONES_CONFIG = {
    "zones": [],
    "queue_regions": [
        {
            "id": "QUEUE_COUNTER_1",
            "name": "Counter 1 Queue",
            "counter_id": "COUNTER_1",
            "polygon": [[0, 620], [200, 620], [200, 800], [0, 800]],
            "max_capacity": 10,
        },
        {
            "id": "QUEUE_COUNTER_2",
            "name": "Counter 2 Queue",
            "counter_id": "COUNTER_2",
            "polygon": [[220, 620], [440, 620], [440, 800], [220, 800]],
            "max_capacity": 10,
        },
    ],
    "billing_counters": [
        {"id": "COUNTER_1", "name": "Counter 1", "queue_region_id": "QUEUE_COUNTER_1"},
        {"id": "COUNTER_2", "name": "Counter 2", "queue_region_id": "QUEUE_COUNTER_2"},
    ],
}

THRESHOLDS = {
    "queue": {
        "min_people_to_form_queue": 3,
        "queue_congestion_length": 7,
        "queue_overloaded_length": 12,
        "grace_period_seconds": 2,  # Short for tests
    },
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
}


def make_event(track_id: int, cx: float, cy: float) -> dict:
    return {
        "timestamp": "2026-09-13T12:00:00",
        "camera_id": "CAM_BILLING",
        "track_id": track_id,
        "class": "person",
        "confidence": 0.95,
        "bbox": [cx - 15, cy - 30, cx + 15, cy + 30],
        "center": [cx, cy],
    }


# ── QueueDetector tests ───────────────────────────────────────────────────────

class TestQueueDetector:
    def setup_method(self):
        self.detector = QueueDetector(ZONES_CONFIG, THRESHOLDS)

    def test_detects_queue_when_enough_people(self):
        events = [
            make_event(1, 50, 700),
            make_event(2, 80, 720),
            make_event(3, 110, 680),
            make_event(4, 140, 750),
        ]
        result = self.detector.detect(events)
        assert result["QUEUE_COUNTER_1"]["queue_detected"] is True

    def test_no_queue_with_few_people(self):
        events = [
            make_event(1, 50, 700),
            make_event(2, 80, 720),
        ]
        result = self.detector.detect(events)
        assert result["QUEUE_COUNTER_1"]["queue_detected"] is False

    def test_empty_frame_no_queues(self):
        result = self.detector.detect([])
        for region_id, data in result.items():
            assert data["people_in_queue"] == 0
            assert data["queue_detected"] is False

    def test_people_counted_in_correct_queue(self):
        # Put people in COUNTER_1 area
        counter1_events = [make_event(i, 50 + i * 10, 700) for i in range(5)]
        # Put people in COUNTER_2 area
        counter2_events = [make_event(i + 100, 280 + i * 10, 700) for i in range(2)]

        result = self.detector.detect(counter1_events + counter2_events)
        assert result["QUEUE_COUNTER_1"]["people_in_queue"] == 5
        assert result["QUEUE_COUNTER_2"]["people_in_queue"] == 2

    def test_no_double_counting_same_track_id(self):
        events = [make_event(1, 50, 700), make_event(1, 55, 705)]  # Same track_id
        result = self.detector.detect(events)
        assert result["QUEUE_COUNTER_1"]["people_in_queue"] == 1

    def test_utilization_capped_at_1(self):
        # Put 20 people in a region with max_capacity=10
        events = [make_event(i, 50 + (i % 10) * 10, 700 + (i // 10) * 20) for i in range(20)]
        result = self.detector.detect(events)
        util = result["QUEUE_COUNTER_1"]["utilization"]
        assert 0.0 <= util <= 1.0


# ── WaitingTimeTracker tests ──────────────────────────────────────────────────

class TestWaitingTimeTracker:
    def setup_method(self):
        self.tracker = WaitingTimeTracker(THRESHOLDS)

    def _make_detection(self, track_ids: list[int], region_id: str = "QUEUE_COUNTER_1") -> dict:
        return {
            region_id: {
                "counter_id": "COUNTER_1",
                "people_in_queue": len(track_ids),
                "queue_detected": len(track_ids) >= 3,
                "track_ids": track_ids,
            }
        }

    def test_tracks_entry(self):
        detections = self._make_detection([1, 2, 3])
        self.tracker.update(detections, "2026-09-13T12:00:00")
        active = self.tracker._active.get("QUEUE_COUNTER_1", {})
        assert 1 in active
        assert 2 in active

    def test_person_counted_as_waiting_after_grace(self):
        detections_with = self._make_detection([1, 2, 3])
        detections_without = self._make_detection([])

        # T=00:00: Persons 1,2,3 enter the queue (entry_time set, last_seen=T0)
        self.tracker.update(detections_with, "2026-09-13T12:00:00")
        # T=00:15: Persons absent; elapsed since last_seen = 15s > 2s grace → they exit
        self.tracker.update(detections_without, "2026-09-13T12:00:15")
        # T=00:30: Still absent; tracks should already be removed
        self.tracker.update(detections_without, "2026-09-13T12:00:30")

        completed = self.tracker._completed.get("QUEUE_COUNTER_1", [])
        # All 3 persons should have completed their wait
        assert len(completed) == 3  # All 3 exited after grace period

    def test_person_not_removed_within_grace_period(self):
        detections_with = self._make_detection([1])
        detections_without = self._make_detection([])

        self.tracker.update(detections_with, "2026-09-13T12:00:00")
        # Only 1 second elapsed — within 2s grace period
        self.tracker.update(detections_without, "2026-09-13T12:00:01")

        active = self.tracker._active.get("QUEUE_COUNTER_1", {})
        assert 1 in active  # Still considered in queue

    def test_statistics_with_no_data(self):
        detections = self._make_detection([1, 2])
        stats = self.tracker.get_statistics(detections)
        assert stats["QUEUE_COUNTER_1"]["avg_waiting_time_seconds"] == 0.0
        assert stats["QUEUE_COUNTER_1"]["completed_waits"] == 0

    def test_reset_clears_state(self):
        detections = self._make_detection([1, 2, 3])
        self.tracker.update(detections, "2026-09-13T12:00:00")
        self.tracker.reset()
        assert self.tracker._active == {}
        assert self.tracker._completed == {}


# ── CounterAnalytics tests ────────────────────────────────────────────────────

class TestCounterAnalytics:
    def setup_method(self):
        self.analytics = CounterAnalytics(ZONES_CONFIG, THRESHOLDS)

    def _make_queue_detections(self, length: int, avg_wait: float = 0.0) -> tuple[dict, dict]:
        queue_det = {
            "QUEUE_COUNTER_1": {
                "counter_id": "COUNTER_1",
                "people_in_queue": length,
                "queue_detected": length >= 3,
                "max_capacity": 10,
                "utilization": min(length / 10.0, 1.0),
                "track_ids": list(range(length)),
            }
        }
        wait_stats = {
            "QUEUE_COUNTER_1": {
                "avg_waiting_time_seconds": avg_wait,
                "median_waiting_time_seconds": avg_wait,
                "max_waiting_time_seconds": avg_wait * 1.5,
                "min_waiting_time_seconds": 0.0,
                "completed_waits": 3,
                "currently_waiting": length,
            }
        }
        return queue_det, wait_stats

    def test_normal_status_short_queue(self):
        qd, ws = self._make_queue_detections(2, avg_wait=30)
        result = self.analytics.analyze(qd, ws)
        assert result["COUNTER_1"]["status"] == "NORMAL"

    def test_busy_status_medium_queue(self):
        qd, ws = self._make_queue_detections(6, avg_wait=50)
        result = self.analytics.analyze(qd, ws)
        assert result["COUNTER_1"]["status"] == "BUSY"

    def test_overloaded_status_long_queue(self):
        qd, ws = self._make_queue_detections(12, avg_wait=100)
        result = self.analytics.analyze(qd, ws)
        assert result["COUNTER_1"]["status"] == "OVERLOADED"

    def test_overloaded_by_wait_time(self):
        qd, ws = self._make_queue_detections(2, avg_wait=300)  # High wait, short queue
        result = self.analytics.analyze(qd, ws)
        assert result["COUNTER_1"]["status"] == "OVERLOADED"

    def test_counter_with_no_detections_is_normal(self):
        result = self.analytics.analyze({}, {})
        # Both configured counters should default to NORMAL
        assert result["COUNTER_1"]["status"] == "NORMAL"
        assert result["COUNTER_2"]["status"] == "NORMAL"

    def test_empty_frame_returns_all_counters(self):
        result = self.analytics.analyze({}, {})
        assert "COUNTER_1" in result
        assert "COUNTER_2" in result
