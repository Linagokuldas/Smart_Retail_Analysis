"""
Integration tests for the full LinaPipeline.

Runs the complete pipeline using mock tracking events.
No real camera, no external dependencies required.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pytest

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.pipeline.lina_pipeline import LinaPipeline


# ── Config fixtures ───────────────────────────────────────────────────────────

def load_configs() -> tuple[dict, dict, dict]:
    def load(name):
        p = LINA_ROOT / "config" / name
        with open(p) as f:
            return json.load(f)
    return load("zones.json"), load("thresholds.json"), load("prediction_config.json")


def make_events(n_people: int, base_ts: str = "2026-09-13T12:00:00") -> list[dict]:
    """Generate minimal valid tracking events for testing."""
    events = []
    for i in range(n_people):
        cx = 100 + (i % 10) * 40
        cy = 150 + (i // 10) * 40
        events.append({
            "timestamp": base_ts,
            "camera_id": "CAM01",
            "track_id": i + 1,
            "class": "person",
            "confidence": 0.90,
            "bbox": [cx - 20, cy - 40, cx + 20, cy + 40],
            "center": [cx, cy],
        })
    return events


# ── Pipeline tests ────────────────────────────────────────────────────────────

class TestLinaPipeline:
    def setup_method(self):
        zones_cfg, thresholds, pred_cfg = load_configs()
        self.pipeline = LinaPipeline(
            zones_config=zones_cfg,
            thresholds=thresholds,
            prediction_config=pred_cfg,
            store_capacity=200,
            frame_width=1000,
            frame_height=800,
            output_dir=None,  # Don't save files during tests
            heatmap_dir=None,
        )

    def test_pipeline_runs_without_error(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        assert isinstance(output, dict)

    def test_output_has_required_keys(self):
        events = make_events(15)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        required = [
            "frame_timestamp", "people", "zones", "queues",
            "peak_hours", "prediction", "recommendations"
        ]
        for key in required:
            assert key in output, f"Missing key in output: {key}"

    def test_people_total_is_correct(self):
        events = make_events(12)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        total = output["people"]["total"]
        # Total should be <= 12 (some may fall outside configured zones)
        assert total >= 0
        assert isinstance(total, int)

    def test_zones_in_output(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        zones = output.get("zones", {})
        assert isinstance(zones, dict)
        for zone_data in zones.values():
            assert "count" in zone_data
            assert "crowd_level" in zone_data
            assert "normalized_density" in zone_data

    def test_queues_in_output(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        queues = output.get("queues", {})
        assert isinstance(queues, dict)

    def test_recommendations_is_list(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        assert isinstance(output["recommendations"], list)

    def test_empty_events_returns_empty(self):
        output = self.pipeline.process_events([], save_json=False, save_heatmap=False)
        assert output == {}

    def test_invalid_events_handled_gracefully(self):
        invalid_events = [
            {"track_id": "bad", "center": None},
            {"timestamp": "2026-09-13T12:00:00"},  # Missing required fields
            None,
        ]
        output = self.pipeline.process_events(invalid_events, save_json=False, save_heatmap=False)
        # Should not raise — should return empty or partial
        assert isinstance(output, dict)

    def test_schema_version_in_output(self):
        events = make_events(5)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        assert "schema_version" in output

    def test_synthetic_label_in_output(self):
        events = make_events(5)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        assert "SYNTHETIC" in output.get("data_classification", "")

    def test_multiple_frames_processed(self):
        events = []
        for i in range(5):
            ts = f"2026-09-13T12:0{i}:00"
            events.extend(make_events(8, base_ts=ts))
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        assert isinstance(output, dict)

    def test_crowd_level_valid_values(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for zone_data in output.get("zones", {}).values():
            assert zone_data["crowd_level"] in valid_levels

    def test_counter_status_valid_values(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        valid_statuses = {"NORMAL", "BUSY", "OVERLOADED"}
        for counter_data in output.get("queues", {}).values():
            assert counter_data["status"] in valid_statuses

    def test_heatmap_not_saved_when_disabled(self):
        events = make_events(10)
        output = self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        # heatmap_path should be None or not present when disabled
        heatmap = output.get("heatmap_path")
        assert heatmap is None

    def test_session_summary(self):
        events = make_events(10)
        self.pipeline.process_events(events, save_json=False, save_heatmap=False)
        summary = self.pipeline.get_session_summary()
        assert "frames_processed" in summary
        assert summary["frames_processed"] >= 1

    def test_load_historical_data_from_csv(self, tmp_path):
        """Test that historical CSV can be loaded into pipeline."""
        import csv
        csv_path = tmp_path / "test_history.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "people_count", "queue_length"])
            writer.writeheader()
            for i in range(50):
                ts = datetime(2026, 9, 1, 10, 0, 0) + timedelta(minutes=i * 5)
                writer.writerow({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"), "people_count": 30 + i, "queue_length": 2})

        result = self.pipeline.load_historical_data(str(csv_path))
        assert result is True
