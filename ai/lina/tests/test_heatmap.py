"""
Tests for the HeatmapGenerator module.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.heatmap.heatmap_generator import HeatmapGenerator


def make_event(track_id: int, cx: float, cy: float) -> dict:
    return {
        "timestamp": "2026-09-13T12:00:00",
        "camera_id": "CAM01",
        "track_id": track_id,
        "class": "person",
        "confidence": 0.92,
        "bbox": [cx - 20, cy - 40, cx + 20, cy + 40],
        "center": [cx, cy],
    }


class TestHeatmapGenerator:
    def setup_method(self):
        # Use tmp output dir to avoid side effects
        self.heatmap = HeatmapGenerator(width=200, height=150, output_dir=None)

    def test_initial_accumulator_is_zero(self):
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.shape == (150, 200)
        assert grid.sum() == 0.0

    def test_add_frame_increments_accumulator(self):
        events = [make_event(1, 100, 75)]
        self.heatmap.add_frame(events)
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() > 0

    def test_add_frame_clamps_out_of_bounds(self):
        """Coordinates outside frame should be clamped, not cause errors."""
        events = [
            make_event(1, -50, -50),   # Negative coords
            make_event(2, 5000, 5000),  # Way out of bounds
        ]
        self.heatmap.add_frame(events)
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() >= 0  # No exception raised

    def test_multiple_frames_accumulate(self):
        events = [make_event(1, 100, 75)]
        self.heatmap.add_frame(events)
        self.heatmap.add_frame(events)
        grid = self.heatmap.get_numerical_heatmap()
        assert grid[75, 100] == 2.0  # Accumulated twice

    def test_normalized_heatmap_max_is_1(self):
        events = [make_event(1, 100, 75)]
        self.heatmap.add_frame(events)
        normalized = self.heatmap.get_normalized_heatmap()
        assert normalized.max() == pytest.approx(1.0)

    def test_normalized_empty_heatmap_is_zero(self):
        normalized = self.heatmap.get_normalized_heatmap()
        assert normalized.max() == 0.0

    def test_add_multiple_frames_dict(self):
        frames = {
            "2026-09-13T12:00:00": [make_event(1, 50, 50)],
            "2026-09-13T12:00:30": [make_event(2, 100, 75)],
            "2026-09-13T12:01:00": [make_event(3, 150, 100)],
        }
        self.heatmap.add_multiple_frames(frames)
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() == 3.0

    def test_get_top_hotspots_returns_sorted(self):
        # Add points at known locations
        events = [
            make_event(1, 50, 50),
            make_event(2, 50, 50),  # Same spot — accumulated count = 2
            make_event(3, 100, 75),  # count = 1
        ]
        self.heatmap.add_frame(events)
        hotspots = self.heatmap.get_top_hotspots(top_n=2)
        assert len(hotspots) == 2
        assert hotspots[0]["count"] >= hotspots[1]["count"]  # Sorted descending

    def test_reset_clears_accumulator(self):
        self.heatmap.add_frame([make_event(1, 100, 75)])
        self.heatmap.reset()
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() == 0.0

    def test_invalid_center_skipped(self):
        events = [{"timestamp": "2026-09-13T12:00:00", "track_id": 1, "center": None}]
        self.heatmap.add_frame(events)  # Should not raise
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() == 0.0

    def test_empty_frame_does_not_error(self):
        self.heatmap.add_frame([])
        grid = self.heatmap.get_numerical_heatmap()
        assert grid.sum() == 0.0
