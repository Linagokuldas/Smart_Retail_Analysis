"""
Tests for crowd analytics modules:
- ZoneCounter (zone assignment, point-in-polygon, deduplication)
- DensityCalculator
- CrowdLevelClassifier
"""

import sys
from pathlib import Path
import pytest

LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.crowd.zone_counter import ZoneCounter, point_in_polygon
from src.crowd.density import DensityCalculator
from src.crowd.crowd_level import CrowdLevelClassifier


# ── Fixtures ─────────────────────────────────────────────────────────────────

ZONES_CONFIG = {
    "zones": [
        {
            "id": "ZONE_A",
            "name": "Grocery",
            "polygon": [[0, 0], [400, 0], [400, 300], [0, 300]],
            "area_sqm": 120.0,
        },
        {
            "id": "ZONE_B",
            "name": "Beverages",
            "polygon": [[400, 0], [800, 0], [800, 300], [400, 300]],
            "area_sqm": 120.0,
        },
        {
            "id": "BILLING",
            "name": "Billing",
            "polygon": [[0, 600], [1000, 600], [1000, 800], [0, 800]],
            "area_sqm": 80.0,
        },
    ]
}

THRESHOLDS = {
    "crowd": {
        "density_low": 0.10,
        "density_medium": 0.30,
        "density_high": 0.60,
        "density_critical": 0.80,
    }
}

def make_event(track_id: int, cx: float, cy: float) -> dict:
    return {
        "timestamp": "2026-09-13T12:00:00",
        "camera_id": "CAM01",
        "track_id": track_id,
        "class": "person",
        "confidence": 0.95,
        "bbox": [cx - 20, cy - 40, cx + 20, cy + 40],
        "center": [cx, cy],
    }


# ── point_in_polygon tests ────────────────────────────────────────────────────

class TestPointInPolygon:
    def test_inside_square(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon([50, 50], poly) is True

    def test_outside_square(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        assert point_in_polygon([150, 50], poly) is False

    def test_corner_case(self):
        poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
        # Points very near edge — should not raise exception
        result = point_in_polygon([0.1, 0.1], poly)
        assert isinstance(result, bool)

    def test_triangle(self):
        poly = [[0, 0], [200, 0], [100, 200]]
        assert point_in_polygon([100, 50], poly) is True
        assert point_in_polygon([10, 190], poly) is False


# ── ZoneCounter tests ─────────────────────────────────────────────────────────

class TestZoneCounter:
    def setup_method(self):
        self.counter = ZoneCounter(ZONES_CONFIG)

    def test_person_in_zone_a(self):
        events = [make_event(1, 200, 150)]  # Inside ZONE_A
        assignments = self.counter.assign_zones(events)
        assert len(assignments["ZONE_A"]) == 1
        assert len(assignments["ZONE_B"]) == 0

    def test_person_in_zone_b(self):
        events = [make_event(1, 600, 150)]  # Inside ZONE_B
        assignments = self.counter.assign_zones(events)
        assert len(assignments["ZONE_B"]) == 1
        assert len(assignments["ZONE_A"]) == 0

    def test_no_double_counting_same_track_id(self):
        """Same track_id appearing twice in one frame → counted once."""
        events = [
            make_event(1, 200, 150),
            make_event(1, 210, 155),  # Same track_id, slight position diff
        ]
        assignments = self.counter.assign_zones(events)
        assert len(assignments["ZONE_A"]) == 1  # Only counted once

    def test_multiple_people_in_zone(self):
        events = [
            make_event(1, 100, 100),
            make_event(2, 200, 200),
            make_event(3, 300, 150),
        ]
        assignments = self.counter.assign_zones(events)
        assert len(assignments["ZONE_A"]) == 3

    def test_person_outside_all_zones(self):
        events = [make_event(1, 950, 400)]  # Not in any configured zone
        assignments = self.counter.assign_zones(events)
        total_assigned = sum(len(v) for v in assignments.values())
        assert total_assigned == 0

    def test_empty_frame(self):
        assignments = self.counter.assign_zones([])
        assert all(len(v) == 0 for v in assignments.values())

    def test_count_per_zone_total(self):
        events = [make_event(1, 200, 150), make_event(2, 600, 150)]
        assignments = self.counter.assign_zones(events)
        counts = self.counter.count_per_zone(assignments)
        assert counts.pop("__total__") == 2
        assert counts["ZONE_A"]["people_count"] == 1
        assert counts["ZONE_B"]["people_count"] == 1

    def test_get_zone_area(self):
        area = self.counter.get_zone_area("ZONE_A")
        assert area == 120.0

    def test_unknown_zone_area_returns_default(self):
        area = self.counter.get_zone_area("NONEXISTENT_ZONE")
        assert area == 1.0

    def test_empty_zone_config(self):
        counter = ZoneCounter({})
        events = [make_event(1, 100, 100)]
        assignments = counter.assign_zones(events)
        assert assignments == {}


# ── DensityCalculator tests ───────────────────────────────────────────────────

class TestDensityCalculator:
    def setup_method(self):
        self.zone_counter = ZoneCounter(ZONES_CONFIG)
        self.calc = DensityCalculator(self.zone_counter, max_density_ppsm=2.0)

    def test_density_calculation(self):
        zone_counts = {
            "ZONE_A": {"name": "Grocery", "people_count": 12, "track_ids": []},
        }
        result = self.calc.calculate(zone_counts)
        expected_density = 12 / 120.0  # 0.1 ppsm
        assert abs(result["ZONE_A"]["density_ppsm"] - expected_density) < 0.001

    def test_normalized_density_clamped_at_1(self):
        zone_counts = {
            "ZONE_A": {"name": "Grocery", "people_count": 1000, "track_ids": []},
        }
        result = self.calc.calculate(zone_counts)
        assert result["ZONE_A"]["normalized_density"] == 1.0

    def test_zero_people(self):
        zone_counts = {
            "ZONE_A": {"name": "Grocery", "people_count": 0, "track_ids": []},
        }
        result = self.calc.calculate(zone_counts)
        assert result["ZONE_A"]["density_ppsm"] == 0.0
        assert result["ZONE_A"]["normalized_density"] == 0.0

    def test_excludes_total_key(self):
        zone_counts = {
            "ZONE_A": {"name": "Grocery", "people_count": 5, "track_ids": []},
            "__total__": 5,
        }
        result = self.calc.calculate(zone_counts)
        assert "__total__" not in result


# ── CrowdLevelClassifier tests ────────────────────────────────────────────────

class TestCrowdLevelClassifier:
    def setup_method(self):
        self.classifier = CrowdLevelClassifier(THRESHOLDS)

    def test_low_level(self):
        assert self.classifier.classify(0.05) == "LOW"

    def test_medium_level(self):
        assert self.classifier.classify(0.35) == "MEDIUM"  # 0.30 <= 0.35 < 0.60

    def test_high_level(self):
        assert self.classifier.classify(0.70) == "HIGH"

    def test_critical_level(self):
        assert self.classifier.classify(0.90) == "CRITICAL"

    def test_exact_boundary_medium(self):
        # At exactly medium threshold
        assert self.classifier.classify(0.30) == "MEDIUM"

    def test_exact_boundary_critical(self):
        assert self.classifier.classify(0.80) == "CRITICAL"

    def test_classify_all_zones(self):
        density_results = {
            "ZONE_A": {"name": "Grocery", "normalized_density": 0.05},
            "ZONE_B": {"name": "Beverages", "normalized_density": 0.85},
        }
        result = self.classifier.classify_all_zones(density_results)
        assert result["ZONE_A"]["crowd_level"] == "LOW"
        assert result["ZONE_B"]["crowd_level"] == "CRITICAL"

    def test_out_of_range_density_clamped(self):
        # Values outside [0,1] should be handled gracefully
        assert self.classifier.classify(-0.5) == "LOW"
        assert self.classifier.classify(1.5) == "CRITICAL"
