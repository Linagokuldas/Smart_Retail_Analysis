"""
Crowd Level Classifier.

Classifies crowd density into discrete levels:
    LOW | MEDIUM | HIGH | CRITICAL

Thresholds are loaded from config/thresholds.json and are NOT hard-coded.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

LEVEL_LOW = "LOW"
LEVEL_MEDIUM = "MEDIUM"
LEVEL_HIGH = "HIGH"
LEVEL_CRITICAL = "CRITICAL"


class CrowdLevelClassifier:
    """
    Classifies normalized crowd density into human-readable levels.
    """

    def __init__(self, thresholds: dict):
        """
        Args:
            thresholds: Full thresholds.json dict. Reads from 'crowd' sub-key.
        """
        crowd_cfg = thresholds.get("crowd", {})
        self.low = float(crowd_cfg.get("density_low", 0.10))
        self.medium = float(crowd_cfg.get("density_medium", 0.30))
        self.high = float(crowd_cfg.get("density_high", 0.60))
        self.critical = float(crowd_cfg.get("density_critical", 0.80))

        logger.debug(
            "CrowdLevelClassifier thresholds: LOW=%.2f MEDIUM=%.2f HIGH=%.2f CRITICAL=%.2f",
            self.low, self.medium, self.high, self.critical,
        )

    def classify(self, normalized_density: float) -> str:
        """
        Classify a single normalized density value.

        Args:
            normalized_density: Float in [0, 1] from DensityCalculator.

        Returns:
            One of: "LOW", "MEDIUM", "HIGH", "CRITICAL".
        """
        d = max(0.0, min(1.0, float(normalized_density)))

        if d >= self.critical:
            return LEVEL_CRITICAL
        elif d >= self.high:
            return LEVEL_HIGH
        elif d >= self.medium:
            return LEVEL_MEDIUM
        else:
            return LEVEL_LOW

    def classify_all_zones(self, density_results: dict[str, dict]) -> dict[str, dict]:
        """
        Classify crowd level for every zone in the density results.

        Args:
            density_results: Output of DensityCalculator.calculate().

        Returns:
            Updated dict with 'crowd_level' added to each zone.
        """
        classified = {}
        for zone_id, zone_data in density_results.items():
            entry = dict(zone_data)  # Shallow copy
            nd = entry.get("normalized_density", 0.0)
            entry["crowd_level"] = self.classify(nd)
            classified[zone_id] = entry

        return classified
