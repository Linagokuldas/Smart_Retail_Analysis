"""
Crowd Density Calculator.

Computes crowd density per zone as:
    density = people_count / zone_area_sqm

Density is expressed as people per square metre (ppsm).
A normalized_density [0, 1] is also provided relative to a configurable
max capacity assumption, for threshold comparison.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default: assume a "full" zone is 2 people/sqm (can be overridden via config)
DEFAULT_MAX_DENSITY_PPSM = 2.0


class DensityCalculator:
    """
    Calculates per-zone crowd density from zone counts and configured areas.
    """

    def __init__(self, zone_counter, max_density_ppsm: float = DEFAULT_MAX_DENSITY_PPSM):
        """
        Args:
            zone_counter: ZoneCounter instance (provides area lookups).
            max_density_ppsm: Maximum density (people/sqm) considered 100% full.
                              Used to compute normalized density for threshold comparison.
        """
        self.zone_counter = zone_counter
        self.max_density_ppsm = max(max_density_ppsm, 0.01)  # Guard division by zero

    def calculate(self, zone_counts: dict[str, dict]) -> dict[str, dict]:
        """
        Compute density metrics for all zones.

        Args:
            zone_counts: Output of ZoneCounter.count_per_zone() — zone_id → {people_count, ...}.

        Returns:
            Dict zone_id → {people_count, area_sqm, density_ppsm, normalized_density}.
        """
        results = {}

        for zone_id, zone_data in zone_counts.items():
            if zone_id == "__total__":
                continue

            people_count = zone_data.get("people_count", 0)
            area_sqm = self.zone_counter.get_zone_area(zone_id)

            if area_sqm <= 0:
                logger.warning("Zone '%s' has area <= 0; using 1.0 as fallback", zone_id)
                area_sqm = 1.0

            density_ppsm = people_count / area_sqm
            normalized = min(density_ppsm / self.max_density_ppsm, 1.0)

            results[zone_id] = {
                "name": zone_data.get("name", zone_id),
                "people_count": people_count,
                "area_sqm": area_sqm,
                "density_ppsm": round(density_ppsm, 4),
                "normalized_density": round(normalized, 4),
            }

        return results
