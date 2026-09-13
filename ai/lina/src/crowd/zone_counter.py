"""
Zone Counter — assigns tracked persons to store zones and counts them.

Uses polygon-based point-in-polygon test (ray casting) to determine
which zone each person's center coordinate falls into.
Each unique track_id is counted only once per frame to prevent double-counting.
"""

from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def point_in_polygon(point: list[float], polygon: list[list[float]]) -> bool:
    """
    Determine whether a 2D point lies inside a polygon using ray casting.

    Args:
        point: [x, y] coordinate.
        polygon: List of [x, y] vertices defining the polygon (closed implicitly).

    Returns:
        True if point is inside or on the edge of the polygon.
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    px, py = polygon[0][0], polygon[0][1]
    for i in range(1, n + 1):
        qx, qy = polygon[i % n][0], polygon[i % n][1]
        if ((py <= y < qy) or (qy <= y < py)) and (x < (qx - px) * (y - py) / (qy - py + 1e-10) + px):
            inside = not inside
        px, py = qx, qy

    return inside


class ZoneCounter:
    """
    Assigns tracked persons to configured zones and computes per-zone counts.
    """

    def __init__(self, zones_config: dict):
        """
        Args:
            zones_config: Loaded zones.json dict containing "zones" list.
        """
        self.zones = zones_config.get("zones", [])
        if not self.zones:
            logger.warning("ZoneCounter initialized with no zones configured")

    def assign_zones(self, frame_events: list[dict]) -> dict[str, list[dict]]:
        """
        Assign each person detection to their zone(s).

        A person is assigned to the first zone whose polygon contains their center.
        Each track_id appears at most once per zone per frame (deduplication).

        Args:
            frame_events: List of validated tracking event dicts for one frame.

        Returns:
            Dict mapping zone_id → list of tracking events in that zone.
        """
        zone_map: dict[str, list[dict]] = {z["id"]: [] for z in self.zones}
        seen_per_zone: dict[str, set] = {z["id"]: set() for z in self.zones}

        for event in frame_events:
            center = event.get("center")
            if not center or len(center) < 2:
                logger.debug("Event track_id=%s has invalid center; skipping zone assignment", event.get("track_id"))
                continue

            track_id = event["track_id"]
            assigned = False

            for zone in self.zones:
                zone_id = zone["id"]
                try:
                    if point_in_polygon(center, zone["polygon"]):
                        if track_id not in seen_per_zone[zone_id]:
                            zone_map[zone_id].append(event)
                            seen_per_zone[zone_id].add(track_id)
                        assigned = True
                        break  # First-match: a person is in one zone only
                except (KeyError, TypeError, ZeroDivisionError) as exc:
                    logger.warning("Zone '%s' polygon error: %s", zone.get("id"), exc)

            if not assigned:
                logger.debug("track_id=%s center=%s not assigned to any zone", track_id, center)

        return zone_map

    def count_per_zone(self, zone_assignments: dict[str, list[dict]]) -> dict[str, dict]:
        """
        Compute crowd count metrics per zone.

        Args:
            zone_assignments: Output of assign_zones().

        Returns:
            Dict mapping zone_id → {people_count, track_ids}.
        """
        results = {}
        total = 0

        for zone in self.zones:
            zone_id = zone["id"]
            events = zone_assignments.get(zone_id, [])
            count = len(events)
            track_ids = [e["track_id"] for e in events]
            total += count

            results[zone_id] = {
                "name": zone.get("name", zone_id),
                "people_count": count,
                "track_ids": track_ids,
            }

        results["__total__"] = total
        return results

    def get_zone_area(self, zone_id: str) -> float:
        """
        Get configured area (in sqm) for a zone.

        Args:
            zone_id: Zone identifier string.

        Returns:
            Area in sqm, or 1.0 as fallback to avoid division by zero.
        """
        for zone in self.zones:
            if zone["id"] == zone_id:
                return float(zone.get("area_sqm", 1.0))
        logger.warning("Area for zone '%s' not found; defaulting to 1.0 sqm", zone_id)
        return 1.0
