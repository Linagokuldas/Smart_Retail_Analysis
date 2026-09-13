"""
Rule-Based Recommendation Engine.

Generates actionable recommendations based on:
- Current crowd levels per zone
- Queue lengths and counter status
- Waiting time statistics
- Predicted future crowd

All thresholds are loaded from config — NOT hard-coded.
"""

from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"


class RecommendationEngine:
    """
    Generates human-readable, actionable recommendations from analytics results.
    """

    def __init__(self, thresholds: dict):
        """
        Args:
            thresholds: Loaded thresholds.json dict.
        """
        rec_cfg = thresholds.get("recommendations", {})
        self.open_counter_queue = int(rec_cfg.get("open_counter_queue_length", 7))
        self.staff_density = float(rec_cfg.get("staff_allocation_density", 0.60))
        self.predicted_warning_ratio = float(rec_cfg.get("predicted_crowd_warning_ratio", 0.80))

        wait_cfg = thresholds.get("waiting_time", {})
        self.busy_wait_sec = float(wait_cfg.get("busy_threshold_seconds", 120))
        self.critical_wait_sec = float(wait_cfg.get("critical_threshold_seconds", 300))

    def generate(
        self,
        zone_analytics: dict[str, dict],
        counter_analytics: dict[str, dict],
        wait_stats: dict[str, dict],
        prediction: Optional[dict] = None,
        current_total: int = 0,
        store_capacity: int = 200,
    ) -> list[dict]:
        """
        Generate all recommendations for the current analytics snapshot.

        Args:
            zone_analytics: Classified zone analytics dict (zone_id → {crowd_level, normalized_density, ...}).
            counter_analytics: Counter analytics dict (counter_id → {queue_length, status, avg_wait}).
            wait_stats: Waiting time stats dict (region_id → {avg_waiting_time_seconds, ...}).
            prediction: Optional prediction dict {next_15_min, next_30_min}.
            current_total: Current total people in store.
            store_capacity: Maximum configured store capacity.

        Returns:
            List of recommendation dicts sorted by severity.
        """
        recommendations = []
        ts = datetime.now().isoformat()

        # 1. Zone crowd density recommendations
        for zone_id, zone_data in zone_analytics.items():
            level = zone_data.get("crowd_level", "LOW")
            density = zone_data.get("normalized_density", 0.0)
            name = zone_data.get("name", zone_id)

            if level == "CRITICAL":
                recommendations.append(self._build(
                    rec_type="ZONE_CRITICAL_DENSITY",
                    severity=SEVERITY_CRITICAL,
                    message=f"Critical crowd density in {name}.",
                    recommendation=f"Immediately redirect customers away from {name} and deploy additional staff.",
                    zone=zone_id,
                    timestamp=ts,
                ))
            elif level == "HIGH" and density >= self.staff_density:
                recommendations.append(self._build(
                    rec_type="ZONE_HIGH_DENSITY",
                    severity=SEVERITY_HIGH,
                    message=f"High crowd concentration in {name}.",
                    recommendation=f"Consider staff reallocation to {name} to manage flow.",
                    zone=zone_id,
                    timestamp=ts,
                ))

        # 2. Queue / counter recommendations
        for counter_id, counter_data in counter_analytics.items():
            queue_length = counter_data.get("queue_length", 0)
            status = counter_data.get("status", "NORMAL")
            avg_wait = counter_data.get("avg_waiting_time_seconds", 0.0)

            if status == "OVERLOADED":
                recommendations.append(self._build(
                    rec_type="COUNTER_OVERLOADED",
                    severity=SEVERITY_CRITICAL,
                    message=f"{counter_id} is overloaded (queue: {queue_length}).",
                    recommendation="Open an additional billing counter immediately.",
                    counter=counter_id,
                    timestamp=ts,
                ))
            elif status == "BUSY" and queue_length >= self.open_counter_queue:
                recommendations.append(self._build(
                    rec_type="QUEUE_CONGESTION",
                    severity=SEVERITY_HIGH,
                    message=f"{counter_id} queue exceeds threshold (length: {queue_length}).",
                    recommendation="Consider opening another billing counter.",
                    counter=counter_id,
                    timestamp=ts,
                ))

            if avg_wait >= self.critical_wait_sec:
                recommendations.append(self._build(
                    rec_type="CRITICAL_WAIT_TIME",
                    severity=SEVERITY_CRITICAL,
                    message=f"Critical waiting time at {counter_id}: {avg_wait:.0f}s.",
                    recommendation="Urgent: open additional counter or assign more cashiers.",
                    counter=counter_id,
                    timestamp=ts,
                ))
            elif avg_wait >= self.busy_wait_sec:
                recommendations.append(self._build(
                    rec_type="HIGH_WAIT_TIME",
                    severity=SEVERITY_HIGH,
                    message=f"Billing congestion at {counter_id}: avg wait {avg_wait:.0f}s.",
                    recommendation="Monitor and prepare to open an additional counter.",
                    counter=counter_id,
                    timestamp=ts,
                ))

        # 3. Predicted crowd recommendations
        if prediction:
            next_30 = prediction.get("next_30_min")
            if next_30 is not None and store_capacity > 0 and next_30 >= self.predicted_warning_ratio * store_capacity:
                recommendations.append(self._build(
                    rec_type="PREDICTED_HIGH_CROWD",
                    severity=SEVERITY_MEDIUM,
                    message=f"High crowd expected in 30 minutes (~{next_30} people).",
                    recommendation="Prepare staff and open additional counters proactively.",
                    timestamp=ts,
                ))

        # 4. Store capacity warning
        if store_capacity > 0 and current_total is not None and current_total >= 0.90 * store_capacity:
            recommendations.append(self._build(
                rec_type="STORE_NEAR_CAPACITY",
                severity=SEVERITY_CRITICAL,
                message=f"Store near capacity: {current_total}/{store_capacity} people.",
                recommendation="Consider entry management — limit new customers entering.",
                timestamp=ts,
            ))

        # Sort by severity (CRITICAL first)
        severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MEDIUM: 2, SEVERITY_LOW: 3}
        recommendations.sort(key=lambda r: severity_order.get(r.get("severity", "LOW"), 3))

        logger.info("Generated %d recommendations", len(recommendations))
        return recommendations

    @staticmethod
    def _build(
        rec_type: str,
        severity: str,
        message: str,
        recommendation: str,
        timestamp: str,
        zone: Optional[str] = None,
        counter: Optional[str] = None,
    ) -> dict:
        """Build a single recommendation dict."""
        rec = {
            "type": rec_type,
            "severity": severity,
            "message": message,
            "recommendation": recommendation,
            "timestamp": timestamp,
        }
        if zone:
            rec["zone_id"] = zone
        if counter:
            rec["counter_id"] = counter
        return rec
