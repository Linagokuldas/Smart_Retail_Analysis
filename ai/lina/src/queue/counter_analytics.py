"""
Billing Counter Analytics.

Aggregates per-queue detection and waiting time data into per-counter
status reports.

Counter Status Levels:
    NORMAL   — queue and wait time within thresholds
    BUSY     — queue exceeds busy threshold
    OVERLOADED — queue exceeds overloaded threshold
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

STATUS_NORMAL = "NORMAL"
STATUS_BUSY = "BUSY"
STATUS_OVERLOADED = "OVERLOADED"


class CounterAnalytics:
    """
    Generates per-billing-counter analytics by combining queue detections
    and waiting time statistics.
    """

    def __init__(self, zones_config: dict, thresholds: dict):
        """
        Args:
            zones_config: Loaded zones.json with "billing_counters" list.
            thresholds: Loaded thresholds.json with "counter" sub-key.
        """
        self.counters = {c["id"]: c for c in zones_config.get("billing_counters", [])}
        counter_cfg = thresholds.get("counter", {})
        self.busy_queue = int(counter_cfg.get("busy_queue_length", 5))
        self.overloaded_queue = int(counter_cfg.get("overloaded_queue_length", 10))
        self.busy_wait = float(counter_cfg.get("busy_avg_wait_seconds", 120))
        self.overloaded_wait = float(counter_cfg.get("overloaded_avg_wait_seconds", 240))

    def analyze(
        self,
        queue_detections: dict[str, dict],
        wait_stats: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Produce per-counter analytics.

        Args:
            queue_detections: Output of QueueDetector.detect().
            wait_stats: Output of WaitingTimeTracker.get_statistics().

        Returns:
            Dict mapping counter_id → {queue_length, avg_wait, max_wait, status, utilization}.
        """
        results = {}

        for region_id, detection in queue_detections.items():
            counter_id = detection.get("counter_id", region_id)
            queue_length = detection.get("people_in_queue", 0)
            utilization = detection.get("utilization", 0.0)

            wait = wait_stats.get(region_id, {})
            avg_wait = wait.get("avg_waiting_time_seconds", 0.0)
            max_wait = wait.get("max_waiting_time_seconds", 0.0)

            status = self._classify_status(queue_length, avg_wait)

            results[counter_id] = {
                "queue_region_id": region_id,
                "queue_length": queue_length,
                "avg_waiting_time_seconds": avg_wait,
                "max_waiting_time_seconds": max_wait,
                "utilization": utilization,
                "status": status,
                "queue_detected": detection.get("queue_detected", False),
            }

        # Add counters with no detections as NORMAL
        for counter_id, counter_info in self.counters.items():
            if counter_id not in results:
                results[counter_id] = {
                    "queue_region_id": counter_info.get("queue_region_id", ""),
                    "queue_length": 0,
                    "avg_waiting_time_seconds": 0.0,
                    "max_waiting_time_seconds": 0.0,
                    "utilization": 0.0,
                    "status": STATUS_NORMAL,
                    "queue_detected": False,
                }

        return results

    def _classify_status(self, queue_length: int, avg_wait_seconds: float) -> str:
        """
        Classify counter status based on configurable thresholds.

        Args:
            queue_length: Current number of people in this counter's queue.
            avg_wait_seconds: Average waiting time in seconds.

        Returns:
            "NORMAL", "BUSY", or "OVERLOADED".
        """
        if queue_length >= self.overloaded_queue or avg_wait_seconds >= self.overloaded_wait:
            return STATUS_OVERLOADED
        elif queue_length >= self.busy_queue or avg_wait_seconds >= self.busy_wait:
            return STATUS_BUSY
        return STATUS_NORMAL
