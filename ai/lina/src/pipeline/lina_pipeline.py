"""
Lina Analytics Pipeline.

Orchestrates the full analytics chain from validated tracking events
to a unified JSON output:

INPUT
 ↓
Validation (InputAdapter)
 ↓
Zone Assignment (ZoneCounter)
 ↓
Crowd Count
 ↓
Density (DensityCalculator)
 ↓
Crowd Level (CrowdLevelClassifier)
 ↓
Heatmap (HeatmapGenerator)
 ↓
Queue Analysis (QueueDetector)
 ↓
Waiting Time (WaitingTimeTracker)
 ↓
Counter Analytics (CounterAnalytics)
 ↓
Historical Analytics (HistoricalAnalytics)
 ↓
Peak Hours (PeakHoursAnalyzer)
 ↓
Prediction (CrowdPredictor)
 ↓
Recommendations (RecommendationEngine)
 ↓
Unified JSON Output (OutputAdapter)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.adapters.input_adapter import InputAdapter
from src.adapters.output_adapter import OutputAdapter
from src.crowd.zone_counter import ZoneCounter
from src.crowd.density import DensityCalculator
from src.crowd.crowd_level import CrowdLevelClassifier
from src.heatmap.heatmap_generator import HeatmapGenerator
from src.queue.queue_detector import QueueDetector
from src.queue.waiting_time import WaitingTimeTracker
from src.queue.counter_analytics import CounterAnalytics
from src.analytics.peak_hours import PeakHoursAnalyzer
from src.analytics.historical import HistoricalAnalytics
from src.analytics.recommendations import RecommendationEngine
from src.prediction.predict import CrowdPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinaPipeline:
    """
    End-to-end crowd, queue, and predictive analytics pipeline for Lina's module.
    """

    def __init__(
        self,
        zones_config: dict,
        thresholds: dict,
        prediction_config: dict,
        store_capacity: int = 200,
        frame_width: int = 1000,
        frame_height: int = 800,
        min_confidence: float = 0.50,
        output_dir: Optional[str | Path] = None,
        heatmap_dir: Optional[str | Path] = None,
    ):
        """
        Initialize all pipeline components.

        Args:
            zones_config: Loaded zones.json dict.
            thresholds: Loaded thresholds.json dict.
            prediction_config: Loaded prediction_config.json dict.
            store_capacity: Maximum store capacity for capacity warnings.
            frame_width: Camera/store frame width in pixels.
            frame_height: Camera/store frame height in pixels.
            min_confidence: Minimum detection confidence threshold.
            output_dir: Override output directory for JSON files.
            heatmap_dir: Override output directory for heatmap images.
        """
        self.zones_config = zones_config
        self.thresholds = thresholds
        self.prediction_config = prediction_config
        self.store_capacity = store_capacity

        # Instantiate all components
        self.input_adapter = InputAdapter(min_confidence=min_confidence)
        self.output_adapter = OutputAdapter(output_dir=output_dir)
        self.zone_counter = ZoneCounter(zones_config)
        self.density_calc = DensityCalculator(self.zone_counter)
        self.crowd_classifier = CrowdLevelClassifier(thresholds)
        self.heatmap_gen = HeatmapGenerator(
            width=frame_width, height=frame_height, output_dir=heatmap_dir
        )
        self.queue_detector = QueueDetector(zones_config, thresholds)
        self.wait_tracker = WaitingTimeTracker(thresholds)
        self.counter_analytics = CounterAnalytics(zones_config, thresholds)
        self.peak_analyzer = PeakHoursAnalyzer(thresholds)
        self.historical = HistoricalAnalytics()
        self.recommendation_engine = RecommendationEngine(thresholds)
        self.predictor = CrowdPredictor(prediction_config)

        # Load prediction model (optional — continues without it)
        self.predictor.load()

        # Accumulate crowd counts for in-session analytics
        self._session_counts: list[dict] = []

        logger.info("LinaPipeline initialized — all components ready")

    def load_historical_data(self, filepath: str) -> bool:
        """
        Load historical crowd data for peak-hour and trend analytics.

        Args:
            filepath: Path to CSV file with historical crowd counts.

        Returns:
            True if loaded successfully.
        """
        return self.historical.load_csv(filepath)

    def process_events(
        self,
        raw_events: list[dict],
        save_json: bool = True,
        save_heatmap: bool = True,
    ) -> dict:
        """
        Process a batch of raw tracking events through the full pipeline.

        Args:
            raw_events: List of raw tracking event dicts from upstream CV.
            save_json: Whether to save the output JSON to disk.
            save_heatmap: Whether to save the heatmap image to disk.

        Returns:
            Unified analytics output dict.
        """
        # Step 1: Validate and normalize
        valid_events = self.input_adapter.from_list(raw_events)
        frames = self.input_adapter.group_by_frame(valid_events)

        if not frames:
            logger.warning("No valid frames to process")
            return {}

        latest_timestamp = max(frames.keys())
        latest_frame_events = frames[latest_timestamp]

        # Step 2: Zone assignment and crowd count
        zone_assignments = self.zone_counter.assign_zones(latest_frame_events)
        zone_counts = self.zone_counter.count_per_zone(zone_assignments)
        total_people = zone_counts.pop("__total__", 0)

        # Step 3: Density calculation
        density_results = self.density_calc.calculate(zone_counts)

        # Step 4: Crowd level classification
        zone_analytics = self.crowd_classifier.classify_all_zones(density_results)

        # Step 5: Heatmap accumulation
        self.heatmap_gen.add_multiple_frames(frames)
        heatmap_path = None
        if save_heatmap:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_path = self.heatmap_gen.save_heatmap_image(f"heatmap_{ts_str}.png")

        # Step 6: Queue detection
        queue_detections = self.queue_detector.detect(latest_frame_events)

        # Step 7: Waiting time
        self.wait_tracker.update(queue_detections, latest_timestamp)
        wait_stats = self.wait_tracker.get_statistics(queue_detections)

        # Step 8: Counter analytics
        counter_results = self.counter_analytics.analyze(queue_detections, wait_stats)

        # Step 9: Accumulate session count for in-session analytics
        self._session_counts.append({
            "timestamp": latest_timestamp,
            "people_count": total_people,
            "queue_length": sum(d.get("people_in_queue", 0) for d in queue_detections.values()),
        })

        # Step 10: Peak hours (uses historical if available, else session data)
        peak_data = self._compute_peak_hours(total_people)

        # Step 11: Prediction
        prediction = self._compute_prediction()

        # Step 12: Recommendations
        recommendations = self.recommendation_engine.generate(
            zone_analytics=zone_analytics,
            counter_analytics=counter_results,
            wait_stats=wait_stats,
            prediction=prediction,
            current_total=total_people,
            store_capacity=self.store_capacity,
        )

        # Step 13: Assemble unified output
        # Format zone analytics for output (clean up internal fields)
        zones_output = {}
        for zone_id, data in zone_analytics.items():
            zones_output[zone_id] = {
                "name": data.get("name", zone_id),
                "count": data.get("people_count", 0),
                "density_ppsm": data.get("density_ppsm", 0.0),
                "normalized_density": data.get("normalized_density", 0.0),
                "crowd_level": data.get("crowd_level", "LOW"),
            }

        # Format queue output
        queues_output = {}
        for counter_id, c_data in counter_results.items():
            queues_output[counter_id] = {
                "queue_length": c_data.get("queue_length", 0),
                "average_waiting_time_seconds": c_data.get("avg_waiting_time_seconds", 0.0),
                "max_waiting_time_seconds": c_data.get("max_waiting_time_seconds", 0.0),
                "utilization": c_data.get("utilization", 0.0),
                "status": c_data.get("status", "NORMAL"),
                "queue_detected": c_data.get("queue_detected", False),
            }

        unified_output = self.output_adapter.build_output(
            timestamp=latest_timestamp,
            people_total=total_people,
            zones=zones_output,
            queues=queues_output,
            peak_hours=peak_data,
            prediction=prediction,
            recommendations=recommendations,
            heatmap_path=str(heatmap_path) if heatmap_path else None,
        )

        if save_json:
            self.output_adapter.save_to_json(unified_output)

        logger.info(
            "Pipeline complete: %d people, %d zones, %d queue regions, %d recommendations",
            total_people, len(zones_output), len(queues_output), len(recommendations),
        )

        return unified_output

    def process_json_file(
        self, filepath: str, save_json: bool = True, save_heatmap: bool = True
    ) -> dict:
        """
        Process tracking events from a JSON file.

        Args:
            filepath: Path to JSON file containing tracking events.
            save_json: Whether to save output JSON.
            save_heatmap: Whether to save heatmap image.

        Returns:
            Unified analytics output dict.
        """
        events = self.input_adapter.from_json_file(filepath)
        return self.process_events(events, save_json=save_json, save_heatmap=save_heatmap)

    def _compute_peak_hours(self, current_total: int) -> dict:
        """Compute peak hour analytics using available data."""
        df = self.historical.dataframe

        if df is None and self._session_counts:
            # Fall back to session-accumulated data
            df = pd.DataFrame(self._session_counts)

        if df is None or df.empty:
            return {"current_status": "UNKNOWN", "note": "No historical data available"}

        peak_analysis = self.peak_analyzer.analyze(df)
        current_status = self.peak_analyzer.classify_current(current_total, df)
        peak_analysis["current_status"] = current_status
        peak_analysis["current_people"] = current_total

        return peak_analysis

    def _compute_prediction(self) -> dict:
        """Generate predictions using session data or historical data."""
        df = self.historical.dataframe

        if df is None and len(self._session_counts) >= 10:
            df = pd.DataFrame(self._session_counts)
        elif df is None:
            return self.predictor._empty_prediction()

        return self.predictor.predict(df)

    def get_session_summary(self) -> dict:
        """Return a summary of all data processed in this session."""
        if not self._session_counts:
            return {}

        counts = [r["people_count"] for r in self._session_counts]
        return {
            "frames_processed": len(self._session_counts),
            "total_avg_people": round(sum(counts) / len(counts), 1),
            "session_peak": max(counts),
            "session_min": min(counts),
        }

    def reset_heatmap(self) -> None:
        """Reset the heatmap accumulator for a new session."""
        self.heatmap_gen.reset()

    def reset_queue_tracking(self) -> None:
        """Reset queue waiting time state."""
        self.wait_tracker.reset()
