"""
Output Adapter for the Lina Analytics Module.

Assembles a unified, backend-ready JSON object from all analytics results.

--- INTEGRATION CONTRACT (for Deepika's Backend) ---
Deepika's backend should:
1. Call Lina's pipeline and receive the dict returned by build_output()
2. OR poll a JSON file written to outputs/json/lina_output.json
3. The output schema is stable and versioned below.

Schema version: 1.0.0
---
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_SCHEMA_VERSION = "1.0.0"


class OutputAdapter:
    """
    Builds and serializes the unified Lina analytics output.
    """

    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Args:
            output_dir: Directory to save JSON output files.
                        Defaults to outputs/json/ relative to module root.
        """
        self.output_dir = self._resolve_output_dir(output_dir)

    def build_output(
        self,
        timestamp: str,
        people_total: int,
        zones: dict[str, dict],
        queues: dict[str, dict],
        peak_hours: dict,
        prediction: dict,
        recommendations: list[dict],
        historical_summary: Optional[dict] = None,
        heatmap_path: Optional[str] = None,
    ) -> dict:
        """
        Assemble the complete analytics output dict.

        Args:
            timestamp: ISO 8601 current timestamp string.
            people_total: Total number of people detected in this frame.
            zones: Zone analytics dict {zone_id: {count, density, level, ...}}.
            queues: Queue analytics dict {counter_id: {length, avg_wait, status}}.
            peak_hours: Peak hour analytics summary dict.
            prediction: Prediction dict {next_15_min, next_30_min, next_60_min}.
            recommendations: List of recommendation dicts.
            historical_summary: Optional historical analytics summary dict.
            heatmap_path: Optional path to saved heatmap image.

        Returns:
            Backend-ready analytics output dict.
        """
        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "frame_timestamp": timestamp,
            "data_classification": "SYNTHETIC DEMO DATA — Not real store analytics",
            "people": {
                "total": int(people_total),
            },
            "zones": zones,
            "queues": queues,
            "peak_hours": peak_hours,
            "prediction": prediction,
            "recommendations": recommendations,
        }

        if historical_summary:
            output["historical"] = historical_summary

        if heatmap_path:
            output["heatmap_path"] = str(heatmap_path)

        return output

    def save_to_json(self, output: dict, filename: str = "lina_output.json") -> Path:
        """
        Save the analytics output dict as a JSON file.

        Args:
            output: The assembled analytics dict.
            filename: Output filename (default: lina_output.json).

        Returns:
            Path to the saved file.
        """
        if self.output_dir is None:
            logger.warning("Output directory not found — skipping JSON save")
            return Path(filename)

        output_path = self.output_dir / filename
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)
            logger.info("Analytics output saved → %s", output_path)
        except OSError as exc:
            logger.error("Failed to save output JSON: %s", exc)

        return output_path

    def _resolve_output_dir(self, override: Optional[str | Path]) -> Optional[Path]:
        """Resolve the output/json directory relative to the module root."""
        if override:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p

        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "outputs" / "json"
            if candidate.exists():
                return candidate
        return None
