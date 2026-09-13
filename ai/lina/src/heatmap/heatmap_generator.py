"""
Heatmap Generator.

Accumulates tracked person positions over time and generates a
density heatmap image using OpenCV.

No new AI model is used — the heatmap is derived purely from
the (x, y) center coordinates of tracked persons.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not installed. Heatmap image saving will be disabled. Numerical heatmap still works.")


class HeatmapGenerator:
    """
    Generates and saves customer density heatmaps from tracked person coordinates.

    The heatmap is accumulated across multiple frames/calls to show
    overall movement patterns over time.
    """

    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        blur_radius: int = 31,
        output_dir: Optional[str | Path] = None,
    ):
        """
        Args:
            width: Frame/store width in pixels.
            height: Frame/store height in pixels.
            blur_radius: Gaussian blur kernel size for heatmap smoothing (must be odd).
            output_dir: Directory to save heatmap images.
        """
        self.width = width
        self.height = height
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        self.output_dir = self._resolve_output_dir(output_dir)

        # Accumulation grid — float32 for precision across many frames
        self._accumulator = np.zeros((height, width), dtype=np.float32)
        self._total_points = 0

    def add_frame(self, events: list[dict]) -> None:
        """
        Accumulate person positions from one frame into the heatmap.

        Args:
            events: List of validated tracking events for a single frame.
        """
        for event in events:
            center = event.get("center")
            if not center or len(center) < 2:
                continue

            cx, cy = int(center[0]), int(center[1])

            # Clamp to grid boundaries
            cx = max(0, min(cx, self.width - 1))
            cy = max(0, min(cy, self.height - 1))

            self._accumulator[cy, cx] += 1.0
            self._total_points += 1

    def add_multiple_frames(self, frames: dict[str, list[dict]]) -> None:
        """
        Process all frames in a sequence.

        Args:
            frames: Dict mapping timestamp → list of tracking events.
        """
        for ts, events in frames.items():
            self.add_frame(events)
        logger.info("Heatmap accumulated %d position points from %d frames", self._total_points, len(frames))

    def get_numerical_heatmap(self) -> np.ndarray:
        """
        Return the raw numerical accumulation grid.

        Returns:
            2D float32 numpy array of shape (height, width).
        """
        return self._accumulator.copy()

    def get_normalized_heatmap(self) -> np.ndarray:
        """
        Return the heatmap normalized to [0, 1].

        Returns:
            2D float32 array normalized to [0, 1].
        """
        grid = self._accumulator.copy()
        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid

    def get_top_hotspots(self, top_n: int = 5) -> list[dict]:
        """
        Return the top N hotspot locations.

        Args:
            top_n: Number of hotspots to return.

        Returns:
            List of {x, y, count} dicts sorted by count descending.
        """
        flat = self._accumulator.flatten()
        top_indices = np.argsort(flat)[::-1][:top_n]
        hotspots = []
        for idx in top_indices:
            y, x = divmod(int(idx), self.width)
            hotspots.append({"x": x, "y": y, "count": float(self._accumulator[y, x])})
        return hotspots

    def save_heatmap_image(self, filename: str = "heatmap.png") -> Optional[Path]:
        """
        Apply Gaussian blur and colormap, then save as a PNG image.

        Args:
            filename: Output filename.

        Returns:
            Path to saved heatmap image, or None if OpenCV is unavailable.
        """
        if not CV2_AVAILABLE:
            logger.warning("cv2 not available — skipping heatmap image save")
            return None

        if self.output_dir is None:
            logger.warning("Heatmap output directory not found — skipping save")
            return None

        if self._accumulator.max() == 0:
            logger.warning("Heatmap accumulator is empty — no data to visualize")
            return None

        # Normalize to 0–255
        normalized = cv2.normalize(self._accumulator, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

        # Apply Gaussian blur for smooth density visualization
        blurred = cv2.GaussianBlur(normalized, (self.blur_radius, self.blur_radius), 0)

        # Apply color map (COLORMAP_JET: blue=low, red=high)
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        output_path = self.output_dir / filename
        success = cv2.imwrite(str(output_path), colored)

        if success:
            logger.info("Heatmap saved → %s", output_path)
        else:
            logger.error("cv2.imwrite failed for path: %s", output_path)
            return None

        return output_path

    def reset(self) -> None:
        """Reset the accumulator to start a new heatmap session."""
        self._accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        self._total_points = 0
        logger.info("Heatmap accumulator reset")

    def _resolve_output_dir(self, override: Optional[str | Path]) -> Optional[Path]:
        """Resolve the heatmap output directory."""
        if override:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p

        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "outputs" / "heatmaps"
            if candidate.exists():
                return candidate
        return None
