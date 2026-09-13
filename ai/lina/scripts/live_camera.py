"""
Live Camera — Real-Time Crowd & Queue Analytics using YOLOv8.

Uses your laptop webcam to detect real people with YOLOv8n,
then feeds them through Lina's full analytics pipeline.

Usage:
    cd ai/lina
    python scripts/live_camera.py

Controls (while window is open):
    Q or ESC  — quit
    H         — toggle heatmap overlay
    Z         — toggle zone overlay
    S         — save current analytics snapshot to JSON

Requirements:
    pip install ultralytics opencv-python
"""

# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Make src importable
LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.pipeline.lina_pipeline import LinaPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

CAMERA_INDEX        = 0          # 0 = default laptop webcam
CAMERA_BACKEND      = cv2.CAP_DSHOW   # DirectShow — most reliable on Windows
YOLO_MODEL          = "yolov8n.pt"  # nano = fastest; change to yolov8s.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.45     # Min detection confidence
ANALYTICS_EVERY_N_FRAMES = 15   # Run full pipeline every N frames (~0.5s at 30fps)
DISPLAY_WIDTH       = 1280
DISPLAY_HEIGHT      = 720
STORE_CAPACITY      = 50        # Set to your actual room/store capacity

# Zone polygon colours (BGR)
ZONE_COLOURS = {
    "LOW":      (0, 200, 0),      # Green
    "MEDIUM":   (0, 165, 255),    # Orange
    "HIGH":     (0, 0, 255),      # Red
    "CRITICAL": (0, 0, 180),      # Dark red
    "DEFAULT":  (200, 200, 200),  # Grey
}

SEVERITY_COLOURS = {
    "CRITICAL": (0, 0, 255),
    "HIGH":     (0, 100, 255),
    "MEDIUM":   (0, 200, 255),
    "LOW":      (0, 255, 0),
}


def load_config(name: str) -> dict:
    p = LINA_ROOT / "config" / name
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Simple Centroid Tracker ───────────────────────────────────────────────────

class CentroidTracker:
    """
    Lightweight centroid-based person tracker.
    Assigns stable track IDs to detections across frames.
    """
    def __init__(self, max_disappeared: int = 20, max_distance: float = 80):
        self.next_id = 1
        self.objects: dict[int, np.ndarray] = {}   # track_id -> centroid
        self.disappeared: dict[int, int] = {}       # track_id -> frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, bboxes: list[list[float]]) -> dict[int, list]:
        """
        Update tracker with new bounding boxes.

        Args:
            bboxes: List of [x1, y1, x2, y2] boxes.

        Returns:
            Dict of {track_id: [x1, y1, x2, y2, cx, cy]}.
        """
        if not bboxes:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]
            return {}

        # Compute centroids for new detections
        new_centroids = []
        for (x1, y1, x2, y2) in bboxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            new_centroids.append(np.array([cx, cy]))

        if not self.objects:
            # Register all as new
            for i, centroid in enumerate(new_centroids):
                self.objects[self.next_id] = centroid
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            existing_ids = list(self.objects.keys())
            existing_cents = list(self.objects.values())

            # Compute distance matrix
            D = np.linalg.norm(
                np.array(existing_cents)[:, np.newaxis] - np.array(new_centroids)[np.newaxis, :],
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                tid = existing_ids[row]
                self.objects[tid] = new_centroids[col]
                self.disappeared[tid] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Mark unmatched existing as disappeared
            for row in set(range(len(existing_ids))) - used_rows:
                tid = existing_ids[row]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]

            # Register unmatched new detections
            for col in set(range(len(new_centroids))) - used_cols:
                self.objects[self.next_id] = new_centroids[col]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        # Build result dict
        result = {}
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            centroid = np.array([cx, cy])
            # Match to closest track
            best_tid, best_dist = None, float("inf")
            for tid, obj_cent in self.objects.items():
                d = np.linalg.norm(centroid - obj_cent)
                if d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is not None and best_dist < self.max_distance:
                result[best_tid] = [x1, y1, x2, y2, cx, cy]

        return result


# ── Drawing Helpers ───────────────────────────────────────────────────────────

def draw_bounding_boxes(frame: np.ndarray, tracked: dict, confs: dict) -> None:
    """Draw person bounding boxes with track ID labels."""
    for tid, (x1, y1, x2, y2, cx, cy) in tracked.items():
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
        label = f"#{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (50, 200, 50), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def draw_zones(frame: np.ndarray, zones_cfg: dict, analytics_zones: dict,
               fw: int, fh: int, orig_w: int, orig_h: int) -> None:
    """Draw zone polygons with crowd level colours."""
    sx, sy = fw / orig_w, fh / orig_h
    for zone in zones_cfg.get("zones", []):
        zid = zone["id"]
        poly = np.array([[int(p[0] * sx), int(p[1] * sy)]
                         for p in zone["polygon"]], dtype=np.int32)
        level = analytics_zones.get(zid, {}).get("crowd_level", "DEFAULT")
        colour = ZONE_COLOURS.get(level, ZONE_COLOURS["DEFAULT"])
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], colour)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [poly], True, colour, 2)
        # Label
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        name = zone.get("name", zid)
        count = analytics_zones.get(zid, {}).get("count", 0)
        cv2.putText(frame, f"{name}: {count}", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend heatmap into frame."""
    if heatmap is None or heatmap.max() == 0:
        return frame
    norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    resized = cv2.resize(coloured, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.65, resized, 0.35, 0)


def draw_stats_panel(frame: np.ndarray, analytics: dict, fps: float, n_people: int) -> None:
    """Draw the semi-transparent analytics panel on the right side."""
    h, w = frame.shape[:2]
    panel_w = 280
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 30)

    y = 20
    def put(text, colour=(220, 220, 220), scale=0.5, bold=False):
        nonlocal y
        thickness = 2 if bold else 1
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness, cv2.LINE_AA)
        y += 22

    put("LINA ANALYTICS", (100, 220, 255), 0.6, bold=True)
    put(f"FPS: {fps:.1f}  |  People: {n_people}", (180, 255, 180), 0.48)
    put(datetime.now().strftime("%H:%M:%S"), (150, 150, 150), 0.45)
    y += 5

    # Zones
    put("-- ZONES --", (100, 200, 255), 0.48, bold=True)
    for zid, zdata in analytics.get("zones", {}).items():
        level = zdata.get("crowd_level", "LOW")
        col = ZONE_COLOURS.get(level, ZONE_COLOURS["DEFAULT"])
        name = zdata.get("name", zid)[:16]
        cnt = zdata.get("count", 0)
        put(f"  {name}: {cnt} [{level}]", col[::-1], 0.42)

    y += 5
    # Queues
    put("-- QUEUES --", (100, 200, 255), 0.48, bold=True)
    for cid, cdata in analytics.get("queues", {}).items():
        status = cdata.get("status", "NORMAL")
        qlen = cdata.get("queue_length", 0)
        wait = cdata.get("average_waiting_time_seconds", 0)
        col = (0, 200, 0) if status == "NORMAL" else (0, 100, 255) if status == "BUSY" else (0, 0, 255)
        put(f"  {cid}: Q={qlen} W={wait:.0f}s [{status}]", col[::-1], 0.40)

    y += 5
    # Prediction
    put("-- PREDICTION --", (100, 200, 255), 0.48, bold=True)
    pred = analytics.get("prediction", {})
    if pred.get("model_available"):
        put(f"  +15min: {pred.get('next_15_min','?')} people", (200, 255, 200), 0.42)
        put(f"  +30min: {pred.get('next_30_min','?')} people", (200, 255, 200), 0.42)
        put(f"  +60min: {pred.get('next_60_min','?')} people", (200, 255, 200), 0.42)
    else:
        put("  No model — run train script", (150, 150, 150), 0.40)

    y += 5
    # Recommendations
    recs = analytics.get("recommendations", [])
    if recs:
        put("-- ALERTS --", (0, 100, 255), 0.48, bold=True)
        for rec in recs[:3]:
            col = SEVERITY_COLOURS.get(rec.get("severity", "LOW"), (200, 200, 200))
            msg = rec.get("message", "")[:38]
            put(f"  {msg}", col[::-1], 0.38)

    # Peak hours
    peak = analytics.get("peak_hours", {})
    y += 5
    put("-- PEAK HOURS --", (100, 200, 255), 0.48, bold=True)
    put(f"  Now: {peak.get('current_status','?')}", (200, 200, 200), 0.42)
    put(f"  Peak: {peak.get('peak_hour','?')}", (200, 200, 200), 0.42)

    put("  [Q/ESC]=Quit [H]=Heatmap", (100, 100, 100), 0.38)
    put("  [Z]=Zones   [S]=Snapshot", (100, 100, 100), 0.38)

    combined = np.hstack([frame, panel])
    frame[:] = cv2.resize(combined, (frame.shape[1], frame.shape[0]))


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LINA — LIVE CAMERA ANALYTICS")
    print("YOLOv8 + Real Webcam")
    print("=" * 60)

    # Load configs
    zones_cfg   = load_config("zones.json")
    thresholds  = load_config("thresholds.json")
    pred_cfg    = load_config("prediction_config.json")

    # Init pipeline
    pipeline = LinaPipeline(
        zones_config=zones_cfg,
        thresholds=thresholds,
        prediction_config=pred_cfg,
        store_capacity=STORE_CAPACITY,
        frame_width=DISPLAY_WIDTH,
        frame_height=DISPLAY_HEIGHT,
        min_confidence=CONFIDENCE_THRESHOLD,
        output_dir=str(LINA_ROOT / "outputs" / "json"),
        heatmap_dir=str(LINA_ROOT / "outputs" / "heatmaps"),
    )

    # Load historical CSV for prediction baseline (optional)
    hist_csv = LINA_ROOT / "data" / "sample" / "historical_counts.csv"
    if hist_csv.exists():
        pipeline.load_historical_data(str(hist_csv))
        print(f"[OK] Historical baseline loaded: {hist_csv.name}")

    # Load YOLOv8
    print(f"\n[*] Loading YOLOv8 model: {YOLO_MODEL} ...")
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        print(f"[OK] YOLOv8 loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        print("       Run: pip install ultralytics")
        sys.exit(1)

    # Open webcam — try DirectShow first, fall back to default
    print(f"\n[*] Opening webcam (index {CAMERA_INDEX}, DirectShow backend) ...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] DirectShow failed, trying default backend...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        print("  - Make sure no other app is using the webcam")
        print("  - Try changing CAMERA_INDEX to 1 in live_camera.py")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Webcam opened: {orig_w}x{orig_h}")
    print("\nPress Q or ESC to quit | H = heatmap | Z = zones | S = snapshot")
    print("-" * 60)

    tracker = CentroidTracker(max_disappeared=25, max_distance=100)

    # State
    frame_count     = 0
    analytics_output = {}
    show_heatmap    = False
    show_zones      = True
    all_events      = []       # events accumulated since last pipeline run
    fps             = 0.0
    t_fps           = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Try to recover — skip up to 5 bad frames before quitting
            bad_frames = getattr(main, '_bad_frames', 0) + 1
            main._bad_frames = bad_frames
            if bad_frames > 5:
                print("[ERROR] Camera read failed repeatedly — exiting")
                break
            print(f"[!] Frame grab failed ({bad_frames}/5) — retrying...")
            time.sleep(0.1)
            continue
        main._bad_frames = 0  # Reset counter on success

        frame_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # ── YOLOv8 Detection ────────────────────────────────────────
        results = model(
            frame,
            classes=[0],           # 0 = person in COCO
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            stream=False,
        )

        bboxes = []
        confs_map = {}
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bboxes.append([x1, y1, x2, y2])
                confs_map[len(bboxes) - 1] = conf

        # ── Track ────────────────────────────────────────────────────
        tracked = tracker.update(bboxes)

        # ── Build events for this frame ──────────────────────────────
        for tid, (x1, y1, x2, y2, cx, cy) in tracked.items():
            conf_idx = min(range(len(bboxes)),
                          key=lambda i: abs(bboxes[i][0] - x1) + abs(bboxes[i][1] - y1),
                          default=0)
            event = {
                "timestamp": timestamp,
                "camera_id": f"WEBCAM_{CAMERA_INDEX}",
                "track_id": tid,
                "class": "person",
                "confidence": round(confs_map.get(conf_idx, CONFIDENCE_THRESHOLD), 2),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "center": [round(cx, 1), round(cy, 1)],
            }
            all_events.append(event)

        # ── Run full pipeline every N frames ─────────────────────────
        if frame_count % ANALYTICS_EVERY_N_FRAMES == 0 and all_events:
            analytics_output = pipeline.process_events(
                all_events,
                save_json=True,
                save_heatmap=False,  # Don't save image every N frames
            )
            all_events = []  # Reset accumulator

        # ── Draw bounding boxes ──────────────────────────────────────
        draw_bounding_boxes(frame, tracked, confs_map)

        # ── Heatmap overlay ──────────────────────────────────────────
        if show_heatmap:
            hm = pipeline.heatmap_gen.get_normalized_heatmap()
            frame = draw_heatmap_overlay(frame, hm)

        # ── Zone overlay ─────────────────────────────────────────────
        if show_zones and analytics_output:
            draw_zones(frame, zones_cfg,
                       analytics_output.get("zones", {}),
                       frame.shape[1], frame.shape[0], orig_w, orig_h)

        # ── People count on frame ────────────────────────────────────
        n_people = len(tracked)
        cv2.putText(frame, f"People: {n_people}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2, cv2.LINE_AA)

        # ── Stats panel ──────────────────────────────────────────────
        panel_frame = frame.copy()
        if analytics_output:
            draw_stats_panel(panel_frame, analytics_output, fps, n_people)
        cv2.imshow("Lina Live Analytics | SIH 2026", panel_frame)

        # ── FPS ──────────────────────────────────────────────────────
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / max(now - t_fps, 0.001)
            t_fps = now

        # ── Key handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break
        elif key in (ord("h"), ord("H")):
            show_heatmap = not show_heatmap
            print(f"[H] Heatmap overlay: {'ON' if show_heatmap else 'OFF'}")
        elif key in (ord("z"), ord("Z")):
            show_zones = not show_zones
            print(f"[Z] Zone overlay: {'ON' if show_zones else 'OFF'}")
        elif key in (ord("s"), ord("S")):
            if analytics_output:
                snap_path = LINA_ROOT / "outputs" / "json" / \
                    f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(snap_path, "w", encoding="utf-8") as f:
                    json.dump(analytics_output, f, indent=2)
                print(f"[S] Snapshot saved -> {snap_path.name}")
            # Also save heatmap on S
            hm_path = pipeline.heatmap_gen.save_heatmap_image(
                f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            if hm_path:
                print(f"[S] Heatmap saved -> {hm_path.name}")

    cap.release()
    cv2.destroyAllWindows()

    # Final session summary
    print("\n" + "=" * 60)
    print("Session ended.")
    summary = pipeline.get_session_summary()
    if summary:
        print(f"  Frames processed : {summary.get('frames_processed')}")
        print(f"  Avg people       : {summary.get('total_avg_people')}")
        print(f"  Session peak     : {summary.get('session_peak')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
