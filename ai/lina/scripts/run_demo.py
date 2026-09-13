# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Lina Analytics Demo Runner.

Runs the full analytics pipeline using synthetic sample data.
No real camera, no Kanimozhi code, no Janavi code, no Deepika backend,
no PostgreSQL, and no hardware required.

Usage:
    cd ai/lina
    python scripts/run_demo.py

Output:
    - Prints analytics summary to console
    - Saves JSON output -> outputs/json/lina_output.json
    - Saves heatmap image -> outputs/heatmaps/heatmap_YYYYMMDD_HHMMSS.png
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Make src importable from scripts/
LINA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LINA_ROOT))

from src.pipeline.lina_pipeline import LinaPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(filename: str) -> dict:
    path = LINA_ROOT / "config" / filename
    if not path.exists():
        print(f"ERROR: Config file missing: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_sample_data() -> None:
    """Ensure sample data exists; if not, generate it."""
    tracks_file = LINA_ROOT / "data" / "sample" / "people_tracks.json"
    if not tracks_file.exists():
        print("Sample data not found. Generating now...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(LINA_ROOT / "scripts" / "generate_sample_data.py")],
            cwd=str(LINA_ROOT),
        )
        if result.returncode != 0:
            print("ERROR: Failed to generate sample data")
            sys.exit(1)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def print_output_summary(output: dict) -> None:
    """Pretty-print the key analytics results."""
    print_section("[ANALYTICS] LINA ANALYTICS OUTPUT SUMMARY")
    print("[WARNING] ALL DATA IS SYNTHETIC DEMO DATA\n")

    print(f"  Timestamp    : {output.get('frame_timestamp', 'N/A')}")
    print(f"  Generated at : {output.get('generated_at', 'N/A')}")

    # People
    people = output.get("people", {})
    print(f"\n  [PEOPLE] Total People : {people.get('total', 0)}")

    # Zones
    print("\n  [ZONES]  Zone Analytics:")
    zones = output.get("zones", {})
    if zones:
        print(f"  {'Zone':<20} {'Count':>6} {'Density':>10} {'Level':<10}")
        print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10}")
        for zone_id, z in zones.items():
            print(
                f"  {z.get('name', zone_id):<20} {z.get('count', 0):>6} "
                f"{z.get('normalized_density', 0):.4f}     {z.get('crowd_level', '-'):<10}"
            )
    else:
        print("  (no zone data)")

    # Queues
    print("\n  [QUEUES] Queue / Counter Analytics:")
    queues = output.get("queues", {})
    if queues:
        print(f"  {'Counter':<15} {'Queue':>6} {'Avg Wait(s)':>12} {'Status':<12}")
        print(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*12}")
        for counter_id, q in queues.items():
            print(
                f"  {counter_id:<15} {q.get('queue_length', 0):>6} "
                f"{q.get('average_waiting_time_seconds', 0):>12.1f} "
                f"{q.get('status', '-'):<12}"
            )
    else:
        print("  (no queue data)")

    # Peak Hours
    peak = output.get("peak_hours", {})
    print(f"\n  [PEAK HOURS] Peak Hours:")
    print(f"     Current status  : {peak.get('current_status', 'N/A')}")
    print(f"     Peak hour       : {peak.get('peak_hour', 'N/A')}")
    print(f"     Busiest day     : {peak.get('busiest_day', 'N/A')}")
    print(f"     Overall avg     : {peak.get('overall_avg_people', 'N/A')}")

    # Prediction
    pred = output.get("prediction", {})
    print(f"\n  [PREDICTION] Crowd Predictions (SYNTHETIC DEMO DATA):")
    if pred.get("model_available"):
        print(f"     Next 15 min  : {pred.get('next_15_min', 'N/A')} people")
        print(f"     Next 30 min  : {pred.get('next_30_min', 'N/A')} people")
        print(f"     Next 60 min  : {pred.get('next_60_min', 'N/A')} people")
    else:
        print(f"     Model not available -- run train_prediction_model.py first")
        print(f"     {pred.get('note', '')}")

    # Recommendations
    recs = output.get("recommendations", [])
    print(f"\n  [RECOMMENDATIONS] Recommendations ({len(recs)} total):")
    if recs:
        for rec in recs[:5]:  # Show top 5
            severity = rec.get("severity", "")
            print(f"     [{severity:<8}] {rec.get('message', '')}")
            print(f"              -> {rec.get('recommendation', '')}")
    else:
        print("  (no recommendations -- system operating normally)")

    # Heatmap
    if output.get("heatmap_path"):
        print(f"\n  [HEATMAP]  Heatmap saved -> {output['heatmap_path']}")


def main() -> None:
    print_section("[START] LINA ANALYTICS MODULE -- DEMO")
    print("Smart Retail Analytics System | SIH 2026")
    print("Module: Crowd, Queue & Predictive Analytics")
    print("[WARNING] ALL DATA IS SYNTHETIC DEMO DATA\n")

    # Load configurations
    print("[1/5] Loading configurations...")
    zones_config = load_config("zones.json")
    thresholds = load_config("thresholds.json")
    pred_config = load_config("prediction_config.json")
    print("      [OK] Configurations loaded")

    # Ensure sample data exists
    print("\n[2/5] Checking sample data...")
    check_sample_data()
    print("      [OK] Sample data ready")

    # Initialize pipeline
    print("\n[3/5] Initializing analytics pipeline...")
    pipeline = LinaPipeline(
        zones_config=zones_config,
        thresholds=thresholds,
        prediction_config=pred_config,
        store_capacity=200,
        frame_width=1000,
        frame_height=800,
        min_confidence=0.50,
    )
    print("      [OK] Pipeline initialized")

    # Load historical data
    historical_csv = LINA_ROOT / "data" / "sample" / "historical_counts.csv"
    if historical_csv.exists():
        loaded = pipeline.load_historical_data(str(historical_csv))
        if loaded:
            print(f"      [OK] Historical data loaded from {historical_csv.name}")
    else:
        print("      [WARNING] No historical data -- run generate_sample_data.py first")

    # Process sample tracking events
    print("\n[4/5] Processing tracking events...")
    tracks_file = LINA_ROOT / "data" / "sample" / "people_tracks.json"

    output = pipeline.process_json_file(
        str(tracks_file),
        save_json=True,
        save_heatmap=True,
    )

    if not output:
        print("ERROR: Pipeline produced no output")
        sys.exit(1)

    print("      [OK] Analytics pipeline complete")

    # Print summary
    print("\n[5/5] Analytics Summary:")
    print_output_summary(output)

    # Save JSON
    json_out = LINA_ROOT / "outputs" / "json" / "lina_output.json"
    print(f"\n[OUTPUT] Full JSON output -> {json_out}")

    print("\n" + "=" * 60)
    print("Demo complete! [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()


