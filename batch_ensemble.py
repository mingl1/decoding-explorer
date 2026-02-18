import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import (get_latest_trace, label_beads_with_proteins,
                      load_protein_profiles)


def create_trace(run, output_dir):
    """Run labeling_benchmark.py to generate a trace CSV and return its path.

    Expects run["create_trace"] with keys:
      - cycle1, cycle2       (required) paths to TIFF files
      - method               (optional, default "voronoi_median")
      - bead_csv             (optional) path to bead centers CSV
      - crop_size            (optional) integer crop size
      - extra_args           (optional) list of extra CLI args passed verbatim
    """
    ct = run["create_trace"]
    name = run["name"]
    method = ct.get("method", "voronoi_median")

    trace_dir = os.path.join(output_dir, name, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    csv_path = os.path.join(trace_dir, f"{method}.csv")
    cache_meta_path = os.path.join(trace_dir, ".cache_meta.json")
    cycle1_name = os.path.basename(ct["cycle1"])

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeling_benchmark.py")
    cmd = [
        sys.executable, script,
        "--cycle1", ct["cycle1"],
        "--cycle2", ct["cycle2"],
        "--protein-csvs", *run["protein_profiles"],
        "--methods", method,
        "--output-folder", trace_dir,
        "--border-erosion", str(ct.get("border_erosion", 2)),
        "--min-margin-ratio", ct.get("min_margin_ratio", "1.0-1.5"),
        "--trace-output", os.path.join(trace_dir, "trace.json"),
    ]
    if "bead_csv" in ct:
        cmd += ["--bead-csv", ct["bead_csv"]]
    if "crop_size" in ct:
        cmd += ["--crop-size", str(ct["crop_size"])]
    if "extra_args" in ct:
        cmd += ct["extra_args"]

    print(f"  Creating trace via labeling_benchmark ({method})...")
    subprocess.run(cmd, check=True)

    csv_path = os.path.join(trace_dir, f"{method}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected trace CSV not found: {csv_path}")

    with open(cache_meta_path, "w") as f:
        json.dump({"cycle1": cycle1_name}, f)

    return csv_path


def process_run(run, output_dir):
    name = run["name"]
    print(f"\n--- Processing: {name} ---")

    bead_df = pd.read_csv(run["bead_df"])
    results_df = pd.read_csv(run["results_df"])
    protein_df = load_protein_profiles(run["protein_profiles"])

    if "other_df" in run:
        other_df = pd.read_csv(run["other_df"])
    elif "create_trace" in run:
        other_df = pd.read_csv(create_trace(run, output_dir))
    else:
        other_df = get_latest_trace(run["trace_folder"])

    # round x,y
    other_df["x"] = np.rint(other_df["x"]).astype(np.float32)
    other_df["y"] = np.rint(other_df["y"]).astype(np.float32)
    results_df["x"] = np.rint(results_df["x"]).astype(np.float32)
    results_df["y"] = np.rint(results_df["y"]).astype(np.float32)

    merged_df = results_df.merge(other_df, on=["x", "y"], suffixes=("", "_other"), how="left")

    # cy0/cy1 correction: if cy0==255 but cy0_other is not, use other_df's values
    mask_255_cy0 = merged_df["cy0"] == 255
    mask_255_cy0_other = (merged_df["cy0_other"] == 255) | (merged_df["cy0_other"].isna())
    mask_update = mask_255_cy0 & ~mask_255_cy0_other
    merged_df.loc[mask_update, "cy0"] = merged_df.loc[mask_update, "cy0_other"]
    merged_df.loc[mask_update, "cy1"] = merged_df.loc[mask_update, "cy1_other"]

    labeled_df = label_beads_with_proteins(merged_df.drop(columns=["Protein name"], errors="ignore"), protein_df)

    # save outputs
    run_out = os.path.join(output_dir, name)
    os.makedirs(run_out, exist_ok=True)

    cycle_cols = [c for c in protein_df.columns if c.startswith("cy")]
    save_cols = ["x", "y"] + cycle_cols + ["Protein name"]
    save_cols = [c for c in save_cols if c in labeled_df.columns]
    labeled_df[save_cols].to_csv(os.path.join(run_out, "beads.csv"), index=False)

    counts = labeled_df["Protein name"].value_counts()
    total = len(bead_df)
    invalid = int(counts.get("Invalid", 0))
    filtered = int(counts.get("Filtered", 0))
    valid = total - invalid - filtered
    protein_counts = {
        k: int(v) for k, v in counts.items() if k not in ("Invalid", "Filtered")
    }

    stats = {
        "name": name,
        "protein_profiles": run["protein_profiles"],
        "total_beads": total,
        "valid": valid,
        "valid_pct": round(100 * valid / total, 1) if total else 0,
        "invalid": invalid,
        "invalid_pct": round(100 * invalid / total, 1) if total else 0,
        "filtered": filtered,
        "filtered_pct": round(100 * filtered / total, 1) if total else 0,
        "protein_counts": protein_counts,
    }

    # Include margin ratio sweep results if available
    sweep_path = os.path.join(run_out, "trace", "margin_sweep.json")
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            stats["margin_sweep"] = json.load(f)

    with open(os.path.join(run_out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Total: {total}  Valid: {valid} ({stats['valid_pct']}%)  "
          f"Invalid: {invalid} ({stats['invalid_pct']}%)  "
          f"Filtered: {filtered} ({stats['filtered_pct']}%)")
    print(f"  Saved to {run_out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to batch config JSON")
    parser.add_argument("--parallel", action="store_true", help="Process runs in parallel")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: number of runs)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    with open(config_path) as f:
        config = json.load(f)

    # input_dir: where run folders live; resolve relative to config file location
    raw_input = config.get("input_dir", ".")
    input_dir = Path(raw_input) if Path(raw_input).is_absolute() else (config_dir / raw_input)

    # Resolve run paths relative to input_dir
    def resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else (input_dir / p))

    for run in config["runs"]:
        run["bead_df"] = resolve(run["bead_df"])
        run["results_df"] = resolve(run["results_df"])
        run["protein_profiles"] = [resolve(p) for p in run["protein_profiles"]]
        if "other_df" in run:
            run["other_df"] = resolve(run["other_df"])
        if "trace_folder" in run:
            run["trace_folder"] = resolve(run["trace_folder"])
        if "create_trace" in run:
            ct = run["create_trace"]
            ct["cycle1"] = resolve(ct["cycle1"])
            ct["cycle2"] = resolve(ct["cycle2"])
            if "bead_csv" in ct:
                ct["bead_csv"] = resolve(ct["bead_csv"])

    # output_dir: where results are written; resolve relative to input_dir
    raw_output = config.get("output_dir", "./batch_results")
    output_dir = str(Path(raw_output) if Path(raw_output).is_absolute() else (input_dir / raw_output))
    os.makedirs(output_dir, exist_ok=True)

    runs = config["runs"]

    if args.parallel:
        workers = args.workers or len(runs)
        print(f"Running {len(runs)} runs in parallel (workers={workers})...")
        print("Note: output from runs may be interleaved.\n")
        failed = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_run, run, output_dir): run["name"] for run in runs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\n[ERROR] {name}: {e}")
                    failed.append(name)
        if failed:
            print(f"\nBatch complete with errors: {failed}")
        else:
            print("\nBatch complete.")
    else:
        for run in runs:
            process_run(run, output_dir)
        print("\nBatch complete.")
