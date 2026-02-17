import json
import os
import sys

import numpy as np
import pandas as pd

from ensemble import get_latest_trace, label_beads_with_proteins, load_protein_profiles


def process_run(run, output_dir):
    name = run["name"]
    print(f"\n--- Processing: {name} ---")

    bead_df = pd.read_csv(run["bead_df"])
    results_df = pd.read_csv(run["results_df"])
    protein_df = load_protein_profiles(run["protein_profiles"])

    if "other_df" in run:
        other_df = pd.read_csv(run["other_df"])
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

    labeled_df = label_beads_with_proteins(merged_df, protein_df)

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

    with open(os.path.join(run_out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Total: {total}  Valid: {valid} ({stats['valid_pct']}%)  "
          f"Invalid: {invalid} ({stats['invalid_pct']}%)  "
          f"Filtered: {filtered} ({stats['filtered_pct']}%)")
    print(f"  Saved to {run_out}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python batch_ensemble.py config.json")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    output_dir = config.get("output_dir", "./batch_results")
    os.makedirs(output_dir, exist_ok=True)

    for run in config["runs"]:
        process_run(run, output_dir)

    print("\nBatch complete.")
