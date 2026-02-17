import argparse
import os

import numpy as np
import pandas as pd


def load_protein_profiles(file_paths: list[str]) -> pd.DataFrame:
    """Load protein profiles from CSV/Excel files."""
    dfs = []
    for path in file_paths:
        if path.endswith(".csv"):
            dfs.append(pd.read_csv(path))
        else:
            dfs.append(pd.read_excel(path))
    merged = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    cols = merged.columns.tolist()
    renamed = {cols[0]: "Protein name"}
    for i in range(1, len(cols)):
        renamed[cols[i]] = f"cy{i - 1}"
    merged.rename(columns=renamed, inplace=True)
    print(f"Loaded {len(merged)} proteins, {len(cols) - 1} cycles")
    return merged


def label_beads_with_proteins(bead_df, protein_df):
    cycle_cols = [c for c in protein_df.columns if c.startswith("cy")]
    for col in cycle_cols:
        if col in bead_df.columns:
            bead_df[col] = bead_df[col].astype(int)
            protein_df[col] = protein_df[col].astype(int)
    result = bead_df.merge(protein_df, on=cycle_cols, how="left")
    result["Protein name"] = result["Protein name"].fillna("Invalid")
    filter_mask = pd.Series(False, index=result.index)
    for col in cycle_cols:
        filter_mask |= result[col] >= 254
    result.loc[filter_mask, "Protein name"] = "Filtered"
    return result


def get_latest_trace(folder, method="voronoi_median"):
    # folder names are prefixed with dataetime, so sorting by name gives latest
    subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    if not subdirs:
        print("No subdirectories found in trace folder!")
        return None
    latest_subdir = sorted(
        subdirs, key=lambda x: os.path.getctime(os.path.join(folder, x))
    )[-1]
    print(f"Latest trace subdir: {latest_subdir}")
    trace_path = os.path.join(folder, latest_subdir, f"{method}.csv")
    if not os.path.exists(trace_path):
        print(f"No trace.csv found in latest subdir: {latest_subdir}")
        return None
    trace_df = pd.read_csv(trace_path)
    return trace_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble bead labeling with protein profiles.")
    parser.add_argument("--bead-df", required=True, help="Path to bead_df CSV")
    parser.add_argument("--results-df", required=True, help="Path to results_df CSV")
    parser.add_argument("--protein-profiles", nargs="+", required=True, help="Path(s) to protein profile CSV/Excel files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--other-df", help="Path to other (trace) CSV")
    group.add_argument("--trace-folder", help="Folder to auto-select latest trace from")
    args = parser.parse_args()

    bead_df = pd.read_csv(args.bead_df)
    results_df = pd.read_csv(args.results_df)
    protein_df = load_protein_profiles(args.protein_profiles)
    if args.other_df:
        other_df = pd.read_csv(args.other_df)
    else:
        other_df = get_latest_trace(args.trace_folder)

    # round x,y
    other_df["x"] = np.rint(other_df["x"]).astype(np.float32)
    other_df["y"] = np.rint(other_df["y"]).astype(np.float32)
    results_df["x"] = np.rint(results_df["x"]).astype(np.float32)
    results_df["y"] = np.rint(results_df["y"]).astype(np.float32)
    # merge df on x,y, cy0,cy1 should be suffixed with _other for the trace df, keep all rows from results_df (left join)
    merged_df = results_df.merge(
        other_df, on=["x", "y"], suffixes=("", "_other"), how="left"
    )
    # find what % of beads had 255 for cy0 but not in the other df, and vice versa
    mask_255_cy0 = merged_df["cy0"] == 255
    mask_255_cy0_other = (merged_df["cy0_other"] == 255) | (
        merged_df["cy0_other"].isna()
    )  # treat missing as not 255
    both_255 = (mask_255_cy0 & mask_255_cy0_other).sum()
    only_cy0_255 = (mask_255_cy0 & ~mask_255_cy0_other).sum()
    only_other_255 = (~mask_255_cy0 & mask_255_cy0_other).sum()
    total = len(merged_df)
    print(f"Both cy0 and cy0_other 255: {both_255} ({100 * both_255 / total:.1f}%)")
    print(f"Only cy0 255: {only_cy0_255} ({100 * only_cy0_255 / total:.1f}%)")
    print(f"Only cy0_other 255: {only_other_255} ({100 * only_other_255 / total:.1f}%)")
    # if cy0 is 255 but cy0_other is not, assign cy0 cy1 to the other df's values (assuming the other df is more accurate in this case)
    mask_update = mask_255_cy0 & ~mask_255_cy0_other
    merged_df.loc[mask_update, "cy0"] = merged_df.loc[mask_update, "cy0_other"]
    merged_df.loc[mask_update, "cy1"] = merged_df.loc[mask_update, "cy1_other"]
    results_df_with_protein = label_beads_with_proteins(merged_df, protein_df)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nTotal beads: {len(bead_df)}")

    counts = results_df_with_protein["Protein name"].value_counts()
    valid = len(bead_df) - counts.get("Invalid", 0) - counts.get("Filtered", 0)

    # print("\nPreview:")
    # print(results_df.head(10))
    print(f"Valid: {valid} ({100 * valid / len(bead_df):.1f}%)")
    print(
        f"invalid_pct: {100 * counts.get('Invalid', 0) / (counts.mean() if counts.mean() > 0 else 1):.1f}%"
    )
    print(f"Invalid: {counts.get('Invalid', 0)}")
    print(f"Filtered: {counts.get('Filtered', 0)}")
