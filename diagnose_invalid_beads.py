"""
Diagnostic functions to understand why resolved beads are becoming invalid.
"""

import numpy as np
import pandas as pd
from collections import Counter


def print_comprehensive_stats(results_df):
    """Print comprehensive statistics matching the user's format."""
    print("="*60)
    print("PROTEIN ASSIGNMENT STATISTICS")
    print("="*60)

    # Overall counts
    total_beads = len(results_df)
    invalid_count = (results_df["Protein name"] == "Invalid").sum()
    filtered_count = (results_df["Protein name"] == "Filtered").sum()
    valid_count = total_beads - invalid_count - filtered_count

    print(f"\nTotal beads: {total_beads}")
    print(f"  Valid: {valid_count} ({100*valid_count/total_beads:.1f}%)")
    print(f"  Invalid: {invalid_count} ({100*invalid_count/total_beads:.1f}%)")
    print(f"  Filtered: {filtered_count} ({100*filtered_count/total_beads:.1f}%)")

    # Valid protein statistics
    valid_results = results_df[~results_df["Protein name"].isin(["Filtered", "Invalid"])]
    if len(valid_results) > 0:
        protein_counts = valid_results["Protein name"].value_counts()
        unique_proteins = len(protein_counts)
        mean_beads_per_protein = valid_count / unique_proteins

        print(f"\nUnique proteins detected: {unique_proteins}")
        print(f"Mean beads per protein: {mean_beads_per_protein:.1f}")
        print(f"Invalid wrt mean: {100*invalid_count/mean_beads_per_protein:.1f}%")
        print(f"  Min: {protein_counts.min()}")
        print(f"  Max: {protein_counts.max()}")
        print(f"  Median: {protein_counts.median():.0f}")

        print(f"\nProtein distribution (top 20):")
        print(protein_counts.head(20).to_string())


def analyze_invalid_codes(results_df, num_cycles):
    """Analyze which cycle codes are causing invalid assignments."""
    invalid_beads = results_df[results_df["Protein name"] == "Invalid"]

    if len(invalid_beads) == 0:
        print("No invalid beads to analyze.")
        return

    print("\n" + "="*60)
    print("INVALID CODE ANALYSIS")
    print("="*60)
    print(f"\nTotal invalid beads: {len(invalid_beads)}")

    # Get cycle columns
    cycle_cols = [f"cy{i}" for i in range(num_cycles)]

    # Find most common invalid codes
    invalid_codes = invalid_beads[cycle_cols].apply(tuple, axis=1)
    code_counts = Counter(invalid_codes)

    print(f"\nMost common invalid codes (top 20):")
    print("Cycle codes -> Count")
    for code, count in code_counts.most_common(20):
        pct = 100 * count / len(invalid_beads)
        code_str = ", ".join([str(c) for c in code])
        print(f"  ({code_str}) -> {count} beads ({pct:.1f}%)")

    # Analyze which cycles have unusual values
    print(f"\nCycle value distributions in invalid beads:")
    for col in cycle_cols:
        values = invalid_beads[col].value_counts().sort_index()
        print(f"\n{col}:")
        for val, cnt in values.items():
            if val != 255:  # Skip filtered marker
                pct = 100 * cnt / len(invalid_beads)
                print(f"  Layer {val}: {cnt} ({pct:.1f}%)")


def compare_ambiguous_resolution(bead_df_with_intensities, results_df, num_cycles, num_layers):
    """Compare intensity differences for beads that became invalid vs valid."""

    # Get ambiguous beads
    ambiguous_masks = []
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        count = (bead_df_with_intensities[layer_cols] > 0).sum(axis=1)
        ambiguous_masks.append(count > 1)

    overall_ambiguous = pd.concat(ambiguous_masks, axis=1).any(axis=1)
    ambiguous_beads = bead_df_with_intensities[overall_ambiguous].copy()

    # Add resolution outcome
    ambiguous_beads["outcome"] = results_df.loc[overall_ambiguous, "Protein name"]
    ambiguous_beads["is_invalid"] = ambiguous_beads["outcome"] == "Invalid"
    ambiguous_beads["is_valid"] = ~ambiguous_beads["outcome"].isin(["Invalid", "Filtered"])

    print("\n" + "="*60)
    print("AMBIGUOUS BEAD RESOLUTION QUALITY")
    print("="*60)

    print(f"\nTotal ambiguous beads: {len(ambiguous_beads)}")
    print(f"  Became valid: {ambiguous_beads['is_valid'].sum()} ({100*ambiguous_beads['is_valid'].mean():.1f}%)")
    print(f"  Became invalid: {ambiguous_beads['is_invalid'].sum()} ({100*ambiguous_beads['is_invalid'].mean():.1f}%)")

    # Compare intensity differences for valid vs invalid outcomes
    print("\nIntensity separation (1st - 2nd highest) by outcome:")

    for cycle_idx in range(num_cycles):
        intensity_cols = [f"cy{cycle_idx}_{j}_intensity" for j in range(num_layers)]
        layer_cols = [f"cy{cycle_idx}_{j}" for j in range(num_layers)]

        # Get beads ambiguous in this cycle
        count = (ambiguous_beads[layer_cols] > 0).sum(axis=1)
        cycle_ambiguous = ambiguous_beads[count > 1].copy()

        if len(cycle_ambiguous) == 0:
            continue

        # Calculate intensity differences
        diffs_valid = []
        diffs_invalid = []

        for idx in cycle_ambiguous.index:
            intensities = cycle_ambiguous.loc[idx, intensity_cols].values
            labels = cycle_ambiguous.loc[idx, layer_cols].values
            active_intensities = intensities[labels > 0]

            if len(active_intensities) >= 2:
                sorted_int = np.sort(active_intensities)[::-1]
                diff = sorted_int[0] - sorted_int[1]

                if cycle_ambiguous.loc[idx, "is_valid"]:
                    diffs_valid.append(diff)
                elif cycle_ambiguous.loc[idx, "is_invalid"]:
                    diffs_invalid.append(diff)

        if diffs_valid and diffs_invalid:
            print(f"\nCycle {cycle_idx}:")
            print(f"  Valid outcomes (n={len(diffs_valid)}):")
            print(f"    Mean: {np.mean(diffs_valid):.4f}, Median: {np.median(diffs_valid):.4f}")
            print(f"  Invalid outcomes (n={len(diffs_invalid)}):")
            print(f"    Mean: {np.mean(diffs_invalid):.4f}, Median: {np.median(diffs_invalid):.4f}")
            print(f"  Difference: {np.mean(diffs_valid) - np.mean(diffs_invalid):.4f}")


def suggest_threshold(bead_df_with_metrics, results_df, num_cycles, num_layers, metric_type="intensity"):
    """
    Suggest a threshold to filter ambiguous beads.

    Args:
        bead_df_with_metrics: DataFrame with cy{i}_{j}_intensity or cy{i}_{j}_score columns
        results_df: Results DataFrame with Protein name assignments
        num_cycles: Number of cycles
        num_layers: Number of layers
        metric_type: Either "intensity" or "score" to determine which metric to use
    """

    # Determine metric column suffix
    metric_suffix = "_intensity" if metric_type == "intensity" else "_score"

    # Get ambiguous beads with outcomes
    ambiguous_masks = []
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        count = (bead_df_with_metrics[layer_cols] > 0).sum(axis=1)
        ambiguous_masks.append(count > 1)

    overall_ambiguous = pd.concat(ambiguous_masks, axis=1).any(axis=1)
    ambiguous_beads = bead_df_with_metrics[overall_ambiguous].copy()
    ambiguous_beads["outcome"] = results_df.loc[overall_ambiguous, "Protein name"]

    # Calculate min metric difference across all cycles for each bead
    min_diffs = []
    for idx in ambiguous_beads.index:
        bead_min_diff = np.inf

        for cycle_idx in range(num_cycles):
            metric_cols = [f"cy{cycle_idx}_{j}{metric_suffix}" for j in range(num_layers)]
            layer_cols = [f"cy{cycle_idx}_{j}" for j in range(num_layers)]

            metrics = ambiguous_beads.loc[idx, metric_cols].values
            labels = ambiguous_beads.loc[idx, layer_cols].values
            active_metrics = metrics[labels > 0]

            if len(active_metrics) >= 2:
                sorted_metrics = np.sort(active_metrics)[::-1]
                diff = sorted_metrics[0] - sorted_metrics[1]
                bead_min_diff = min(bead_min_diff, diff)

        if bead_min_diff != np.inf:
            min_diffs.append((idx, bead_min_diff, ambiguous_beads.loc[idx, "outcome"]))

    df_diffs = pd.DataFrame(min_diffs, columns=["idx", "min_diff", "outcome"])
    df_diffs["is_valid"] = ~df_diffs["outcome"].isin(["Invalid", "Filtered"])

    print("\n" + "="*60)
    print(f"{metric_type.upper()} THRESHOLD ANALYSIS")
    print("="*60)

    # Test different thresholds
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]
    print("\nImpact of minimum intensity difference thresholds:")
    print("Threshold | Kept | Valid% | Invalid% | Filtered%")
    print("-" * 55)

    for thresh in thresholds:
        kept = df_diffs[df_diffs["min_diff"] >= thresh]
        if len(kept) > 0:
            valid_pct = 100 * kept["is_valid"].mean()
            invalid_pct = 100 * (kept["outcome"] == "Invalid").mean()
            filtered_pct = 100 * (kept["outcome"] == "Filtered").mean()
            print(f"{thresh:8.3f}  | {len(kept):5d} | {valid_pct:5.1f}% | {invalid_pct:6.1f}% | {filtered_pct:7.1f}%")

    print("\nRecommendation: Choose a threshold where Invalid% is minimized")
    print("while keeping a reasonable number of beads.")
