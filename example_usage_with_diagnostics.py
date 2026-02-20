"""
Example usage with diagnostics to minimize invalid assignments.

Upload both resolve_ambiguous_beads.py and diagnose_invalid_beads.py to Colab,
then run this after cell 11 (Generate Beads).
"""

from resolve_ambiguous_beads import (
    compute_median_intensities,
    resolve_with_median_intensity,
    print_resolution_stats
)
from diagnose_invalid_beads import (
    print_comprehensive_stats,
    analyze_invalid_codes,
    compare_ambiguous_resolution,
    suggest_threshold
)

# Step 1: Compute median intensities
print("Computing median intensities...")
bead_df_with_intensities = compute_median_intensities(
    bead_df,
    cycle_images,
    cycle_labels,
    cycle_metadata,
    MAX_SIZE
)

# Step 2: First pass - resolve without threshold to see baseline
print("\n" + "="*60)
print("FIRST PASS: No threshold")
print("="*60)
results_df_unfiltered = resolve_with_median_intensity(
    bead_df_with_intensities,
    NUM_CYCLES,
    num_layers,
    min_intensity_diff=None  # No threshold
)

results_df_unfiltered = label_beads_with_proteins(results_df_unfiltered, protein_df)
print_comprehensive_stats(results_df_unfiltered)

# Step 3: Diagnose why beads are invalid
analyze_invalid_codes(results_df_unfiltered, NUM_CYCLES)
compare_ambiguous_resolution(bead_df_with_intensities, results_df_unfiltered, NUM_CYCLES, num_layers)

# Step 4: Suggest optimal threshold
suggest_threshold(bead_df_with_intensities, results_df_unfiltered, NUM_CYCLES, num_layers)

# Step 5: Apply recommended threshold (adjust based on output above)
RECOMMENDED_THRESHOLD = 0.05  # Adjust based on suggest_threshold output

print("\n" + "="*60)
print(f"SECOND PASS: With threshold = {RECOMMENDED_THRESHOLD}")
print("="*60)
results_df_filtered = resolve_with_median_intensity(
    bead_df_with_intensities,
    NUM_CYCLES,
    num_layers,
    min_intensity_diff=RECOMMENDED_THRESHOLD
)

results_df_filtered = label_beads_with_proteins(results_df_filtered, protein_df)
print_comprehensive_stats(results_df_filtered)

# Step 6: Compare old vs new
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

old_valid = len(results_df_unfiltered) - (results_df_unfiltered["Protein name"] == "Invalid").sum() - (results_df_unfiltered["Protein name"] == "Filtered").sum()
new_valid = len(results_df_filtered) - (results_df_filtered["Protein name"] == "Invalid").sum() - (results_df_filtered["Protein name"] == "Filtered").sum()

old_invalid = (results_df_unfiltered["Protein name"] == "Invalid").sum()
new_invalid = (results_df_filtered["Protein name"] == "Invalid").sum()

old_filtered = (results_df_unfiltered["Protein name"] == "Filtered").sum()
new_filtered = (results_df_filtered["Protein name"] == "Filtered").sum()

print(f"\nNo threshold:")
print(f"  Valid: {old_valid} ({100*old_valid/len(results_df_unfiltered):.1f}%)")
print(f"  Invalid: {old_invalid} ({100*old_invalid/len(results_df_unfiltered):.1f}%)")
print(f"  Filtered: {old_filtered} ({100*old_filtered/len(results_df_unfiltered):.1f}%)")

print(f"\nWith threshold = {RECOMMENDED_THRESHOLD}:")
print(f"  Valid: {new_valid} ({100*new_valid/len(results_df_filtered):.1f}%)")
print(f"  Invalid: {new_invalid} ({100*new_invalid/len(results_df_filtered):.1f}%)")
print(f"  Filtered: {new_filtered} ({100*new_filtered/len(results_df_filtered):.1f}%)")

print(f"\nChange:")
print(f"  Valid: {new_valid - old_valid:+d} ({100*(new_valid - old_valid)/old_valid:+.1f}%)")
print(f"  Invalid: {new_invalid - old_invalid:+d} ({100*(new_invalid - old_invalid)/old_invalid:+.1f}%)")
print(f"  Filtered: {new_filtered - old_filtered:+d} ({100*(new_filtered - old_filtered)/old_filtered:+.1f}%)")

# Use the filtered version as final results
results_df = results_df_filtered
print("\n✓ Using threshold-filtered results as final output")
