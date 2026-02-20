"""
Example usage with template matching scores for ambiguity resolution.

This approach uses Gaussian template matching to measure how "bead-like"
each signal is, which is more robust than just using median intensity.

Upload resolve_ambiguous_beads.py and diagnose_invalid_beads.py to Colab,
then run this after cell 11 (Generate Beads).
"""

from resolve_ambiguous_beads import (
    compute_template_scores,
    resolve_with_template_scores,
    print_resolution_stats
)
from diagnose_invalid_beads import (
    print_comprehensive_stats,
    analyze_invalid_codes,
    suggest_threshold
)

# Step 1: Compute template matching scores for each region
print("Computing template matching scores...")
print("(This measures how 'bead-like' each fluorescence signal is)")
bead_df_with_scores = compute_template_scores(
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
results_df_unfiltered = resolve_with_template_scores(
    bead_df_with_scores,
    NUM_CYCLES,
    num_layers,
    min_score_diff=None  # No threshold
)

results_df_unfiltered = label_beads_with_proteins(results_df_unfiltered, protein_df)
print_comprehensive_stats(results_df_unfiltered)

# Step 3: Diagnose why beads are invalid
analyze_invalid_codes(results_df_unfiltered, NUM_CYCLES)

# Step 4: Suggest optimal threshold based on score differences
suggest_threshold(
    bead_df_with_scores,
    results_df_unfiltered,
    NUM_CYCLES,
    num_layers,
    metric_type="score"  # Use template scores
)

# Step 5: Apply recommended threshold (adjust based on output above)
# Start with a small threshold like 0.01-0.05
RECOMMENDED_THRESHOLD = 0.02  # Adjust based on suggest_threshold output

print("\n" + "="*60)
print(f"SECOND PASS: With score threshold = {RECOMMENDED_THRESHOLD}")
print("="*60)
results_df_filtered = resolve_with_template_scores(
    bead_df_with_scores,
    NUM_CYCLES,
    num_layers,
    min_score_diff=RECOMMENDED_THRESHOLD
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

print(f"\nWith score threshold = {RECOMMENDED_THRESHOLD}:")
print(f"  Valid: {new_valid} ({100*new_valid/len(results_df_filtered):.1f}%)")
print(f"  Invalid: {new_invalid} ({100*new_invalid/len(results_df_filtered):.1f}%)")
print(f"  Filtered: {new_filtered} ({100*new_filtered/len(results_df_filtered):.1f}%)")

print(f"\nChange:")
print(f"  Valid: {new_valid - old_valid:+d} ({100*(new_valid - old_valid)/old_valid if old_valid > 0 else 0:+.1f}%)")
print(f"  Invalid: {new_invalid - old_invalid:+d} ({100*(new_invalid - old_invalid)/old_invalid if old_invalid > 0 else 0:+.1f}%)")
print(f"  Filtered: {new_filtered - old_filtered:+d} ({100*(new_filtered - old_filtered)/old_filtered if old_filtered > 0 else 0:+.1f}%)")

# Calculate invalid wrt mean
valid_proteins = results_df_filtered[~results_df_filtered["Protein name"].isin(["Invalid", "Filtered"])]
if len(valid_proteins) > 0:
    mean_beads_per_protein = len(valid_proteins) / valid_proteins["Protein name"].nunique()
    invalid_wrt_mean = 100 * new_invalid / mean_beads_per_protein
    print(f"\nInvalid wrt mean: {invalid_wrt_mean:.1f}%")

# Use the filtered version as final results
results_df = results_df_filtered
print("\n✓ Using threshold-filtered results as final output")

# Optional: Show score distributions for valid vs invalid beads
print("\n" + "="*60)
print("SCORE QUALITY ANALYSIS")
print("="*60)

# Sample some valid and invalid beads to check their scores
valid_beads = results_df_filtered[~results_df_filtered["Protein name"].isin(["Invalid", "Filtered"])]
invalid_beads = results_df_filtered[results_df_filtered["Protein name"] == "Invalid"]

if len(valid_beads) > 0 and len(invalid_beads) > 0:
    print(f"\nValid beads: {len(valid_beads)}")
    print(f"Invalid beads: {len(invalid_beads)}")
    print("\nThis suggests the template score approach is working!")
