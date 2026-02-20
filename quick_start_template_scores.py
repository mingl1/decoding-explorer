"""
Quick Start: Template Score Resolution
Add this cell after cell 11 in your Colab notebook.
"""

#@title Resolve Ambiguous Beads with Template Scores

# Upload resolve_ambiguous_beads.py first
from resolve_ambiguous_beads import compute_template_scores, resolve_with_template_scores

# Compute template matching scores (measures how "bead-like" each signal is)
print("Computing template scores...")
bead_df_with_scores = compute_template_scores(
    bead_df, cycle_images, cycle_labels, cycle_metadata, MAX_SIZE
)

# Resolve with threshold
# min_score_diff: minimum score difference between 1st and 2nd highest layers
# - Start with 0.01-0.02 for lenient filtering
# - Increase to 0.03-0.05 for stricter filtering
# - Set to None to accept all ambiguous beads
SCORE_THRESHOLD = 0.02  #@param {type:"number"}

print(f"\nResolving ambiguous beads (threshold={SCORE_THRESHOLD})...")
results_df_resolved = resolve_with_template_scores(
    bead_df_with_scores,
    NUM_CYCLES,
    num_layers,
    min_score_diff=SCORE_THRESHOLD
)

# Label with proteins
results_df_resolved = label_beads_with_proteins(results_df_resolved, protein_df)

# Print statistics
print("\n" + "="*60)
print("RESULTS")
print("="*60)

total = len(results_df_resolved)
invalid = (results_df_resolved["Protein name"] == "Invalid").sum()
filtered = (results_df_resolved["Protein name"] == "Filtered").sum()
valid = total - invalid - filtered

print(f"\nTotal beads: {total}")
print(f"  Valid: {valid} ({100*valid/total:.1f}%)")
print(f"  Invalid: {invalid} ({100*invalid/total:.1f}%)")
print(f"  Filtered: {filtered} ({100*filtered/total:.1f}%)")

# Calculate invalid wrt mean
valid_proteins = results_df_resolved[~results_df_resolved["Protein name"].isin(["Invalid", "Filtered"])]
if len(valid_proteins) > 0:
    unique_proteins = valid_proteins["Protein name"].nunique()
    mean_beads = valid / unique_proteins
    invalid_wrt_mean = 100 * invalid / mean_beads
    print(f"\nUnique proteins: {unique_proteins}")
    print(f"Mean beads per protein: {mean_beads:.1f}")
    print(f"Invalid wrt mean: {invalid_wrt_mean:.1f}%")

# Use this as final results
results_df = results_df_resolved
