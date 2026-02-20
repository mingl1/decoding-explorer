"""
Example usage in Colab notebook after running cell 11 (Generate Beads)

Copy and paste this into a new cell in your Colab notebook.
"""

# Upload the resolve_ambiguous_beads.py file to Colab first, then:

from resolve_ambiguous_beads import (
    compute_median_intensities,
    resolve_with_median_intensity,
    print_resolution_stats
)

# Step 1: Compute median intensities for each bead's assigned label
print("Computing median intensities...")
bead_df_with_intensities = compute_median_intensities(
    bead_df,           # From cell 11
    cycle_images,      # From cell 10
    cycle_labels,      # From cell 11
    cycle_metadata,    # From cell 10
    MAX_SIZE           # From cell 5
)

# Step 2: Resolve ambiguous beads using median intensity
print("\nResolving ambiguous beads...")
results_df_resolved = resolve_with_median_intensity(
    bead_df_with_intensities,
    NUM_CYCLES,        # From cell 10
    num_layers         # From cell 11
)

# Step 3: Show detailed statistics
print_resolution_stats(
    bead_df_with_intensities,
    NUM_CYCLES,
    num_layers
)

# Step 4: Label with proteins
print("\nLabeling with protein names...")
results_df_final = label_beads_with_proteins(results_df_resolved, protein_df)

# Step 5: View results
print("\n" + "="*60)
print("FINAL RESULTS WITH AMBIGUITY RESOLUTION")
print("="*60)
print(f"\nTotal beads: {len(results_df_final)}")

counts = results_df_final["Protein name"].value_counts()
valid = len(results_df_final) - counts.get("Invalid", 0) - counts.get("Filtered", 0)
print(f"Valid: {valid} ({100*valid/len(results_df_final):.1f}%)")
print(f"Invalid: {counts.get('Invalid', 0)}")
print(f"Filtered: {counts.get('Filtered', 0)}")

print("\nProtein distribution:")
print(results_df_final["Protein name"].value_counts().head(20).to_string())

print("\nPreview:")
display(results_df_final.head(10))

# Optional: Compare with old results
if 'results_df' in locals():
    print("\n" + "="*60)
    print("COMPARISON WITH OLD METHOD (filtered ambiguous beads)")
    print("="*60)
    old_valid = len(results_df) - results_df["Protein name"].value_counts().get("Invalid", 0) - results_df["Protein name"].value_counts().get("Filtered", 0)
    print(f"Old method valid beads: {old_valid} / {len(results_df)} ({100*old_valid/len(results_df):.1f}%)")
    print(f"New method valid beads: {valid} / {len(results_df_final)} ({100*valid/len(results_df_final):.1f}%)")
    print(f"Improvement: +{valid - old_valid} valid beads (+{100*(valid-old_valid)/old_valid:.1f}%)")