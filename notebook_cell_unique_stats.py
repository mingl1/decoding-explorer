#@title 11. View Full Results (Unique Beads)

# Remove duplicates by x,y coordinates
# (Some proteins have same activation codes, causing duplicates)
unique_beads = results_df.drop_duplicates(subset=['x', 'y'], keep='first')

num_duplicates = len(results_df) - len(unique_beads)
if num_duplicates > 0:
    print(f"Note: Removed {num_duplicates} duplicate entries (same x,y coordinates)\n")

print("="*60)
print("PROTEIN ASSIGNMENT STATISTICS (Unique Beads)")
print("="*60)

# Overall counts
total_beads = len(unique_beads)
invalid_count = (unique_beads["Protein name"] == "Invalid").sum()
filtered_count = (unique_beads["Protein name"] == "Filtered").sum()
valid_count = total_beads - invalid_count - filtered_count

print(f"\nTotal unique bead positions: {total_beads}")
print(f"  Valid: {valid_count} ({100*valid_count/total_beads:.1f}%)")
print(f"  Invalid: {invalid_count} ({100*invalid_count/total_beads:.1f}%)")
print(f"  Filtered: {filtered_count} ({100*filtered_count/total_beads:.1f}%)")

# Valid protein statistics
valid_results = unique_beads[~unique_beads["Protein name"].isin(["Filtered", "Invalid"])]
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

    # Get cycle columns (cy0, cy1, etc.)
    cycle_cols = [col for col in unique_beads.columns if col.startswith('cy') and col[2:].isdigit()]

    # Group by protein name and show activation codes
    protein_summary = valid_results.groupby('Protein name').agg({
        **{col: 'first' for col in cycle_cols},  # Get activation codes
        'x': 'count'  # Count occurrences
    }).rename(columns={'x': 'Count'})

    # Sort by count descending
    protein_summary = protein_summary.sort_values('Count', ascending=False)

    # Format the output
    print(protein_summary.head(20).to_string())

    # Show proteins with very few beads (potential issues)
    low_count_proteins = protein_counts[protein_counts < 5]
    if len(low_count_proteins) > 0:
        print(f"\nProteins with <5 beads ({len(low_count_proteins)} proteins):")
        print(low_count_proteins.to_string())
else:
    print("\nNo valid proteins detected!")

# Optional: Replace results_df with deduplicated version
# results_df = unique_beads
