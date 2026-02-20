# Template Score Resolution for Ambiguous Beads

## Overview

This approach resolves ambiguous bead assignments using **template matching scores** instead of median intensity. The template score measures how well each fluorescence signal matches a Gaussian blob pattern - the expected shape of a well-centered bead.

## Why Template Scores?

**Problem with median intensity:**
- Only measures brightness, not spatial pattern
- Can't distinguish between true bead signal and noise/artifacts
- Led to high invalid rates (77% wrt mean in your data)

**Benefits of template scores:**
- Measures both intensity AND spatial pattern
- Scores how "bead-like" the signal is
- Better distinguishes true signal from background noise
- Same metric used in ROI Inspector for visual assessment

## How It Works

### 1. Template Matching Score Calculation

For each labeled region:
```python
def compute_template_match_score(roi):
    # Use 5x5 Gaussian kernel
    gaussian_5x5 = gaussian_kernel(5)

    # Adjust contrast (10-90 percentile)
    roi_adjusted = adjust_contrast(roi, 10, 90)

    # Correlate with Gaussian template
    score = correlate2d(roi_adjusted, gaussian_5x5, mode="valid")

    return median(score)
```

**Higher score = more Gaussian-like = more bead-like**

### 2. Ambiguity Resolution

For beads with multiple activations in a cycle:
1. Compute template score for each active layer
2. Find the two highest scores
3. Calculate score difference: `diff = score_1st - score_2nd`
4. If `diff >= min_score_diff`: Accept the highest scoring layer
5. If `diff < min_score_diff`: Reject (mark as filtered)

### 3. Threshold Selection

The `min_score_diff` parameter controls strictness:

| Threshold | Effect |
|-----------|--------|
| `None` | Accept all ambiguous (no filtering) |
| `0.01-0.02` | Lenient - only filter very ambiguous beads |
| `0.03-0.05` | Moderate - balance valid/invalid rates |
| `0.05-0.10` | Strict - minimize invalids, more filtered |

Use `suggest_threshold()` to find the optimal value for your data.

## Usage

### Quick Start (in Colab)

```python
from resolve_ambiguous_beads import compute_template_scores, resolve_with_template_scores

# Step 1: Compute template scores
bead_df_with_scores = compute_template_scores(
    bead_df, cycle_images, cycle_labels, cycle_metadata, MAX_SIZE
)

# Step 2: Resolve with threshold
results_df = resolve_with_template_scores(
    bead_df_with_scores,
    NUM_CYCLES,
    num_layers,
    min_score_diff=0.02  # Adjust this value
)

# Step 3: Label with proteins
results_df = label_beads_with_proteins(results_df, protein_df)
```

### With Diagnostics

```python
from diagnose_invalid_beads import suggest_threshold, print_comprehensive_stats

# Try without threshold first
results_no_thresh = resolve_with_template_scores(
    bead_df_with_scores, NUM_CYCLES, num_layers, min_score_diff=None
)
results_no_thresh = label_beads_with_proteins(results_no_thresh, protein_df)

# Find optimal threshold
suggest_threshold(
    bead_df_with_scores,
    results_no_thresh,
    NUM_CYCLES,
    num_layers,
    metric_type="score"
)

# Apply recommended threshold
results_final = resolve_with_template_scores(
    bead_df_with_scores, NUM_CYCLES, num_layers, min_score_diff=0.02
)
results_final = label_beads_with_proteins(results_final, protein_df)
print_comprehensive_stats(results_final)
```

## Expected Improvements

Based on your data showing:
- 90.1% valid, 3.3% invalid, 6.5% filtered (no threshold)
- Invalid wrt mean: 77.7%

With template scores + optimal threshold, expect:
- **Lower invalid rate** (target: <10% wrt mean)
- **Higher filtered rate** (ambiguous beads properly rejected)
- **Similar or higher valid rate** (keep clean + high-confidence ambiguous)

## Key Functions

| Function | Purpose |
|----------|---------|
| `compute_template_scores()` | Calculate template scores for all regions |
| `resolve_with_template_scores()` | Resolve ambiguous beads using scores |
| `suggest_threshold()` | Find optimal threshold for your data |
| `print_comprehensive_stats()` | Show detailed statistics |
| `analyze_invalid_codes()` | Diagnose which codes are invalid |

## Files

- `resolve_ambiguous_beads.py` - Core resolution functions
- `diagnose_invalid_beads.py` - Diagnostic and analysis tools
- `example_usage_template_scores.py` - Complete example with diagnostics
- `quick_start_template_scores.py` - Simple quick-start snippet

## Notes

- Template scores are computed using `regionprops` for efficiency
- Scores are precomputed and stored (not recalculated during resolution)
- Uses same Gaussian kernel as ROI Inspector for consistency
- Fully vectorized for performance (processes thousands of beads quickly)
