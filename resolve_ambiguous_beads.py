"""
Functions to resolve ambiguous bead assignments using template matching scores.

Template matching measures how "bead-like" each fluorescence signal is by correlating
with a Gaussian kernel. This is more robust than using median intensity alone.

## Quick Start - Template Score Resolution (Recommended)

Usage in Colab after running the main pipeline:

    from resolve_ambiguous_beads import compute_template_scores, resolve_with_template_scores

    # Step 1: Compute template matching scores for each region
    bead_df_with_scores = compute_template_scores(
        bead_df, cycle_images, cycle_labels, cycle_metadata, MAX_SIZE
    )

    # Step 2: Resolve ambiguous beads with optional threshold
    # min_score_diff: minimum score difference between 1st and 2nd highest layers
    # Start with 0.01-0.02 for lenient, 0.03-0.05 for stricter filtering
    results_df = resolve_with_template_scores(
        bead_df_with_scores, NUM_CYCLES, num_layers, min_score_diff=0.02
    )

    # Step 3: Label with proteins
    results_df = label_beads_with_proteins(results_df, protein_df)

## Alternative 1 - NCC Cross-Cycle Resolution

Use clean signals from one cycle to resolve ambiguous signals in other cycles:

    from resolve_ambiguous_beads import resolve_using_clean_cycle_ncc

    # For beads where one cycle is clean (count==1) and others are ambiguous (count!=1),
    # extract a 9x9 patch from the clean cycle and use NCC to disambiguate other cycles
    bead_df_ncc = resolve_using_clean_cycle_ncc(
        bead_df, cycle_images, cycle_metadata, MAX_SIZE, NUM_CYCLES, num_layers
    )
    # Returns cy{i}_ncc_resolved columns with layer assignments (255 if unresolved)

## Alternative 2 - Median Intensity Resolution

    from resolve_ambiguous_beads import compute_median_intensities, resolve_with_median_intensity

    bead_df_with_intensities = compute_median_intensities(
        bead_df, cycle_images, cycle_labels, cycle_metadata, MAX_SIZE
    )
    results_df = resolve_with_median_intensity(
        bead_df_with_intensities, NUM_CYCLES, num_layers, min_intensity_diff=0.05
    )
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate2d
from skimage.measure import regionprops
from tqdm.auto import tqdm


def adjust_contrast(img, min_percentile=2, max_percentile=98):
    """Adjust image contrast using percentile-based clipping for float images"""
    # Calculate percentiles
    minval = np.percentile(img, min_percentile)
    maxval = np.percentile(img, max_percentile)

    # Avoid division by zero
    if maxval - minval < 1e-12:
        return np.zeros_like(img)

    # Clip and rescale to [0.0, 1.0]
    img_adjusted = np.clip(img, minval, maxval)
    img_adjusted = (img_adjusted - minval) / (maxval - minval)

    return img_adjusted


def gaussian_kernel(size: int, sigma=None) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel.

    Args:
        size (int): Kernel size (must be odd).
        sigma (float): Standard deviation. If None, defaults to size/6.

    Returns:
        np.ndarray: (size, size) Gaussian kernel normalized to sum=1
    """
    if size % 2 == 0:
        raise ValueError("Size must be odd.")

    if sigma is None:
        sigma = size / 6.0  # heuristic so Gaussian spans kernel nicely

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    return kernel / np.sum(kernel)


def compute_template_scores(
    bead_df, cycle_images, cycle_labels, cycle_metadata, max_size, ambiguous_only=True
):
    """
    Compute template matching scores for each bead's assigned label.

    Args:
        bead_df: DataFrame with x, y, and cy{i}_{j} columns (label assignments)
        cycle_images: List of cycle image arrays
        cycle_labels: List of label arrays from get_labels_from_cycles
        cycle_metadata: List of MetaData objects
        max_size: Maximum size for cropping
        ambiguous_only: If True, only compute scores for ambiguous beads (much faster)

    Returns:
        DataFrame with additional cy{i}_{j}_score columns
    """
    bead_df = bead_df.copy()

    # Identify ambiguous beads if we're only computing for them
    num_cycles = len(cycle_images)
    num_layers = len(cycle_metadata[0].flors_layers) if cycle_metadata else 0

    if ambiguous_only and num_layers > 0:
        # Find beads that have 2+ activations in any cycle
        ambiguous_mask = pd.Series(False, index=bead_df.index)
        for cycle_idx in range(num_cycles):
            layer_cols = [f"cy{cycle_idx}_{j}" for j in range(num_layers)]
            if all(col in bead_df.columns for col in layer_cols):
                count = (bead_df[layer_cols] > 0).sum(axis=1)
                ambiguous_mask |= count > 1

        print(
            f"Computing scores for {ambiguous_mask.sum()} ambiguous beads (skipping {(~ambiguous_mask).sum()} clean beads)"
        )
    else:
        ambiguous_mask = pd.Series(True, index=bead_df.index)
        print(f"Computing scores for all {len(bead_df)} beads")

    total_layers = sum(len(metadata.flors_layers) for metadata in cycle_metadata)
    pbar = tqdm(total=total_layers, desc="Computing template scores")

    # Pre-compute Gaussian kernel once
    gaussian_kernel_5x5 = gaussian_kernel(5)

    for cycle_idx, (cycle_img, cycle_label_list, metadata) in enumerate(
        zip(cycle_images, cycle_labels, cycle_metadata)
    ):
        # Crop the cycle image
        cycle_img_cropped = cycle_img[:, :max_size, :max_size]

        # For each fluorescence layer
        for layer_idx, flor_layer_idx in enumerate(metadata.flors_layers):
            col_name = f"cy{cycle_idx}_{layer_idx}"
            score_col_name = f"{col_name}_score"

            # Get unique label IDs for ambiguous beads only (excluding 0)
            assigned_labels = bead_df[col_name].values
            ambiguous_labels = bead_df.loc[ambiguous_mask, col_name].values
            unique_labels = set(
                ambiguous_labels[ambiguous_labels > 0]
            )  # Use set for O(1) lookup

            # Skip if no ambiguous beads assigned to this layer
            if len(unique_labels) == 0:
                bead_df[score_col_name] = 0.0
                pbar.update(1)
                continue

            # Get the intensity image and label map
            img = cycle_img_cropped[flor_layer_idx].astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            labels = cycle_label_list[layer_idx]

            # Adjust contrast once for the entire image
            img_adjusted = adjust_contrast(img, 10, 90)

            # Use regionprops but only compute scores for assigned labels
            props = regionprops(labels, intensity_image=img_adjusted)
            label_to_score = {}

            for prop in props:
                # Only compute score if this label is assigned to at least one bead
                if prop.label not in unique_labels:
                    continue

                # Skip very small ROIs early (before extracting intensity)
                if prop.bbox_area < 25:  # Less than 5x5 pixels
                    label_to_score[prop.label] = 0.0
                    continue

                # Extract the ROI for this region
                roi = prop.intensity_image

                # Compute template matching score only if ROI is large enough
                if roi.shape[0] >= 5 and roi.shape[1] >= 5:
                    # Use correlation (scipy's correlate2d is optimized)
                    score_map = correlate2d(roi, gaussian_kernel_5x5, mode="valid")
                    label_to_score[prop.label] = float(np.median(score_map))
                else:
                    label_to_score[prop.label] = 0.0

            # Vectorized lookup using label IDs
            scores = np.array(
                [label_to_score.get(label_id, 0.0) for label_id in assigned_labels],
                dtype=np.float32,
            )
            bead_df[score_col_name] = scores

            pbar.update(1)

    pbar.close()
    print(f"Added template score columns for {len(bead_df)} beads")
    return bead_df


def compute_median_intensities(
    bead_df, cycle_images, cycle_labels, cycle_metadata, max_size
):
    """
    Compute median intensity for each bead's assigned label using regionprops.

    Args:
        bead_df: DataFrame with x, y, and cy{i}_{j} columns (label assignments)
        cycle_images: List of cycle image arrays
        cycle_labels: List of label arrays from get_labels_from_cycles
        cycle_metadata: List of MetaData objects
        max_size: Maximum size for cropping

    Returns:
        DataFrame with additional cy{i}_{j}_intensity columns
    """
    bead_df = bead_df.copy()

    total_layers = sum(len(metadata.flors_layers) for metadata in cycle_metadata)
    pbar = tqdm(total=total_layers, desc="Computing median intensities")

    for cycle_idx, (cycle_img, cycle_label_list, metadata) in enumerate(
        zip(cycle_images, cycle_labels, cycle_metadata)
    ):
        # Crop the cycle image
        cycle_img_cropped = cycle_img[:, :max_size, :max_size]

        # For each fluorescence layer
        for layer_idx, flor_layer_idx in enumerate(metadata.flors_layers):
            # Get the intensity image and label map
            img = cycle_img_cropped[flor_layer_idx].astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            labels = cycle_label_list[layer_idx]

            # Use regionprops to compute median intensity for all regions at once
            props = regionprops(labels, intensity_image=img)
            label_to_median = {
                prop.label: np.median(prop.intensity_image[prop.image])
                for prop in props
            }

            # Assign median intensity to each bead
            col_name = f"cy{cycle_idx}_{layer_idx}"
            intensity_col_name = f"{col_name}_intensity"

            # Vectorized lookup using label IDs
            label_ids = bead_df[col_name].values
            intensities = np.array(
                [label_to_median.get(label_id, 0.0) for label_id in label_ids],
                dtype=np.float32,
            )
            bead_df[intensity_col_name] = intensities

            pbar.update(1)

    pbar.close()
    print(f"Added median intensity columns for {len(bead_df)} beads")
    return bead_df


def resolve_with_template_scores(bead_df, num_cycles, num_layers, min_score_diff=None):
    """
    Resolve layer assignments using template matching scores for ambiguous beads.

    For beads with multiple activations in a cycle, selects the layer with highest template score.
    Optionally filters ambiguous beads that don't have sufficient score separation.

    Args:
        bead_df: DataFrame with cy{i}_{j} and cy{i}_{j}_score columns
        num_cycles: Number of cycles
        num_layers: Number of fluorescence layers per cycle
        min_score_diff: Optional minimum score difference (1st - 2nd highest) required
                       to accept ambiguous bead resolution. If None, accepts all.
                       Recommended: 0.01-0.05 depending on your data.

    Returns:
        DataFrame with x, y, and cy{i} columns (resolved assignments)
    """
    bead_df = bead_df.copy()
    final_df = bead_df[["x", "y"]].copy()

    # Count activations per cycle
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        bead_df[f"cy{i}_count"] = (bead_df[layer_cols] > 0).sum(axis=1)

    stats = {"clean": 0, "resolved_ambiguous": 0, "zero_signal": 0}

    # Process each cycle
    for i in tqdm(range(num_cycles), desc="Resolving cycles"):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        score_cols = [f"cy{i}_{j}_score" for j in range(num_layers)]

        # Initialize all as 255 (filtered)
        final_df[f"cy{i}"] = 255

        # Clean beads (exactly 1 activation) - vectorized
        clean_mask = bead_df[f"cy{i}_count"] == 1
        if clean_mask.any():
            # For clean beads, find which layer is active
            clean_beads = bead_df.loc[clean_mask, layer_cols]
            # Get the column index where value > 0, then extract layer number
            assigned_layers = (
                (clean_beads > 0).idxmax(axis=1).str.split("_").str[-1].astype(int)
            )
            final_df.loc[clean_mask, f"cy{i}"] = assigned_layers
            if i == 0:
                stats["clean"] = clean_mask.sum()

        # Ambiguous beads (2+ activations) - resolve using template scores (vectorized)
        ambiguous_mask = bead_df[f"cy{i}_count"] > 1
        if ambiguous_mask.any():
            # Get labels and scores for ambiguous beads
            labels = bead_df.loc[ambiguous_mask, layer_cols].values
            scores = bead_df.loc[ambiguous_mask, score_cols].values

            # Mask out inactive layers by setting their score to -inf
            masked_scores = np.where(labels > 0, scores, -np.inf)

            # Find the layer with highest score for each bead (vectorized)
            best_layers = np.argmax(masked_scores, axis=1)

            # Apply score difference threshold if specified
            if min_score_diff is not None:
                # Calculate score difference between 1st and 2nd highest for each bead
                sorted_scores = np.sort(masked_scores, axis=1)[:, ::-1]
                score_diffs = sorted_scores[:, 0] - sorted_scores[:, 1]

                # Only accept beads with sufficient separation
                accept_mask = score_diffs >= min_score_diff

                # Apply assignments only for beads passing threshold
                ambiguous_indices = bead_df[ambiguous_mask].index
                accepted_indices = ambiguous_indices[accept_mask]
                final_df.loc[accepted_indices, f"cy{i}"] = best_layers[accept_mask]

                # Rejected beads remain at 255 (filtered)
                if i == 0:
                    stats["resolved_ambiguous"] = accept_mask.sum()
                    stats["ambiguous_filtered"] = (~accept_mask).sum()
            else:
                # No threshold - accept all ambiguous resolutions
                final_df.loc[ambiguous_mask, f"cy{i}"] = best_layers
                if i == 0:
                    stats["resolved_ambiguous"] = ambiguous_mask.sum()

        # Zero signal beads (already set to 255)
        zero_mask = bead_df[f"cy{i}_count"] == 0
        if i == 0:
            stats["zero_signal"] = zero_mask.sum()

    # Convert to uint8
    for i in range(num_cycles):
        final_df[f"cy{i}"] = final_df[f"cy{i}"].astype(np.uint8)

    print("\nResolution summary:")
    print(
        f"  Clean assignments: {stats['clean']} / {len(bead_df)} ({100 * stats['clean'] / len(bead_df):.1f}%)"
    )
    print(
        f"  Resolved ambiguous: {stats['resolved_ambiguous']} / {len(bead_df)} ({100 * stats['resolved_ambiguous'] / len(bead_df):.1f}%)"
    )
    if "ambiguous_filtered" in stats:
        print(
            f"  Ambiguous filtered (low separation): {stats['ambiguous_filtered']} / {len(bead_df)} ({100 * stats['ambiguous_filtered'] / len(bead_df):.1f}%)"
        )
    print(
        f"  Zero signal: {stats['zero_signal']} / {len(bead_df)} ({100 * stats['zero_signal'] / len(bead_df):.1f}%)"
    )

    return final_df


def resolve_with_median_intensity(
    bead_df, num_cycles, num_layers, min_intensity_diff=None
):
    """
    Resolve layer assignments for each cycle using median intensity for ambiguous beads.

    For beads with multiple activations in a cycle, selects the layer with highest median intensity.
    Optionally filters ambiguous beads that don't have sufficient intensity separation.

    Args:
        bead_df: DataFrame with cy{i}_{j} and cy{i}_{j}_intensity columns
        num_cycles: Number of cycles
        num_layers: Number of fluorescence layers per cycle
        min_intensity_diff: Optional minimum intensity difference (1st - 2nd highest) required
                           to accept ambiguous bead resolution. If None, accepts all.
                           Recommended: 0.03-0.05 to reduce invalid assignments.

    Returns:
        DataFrame with x, y, and cy{i} columns (resolved assignments)
    """
    bead_df = bead_df.copy()
    final_df = bead_df[["x", "y"]].copy()

    # Count activations per cycle
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        bead_df[f"cy{i}_count"] = (bead_df[layer_cols] > 0).sum(axis=1)

    stats = {"clean": 0, "resolved_ambiguous": 0, "zero_signal": 0}

    # Process each cycle
    for i in tqdm(range(num_cycles), desc="Resolving cycles"):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        intensity_cols = [f"cy{i}_{j}_intensity" for j in range(num_layers)]

        # Initialize all as 255 (filtered)
        final_df[f"cy{i}"] = 255

        # Clean beads (exactly 1 activation) - vectorized
        clean_mask = bead_df[f"cy{i}_count"] == 1
        if clean_mask.any():
            # For clean beads, find which layer is active
            clean_beads = bead_df.loc[clean_mask, layer_cols]
            # Get the column index where value > 0, then extract layer number
            assigned_layers = (
                (clean_beads > 0).idxmax(axis=1).str.split("_").str[-1].astype(int)
            )
            final_df.loc[clean_mask, f"cy{i}"] = assigned_layers
            if i == 0:
                stats["clean"] = clean_mask.sum()

        # Ambiguous beads (2+ activations) - resolve using median intensity (vectorized)
        ambiguous_mask = bead_df[f"cy{i}_count"] > 1
        if ambiguous_mask.any():
            # Get labels and intensities for ambiguous beads
            labels = bead_df.loc[ambiguous_mask, layer_cols].values
            intensities = bead_df.loc[ambiguous_mask, intensity_cols].values

            # Mask out inactive layers by setting their intensity to -inf
            masked_intensities = np.where(labels > 0, intensities, -np.inf)

            # Find the layer with highest intensity for each bead (vectorized)
            best_layers = np.argmax(masked_intensities, axis=1)

            # Apply intensity difference threshold if specified
            if min_intensity_diff is not None:
                # Calculate intensity difference between 1st and 2nd highest for each bead
                sorted_intensities = np.sort(masked_intensities, axis=1)[:, ::-1]
                intensity_diffs = sorted_intensities[:, 0] - sorted_intensities[:, 1]

                # Only accept beads with sufficient separation
                accept_mask = intensity_diffs >= min_intensity_diff

                # Apply assignments only for beads passing threshold
                ambiguous_indices = bead_df[ambiguous_mask].index
                accepted_indices = ambiguous_indices[accept_mask]
                final_df.loc[accepted_indices, f"cy{i}"] = best_layers[accept_mask]

                # Rejected beads remain at 255 (filtered)
                if i == 0:
                    stats["resolved_ambiguous"] = accept_mask.sum()
                    stats["ambiguous_filtered"] = (~accept_mask).sum()
            else:
                # No threshold - accept all ambiguous resolutions
                final_df.loc[ambiguous_mask, f"cy{i}"] = best_layers
                if i == 0:
                    stats["resolved_ambiguous"] = ambiguous_mask.sum()

        # Zero signal beads (already set to 255)
        zero_mask = bead_df[f"cy{i}_count"] == 0
        if i == 0:
            stats["zero_signal"] = zero_mask.sum()

    # Convert to uint8
    for i in range(num_cycles):
        final_df[f"cy{i}"] = final_df[f"cy{i}"].astype(np.uint8)

    print("\nResolution summary:")
    print(
        f"  Clean assignments: {stats['clean']} / {len(bead_df)} ({100 * stats['clean'] / len(bead_df):.1f}%)"
    )
    print(
        f"  Resolved ambiguous: {stats['resolved_ambiguous']} / {len(bead_df)} ({100 * stats['resolved_ambiguous'] / len(bead_df):.1f}%)"
    )
    if "ambiguous_filtered" in stats:
        print(
            f"  Ambiguous filtered (low separation): {stats['ambiguous_filtered']} / {len(bead_df)} ({100 * stats['ambiguous_filtered'] / len(bead_df):.1f}%)"
        )
    print(
        f"  Zero signal: {stats['zero_signal']} / {len(bead_df)} ({100 * stats['zero_signal'] / len(bead_df):.1f}%)"
    )

    return final_df


def compute_ncc(patch1, patch2):
    """
    Compute normalized cross-correlation between two patches.

    NCC is mean-centered and normalized by standard deviation.

    Args:
        patch1: First patch (reference)
        patch2: Second patch (to compare)

    Returns:
        NCC score (higher is more similar, range approximately [-1, 1])
    """
    # Mean-center
    p1 = patch1 - np.mean(patch1)
    p2 = patch2 - np.mean(patch2)

    # Normalize by std
    std1 = np.std(p1)
    std2 = np.std(p2)

    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0

    p1 = p1 / std1
    p2 = p2 / std2

    # Compute correlation
    return np.mean(p1 * p2)


def resolve_using_clean_cycle_ncc(
    bead_df,
    cycle_images,
    cycle_metadata,
    max_size,
    num_cycles,
    num_layers,
    patch_size=9,
):
    """
    Resolve ambiguous cycles using NCC with patches from clean cycles.

    Strategy: For beads where one cycle is clean (count==1), extract a patch from
    the clean cycle's assigned layer at the bead's (x,y) position. Use NCC to match
    this patch against all layers in ambiguous cycles (count!=1) and assign the
    layer with highest NCC score.

    Args:
        bead_df: DataFrame with cy{i}_{j} columns and cy{i}_count columns
        cycle_images: List of cycle image arrays
        cycle_metadata: List of MetaData objects
        max_size: Maximum size for cropping
        num_cycles: Number of cycles
        num_layers: Number of fluorescence layers per cycle
        patch_size: Size of patch to extract (default 9x9)

    Returns:
        DataFrame with cy{i}_ncc_resolved columns (assigned layer or 255 if unresolved)
    """
    bead_df = bead_df.copy()

    # Compute count columns if they don't exist
    for i in range(num_cycles):
        if f"cy{i}_count" not in bead_df.columns:
            layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
            bead_df[f"cy{i}_count"] = (bead_df[layer_cols] > 0).sum(axis=1)

    # Initialize NCC resolved columns
    for i in range(num_cycles):
        bead_df[f"cy{i}_ncc_resolved"] = 255

    # Count how many cycles are clean (count==1) for each bead
    count_cols = [f"cy{i}_count" for i in range(num_cycles)]
    clean_cycle_counts = (bead_df[count_cols] == 1).sum(axis=1)

    # Only process beads with at least 1 clean cycle
    eligible_mask = clean_cycle_counts >= 1

    if eligible_mask.sum() == 0:
        print("No beads with at least 1 clean cycle found")
        return bead_df

    print(
        f"Processing {eligible_mask.sum()} beads with 1+ clean cycle and 1+ ambiguous cycles"
    )

    stats = {"resolved_beads": 0, "resolved_cycles": 0, "skipped_boundary": 0}
    half_patch = patch_size // 2

    # Process each eligible bead
    for bead_idx in tqdm(bead_df[eligible_mask].index, desc="NCC resolution"):
        bead = bead_df.loc[bead_idx]
        x, y = int(bead["x"]), int(bead["y"])

        # Check if patch is within bounds
        if (
            x < half_patch
            or y < half_patch
            or x >= max_size - half_patch
            or y >= max_size - half_patch
        ):
            stats["skipped_boundary"] += 1
            continue

        # Find the first clean cycle to use as reference
        clean_cycle_idx = None
        for i in range(num_cycles):
            if bead[f"cy{i}_count"] == 1:
                clean_cycle_idx = i
                break

        if clean_cycle_idx is None:
            continue

        # Get the assigned layer in the clean cycle
        assigned_layer_idx = None
        for j in range(num_layers):
            if bead[f"cy{clean_cycle_idx}_{j}"] > 0:
                assigned_layer_idx = j
                break

        if assigned_layer_idx is None:
            continue

        # Extract patch from clean cycle
        clean_img = cycle_images[clean_cycle_idx][:, :max_size, :max_size]
        flor_layer_idx = cycle_metadata[clean_cycle_idx].flors_layers[
            assigned_layer_idx
        ]
        clean_layer_img = clean_img[flor_layer_idx].astype(np.float32)

        # Normalize the image
        clean_layer_img = (clean_layer_img - clean_layer_img.min()) / (
            clean_layer_img.max() - clean_layer_img.min() + 1e-8
        )

        # Extract 9x9 patch centered at (x, y)
        reference_patch = clean_layer_img[
            y - half_patch : y + half_patch + 1, x - half_patch : x + half_patch + 1
        ]

        if reference_patch.shape != (patch_size, patch_size):
            continue

        # Copy clean cycle assignment
        bead_df.loc[bead_idx, f"cy{clean_cycle_idx}_ncc_resolved"] = assigned_layer_idx

        # For each ambiguous cycle, compute NCC with all layers
        resolved_count = 0
        for i in range(num_cycles):
            # Skip clean cycle and non-ambiguous cycles
            if i == clean_cycle_idx or bead[f"cy{i}_count"] <= 1:
                continue

            # This cycle is ambiguous - compute NCC for each layer
            cycle_img = cycle_images[i][:, :max_size, :max_size]
            ncc_scores = []

            for j in range(num_layers):
                flor_idx = cycle_metadata[i].flors_layers[j]
                layer_img = cycle_img[flor_idx].astype(np.float32)
                layer_img = (layer_img - layer_img.min()) / (
                    layer_img.max() - layer_img.min() + 1e-8
                )

                # Extract patch at same (x, y)
                layer_patch = layer_img[
                    y - half_patch : y + half_patch + 1,
                    x - half_patch : x + half_patch + 1,
                ]

                if layer_patch.shape != (patch_size, patch_size):
                    ncc_scores.append(-np.inf)
                else:
                    # Compute NCC
                    ncc = compute_ncc(reference_patch, layer_patch)
                    ncc_scores.append(ncc)

            # Assign the layer with highest NCC
            best_layer = np.argmax(ncc_scores)
            bead_df.loc[bead_idx, f"cy{i}_ncc_resolved"] = best_layer
            resolved_count += 1

        if resolved_count > 0:
            stats["resolved_beads"] += 1
            stats["resolved_cycles"] += resolved_count

    # Convert to uint8
    for i in range(num_cycles):
        bead_df[f"cy{i}_ncc_resolved"] = bead_df[f"cy{i}_ncc_resolved"].astype(
            np.uint8
        )

    print("\nNCC Resolution Summary:")
    print(f"  Beads resolved: {stats['resolved_beads']} / {eligible_mask.sum()}")
    print(f"  Cycles resolved: {stats['resolved_cycles']}")
    print(f"  Skipped (boundary): {stats['skipped_boundary']}")

    return bead_df


def print_resolution_stats(bead_df, num_cycles, num_layers):
    """
    Print detailed statistics about ambiguity resolution.

    Args:
        bead_df: Original DataFrame with label assignments and intensities
        num_cycles: Number of cycles
        num_layers: Number of fluorescence layers per cycle
    """
    print("\n" + "=" * 60)
    print("DETAILED RESOLUTION STATISTICS")
    print("=" * 60)

    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        intensity_cols = [f"cy{i}_{j}_intensity" for j in range(num_layers)]

        counts = (bead_df[layer_cols] > 0).sum(axis=1)

        print(f"\nCycle {i}:")
        print("  Activation counts:")
        for count, freq in counts.value_counts().sort_index().items():
            pct = 100 * freq / len(bead_df)
            print(f"    {int(count)} layers: {freq} beads ({pct:.1f}%)")

        # Analyze ambiguous beads (2+ activations)
        ambiguous_mask = counts > 1
        if ambiguous_mask.sum() > 0:
            print(f"\n  Ambiguous bead details ({ambiguous_mask.sum()} beads):")

            # For ambiguous beads, show intensity differences
            intensity_diffs = []
            for bead_idx in bead_df[ambiguous_mask].index:
                intensities = bead_df.loc[bead_idx, intensity_cols].values
                labels = bead_df.loc[bead_idx, layer_cols].values
                active_intensities = intensities[labels > 0]
                if len(active_intensities) > 1:
                    sorted_intensities = np.sort(active_intensities)[::-1]
                    diff = sorted_intensities[0] - sorted_intensities[1]
                    intensity_diffs.append(diff)

            if intensity_diffs:
                print("    Intensity difference (1st - 2nd highest):")
                print(f"      Mean: {np.mean(intensity_diffs):.4f}")
                print(f"      Median: {np.median(intensity_diffs):.4f}")
                print(f"      Min: {np.min(intensity_diffs):.4f}")
                print(f"      Max: {np.max(intensity_diffs):.4f}")

    print("\n" + "=" * 60)
