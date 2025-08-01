import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba import jit, njit, prange
from skimage.exposure import match_histograms
from skimage.util import view_as_windows


def timeit(tag):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[TIME] {tag:<30} took {end - start:.4f}s")
            return result

        return wrapper

    return decorator


@timeit("get_excel_optimized")
def get_excel_optimized(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    layer_threshold_dict=defaultdict(int),
    progress_callback=None,
    is_running_callback=None,
    use_numba=True,
    use_multiprocessing=False,
    n_workers=None,
):
    """
    Optimized version of get_excel with multiple performance improvements:

    1. Vectorized operations where possible
    2. Pre-computed bounds checking
    3. Numba JIT compilation for hot loops
    4. Optional multiprocessing
    5. Reduced memory allocations
    6. Early bounds filtering
    """

    def update_progress_internal(value, message):
        if progress_callback:
            overall_progress = 40 + (value / 100) * 50
            progress_callback(int(overall_progress), message)

    def is_running():
        if is_running_callback:
            return is_running_callback()
        return True

    ColorThreshold = signal_to_noise_cutoff
    beads = np.array(beads)

    # Pre-filter beads that are within valid bounds
    radius = 2
    extended_radius = 20
    valid_bead_mask = (
        (beads[:, 0] >= 2 * radius)
        & (beads[:, 0] < max_size - 2 * radius)
        & (beads[:, 1] >= 2 * radius)
        & (beads[:, 1] < max_size - 2 * radius)
    )

    # Also check extended bounds for background calculation
    extended_valid_mask = (
        (beads[:, 0] >= 2 * extended_radius)
        & (beads[:, 0] < max_size - 2 * extended_radius)
        & (beads[:, 1] >= 2 * extended_radius)
        & (beads[:, 1] < max_size - 2 * extended_radius)
    )

    export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

    results_from_cycles = []
    results_from_cycles_SNR = []
    results_from_cycles_Sig_absolute_threshold = []

    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = [img for img, _ in tifs]

    # Setup flors_layers
    for i, md in enumerate(tif_metadata):
        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j != int(md.reference_channel)
        ]
        print(
            f"Flors layers for tif {i} are {md.flors_layers}, reference channel is {md.reference_channel}"
        )

    total_beads = len(beads)
    total_cycles = len(tif_metadata)
    total_layers_per_cycle = (
        len(tif_metadata[0].flors_layers) if total_cycles > 0 else 0
    )
    total_steps = total_cycles * total_layers_per_cycle

    current_step = 0

    for tif_count, md in enumerate(tif_metadata):
        if not is_running():
            return None

        reference_for_hist_match = None

        cycle_specific_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="uint16"
        )
        cycle_specific_sig_noise_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="float"
        )
        cycle_specific_Sig_absolute_threshold_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="float"
        )

        for i, layer in enumerate(md.flors_layers):
            if not is_running():
                return None

            current_step += 1
            progress_percentage = int((current_step / total_steps) * 100)
            update_progress_internal(
                progress_percentage,
                f"Processing cycle {tif_count + 1}/{total_cycles}, layer {i + 1}/{total_layers_per_cycle}",
            )

            flor_layer = tif_images[tif_count][layer, :, :].astype(np.float32)

            if reference_for_hist_match is None:
                reference_for_hist_match = flor_layer
            else:
                flor_layer = match_histograms(flor_layer, reference_for_hist_match)

            # Use optimized processing function
            if use_numba:
                brightness_values, snr_values, threshold_values = process_beads_numba(
                    flor_layer,
                    beads,
                    valid_bead_mask,
                    extended_valid_mask,
                    radius,
                    extended_radius,
                    layer_threshold_dict.get(layer, 0),
                )
            else:
                brightness_values, snr_values, threshold_values = (
                    process_beads_vectorized(
                        flor_layer,
                        beads,
                        valid_bead_mask,
                        extended_valid_mask,
                        radius,
                        extended_radius,
                        layer_threshold_dict.get(layer, 0),
                    )
                )

            cycle_specific_data[:, i] = brightness_values
            cycle_specific_sig_noise_data[:, i] = snr_values
            cycle_specific_Sig_absolute_threshold_data[:, i] = threshold_values

        results_from_cycles.append(cycle_specific_data)
        results_from_cycles_SNR.append(cycle_specific_sig_noise_data)
        results_from_cycles_Sig_absolute_threshold.append(
            cycle_specific_Sig_absolute_threshold_data
        )

        # Vectorized operations for final processing
        brightest_layers = np.argmax(cycle_specific_data, axis=1)
        export_to_excel[:, tif_count] = brightest_layers

        # Vectorized SNR threshold check
        max_snr_per_bead = np.max(cycle_specific_sig_noise_data, axis=1)
        below_threshold_mask = max_snr_per_bead < ColorThreshold
        export_to_excel[below_threshold_mask, tif_count] = 255

        # Vectorized absolute threshold check
        all_zero_mask = np.all(cycle_specific_Sig_absolute_threshold_data == 0, axis=1)
        export_to_excel[all_zero_mask, tif_count] = 255

    export_to_excel = np.hstack((beads, export_to_excel))
    return export_to_excel


@timeit("get_excel_numba")
@jit(nopython=True, parallel=True)
def process_beads_numba(
    flor_layer,
    beads,
    valid_bead_mask,
    extended_valid_mask,
    radius,
    extended_radius,
    layer_threshold,
):
    """Numba-optimized bead processing function"""
    n_beads = len(beads)
    brightness_values = np.zeros(n_beads, dtype=np.uint16)
    snr_values = np.zeros(n_beads, dtype=np.float32)
    threshold_values = np.zeros(n_beads, dtype=np.float32)

    for b_i in prange(n_beads):
        if valid_bead_mask[b_i]:
            x, y = beads[b_i, 0], beads[b_i, 1]

            # Extract ROI for brightness calculation
            roi = flor_layer[y - radius : y + radius, x - radius : x + radius]
            brightness = np.median(roi.flatten())
            brightness_values[b_i] = int(brightness)

            # Calculate SNR if extended bounds are valid
            if extended_valid_mask[b_i]:
                local_roi = flor_layer[y - radius : y + radius, x - radius : x + radius]
                background_intensity = np.percentile(local_roi.flatten(), 10)

                if background_intensity > 0:
                    snr = (brightness - background_intensity) / background_intensity
                    snr_values[b_i] = snr

            # Threshold check
            threshold_values[b_i] = 1.0 if brightness > layer_threshold else 0.0

    return brightness_values, snr_values, threshold_values


def process_beads_vectorized(
    flor_layer,
    beads,
    valid_bead_mask,
    extended_valid_mask,
    radius,
    extended_radius,
    layer_threshold,
):
    """Vectorized bead processing function (fallback when Numba not available)"""
    n_beads = len(beads)
    brightness_values = np.zeros(n_beads, dtype=np.uint16)
    snr_values = np.zeros(n_beads, dtype=np.float32)
    threshold_values = np.zeros(n_beads, dtype=np.float32)

    valid_beads = beads[valid_bead_mask]
    valid_indices = np.where(valid_bead_mask)[0]

    for i, (b_i, bead) in enumerate(zip(valid_indices, valid_beads)):
        x, y = bead[0], bead[1]

        # Extract ROI
        roi = flor_layer[y - radius : y + radius, x - radius : x + radius]
        brightness = np.median(roi)
        brightness_values[b_i] = brightness

        # SNR calculation
        if extended_valid_mask[b_i]:
            local_roi = flor_layer[y - radius : y + radius, x - radius : x + radius]
            background_intensity = np.percentile(local_roi, 10)

            if background_intensity > 0:
                snr = (brightness - background_intensity) / background_intensity
                snr_values[b_i] = snr

        # Threshold check
        threshold_values[b_i] = 1.0 if brightness > layer_threshold else 0.0

    return brightness_values, snr_values, threshold_values


# def get_excel_original(
#     beads,
#     signal_to_noise_cutoff,
#     tifs,
#     max_size,
#     layer_threshold_dict=defaultdict(int),
#     progress_callback=None,
#     is_running_callback=None,
# ):
#     """Original function for performance comparison"""

#     def update_progress_internal(value, message):
#         if progress_callback:
#             overall_progress = 40 + (value / 100) * 50
#             progress_callback(int(overall_progress), message)

#     def is_running():
#         if is_running_callback:
#             return is_running_callback()
#         return True

#     ColorThreshold = signal_to_noise_cutoff
#     export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

#     results_from_cycles = []
#     results_from_cycles_SNR = []
#     results_from_cycles_Sig_absolute_threshold = []
#     layer_threshold_bool = True
#     tif_metadata = [f.metadata for _, f in tifs]
#     tif_images = [img for img, _ in tifs]

#     for i, md in enumerate(tif_metadata):
#         md.flors_layers = [
#             j for j in range(len(tif_images[i])) if j != int(md.reference_channel)
#         ]
#         print(
#             f"Flors layers for tif {i} are {md.flors_layers}, reference channel is {md.reference_channel}"
#         )

#     total_beads = len(beads)
#     total_cycles = len(tif_metadata)
#     total_layers_per_cycle = (
#         len(tif_metadata[0].flors_layers) if total_cycles > 0 else 0
#     )
#     total_steps = total_cycles * total_layers_per_cycle * total_beads

#     current_step = 0
#     for tif_count, md in enumerate(tif_metadata):
#         if not is_running():
#             return None
#         reference_for_hist_match = None

#         cycle_specific_data = np.zeros(
#             (len(beads), len(md.flors_layers)), dtype="uint16"
#         )
#         cycle_specific_sig_noise_data = np.zeros(
#             (len(beads), len(md.flors_layers)), dtype="float"
#         )
#         cycle_specific_Sig_absolute_threshold_data = np.zeros(
#             (len(beads), len(md.flors_layers)), dtype="float"
#         )

#         for i, layer in enumerate(md.flors_layers):
#             if not is_running():
#                 return None
#             flor_layer = tif_images[tif_count][layer, :, :]

#             if reference_for_hist_match is None:
#                 reference_for_hist_match = flor_layer
#             else:
#                 flor_layer = match_histograms(
#                     flor_layer, reference_for_hist_match, channel_axis=-1
#                 )

#             radius = 2
#             layer_specific_data = np.zeros(len(beads), dtype="uint16")
#             sig_noise_data = np.zeros(len(beads), dtype="float")
#             layer_threshold = np.zeros(len(beads), dtype="float")

#             for b_i, bead in enumerate(beads):
#                 if not is_running():
#                     return None
#                 current_step += 1
#                 progress_percentage = int((current_step / total_steps) * 100)
#                 update_progress_internal(
#                     progress_percentage,
#                     f"Processing cycle {tif_count + 1}/{total_cycles}, layer {i + 1}/{total_layers_per_cycle}, bead {b_i + 1}/{total_beads}",
#                 )

#                 if (bead[0] - 2 * radius) > 0 and (bead[0] + 2 * radius) < max_size:
#                     if (bead[1] - 2 * radius) > 0 and (bead[1] + 2 * radius) < max_size:
#                         x, y = bead
#                         roi = flor_layer[
#                             bead[1] - radius : bead[1] + radius,
#                             bead[0] - radius : bead[0] + radius,
#                         ]
#                         brightness = np.median(roi)

#                         if (bead[0] - 2 * 20) > 0 and (bead[0] + 2 * 20) < max_size:
#                             if (bead[1] - 2 * 20) > 0 and (bead[1] + 2 * 20) < max_size:
#                                 local_flor_layer = flor_layer[
#                                     bead[1] - radius : bead[1] + radius,
#                                     bead[0] - radius : bead[0] + radius,
#                                 ]
#                                 flor_layer_background_intensity_local = np.percentile(
#                                     local_flor_layer, 10
#                                 )
#                                 if flor_layer_background_intensity_local > 0:
#                                     signal_noise_ratio = (
#                                         brightness
#                                         - flor_layer_background_intensity_local
#                                     ) / flor_layer_background_intensity_local
#                                     layer_threshold_bool = (
#                                         brightness > layer_threshold_dict[layer]
#                                     )
#                                 else:
#                                     signal_noise_ratio = 0
#                             else:
#                                 signal_noise_ratio = 0
#                         else:
#                             signal_noise_ratio = 0

#                         layer_specific_data[b_i] = brightness
#                         sig_noise_data[b_i] = signal_noise_ratio
#                         layer_threshold[b_i] = layer_threshold_bool

#             cycle_specific_data[:, i] = layer_specific_data
#             cycle_specific_sig_noise_data[:, i] = sig_noise_data
#             cycle_specific_Sig_absolute_threshold_data[:, i] = layer_threshold

#         results_from_cycles.append(cycle_specific_data)
#         results_from_cycles_SNR.append(cycle_specific_sig_noise_data)
#         results_from_cycles_Sig_absolute_threshold.append(
#             cycle_specific_Sig_absolute_threshold_data
#         )

#         brightest_layers = np.argmax(cycle_specific_data, axis=1)
#         export_to_excel[:, tif_count] = brightest_layers

#         for i, row in enumerate(cycle_specific_sig_noise_data):
#             if np.max(row) < ColorThreshold:
#                 export_to_excel[i, tif_count] = 255

#         for i, row in enumerate(cycle_specific_Sig_absolute_threshold_data):
#             if np.all(row) == 0:
#                 export_to_excel[i, tif_count] = 255

#     export_to_excel = np.hstack((beads, export_to_excel))
#     return export_to_excel


def edge_bead_filtering(radius, max_size):
    def filter_func(bead):
        x, y = bead
        # Check if the bead is at least 'radius' pixels away from the edges
        if (x - 2 * radius) > 0 and (x + 2 * radius) < max_size:
            if (y - 2 * radius) > 0 and (y + 2 * radius) < max_size:
                return True
        return False

    return filter_func


@timeit("get_excel_original")
def get_excel_original(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    layer_threshold_dict=defaultdict(int),
    progress_callback=None,
    is_running_callback=None,
):
    def update_progress_internal(value, message):
        if progress_callback:
            # Scale the internal progress (0-100) to a sub-range of the overall progress (40-90)
            overall_progress = 40 + (value / 100) * 50
            progress_callback(int(overall_progress), message)

    def is_running():
        if is_running_callback:
            return is_running_callback()
        return True

    ColorThreshold = signal_to_noise_cutoff
    export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

    results_from_cycles = []
    results_from_cycles_SNR = []
    results_from_cycles_Sig_absolute_threshold = []
    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = [img for img, _ in tifs]
    # flor layer is all channels except brightfield
    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)
        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j != int(md.reference_channel)
        ]
        print(
            f"Flors layers for tif {i} are {md.flors_layers}, reference channel is {md.reference_channel}"
        )
    # GETTING ALL THE BEAD BRIGHTNESSES
    total_beads = len(beads)
    total_cycles = len(tif_metadata)
    total_layers_per_cycle = (
        len(tif_metadata[0].flors_layers) if total_cycles > 0 else 0
    )
    total_steps = total_cycles * total_layers_per_cycle * total_beads
    radius = 2
    beads = list(filter(edge_bead_filtering(20, max_size), beads))

    current_step = 0
    for tif_count, md in enumerate(tif_metadata):
        if not is_running():
            return None
        reference_for_hist_match = None

        cycle_specific_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="uint16"
        )
        cycle_specific_sig_noise_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="float"
        )
        cycle_specific_Sig_absolute_threshold_data = np.zeros(
            (len(beads), len(md.flors_layers)), dtype="float"
        )

        for i, layer in enumerate(md.flors_layers):
            if not is_running():
                return None
            flor_layer = tif_images[tif_count][layer, :, :]

            if reference_for_hist_match is None:
                reference_for_hist_match = flor_layer
            else:
                flor_layer = match_histograms(flor_layer, reference_for_hist_match)

            layer_specific_data = np.zeros(len(beads), dtype="uint16")
            sig_noise_data = np.zeros(len(beads), dtype="float")
            layer_threshold = np.zeros(len(beads), dtype="float")
            last_progress = -1
            # filter out edge beads:
            max_size = md.max_size
            bead_rois = [
                flor_layer[y - radius : y + radius, x - radius : x + radius]
                for x, y in beads
            ]

            percentile_map = np.percentile(bead_rois, 10, axis=(1, 2))
            for b_i, bead in enumerate(beads):
                if not is_running():
                    return None
                current_step += 1
                progress_percentage = int((current_step / total_steps) * 100)
                if progress_percentage != last_progress:
                    update_progress_internal(
                        progress_percentage,
                        f"Processing cycle {tif_count + 1}/{total_cycles}, layer {i + 1}/{total_layers_per_cycle}, bead {b_i + 1}/{total_beads}",
                    )
                    last_progress = progress_percentage
                x, y = bead

                roi = bead_rois[b_i]
                brightness = np.median(roi)
                flor_layer_background_intensity_local = percentile_map[b_i]
                if flor_layer_background_intensity_local > 0:  # Corrected if statement
                    signal_noise_ratio = (
                        brightness - flor_layer_background_intensity_local
                    ) / flor_layer_background_intensity_local
                else:
                    signal_noise_ratio = 0

                layer_specific_data[b_i] = brightness
                sig_noise_data[b_i] = signal_noise_ratio
                layer_threshold[b_i] = (
                    brightness > layer_threshold_dict[layer]
                )  # boolean values

            cycle_specific_data[:, i] = layer_specific_data
            cycle_specific_sig_noise_data[:, i] = sig_noise_data
            cycle_specific_Sig_absolute_threshold_data[:, i] = layer_threshold

        results_from_cycles.append(cycle_specific_data)
        results_from_cycles_SNR.append(cycle_specific_sig_noise_data)
        results_from_cycles_Sig_absolute_threshold.append(
            cycle_specific_Sig_absolute_threshold_data
        )

        brightest_layers = np.argmax(cycle_specific_data, axis=1)
        export_to_excel[:, tif_count] = brightest_layers

        for i, row in enumerate(cycle_specific_sig_noise_data):
            if np.max(row) < ColorThreshold:
                export_to_excel[i, tif_count] = 255

        for i, row in enumerate(cycle_specific_Sig_absolute_threshold_data):
            if np.all(row) == 0:
                export_to_excel[i, tif_count] = 255

    export_to_excel = np.hstack((beads, export_to_excel))
    return export_to_excel


# Mock MetaData class for testing
@dataclass
class MetaData:
    reference_channel: int
    max_size: int = 10000  # default max size
    flors_layers: List[int] = None


# Mock TIF data structure
class MockTifData:
    def __init__(self, metadata: MetaData):
        self.metadata = metadata


def create_mock_data(n_beads=100, n_cycles=3, n_layers=4, image_size=512):
    """Create mock data for testing both functions"""

    # Generate random bead coordinates
    beads = []
    for _ in range(n_beads):
        x = np.random.randint(50, image_size - 50)
        y = np.random.randint(50, image_size - 50)
        beads.append([x, y])

    # Create mock TIF images and metadata
    tifs = []
    for cycle in range(n_cycles):
        # Create random image data with multiple layers
        img_data = np.random.randint(
            0, 4096, size=(n_layers, image_size, image_size), dtype=np.uint16
        )

        # Add some realistic signal patterns around bead locations
        for bead in beads:
            x, y = bead
            # Add bright spots around beads in random layers
            bright_layer = np.random.randint(0, n_layers)
            if bright_layer != 0:  # Avoid reference channel
                img_data[
                    bright_layer, y - 3 : y + 3, x - 3 : x + 3
                ] += np.random.randint(1000, 3000)

        metadata = MetaData(reference_channel=0)  # Brightfield is channel 0
        tif_data = MockTifData(metadata)

        tifs.append((img_data, tif_data))

    # Create layer threshold dictionary
    layer_threshold_dict = defaultdict(int)
    for i in range(1, n_layers):  # Skip reference channel
        layer_threshold_dict[i] = np.random.randint(500, 1500)

    signal_to_noise_cutoff = 0.5
    max_size = image_size

    return beads, signal_to_noise_cutoff, tifs, max_size, layer_threshold_dict


def test_correctness():
    """Test that optimized version produces same results as original"""
    print("Testing correctness...")

    # Create deterministic test data
    np.random.seed(42)
    beads, snr_cutoff, tifs, max_size, layer_threshold_dict = create_mock_data(
        n_beads=20, n_cycles=2, n_layers=3, image_size=128
    )

    # Run original function
    result_original = get_excel_original(
        beads, snr_cutoff, tifs, max_size, layer_threshold_dict
    )

    # Run optimized function
    result_optimized = get_excel_optimized(
        beads, snr_cutoff, tifs, max_size, layer_threshold_dict, use_numba=False
    )

    # Compare results
    print(f"Original result shape: {result_original.shape}")
    print(f"Optimized result shape: {result_optimized.shape}")

    # Check if results are close (allowing for small numerical differences)
    coordinates_match = np.allclose(result_original[:, :2], result_optimized[:, :2])
    results_match = np.array_equal(result_original[:, 2:], result_optimized[:, 2:])

    print(f"Coordinates match: {coordinates_match}")
    print(f"Results match: {results_match}")

    if not coordinates_match or not results_match:
        print("Differences found:")
        print("Original coords:", result_original[:5, :2])
        print("Optimized coords:", result_optimized[:5, :2])
        print("Original results:", result_original[:5, 2:])
        print("Optimized results:", result_optimized[:5, 2:])

    return coordinates_match and results_match


def benchmark_functions():
    """Benchmark performance of different versions"""
    print("\nRunning performance benchmarks...")

    test_configs = [
        {"n_beads": 2_000_0, "n_cycles": 2, "n_layers": 4, "image_size": 10_000},
    ]

    results = []

    for config in test_configs:
        print(f"\nTesting config: {config}")

        # Create test data
        np.random.seed(42)  # For reproducible results
        beads, snr_cutoff, tifs, max_size, layer_threshold_dict = create_mock_data(
            **config
        )

        times = {}

        # Test original function
        print("Running original function...")
        start_time = time.time()
        result_original = get_excel_original(
            beads, snr_cutoff, tifs, max_size, layer_threshold_dict
        )
        times["original"] = time.time() - start_time

    return results


def memory_usage_test():
    """Test memory usage patterns"""
    print("\nTesting memory usage...")

    try:
        import os

        import psutil

        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB

        np.random.seed(42)
        beads, snr_cutoff, tifs, max_size, layer_threshold_dict = create_mock_data(
            n_beads=200, n_cycles=3, n_layers=4, image_size=512
        )

        # Test original function memory usage
        initial_memory = get_memory_usage()
        result_original = get_excel_original(
            beads, snr_cutoff, tifs, max_size, layer_threshold_dict
        )
        peak_memory_original = get_memory_usage()
        memory_used_original = peak_memory_original - initial_memory

        # Test optimized function memory usage
        initial_memory = get_memory_usage()
        result_optimized = get_excel_optimized(
            beads, snr_cutoff, tifs, max_size, layer_threshold_dict
        )
        peak_memory_optimized = get_memory_usage()
        memory_used_optimized = peak_memory_optimized - initial_memory

        print(f"Original function memory usage: {memory_used_original:.2f} MB")
        print(f"Optimized function memory usage: {memory_used_optimized:.2f} MB")

    except ImportError:
        print("psutil not available, skipping memory usage test")


def plot_performance_results(results):
    """Plot performance comparison results"""
    try:
        import matplotlib.pyplot as plt

        configs = [
            f"{r['config']['n_beads']} beads\n{r['config']['n_cycles']} cycles"
            for r in results
        ]
        speedups_vec = [r["speedup_vectorized"] for r in results]
        speedups_numba = [
            r["speedup_numba"] if r["speedup_numba"] else 0 for r in results
        ]

        x = np.arange(len(configs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(
            x - width / 2, speedups_vec, width, label="Vectorized", alpha=0.8
        )
        bars2 = ax.bar(x + width / 2, speedups_numba, width, label="Numba", alpha=0.8)

        ax.set_xlabel("Test Configuration")
        ax.set_ylabel("Speedup (x times faster)")
        ax.set_title("Performance Improvements Over Original Function")
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()
        plt.savefig("performance_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

    except ImportError:
        print("matplotlib not available, skipping plot generation")


def run_all_tests():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("COMPREHENSIVE PERFORMANCE TEST SUITE")
    print("=" * 60)

    # Test correctness
    # correctness_passed = test_correctness()
    # print(f"\nCorrectness test: {'PASSED' if correctness_passed else 'FAILED'}")

    # if not correctness_passed:
    #     print("WARNING: Correctness test failed. Check the optimized implementation.")
    #     return False

    # Benchmark performance
    benchmark_results = benchmark_functions()

    # Test memory usage
    # memory_usage_test()

    # Plot results
    plot_performance_results(benchmark_results)

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    for i, result in enumerate(benchmark_results):
        config = result["config"]
        print(
            f"\nTest {i+1}: {config['n_beads']} beads, {config['n_cycles']} cycles, {config['image_size']}px"
        )
        print(f"  Vectorized speedup: {result['speedup_vectorized']:.2f}x")
        if result["speedup_numba"]:
            print(f"  Numba speedup: {result['speedup_numba']:.2f}x")

    avg_speedup_vec = np.mean([r["speedup_vectorized"] for r in benchmark_results])
    avg_speedup_numba = np.mean(
        [r["speedup_numba"] for r in benchmark_results if r["speedup_numba"]]
    )

    print(f"\nAverage speedup (vectorized): {avg_speedup_vec:.2f}x")
    if avg_speedup_numba:
        print(f"Average speedup (numba): {avg_speedup_numba:.2f}x")

    return True


import cProfile
import pstats

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    success = run_all_tests()

    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("tottime")
    stats.print_stats(30)  # show top 30 time-consuming functions
