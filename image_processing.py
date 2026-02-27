import json
import numbers
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import cv2

# import diplib as dip
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import ndimage as ndi
from skimage import filters, img_as_float, img_as_uint, measure
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from sklearn.cluster import KMeans
from tqdm import tqdm

from benchmark_tracer import _median_per_region, voronoi_from_centers_tiled
from ensemble import compute_bead_profile_metrics
from model.file_item import MetaData


def log(*msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(map(str, msg))}")


class TileMap:
    def __init__(self, name: str, image: np.ndarray, overlap: int, height_width: int):
        """
        :param name:
        :param image:
        :param overlap: pixel amount of overlap
        :param height_width:
        """

        self.name = name
        self.image = image

        self.height_width = height_width

        self.tile_center_points = self.blockify(height_width) * self.image.shape[0]

        self.tile_size = self.tile_center_points[0][0][0]

        self.overlap = overlap

    def __len__(self):
        return self.height_width * self.height_width

    @staticmethod
    def find_mask(moving_array):
        def blur(img):
            img = img.copy()
            kernel = np.ones((5, 5), np.float64) / 225
            dst = cv2.filter2D(img, -1, kernel)
            return dst

        def threshold(im, percentile):
            p = np.percentile(im, percentile)
            im = im.copy()
            im[im < p] = 0
            im[im >= p] = 255
            return im

        small = cv2.resize(
            moving_array,
            (np.array(moving_array.shape) / 10).astype(int),
            interpolation=cv2.INTER_LINEAR,
        )

        im = np.invert(threshold(blur(small), 20))

        out = dip.AreaOpening(im, filterSize=150, connectivity=2)
        out = np.array(out)

        big = cv2.resize(
            out,
            (np.array(moving_array.shape)).astype(int),
            interpolation=cv2.INTER_LINEAR,
        )
        big[moving_array == 0] = 255

        return np.invert((big / 255).astype(bool))

    def get_tile_by_center(self, image, x, y):
        tile_size = round(self.tile_size) + self.overlap

        ymin = int(round(y - tile_size))
        ymax = int(round(y + tile_size))
        xmin = int(round(x - tile_size))
        xmax = int(round(x + tile_size))

        img_h, img_w = image.shape[:2]

        crop_ymin = max(0, ymin)
        crop_ymax = min(img_h, ymax)
        crop_xmin = max(0, xmin)
        crop_xmax = min(img_w, xmax)

        tile = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        pad_top = max(0, 0 - ymin)
        pad_bottom = max(0, ymax - img_h)
        pad_left = max(0, 0 - xmin)
        pad_right = max(0, xmax - img_w)

        tile_padded = cv2.copyMakeBorder(
            tile,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

        return tile_padded

    def get_bounds_of_tile(self, x, y):
        # log("Got ", x, y)
        tile_size = round(self.tile_size) + self.overlap
        ymin = (
            self.overlap
            if self.keep_in_bounds(int(y - tile_size)) == int(y - tile_size)
            else 0
        )
        ymax = (
            self.overlap
            if self.keep_in_bounds(int(y + tile_size)) == int(y + tile_size)
            else 0
        )
        xmin = (
            self.overlap
            if self.keep_in_bounds(int(x - tile_size)) == int(x - tile_size)
            else 0
        )
        xmax = (
            self.overlap
            if self.keep_in_bounds(int(x + tile_size)) == int(x + tile_size)
            else 0
        )

        return {
            "center": (x, y),
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
        }

    def __iter__(self):
        for i in self.tile_center_points:
            for j in i:
                # log("THIS IS THE TILE WE TALKIGN ABOUT", j)
                tile = self.get_tile_by_center(self.image, j[0], j[1])
                bounds = self.get_bounds_of_tile(j[0], j[1])

                yield (tile, bounds)

    def keep_in_bounds(self, num):
        if num < 0:
            return 0
        if num > self.image.shape[0]:
            return self.image.shape[0]

        return int(num)

    @staticmethod
    def blockify(cuts):
        centerpoints = []
        for i in range(cuts):
            row = []
            for j in range(cuts):
                # log((i + 1), cuts, (j + 1), cuts)
                row.append(
                    np.array([(2 * i + 1) / (cuts * 2), (2 * j + 1) / (cuts * 2)])
                )
                # log((2*i + 1) / (cuts *2))

            centerpoints.append(np.array(row))

        return np.array(centerpoints)

    def get_neighbor_pairs(self):
        """
        Return a list of (tile_idx_a, tile_idx_b) pairs for horizontally and vertically adjacent tiles.
        Indices correspond to enumeration order of __iter__.
        """
        pairs = []
        rows, cols = self.height_width, self.height_width

        def idx(r, c):
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                # Right neighbor
                if c + 1 < cols:
                    pairs.append((idx(r, c), idx(r, c + 1)))
                # Down neighbor
                if r + 1 < rows:
                    pairs.append((idx(r, c), idx(r + 1, c)))

        return pairs


def beadfinding(
    brightfield: np.ndarray,
    num_tiles: int = 10,
    px_overlap: int = 100,
    workers: int = 10,
    is_running_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds beads in a brightfield image using a tiled, parallel approach.

    This optimized version:
    1.  Eliminates unnecessary disk I/O for performance.
    2.  Vectorizes centroid calculation using SciPy for a massive speedup.
    3.  Correctly applies morphological operations on a boolean mask.
    4.  Simplifies the tile-stitching logic.
    """
    # --- 1. Avoid Unnecessary File I/O ---
    # The brightfield image is already in memory, so saving and reloading it
    # from a temporary file is unnecessary overhead.

    # tileset = TileMap("tm", brightfield, px_overlap, num_tiles)
    # stitched_mask = np.zeros_like(brightfield, dtype=bool)
    # img_h, img_w = brightfield.shape

    # def process_tile(tile_and_bounds):
    #     if is_running_callback and not is_running_callback():
    #         return None

    #     tile, bounds = tile_and_bounds
    #     mask = quadtree_threshold(tile)

    #     # Simplified logic to calculate the destination slice in the full image
    #     center_x, center_y = bounds["center"]
    #     tile_radius = tileset.tile_size + px_overlap

    #     ymin_global = int(round(center_y - tile_radius))
    #     ymax_global = int(round(center_y + tile_radius))
    #     xmin_global = int(round(center_x - tile_radius))
    #     xmax_global = int(round(center_x + tile_radius))

    #     # Define the destination slice in the main stitched_mask
    #     dest_slice = (
    #         slice(max(0, ymin_global), min(img_h, ymax_global)),
    #         slice(max(0, xmin_global), min(img_w, xmax_global)),
    #     )

    #     # Define the source slice from the generated mask to account for edges
    #     src_slice = (
    #         slice(max(0, -ymin_global), mask.shape[0] - max(0, ymax_global - img_h)),
    #         slice(max(0, -xmin_global), mask.shape[1] - max(0, xmax_global - img_w)),
    #     )

    #     return mask[src_slice], dest_slice

    # with ThreadPoolExecutor(max_workers=workers) as executor:
    #     futures = [executor.submit(process_tile, tb) for tb in tileset]
    #     progress_bar = tqdm(
    #         as_completed(futures), total=len(futures), desc="Processing tiles"
    #     )

    #     for future in progress_bar:
    #         if is_running_callback and not is_running_callback():
    #             executor.shutdown(wait=False, cancel_futures=True)
    #             return None, None  # Indicate cancellation

    #         result = future.result()
    #         if result:
    #             mask_chunk, dest_slice = result
    #             stitched_mask[dest_slice] |= mask_chunk

    # # --- 2. Correct Morphological Operations ---
    # # Morphological operations should run on boolean masks, not intensity images.
    # closed_mask = closing(stitched_mask, square(1))
    # cleared_mask = clear_border(closed_mask)
    # label_image, num_labels = label(cleared_mask, return_num=True)
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    label_image = stardist_segmentation(
        brightfield,
        model,
        1,
        99,
        1500,
        scale=2,
    )
    num_labels = label_image.max()
    if num_labels == 0:
        print("No beads found after filtering.")
        return np.array([]), np.zeros_like(brightfield, dtype=int)

    print(f"Found {num_labels} potential beads after labeling.")

    # --- 3. Vectorized Centroid Calculation ---
    # Create an intensity image where non-bead areas are zero
    # intensity_image = np.where(cleared_mask, brightfield, 0)

    # Calculate all weighted centroids in a single, highly optimized operation.
    # This is orders of magnitude faster than looping through regionprops.
    indices = np.arange(1, num_labels + 1)
    centers_yx = ndimage.center_of_mass(brightfield, label_image, indices)

    # Convert list of (y, x) tuples to a NumPy array of [x, y] coordinates
    centers = np.array([(x, y) for y, x in centers_yx if not np.isnan(y)])
    centers = np.rint(centers).astype(np.uint16)

    print(f"Final bead count: {len(centers)}")

    # Note: The 'coords' output was removed. Collecting pixel coordinates for each
    # bead requires a loop and is the primary performance bottleneck. This version
    # prioritizes speed by returning only the centroids and the label image.
    return centers

    # old method missed around 5% of beads, even after second pass
    # new method gets more beads and is like 2 passes in one
    # since second pass doesn't rarely adds new beads
    # (2688242-2548509)/2688242 = 0.0519793233

    # in addition, new method centroids are more centered than before
    # think before they were skewed top left due to always rounding down


def merge_global_duplicates(beads, rois, bead_radius=5):
    """
    beads: list of (x, y) in global coordinates
    rois: list of (x0, y0, x1, y1) in global coordinates
    """
    keep_beads = []
    keep_rois = []

    used = np.zeros(len(beads), dtype=bool)
    for i in range(len(beads)):
        if used[i]:
            continue
        group_idxs = [i]
        for j in range(i + 1, len(beads)):
            if used[j]:
                continue
            # Compare centroid distance instead of neighbors only
            dist = np.linalg.norm(np.array(beads[i]) - np.array(beads[j]))
            if dist <= bead_radius:
                group_idxs.append(j)
                used[j] = True
        # Merge group (average centers, union bbox)
        merged_center = np.mean([beads[k] for k in group_idxs], axis=0)
        merged_box = np.array([rois[k] for k in group_idxs])
        x0, y0 = merged_box[:, 0].min(), merged_box[:, 1].min()
        x1, y1 = merged_box[:, 2].max(), merged_box[:, 3].max()

        keep_beads.append(tuple(merged_center))
        keep_rois.append((x0, y0, x1, y1))
    return keep_beads, keep_rois


def quadtree_threshold(img, min_size=32, max_std=10):
    H, W = img.shape
    mask = np.zeros_like(img, dtype=bool)

    def process_tile(x0, y0, width, height):
        tile = img[y0 : y0 + height, x0 : x0 + width]
        std = tile.std()

        if std <= max_std or width <= min_size or height <= min_size:
            try:
                t = threshold_otsu(tile)
            except ValueError:
                t = tile.mean()
            mask[y0 : y0 + height, x0 : x0 + width] = tile > t
        else:
            w_half = width // 2
            h_half = height // 2
            process_tile(x0, y0, w_half, h_half)
            process_tile(x0 + w_half, y0, width - w_half, h_half)
            process_tile(x0, y0 + h_half, w_half, height - h_half)
            process_tile(x0 + w_half, y0 + h_half, width - w_half, height - h_half)

    process_tile(0, 0, W, H)
    return mask


def find_beads(brightfield):
    mask_value = quadtree_threshold(brightfield)
    brighter_regions = np.where(brightfield, mask_value, 0)

    # attempt at getting beads from dark spots, doesn't seem to work well yet
    # amask = black_tophat(mask_value,footprint=[(np.ones((9, 1)), 1), (np.ones((1, 9)), 1)])
    # assert amask is not None
    # amask = isotropic_erosion(~amask, 3)
    # amask = isotropic_dilation(amask, 3)

    # masked_bf = np.where(amask, brightfield, 0)
    # masked_pixels = brightfield[amask]

    # otsu_thresh = threshold_li(masked_pixels)

    # masked_bf = masked_bf > otsu_thresh
    masked_bf = 0
    bright_and_dark = brighter_regions | masked_bf
    bw = closing(bright_and_dark, square(1))
    cleared = clear_border(bw)  # optional
    # cleared = bw

    label_image = label(cleared)
    centers = []
    coords = []
    for region in regionprops(label_image, brightfield):
        y, x = region.centroid_weighted
        coords.append(region.coords)
        centers.append([x, y])

    centers = np.array(centers)
    centers = np.rint(centers).astype(np.uint16)
    return centers, coords


def scale(arr):
    return ((arr - arr.min()) * (1 / (arr.max() - arr.min()) * 255)).astype("uint8")


def preprocess_brightfield(brightfield, max_size):
    return scale(brightfield)[:max_size, :max_size]


def adjust_contrast(img, min=2, max=98):
    # pixvals = np.array(img)
    minval = np.percentile(img, min)  # room for experimentation
    maxval = np.percentile(img, max)  # room for experimentation
    img = np.clip(img, minval, maxval)
    img = ((img - minval) / (maxval - minval)) * 255
    return img.astype(np.uint8)


def bead_filter(bead, min, max):
    if bead[0] > min and bead[0] < max and bead[1] > min or bead[0] < max:
        return bead
    return [0, 0]


def bead_center(bead_contour):
    M = cv2.moments(bead_contour)
    if M["m00"] != 0:
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])

        return [cX, cY]
    return [0, 0]


def thresholding(image):
    def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def adjust_contrast(img, min=2, max=98):
        img = np.nan_to_num(img, nan=0.0, posinf=255, neginf=0)
        minval = np.percentile(img, min)
        maxval = np.percentile(img, max)
        epsilon = 1e-8
        img = np.clip(img, minval, maxval)
        img = ((img - minval) / (maxval - minval + epsilon)) * 255
        return img.astype(np.uint8)

    image_modified = np.invert(adjust_contrast(unsharp_mask(image)))

    # this is just the standard thresholding function from opencv
    # more can be found https://docs.opencv2.org/4.x/d7/dd0/tutorial_js_thresholding.html
    image_modified = cv2.adaptiveThreshold(
        unsharp_mask(image_modified),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        50,
    )

    # this removes large connected components
    image_modified = image_modified + np.invert(
        np.array(dip.AreaClosing(image_modified, filterSize=20, connectivity=2))
    )

    return image_modified


def blackout_dots(image, coords):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for xy in coords:
        cx = xy[0]
        cy = xy[1]
        # cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 3, (0, 0, 0), -1)
    return image


def draw_dots(image, coords):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for xy in coords:
        cx = xy[0]
        cy = xy[1]
        # cv2.drawContours(image, [i], -1, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 1, (0, 255, 0), -1)
    return image


def abs_threshold(im, p):
    im = im.copy()
    im[im < p] = 0
    im[im >= p] = 255
    return im


def second_pass_beadfinding(brightfield, beads):
    beads_found = blackout_dots(brightfield, beads)

    reduced = cv2.cvtColor(beads_found, cv2.COLOR_BGR2GRAY)

    reduced = cv2.blur(reduced, (3, 3))
    reduced = abs_threshold(reduced, 120)

    contours, _ = cv2.findContours(reduced, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    missed_beads = np.array([bead_center(x) for x in contours], dtype=np.int16)
    beads = np.concatenate((missed_beads, beads), axis=0)

    beads = np.unique(beads, axis=0)
    return beads


def edge_bead_filtering(radius, max_size):
    def filter_func(bead):
        x, y = bead
        # Check if the bead is at least 'radius' pixels away from the edges
        if (x - 2 * radius) > 0 and (x + 2 * radius) < max_size:
            if (y - 2 * radius) > 0 and (y + 2 * radius) < max_size:
                return True
        return False

    return filter_func


# def get_excel(
#     beads,
#     signal_to_noise_cutoff,
#     tifs,
#     max_size,
#     layer_threshold_dict=defaultdict(int),
#     progress_callback=None,
#     is_running_callback=None,
# ):
#     def update_progress_internal(value, message):
#         if progress_callback:
#             # Scale the internal progress (0-100) to a sub-range of the overall progress (40-90)
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
#     # flor layer is all channels except brightfield
#     for i, md in enumerate(tif_metadata):
#         assert isinstance(md, MetaData)
#         md.flors_layers = [
#             j for j in range(len(tif_images[i])) if j != int(md.reference_channel)
#         ]
#         print(
#             f"Flors layers for tif {i} are {md.flors_layers}, reference channel is {md.reference_channel}"
#         )
#     # GETTING ALL THE BEAD BRIGHTNESSES
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
#             last_progress = -1
#             # filter out edge beads:
#             max_size = md.max_size
#             beads = filter(edge_bead_filtering(radius, max_size), beads)
#             for b_i, bead in enumerate(beads):
#                 if not is_running():
#                     return None
#                 current_step += 1
#                 progress_percentage = int((current_step / total_steps) * 100)
#                 if progress_percentage != last_progress:
#                     update_progress_internal(
#                         progress_percentage,
#                         f"Processing cycle {tif_count + 1}/{total_cycles}, layer {i + 1}/{total_layers_per_cycle}, bead {b_i + 1}/{total_beads}",
#                     )
#                     last_progress = progress_percentage
#                 x, y = bead

#                 roi = flor_layer[
#                     y - radius : y + radius,
#                     x - radius : x + radius,
#                 ]
#                 brightness = np.median(roi)

#                 if (x - 2 * 20) > 0 and (x + 2 * 20) < max_size:
#                     if (y - 2 * 20) > 0 and (y + 2 * 20) < max_size:
#                         local_flor_layer = flor_layer[
#                             y - radius : y + radius,
#                             x - radius : x + radius,
#                         ]
#                         flor_layer_background_intensity_local = np.percentile(
#                             local_flor_layer, 10
#                         )
#                         if (
#                             flor_layer_background_intensity_local > 0
#                         ):  # Corrected if statement
#                             signal_noise_ratio = (
#                                 brightness - flor_layer_background_intensity_local
#                             ) / flor_layer_background_intensity_local
#                         else:
#                             signal_noise_ratio = 0
#                     else:
#                         signal_noise_ratio = 0
#                 else:
#                     signal_noise_ratio = 0

#                 layer_specific_data[b_i] = brightness
#                 sig_noise_data[b_i] = signal_noise_ratio
#                 layer_threshold[b_i] = brightness > layer_threshold_dict[layer]  # boolean values

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

import numpy as np
from scipy.signal import correlate2d


def process_bead_batch(args):
    """
    Process a batch of beads for a single fluorescence layer.
    Returns:
        layer_specific_data (uint16): brightness per bead
        sig_noise_data (float): signal-to-noise ratio
        layer_threshold_data (float): thresholded brightness
    """
    (
        flor_layer,
        beads_batch,
        radius,
        layer_threshold,
        roi_coords_batch,
        start_idx,
        end_idx,
    ) = args

    batch_size = end_idx - start_idx
    layer_specific_data = np.zeros(batch_size, dtype="float32")
    sig_noise_data = np.zeros(batch_size, dtype="float32")
    layer_threshold_data = np.zeros(batch_size, dtype="uint8")

    # --- ROI extraction ---
    if roi_coords_batch is not None:
        # custom ROI coords (list of arrays of pixel indices)
        bead_rois = [
            flor_layer[coords[:, 0], coords[:, 1]] for coords in roi_coords_batch
        ]
    else:
        bead_rois = extract_bead_rois(flor_layer, beads_batch, radius)

    bead_rois = np.asarray(bead_rois)

    # --- Background percentile (vectorized) ---
    percentile_map = np.percentile(bead_rois.reshape(batch_size, -1), 10, axis=1)

    # --- Gaussian kernel ---
    kernel_size = 5
    gaussian = gaussian_kernel(kernel_size)

    # --- Process each ROI ---
    for b_i, roi in enumerate(bead_rois):
        filtered_roi = correlate2d(roi, gaussian, mode="valid")
        brightness = np.max(filtered_roi)
        background = percentile_map[b_i]

        if background > 0:
            snr = (brightness - background) / background
        else:
            snr = 0.0

        layer_specific_data[b_i] = brightness
        sig_noise_data[b_i] = snr
        layer_threshold_data[b_i] = brightness > 0.3

    return layer_specific_data, sig_noise_data, layer_threshold_data


# def calculate_template_match(roi):


#     # Normalize so sum = 1
#     gaussian_5x5 /= gaussian_5x5.sum()
#     roi = adjust_contrast(roi.astype(np.float32), 10, 90)
#     score = correlate2d(roi.astype(np.float32), gaussian_5x5, mode="same")
#     return np.max(score)
def calculate_template_match(roi):
    gaussian_9x9 = np.array(
        [
            [1, 4, 7, 11, 14, 11, 7, 4, 1],
            [4, 16, 26, 41, 53, 41, 26, 16, 4],
            [7, 26, 42, 67, 87, 67, 42, 26, 7],
            [11, 41, 67, 107, 139, 107, 67, 41, 11],
            [14, 53, 87, 139, 181, 139, 87, 53, 14],
            [11, 41, 67, 107, 139, 107, 67, 41, 11],
            [7, 26, 42, 67, 87, 67, 42, 26, 7],
            [4, 16, 26, 41, 53, 41, 26, 16, 4],
            [1, 4, 7, 11, 14, 11, 7, 4, 1],
        ],
        dtype=np.float32,
    )
    # remove corners
    gaussian_9x9[0, 0] = 0
    gaussian_9x9[0, 1] = 0
    gaussian_9x9[1, 0] = 0
    gaussian_9x9[0, 7] = 0
    gaussian_9x9[0, 8] = 0
    gaussian_9x9[1, 8] = 0
    gaussian_9x9[7, 0] = 0
    gaussian_9x9[8, 0] = 0
    gaussian_9x9[8, 1] = 0
    gaussian_9x9[7, 8] = 0
    gaussian_9x9[8, 7] = 0
    gaussian_9x9[8, 8] = 0

    gaussian_9x9 /= np.sum(gaussian_9x9)
    roi = adjust_contrast(roi.astype(np.float32), 10, 90)
    score = correlate2d(roi.astype(np.float32), gaussian_9x9, mode="same")
    return np.max(score)


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


def extract_bead_rois(flor_layer, beads_batch, radius):
    """Extract same-shaped, padded ROIs for each bead."""
    H, W = flor_layer.shape
    pad = radius
    padded = np.pad(flor_layer, pad_width=pad, mode="constant", constant_values=0)

    # Shift bead coords to match padded indexing
    beads_batch = np.rint(beads_batch).astype(int)
    beads_batch_padded = beads_batch + pad

    # Preallocate consistent ROI array
    roi_size = 2 * radius + 1
    bead_rois = np.empty((len(beads_batch), roi_size, roi_size), dtype=flor_layer.dtype)

    for i, (x, y) in enumerate(beads_batch_padded):
        bead_rois[i] = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]

    return bead_rois


def extract_padded_rois(img, coords, channels, radius=2):
    """
    Extracts zero-padded ROIs for all coords and channels.

    img: np.ndarray, shape (num_channels, H, W)
    coords: np.ndarray, shape (num_beads, 2) with (x, y)
    channels: list of channel indices
    radius: half-size of ROI (2 → 5x5)
    """
    H, W = img.shape[1:]
    size = radius * 2 + 1
    rois = np.zeros((len(coords), len(channels), size, size), dtype=img.dtype)

    for b_idx, (x, y) in enumerate(coords):
        for c_idx, ch in enumerate(channels):
            # Compute source slice bounds
            x1, x2 = x - radius, x + radius + 1
            y1, y2 = y - radius, y + radius + 1

            # Compute destination slice bounds (where real pixels go inside padded ROI)
            dx1 = max(0, -x1)
            dy1 = max(0, -y1)
            dx2 = size - max(0, x2 - W)
            dy2 = size - max(0, y2 - H)

            # Clip source slice to image bounds
            sx1 = max(0, x1)
            sy1 = max(0, y1)
            sx2 = min(W, x2)
            sy2 = min(H, y2)

            # Copy into padded ROI
            rois[b_idx, c_idx, dy1:dy2, dx1:dx2] = img[ch, sy1:sy2, sx1:sx2]

    return rois


import numpy as np


def print_timing_summary(timings):
    """Print a formatted summary of timing results."""
    print("\n=== Watershed Segmentation Timing Summary ===")
    print(f"{'Step':<25} {'Time (ms)':<12} {'Percentage':<12}")
    print("-" * 50)

    total_time = timings["total"]
    for step, time_val in timings.items():
        if step != "total":
            percentage = (time_val / total_time) * 100
            print(f"{step:<25} {time_val * 1000:>8.2f} ms {percentage:>8.1f}%")

    print("-" * 50)
    print(f"{'TOTAL':<25} {total_time * 1000:>8.2f} ms {100.0:>8.1f}%")
    print("=" * 50)


def watershed_segmentation_cv2(
    img: np.ndarray, border: int, tile_size: int = 100, pad_fraction: float = 0.1
):
    """
    Performs watershed segmentation using tile-based adaptive thresholding.

    This function is robust to uneven illumination by calculating thresholds
    locally on overlapping tiles.

    Args:
        img (np.ndarray): Input grayscale image (float, normalized to 0-1).
        border (int): Erosion iterations to apply to the final mask.
        tile_size (int): The size of the square tiles for local thresholding.
        pad_fraction (float): The fraction of the tile size to use as overlap
                              to prevent edge artifacts.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - A binary mask of the segmented objects.
        - An integer-labeled mask where each unique object has a different ID.
    """
    # 1. Get an initial binary mask using tile-based (adaptive) thresholding.
    height, width = img.shape
    binary_mask = np.zeros_like(img, dtype=bool)
    pad_size = int(tile_size * pad_fraction)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Define the padded tile to calculate the threshold from
            y_start_pad = max(0, y - pad_size)
            y_end_pad = min(height, y + tile_size + pad_size)
            x_start_pad = max(0, x - pad_size)
            x_end_pad = min(width, x + tile_size + pad_size)

            tile = img[y_start_pad:y_end_pad, x_start_pad:x_end_pad]
            if tile.size == 0:
                continue

            # Calculate a local threshold for this specific tile
            local_thresh = filters.threshold_yen(tile)

            # Define the main, non-overlapping region to apply the threshold to
            y_start_main = y
            y_end_main = min(height, y + tile_size)
            x_start_main = x
            x_end_main = min(width, x + tile_size)

            main_region = img[y_start_main:y_end_main, x_start_main:x_end_main]

            # Apply the local threshold to the main region of the final mask
            binary_mask[y_start_main:y_end_main, x_start_main:x_end_main] = (
                main_region > local_thresh
            )

    # The rest of the watershed logic proceeds as before, but now using the
    # much more robust, adaptively-generated binary_mask.

    # 2. Create the "sure background" marker.
    sure_background = binary_mask

    # 3. Create the "sure foreground" markers using the distance transform.
    distance_transform = ndi.distance_transform_edt(binary_mask)
    threshold_distance = 0.0001 * distance_transform.max()
    sure_foreground_seeds = distance_transform > threshold_distance
    labeled_fg, _ = measure.label(sure_foreground_seeds, return_num=True)

    # 4. Create the final marker image for the watershed algorithm.
    markers = np.zeros_like(img, dtype=np.int32)
    markers[~sure_background] = 1
    markers[sure_foreground_seeds] = labeled_fg[sure_foreground_seeds] + 1

    # 5. Compute the elevation map and run the watershed algorithm.
    elevation_map = filters.sobel(img)
    elevation_map_cv = cv2.cvtColor(
        (elevation_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    labeled_markers = cv2.watershed(elevation_map_cv, markers)

    # 6. Post-process the results.
    segmentation_mask = (labeled_markers > 1).astype(np.uint8)
    if border > 0:
        segmentation_mask = ndi.binary_erosion(segmentation_mask, iterations=border)

    labeled_objects = measure.label(segmentation_mask)
    final_segmentation_mask = (labeled_objects > 0).astype(np.uint8)

    return final_segmentation_mask, labeled_objects


def _create_adaptive_threshold_maps(
    img: np.ndarray, tile_size: int, low_percentile: int, high_percentile: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates smooth, full-resolution threshold maps from low-res tile-based statistics.
    """
    img_h, img_w = img.shape

    # Determine the dimensions of the low-resolution grid
    tiles_h = int(np.ceil(img_h / tile_size))
    tiles_w = int(np.ceil(img_w / tile_size))

    # Initialize low-resolution placeholder maps
    low_res_low_map = np.zeros((tiles_h, tiles_w), dtype=np.float32)
    low_res_high_map = np.zeros((tiles_h, tiles_w), dtype=np.float32)

    # Calculate percentile for each tile
    for i in range(tiles_h):
        for j in range(tiles_w):
            # Define tile boundaries
            y1, y2 = i * tile_size, min((i + 1) * tile_size, img_h)
            x1, x2 = j * tile_size, min((j + 1) * tile_size, img_w)
            tile = img[y1:y2, x1:x2]
            # low_percentile = get_adaptive_low_percentile(tile, high_percentile)

            if tile.size > 0:
                low_res_low_map[i, j] = np.percentile(tile, low_percentile)
                low_res_high_map[i, j] = np.percentile(tile, high_percentile)

    # Upscale the low-res maps to full image size using cubic interpolation
    # This creates a smooth, spatially varying threshold surface.
    full_res_low_map = cv2.resize(
        low_res_low_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC
    )
    full_res_high_map = cv2.resize(
        low_res_high_map, (img_w, img_h), interpolation=cv2.INTER_CUBIC
    )

    return full_res_low_map, full_res_high_map


def watershed_segmentation_adaptive(
    img: np.ndarray,
    border: int,
    tile_size: int = 64,
    low_percentile: int = 50,
    fill: bool = True,
    high_percentile: int = 65,
):
    """
    Performs watershed segmentation using tile-based adaptive thresholds.

    This function automatically handles variations in background and brightness
    by calculating marker thresholds locally across a grid of tiles.

    Args:
        img (np.ndarray): Input grayscale image (float, normalized to 0-1).
        border (int): Number of erosion iterations for post-processing.
        tile_size (int): The side length (in pixels) of tiles for local thresholding.
                         Should be larger than the objects of interest.
        low_percentile (int): Percentile (0-100) to define the local background.
        high_percentile (int): Percentile (0-100) to define local foreground seeds.
        fill (bool): Whether to fill holes in the segmentation mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - A binary mask of the segmented objects.
        - An integer-labeled mask where each object has a unique ID.
    """
    img_float = img.astype(np.float32)

    adaptive_low_map, adaptive_high_map = _create_adaptive_threshold_maps(
        img_float, tile_size, low_percentile, high_percentile
    )

    elevation_map = filters.sobel(img_float)

    markers = np.zeros_like(img_float, dtype=np.int32)

    markers[img_float <= adaptive_low_map] = 1
    markers[img_float > adaptive_high_map] = 2

    elevation_map_cv = cv2.cvtColor(
        (elevation_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    labeled_markers = cv2.watershed(elevation_map_cv, markers.copy())

    segmentation_mask = (labeled_markers >= 2).astype(np.uint8)

    if border > 0:
        segmentation_mask = ndi.binary_erosion(segmentation_mask, iterations=border)
    if fill:
        segmentation_mask = ndi.binary_fill_holes(
            segmentation_mask, structure=ndi.generate_binary_structure(2, 1)
        )
    labeled_objects = measure.label(segmentation_mask)
    final_segmentation_mask = (labeled_objects > 0).astype(np.uint8)

    return final_segmentation_mask, labeled_objects


def get_labels_from_cycles(
    cycles,
    cycles_metadata: List[MetaData],
    max_size,
    low_percentile=50,
    high_percentile=65,
    fill=True,
    adaptive_thresholding=False,
    tile_size=100,
    border_erode=1,
    use_stardist=False,  # Toggle between watershed and StarDist
    model=None,  # StarDist model (required if use_stardist=True)
    normalize_method="none",  # Exposure normalization method
    progress_units_callback=None,
):
    cycle_labels = []
    total_units = 0
    for metadata in cycles_metadata:
        if metadata.flors_layers is not None:
            total_units += len(metadata.flors_layers)
    done_units = 0
    for i in range(len(cycles)):
        cycles[i] = process_cycle(cycles[i], cycles_metadata[i], normalize_method)[
            :max_size, :max_size
        ]

    for i, cycle in enumerate(cycles):
        flor_layers = cycles_metadata[i].flors_layers
        assert flor_layers is not None, "forgot to initialize flors layer idx"
        max_size = cycles_metadata[i].max_size
        labels = []
        for layer_idx in flor_layers:
            img_array = cycle[layer_idx, :max_size, :max_size].astype(np.float32)

            # Normalize image to a 0-1 range
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min)

            if use_stardist:
                # StarDist path - use CSBDeep normalization
                img_normalized = normalize(img_array, 1, 99.8)

                if model is None:
                    raise ValueError(
                        "StarDist model is required when use_stardist=True"
                    )

                label_image, _ = model.predict_instances(
                    img_normalized,
                    axes="YX",
                    prob_thresh=0.15,  # Hardcoded from user example
                    n_tiles=(3, 3),  # Hardcoded from user example
                )
                labels.append(label_image)
            else:
                # Watershed path - keep existing logic
                # old cold, shouldn't be used, it is always adaptive now
                if not adaptive_thresholding:
                    seg, label = watershed_segmentation_cv2(
                        img=img_array,
                        border=border_erode,
                        # low_percentile=low_percentile,
                        # high_percentile=high_percentile,
                        # tile_size=100,
                    )
                else:
                    seg, label = watershed_segmentation_adaptive(
                        img=img_array,
                        border=border_erode,
                        low_percentile=low_percentile,
                        high_percentile=high_percentile,
                        tile_size=tile_size,
                        fill=fill,
                    )
                # label = stardist_segmentation(img_array, model)
                labels.append(label)
            done_units += 1
            if progress_units_callback:
                progress_units_callback("activation_regions", done_units, total_units)

        cycle_labels.append(labels)
    return cycle_labels


from csbdeep.utils import normalize
from stardist.models import StarDist2D


def _resolve_stardist_n_tiles(model, img, n_tiles):
    if isinstance(n_tiles, numbers.Number):
        tiles = int(n_tiles)
        if tiles == 0:
            guessed = model._guess_n_tiles(img)
            if guessed is None:
                raise RuntimeError(
                    "StarDist tile auto-guess returned None; cannot estimate tiles."
                )
            resolved = tuple(int(v) for v in guessed)
        elif tiles > 0:
            resolved = (tiles, tiles)
        else:
            raise ValueError(f"n_tiles must be >= 0, got {n_tiles}")
    else:
        resolved = tuple(int(v) for v in n_tiles)
    if len(resolved) == 0:
        raise ValueError("n_tiles must contain at least one axis value.")
    if len(resolved) == 1:
        resolved = (resolved[0], resolved[0])
    if any(v <= 0 for v in resolved):
        raise ValueError(f"n_tiles values must be > 0, got {resolved}")
    return resolved


def _count_total_tiles(n_tiles):
    total = 1
    for value in n_tiles:
        total *= max(int(value), 1)
    return max(int(total), 1)


def _predict_instances_with_generator(
    model,
    img,
    prob_thresh=None,
    nms_thresh=None,
    n_tiles=1,
    scale=None,
    progress_units_callback=None,
    progress_stage=None,
    progress_done_offset=0,
    progress_total_units=None,
    progress_message_callback=None,
    nms_message=None,
):
    resolved_tiles = _resolve_stardist_n_tiles(model, img, n_tiles)
    expected_tiles = _count_total_tiles(resolved_tiles)
    done_tiles = 0
    final_result = None

    generator = model._predict_instances_generator(
        img,
        axes="YX",
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        n_tiles=resolved_tiles,
        scale=scale,
        show_tile_progress=True,
    )
    for output in generator:
        if isinstance(output, str):
            if output == "tile":
                done_tiles += 1
                if progress_units_callback and progress_stage:
                    done_total = progress_done_offset + done_tiles
                    total_units = progress_total_units
                    if total_units is None:
                        total_units = progress_done_offset + expected_tiles
                    total_units = max(int(total_units), done_total)
                    progress_units_callback(progress_stage, done_total, total_units)
                continue
            if output == "nms":
                if progress_message_callback and nms_message:
                    progress_message_callback(nms_message)
                continue
            if output in {"predict", "nms"}:
                continue
            raise RuntimeError(
                "Unexpected StarDist generator event "
                f"'{output}'. Verify stardist==0.9.1 compatibility."
            )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "Unexpected StarDist generator output shape. "
                "Verify stardist==0.9.1 compatibility."
            )
        final_result = output

    if final_result is None:
        raise RuntimeError(
            "StarDist generator finished without final output. "
            "Verify stardist==0.9.1 compatibility."
        )

    completed_tiles = max(done_tiles, expected_tiles)
    if progress_units_callback and progress_stage:
        done_total = progress_done_offset + completed_tiles
        total_units = progress_total_units
        if total_units is None:
            total_units = done_total
        total_units = max(int(total_units), done_total)
        progress_units_callback(progress_stage, done_total, total_units)

    labels, details = final_result
    return labels, details, completed_tiles, expected_tiles


def stardist_segmentation(
    img, model, min=1, max=75, block_size=700, new_min_overlap=24, scale=1
):
    # guess_tiles = model._guess_n_tiles(img)
    img = normalize(img, min, max)
    stardist_labels, polygons = model.predict_instances_big(
        img,
        axes="YX",
        block_size=block_size,
        min_overlap=new_min_overlap,
        context=None,
        show_progress=True,
        n_tiles=(1, 1),
        scale=scale,
    )
    # total_tiles = int(guess_tiles[0] * guess_tiles[1])
    # labels_gen = model._predict_instances_generator(
    #     img,
    #     n_tiles=guess_tiles,
    #     scale=scale,
    # )
    # stardist_labels = None
    # for i, (labels, *_) in enumerate(labels_gen):
    #     stardist_labels = labels  # last one is full image
    #     log(f"Processing tile {i}/{total_tiles}")
    #     if i > total_tiles:
    #         break
    # print(f"Stardist found {stardist_labels.max()} objects")
    return stardist_labels


def beadfinding_notebook_stardist(
    brightfield,
    scale=1.0,
    block_size=700,
    n_tiles=1,
    progress_units_callback=None,
    progress_callback=None,
):
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    normalized = normalize(brightfield.astype(np.float32), 1, 99)
    label_image, _, _, _ = _predict_instances_with_generator(
        model=model,
        img=normalized,
        n_tiles=n_tiles,
        scale=scale,
        progress_units_callback=progress_units_callback,
        progress_stage="initial_detection",
        progress_message_callback=progress_callback,
        nms_message="Initial bead detection... Running NMS",
    )
    num_labels = int(label_image.max())
    if num_labels == 0:
        return np.array([]).reshape(0, 2)
    centroids = ndimage.center_of_mass(
        label_image, label_image, range(1, num_labels + 1)
    )
    centers = np.array(centroids, dtype=np.float32)
    centers = np.array([[x, y] for y, x in centers], dtype=np.float32)
    return centers


def quantile_normalize_layers(cycle_img, flor_layers):
    """
    Normalize fluorescence layers to have identical intensity distributions.

    Matches intensity distributions across layers by aligning quantiles.
    This makes all layers have the same statistical distribution.

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices

    Returns:
        Normalized cycle_img with matched distributions
    """
    cycle_img = cycle_img.copy()

    # Extract fluorescence layers
    flor_data = cycle_img[flor_layers].astype(np.float32)

    # Compute target distribution (mean across all layers)
    all_intensities = flor_data.flatten()
    sorted_target = np.sort(all_intensities)

    # For each layer, match to target distribution
    for i, layer_idx in enumerate(flor_layers):
        layer = flor_data[i].flatten()
        sorted_layer = np.sort(layer)
        ranks = np.argsort(np.argsort(layer))  # Get rank order

        # Map ranks to target quantiles
        normalized = sorted_target[ranks].reshape(cycle_img[layer_idx].shape)
        cycle_img[layer_idx] = np.clip(normalized, 0, 65535).astype(np.uint16)

    return cycle_img


def robust_zscore_normalize_layers(cycle_img, flor_layers):
    """
    Z-score normalize using median and MAD (Median Absolute Deviation).

    More robust to outliers than standard z-score. Normalizes each layer
    to have median=0 and MAD-based scale=1.

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices

    Returns:
        Normalized cycle_img with z-score scaling
    """
    cycle_img = cycle_img.copy()

    for layer_idx in flor_layers:
        layer = cycle_img[layer_idx].astype(np.float32)

        # Robust statistics
        median = np.median(layer)
        mad = np.median(np.abs(layer - median))
        scale = 1.4826 * mad  # Scale factor for normal distribution

        if scale > 1e-8:
            # Normalize: (x - median) / scale
            layer_normalized = (layer - median) / scale
        else:
            layer_normalized = layer - median

        # Rescale to uint16 range (shift and scale to use full range)
        # Center around 32768 (middle of uint16 range)
        layer_rescaled = layer_normalized * 5000 + 32768
        cycle_img[layer_idx] = np.clip(layer_rescaled, 0, 65535).astype(np.uint16)

    return cycle_img


def percentile_match_layers(cycle_img, flor_layers, target_percentiles=None):
    """
    Match specific percentiles (1st, 50th, 99th) across layers.

    Uses linear mapping to align key intensity landmarks across all layers.
    Preserves uint16 range and is robust to outliers.

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices
        target_percentiles: (p1, p50, p99) target values, or None for mean

    Returns:
        Normalized cycle_img with matched percentiles
    """
    cycle_img = cycle_img.copy()

    if target_percentiles is None:
        # Compute target as mean across all layers
        all_data = cycle_img[flor_layers].flatten()
        p1, p50, p99 = np.percentile(all_data, [1, 50, 99])
        target_percentiles = (p1, p50, p99)

    for layer_idx in flor_layers:
        layer = cycle_img[layer_idx].astype(np.float32)

        # Get current percentiles
        curr_p1, curr_p50, curr_p99 = np.percentile(layer, [1, 50, 99])

        # Linear mapping: map current percentiles to target
        # Using 1st and 99th for robustness
        if curr_p99 - curr_p1 > 1e-8:
            scale = (target_percentiles[2] - target_percentiles[0]) / (
                curr_p99 - curr_p1
            )
            offset = target_percentiles[0] - scale * curr_p1

            layer_normalized = scale * layer + offset
            cycle_img[layer_idx] = np.clip(layer_normalized, 0, 65535).astype(np.uint16)

    return cycle_img


def clahe_normalize_layers(cycle_img, flor_layers, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE to enhance local contrast in each layer.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
    local contrast while limiting noise amplification.

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices
        clip_limit: Contrast limiting threshold
        tile_size: Size of tiles for local histogram equalization

    Returns:
        Normalized cycle_img with enhanced local contrast
    """
    cycle_img = cycle_img.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    for layer_idx in flor_layers:
        layer = cycle_img[layer_idx]

        # Convert to uint8 for CLAHE
        layer_uint8 = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        # Apply CLAHE
        layer_clahe = clahe.apply(layer_uint8)

        # Convert back to uint16
        cycle_img[layer_idx] = layer_clahe.astype(np.uint16) * 256

    return cycle_img


def gain_offset_correction_layers(cycle_img, flor_layers):
    """
    Correct gain and offset by estimating background and signal statistics.

    Models acquisition as: observed = gain * signal + offset
    Estimates background (offset) and signal scale (gain) per layer,
    then normalizes to common values.

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices

    Returns:
        Normalized cycle_img with corrected gain and offset
    """
    cycle_img = cycle_img.copy()

    # Compute statistics for each layer
    backgrounds = []
    signals = []

    for layer_idx in flor_layers:
        layer = cycle_img[layer_idx].astype(np.float32)

        # Estimate background (5th percentile)
        bg = np.percentile(layer, 5)
        backgrounds.append(bg)

        # Estimate signal (95th percentile after bg subtraction)
        signal = np.percentile(layer - bg, 95)
        signals.append(signal)

    # Target statistics (median across layers)
    target_bg = np.median(backgrounds)
    target_signal = np.median(signals)

    # Correct each layer
    for i, layer_idx in enumerate(flor_layers):
        layer = cycle_img[layer_idx].astype(np.float32)

        # Remove background
        layer_corrected = layer - backgrounds[i]

        # Apply gain correction
        if signals[i] > 1e-8:
            gain = target_signal / signals[i]
            layer_corrected = layer_corrected * gain

        # Add target background
        layer_corrected = layer_corrected + target_bg

        # Clip to valid range
        cycle_img[layer_idx] = np.clip(layer_corrected, 0, 65535).astype(np.uint16)

    return cycle_img


def enhanced_histogram_matching(cycle_img, flor_layers):
    """
    Enhanced version of current histogram matching.

    Improvements over basic histogram matching:
    1. Use layer with best SNR as reference
    2. Apply percentile clipping before matching
    3. Preserve intensity range

    Args:
        cycle_img: (C, H, W) array
        flor_layers: List of fluorescence layer indices

    Returns:
        Normalized cycle_img with matched histograms
    """
    cycle_img = cycle_img.copy()

    # Step 1: Select best reference layer (highest signal-to-noise)
    snr_scores = []
    for layer_idx in flor_layers:
        layer = cycle_img[layer_idx].astype(np.float32)
        signal = np.percentile(layer, 95)
        noise = np.std(layer[layer < np.percentile(layer, 10)])
        snr = signal / (noise + 1e-8)
        snr_scores.append(snr)

    reference_layer_idx = flor_layers[np.argmax(snr_scores)]
    reference = cycle_img[reference_layer_idx]

    # Step 2: Match all other layers to reference
    for layer_idx in flor_layers:
        if layer_idx != reference_layer_idx:
            layer = cycle_img[layer_idx]

            # Clip extreme values before matching
            p1, p99 = np.percentile(layer, [1, 99])
            layer_clipped = np.clip(layer, p1, p99)

            # Match histogram
            matched = match_histograms(layer_clipped, reference)

            cycle_img[layer_idx] = matched.astype(np.uint16)

    return cycle_img


def process_cycle(cycle, metadata: MetaData, normalize_method="none"):
    """
    Process one imaging cycle with configurable normalization.

    - Leave channels before reference untouched
    - Use (reference_channel+1) as reference, equalize it
    - Match all later channels to the reference, then adjust sigmoid
    - Apply exposure normalization to fluorescence layers

    Args:
        cycle: (C, H, W) array
        metadata: MetaData with reference_channel and flors_layers
        normalize_method: One of:
            - 'none': No normalization (current behavior)
            - 'histogram': Enhanced histogram matching (improved current)
            - 'percentile': Percentile-based matching (recommended)
            - 'quantile': Full quantile normalization
            - 'zscore': Robust z-score normalization
            - 'clahe': CLAHE local contrast enhancement
            - 'gain_offset': Gain/offset correction with background subtraction

    Returns:
        Processed cycle with normalized fluorescence layers
    """
    num_layers = cycle.shape[0]
    processed = []

    ref_idx = metadata.reference_channel + 1

    # keep all channels before reference
    processed.extend(cycle[:ref_idx])

    # reference channel
    ref16 = cycle[ref_idx]
    ref_float = img_as_float(ref16)
    # ref_adj = adjust_sigmoid(ref_float, cutoff=0.1, gain=1)
    processed.append(img_as_uint(ref_float))

    # channels after reference
    for j in range(ref_idx + 1, num_layers):
        img16 = cycle[j]
        matched_float = img_as_float(img16)
        matched = match_histograms(matched_float, ref_float)
        # matched = adjust_sigmoid(matched, cutoff=0.1, gain=1)
        processed.append(img_as_uint(matched))

    processed = np.stack(processed, axis=0)

    # Apply exposure normalization to fluorescence layers
    if normalize_method != "none" and metadata.flors_layers is not None:
        if normalize_method == "percentile":
            processed = percentile_match_layers(processed, metadata.flors_layers)
        elif normalize_method == "quantile":
            processed = quantile_normalize_layers(processed, metadata.flors_layers)
        elif normalize_method == "zscore":
            processed = robust_zscore_normalize_layers(processed, metadata.flors_layers)
        elif normalize_method == "clahe":
            processed = clahe_normalize_layers(processed, metadata.flors_layers)
        elif normalize_method == "gain_offset":
            processed = gain_offset_correction_layers(processed, metadata.flors_layers)
        elif normalize_method == "histogram":
            processed = enhanced_histogram_matching(processed, metadata.flors_layers)

    return processed


import numpy as np


def assign_beads_labels(bead_data, cycle_labels):
    bead_data = bead_data.copy()
    for i, cycle_label in enumerate(cycle_labels):
        # Build one big array of shape (n_flour, H, W)
        labs = np.array(cycle_label)  # shape (n_flour, H, W)
        print(labs.shape)
        # Round coords and clip to bounds
        xs = np.rint(bead_data["x"].to_numpy()).astype(int)
        ys = np.rint(bead_data["y"].to_numpy()).astype(int)
        print(len(xs))
        values = labs[:, ys, xs].T
        print(len(values))

        # Replace zeros with 0, keep as float
        arr = values.astype(int)
        arr[arr == 0] = 0  # redundant, but ensures float
        arr = np.nan_to_num(arr, nan=0)

        # argmax along columns
        # argmaxes = np.argmax(arr, axis=1)

        # Build column names and assign
        for flour in range(arr.shape[1]):
            bead_data[f"cy{i}_{flour}"] = arr[:, flour]

        # bead_data[f'cy{i}_argmax'] = argmaxes

    return bead_data


def get_assignment(row, cols):
    col = [c for c in cols if row[c] > 0]
    # assert len(col) == 1
    col = col[0]
    return int(col.split("_")[-1])  # layer number


def get_adaptive_low_percentile(
    image: np.ndarray,
    high_percentile: float,
    min_diff: float = 1.0,
    max_diff: float = 15.0,
    scale_factor: float = 10.0,
) -> float:
    """
    Calculates an adaptive low percentile based on the image's coefficient of variation.

    Args:
        image (np.ndarray): The 2D image array to analyze.
        high_percentile (float): The upper percentile bound (e.g., 95).
        min_diff (float): The smallest possible percentile difference.
        max_diff (float): The largest possible percentile difference.
        scale_factor (float): A factor to map the CV to the percentile difference.

    Returns:
        float: The calculated lower percentile bound.
    """
    # Analyze only non-zero pixels to avoid background bias
    pixels = image[image > 0]
    if pixels.size == 0:
        return high_percentile - min_diff

    # Calculate Coefficient of Variation (CV)
    std_dev = np.std(pixels)
    mean_val = np.mean(pixels)

    if mean_val == 0:
        return high_percentile - min_diff

    cv = std_dev / mean_val

    # Map the CV to a percentile difference
    percentile_diff = cv * scale_factor

    # Clamp the difference to a reasonable range (e.g., between 3 and 15)
    clamped_diff = np.clip(percentile_diff, min_diff, max_diff)

    low_percentile = high_percentile - clamped_diff

    return low_percentile


from utils import resource_path


def _resolve_model_dir(model_name: str) -> str:
    model_dir = resource_path(f"assets/{model_name}")
    if os.path.isdir(model_dir):
        return model_dir
    if os.path.isdir(model_name):
        return model_name
    raise ValueError(f"StarDist model directory not found: {model_name}")


def load_custom_model(model_dir: str):
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return StarDist2D(
            None, name=os.path.basename(model_dir), basedir=os.path.dirname(model_dir)
        )
    for item in os.listdir(model_dir):
        subpath = os.path.join(model_dir, item)
        if os.path.isdir(subpath) and os.path.exists(
            os.path.join(subpath, "config.json")
        ):
            return StarDist2D(None, name=item, basedir=model_dir)
    raise ValueError(f"No StarDist model found in {model_dir}")


def get_labels_from_cycles_with_prob(
    cycles,
    metadata_list: List[MetaData],
    max_size,
    model,
    block_size=700,
    prob_thresh=0.1,
    n_tiles=1,
    nms_thresh=0.1,
    progress_units_callback=None,
    progress_callback=None,
):
    base_prob_thresh = float(prob_thresh)
    cycle_labels = []
    tile_plan: list[list[Tuple[int, ...]]] = []
    total_units = 0
    for i, cycle in enumerate(cycles):
        flors_layers = metadata_list[i].flors_layers
        if flors_layers is None:
            raise ValueError("Fluorescence layers are not initialized.")
        cycle_tiles: list[Tuple[int, ...]] = []
        for layer_idx in flors_layers:
            tile_img = cycle[layer_idx].astype(np.float32)[:max_size, :max_size]
            tiles = _resolve_stardist_n_tiles(model, tile_img, n_tiles)
            cycle_tiles.append(tiles)
            total_units += _count_total_tiles(tiles)
        tile_plan.append(cycle_tiles)

    done_units = 0
    for i, cycle in enumerate(tqdm(cycles, desc="Processing cycles")):
        layers_out = []
        flors_layers = metadata_list[i].flors_layers
        if flors_layers is None:
            raise ValueError("Fluorescence layers are not initialized.")
        for layer_pos, layer_idx in enumerate(flors_layers):
            img = cycle[layer_idx].astype(np.float32)[:max_size, :max_size]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = normalize(img, 1, 99.8)
            effective_prob_thresh = base_prob_thresh
            try:
                lbl, det, completed_tiles, _ = _predict_instances_with_generator(
                    model=model,
                    img=img,
                    prob_thresh=effective_prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=tile_plan[i][layer_pos],
                    progress_units_callback=progress_units_callback,
                    progress_stage="activation_regions",
                    progress_done_offset=done_units,
                    progress_total_units=total_units,
                    progress_message_callback=progress_callback,
                    nms_message="Getting activation regions from cycles - Running NMS",
                )
            except MemoryError:
                retry_prob_thresh = max(effective_prob_thresh, 0.4)
                if retry_prob_thresh <= effective_prob_thresh:
                    raise
                if progress_callback:
                    progress_callback(
                        "Activation NMS memory pressure; retrying with stricter threshold"
                    )
                lbl, det, completed_tiles, _ = _predict_instances_with_generator(
                    model=model,
                    img=img,
                    prob_thresh=retry_prob_thresh,
                    nms_thresh=nms_thresh,
                    n_tiles=tile_plan[i][layer_pos],
                    progress_units_callback=progress_units_callback,
                    progress_stage="activation_regions",
                    progress_done_offset=done_units,
                    progress_total_units=total_units,
                    progress_message_callback=progress_callback,
                    nms_message="Getting activation regions from cycles - Running NMS",
                )
            if not isinstance(det, dict):
                raise RuntimeError(
                    "Unexpected StarDist details payload. "
                    "Verify stardist==0.9.1 compatibility."
                )
            n_obj = int(lbl.max())
            prob_lut = np.zeros(n_obj + 1, dtype=np.float32)
            if n_obj > 0 and "prob" in det and det["prob"] is not None:
                probs = np.asarray(det["prob"], dtype=np.float32)
                m = min(n_obj, probs.shape[0])
                prob_lut[1 : m + 1] = probs[:m]
            layers_out.append({"lbl": lbl.astype(np.int32), "prob_lut": prob_lut})
            done_units += completed_tiles
        cycle_labels.append(layers_out)
    return cycle_labels


def assign_beads_labels_with_prob_patch3x3_fallback(
    bead_df,
    cycle_labels,
    min_center_prob=None,
    min_patch_prob=None,
    min_patch_support=2,
    min_prob_margin=None,
    center_vs_patch_margin=None,
    invalid_value=0,
    prob_col_suffix="_prob",
    chunk_size=200_000,
):
    bead_df = bead_df.copy()
    xs0 = np.rint(bead_df["x"].to_numpy()).astype(np.int32)
    ys0 = np.rint(bead_df["y"].to_numpy()).astype(np.int32)
    n_rows = xs0.size
    dx = np.array(
        [
            -2,
            -1,
            0,
            1,
            2,
            -2,
            -1,
            0,
            1,
            2,
            -2,
            -1,
            0,
            1,
            2,
            -2,
            -1,
            0,
            1,
            2,
            -2,
            -1,
            0,
            1,
            2,
        ],
        dtype=np.int32,
    )
    dy = np.array(
        [
            -2,
            -2,
            -2,
            -2,
            -2,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
        ],
        dtype=np.int32,
    )
    for i, cycle_layers in enumerate(cycle_labels):
        for j, layer in enumerate(cycle_layers):
            lbl = layer["lbl"]
            prob_lut = layer["prob_lut"]
            h, w = lbl.shape[:2]
            out_id = np.empty(n_rows, dtype=np.int32)
            out_prob = np.empty(n_rows, dtype=np.float32)
            chunks = range(0, n_rows, chunk_size)
            total = (n_rows + chunk_size - 1) // chunk_size
            for s in tqdm(chunks, desc=f"assign cy{i}_{j}", total=total):
                e = min(s + chunk_size, n_rows)
                xs = np.clip(xs0[s:e], 0, w - 1)
                ys = np.clip(ys0[s:e], 0, h - 1)
                center_id = lbl[ys, xs].astype(np.int32)
                center_prob = prob_lut[center_id].astype(np.float32)
                trust_center = center_id > 0
                if min_center_prob is not None:
                    trust_center &= center_prob >= float(min_center_prob)
                best_id = np.where(trust_center, center_id, invalid_value).astype(
                    np.int32
                )
                best_prob = np.where(trust_center, center_prob, 0.0).astype(np.float32)

                def _eval_patch(xs_sub, ys_sub):
                    xs_n = xs_sub[:, None] + dx[None, :]
                    ys_n = ys_sub[:, None] + dy[None, :]
                    np.clip(xs_n, 0, w - 1, out=xs_n)
                    np.clip(ys_n, 0, h - 1, out=ys_n)
                    ids = lbl[ys_n, xs_n].astype(np.int32)
                    probs = prob_lut[ids].astype(np.float32)
                    probs_sel = np.where(ids > 0, probs, -np.inf)
                    best_k = np.argmax(probs_sel, axis=1)
                    row = np.arange(ids.shape[0])
                    patch_best_id = ids[row, best_k]
                    patch_best_prob = prob_lut[patch_best_id].astype(np.float32)
                    ok_support = np.ones(ids.shape[0], dtype=bool)
                    if min_patch_support is not None and int(min_patch_support) > 1:
                        support = (ids == patch_best_id[:, None]).sum(axis=1)
                        ok_support = support >= int(min_patch_support)
                    ok_prob = np.ones(ids.shape[0], dtype=bool)
                    if min_patch_prob is not None:
                        ok_prob = patch_best_prob >= float(min_patch_prob)
                    ok_margin = np.ones(ids.shape[0], dtype=bool)
                    if min_prob_margin is not None:
                        finite_mask = np.isfinite(probs_sel)
                        n_finite = finite_mask.sum(axis=1)
                        top2 = np.sort(probs_sel, axis=1)[:, -2:]
                        margin = np.zeros(ids.shape[0], dtype=np.float32)
                        ok_2plus = n_finite >= 2
                        margin[ok_2plus] = top2[ok_2plus, 1] - top2[ok_2plus, 0]
                        ok_margin = margin >= float(min_prob_margin)
                    ok = (
                        ok_support
                        & ok_prob
                        & ok_margin
                        & (patch_best_id > 0)
                        & np.isfinite(patch_best_prob)
                    )
                    return patch_best_id, patch_best_prob, ok

                need_patch = ~trust_center
                if np.any(need_patch):
                    idxs = np.flatnonzero(need_patch)
                    patch_best_id, patch_best_prob, ok = _eval_patch(
                        xs[need_patch], ys[need_patch]
                    )
                    good_idxs = idxs[ok]
                    best_id[good_idxs] = patch_best_id[ok]
                    best_prob[good_idxs] = patch_best_prob[ok]

                if center_vs_patch_margin is not None:
                    maybe_override = trust_center
                    if np.any(maybe_override):
                        idxs = np.flatnonzero(maybe_override)
                        patch_best_id, patch_best_prob, ok = _eval_patch(
                            xs[maybe_override], ys[maybe_override]
                        )
                        delta = patch_best_prob - center_prob[maybe_override]
                        ok_override = ok & (delta >= float(center_vs_patch_margin))
                        good_idxs = idxs[ok_override]
                        best_id[good_idxs] = patch_best_id[ok_override]
                        best_prob[good_idxs] = patch_best_prob[ok_override]

                out_id[s:e] = best_id
                out_prob[s:e] = best_prob

            bead_df[f"cy{i}_{j}"] = out_id
            bead_df[f"cy{i}_{j}{prob_col_suffix}"] = out_prob
    return bead_df


def enforce_single_layer_per_cycle(
    df,
    num_cycles=2,
    num_layers=4,
    min_prob=0.02,
    invalid_value=255,
):
    df = df.copy()
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        prob_cols = [f"cy{i}_{j}_prob" for j in range(num_layers)]
        layers = df[layer_cols].to_numpy(copy=True)
        probs = df[prob_cols].to_numpy(copy=True)
        valid_mask = (
            (layers != invalid_value) & (layers > 0) & (probs >= float(min_prob))
        )
        valid_counts = valid_mask.sum(axis=1)
        enforce_mask = valid_counts == 2
        if not np.any(enforce_mask):
            continue
        masked_probs = probs.copy()
        masked_probs[~valid_mask] = -np.inf
        best_layers = np.argmax(masked_probs, axis=1)
        keep_mask = np.zeros_like(valid_mask, dtype=bool)
        keep_mask[np.arange(len(df)), best_layers] = True
        invalidate_mask = valid_mask & ~keep_mask
        invalidate_mask &= enforce_mask[:, None]
        layers[invalidate_mask] = invalid_value
        probs[invalidate_mask] = 0.0
        df[layer_cols] = layers
        df[prob_cols] = probs
    return df


def resolve_layers_to_cycles(df, num_cycles, num_layers, invalid_value=255):
    final_df = df[["x", "y"]].copy()
    out_cols = [f"cy{i}" for i in range(num_cycles)]
    final_df[out_cols] = np.uint16(invalid_value)
    counts = []
    valid_layer_masks = []
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        labs = df[layer_cols].to_numpy()
        valid = (labs != invalid_value) & (labs > 0)
        valid_layer_masks.append(valid)
        counts.append(valid.sum(axis=1))
    counts = np.stack(counts, axis=1)
    valid_mask = (counts == 1).all(axis=1)
    if valid_mask.any():
        for i in range(num_cycles):
            valid = valid_layer_masks[i]
            layer_idx = valid.argmax(axis=1).astype(np.uint16)
            final_df.loc[valid_mask, f"cy{i}"] = layer_idx[valid_mask]
    return final_df


DEFAULT_ENSEMBLE_RATIO_START = 1.0
DEFAULT_ENSEMBLE_RATIO_END = 1.5
DEFAULT_ENSEMBLE_RATIO_STEP = 0.05
DEFAULT_VORONOI_MIN_ASSIGNED_VALUE = 0.1


def _validate_fluorescence_layers(tif_metadata: List[MetaData]) -> int:
    if len(tif_metadata) == 0:
        raise ValueError("No cycle metadata found.")
    expected_layers = None
    for i, md in enumerate(tif_metadata):
        flors_layers = md.flors_layers
        if flors_layers is None or len(flors_layers) == 0:
            raise ValueError(f"Cycle {i} has no fluorescence layers.")
        if expected_layers is None:
            expected_layers = len(flors_layers)
        elif len(flors_layers) != expected_layers:
            raise ValueError("Cycle fluorescence-layer counts are inconsistent.")
    assert expected_layers is not None
    return expected_layers


def _validate_ensemble_ratio_range(start: float, end: float, step: float):
    if step <= 0:
        raise ValueError("Ensemble ratio step must be > 0.")
    if end < start:
        raise ValueError("Ensemble ratio end must be >= start.")


def _build_ratio_values(start: float, end: float, step: float) -> list[float]:
    _validate_ensemble_ratio_range(start, end, step)
    values = []
    current = float(start)
    epsilon = abs(step) * 1e-9 + 1e-12
    while current <= end + epsilon:
        values.append(round(current, 4))
        current += step
    if len(values) == 0:
        values = [round(start, 4)]
    return values


def _round_xy_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    rounded["x"] = np.rint(rounded["x"]).astype(np.float32)
    rounded["y"] = np.rint(rounded["y"]).astype(np.float32)
    return rounded


def _compute_ensemble_preview_metrics(beads_df: pd.DataFrame) -> dict:
    total = int(len(beads_df))
    if total == 0:
        return {
            "total": 0,
            "valid": 0,
            "valid_pct": 0.0,
            "invalid_count": 0,
            "invalid_pct": 0.0,
            "filtered_count": 0,
            "filtered_pct": 0.0,
            "mean_beads_per_protein": 0.0,
            "invalid_error_percentage": 0.0,
        }
    error_stats = get_error(beads_df)
    invalid_count = int(error_stats.get("invalid_count", 0))
    filtered_count = int(error_stats.get("filtered_count", 0))
    valid_count = max(total - invalid_count - filtered_count, 0)
    filtered_pct = float(
        error_stats.get("filtered_percentage", 100.0 * filtered_count / total)
    )
    return {
        "total": total,
        "valid": valid_count,
        "valid_pct": 100.0 * valid_count / total,
        "invalid_count": invalid_count,
        "invalid_pct": 100.0 * invalid_count / total,
        "filtered_count": filtered_count,
        "filtered_pct": filtered_pct,
        "mean_beads_per_protein": float(error_stats.get("mean_beads_per_protein", 0.0)),
        "invalid_error_percentage": float(
            error_stats.get("invalid_error_percentage", 0.0)
        ),
    }


def _build_voronoi_resolved_from_cache(
    pre_ensemble_beads: pd.DataFrame,
    ensemble_cache: dict,
    ratio: float,
) -> pd.DataFrame:
    resolved = pre_ensemble_beads[["x", "y"]].copy()
    assigned_layers = ensemble_cache["assigned_layers"]
    assigned_values = ensemble_cache["assigned_values"]
    assigned_margins = ensemble_cache.get("assigned_margins", {})
    threshold = float(
        ensemble_cache.get("min_assigned_value", DEFAULT_VORONOI_MIN_ASSIGNED_VALUE)
    )
    surviving = np.ones(len(pre_ensemble_beads), dtype=bool)
    for vals in assigned_values.values():
        surviving &= vals >= threshold
    if ratio > 1.0 and len(assigned_margins) > 0:
        for margins in assigned_margins.values():
            surviving &= margins >= ratio
    for cy_key, layers in assigned_layers.items():
        col = np.full(len(pre_ensemble_beads), 255, dtype=np.uint16)
        if len(col) > 0:
            col[surviving] = layers[surviving].astype(np.uint16)
        resolved[cy_key] = col
    return resolved


def _merge_with_voronoi_candidates(
    stardist_beads: pd.DataFrame,
    voronoi_beads: pd.DataFrame,
) -> pd.DataFrame:
    if stardist_beads is None:
        return pd.DataFrame()
    if stardist_beads.empty:
        return stardist_beads.copy()
    if voronoi_beads is None or voronoi_beads.empty:
        return stardist_beads.copy()
    required = {"x", "y", "cy0", "cy1"}
    if not required.issubset(voronoi_beads.columns):
        return stardist_beads.copy()

    base = _round_xy_for_merge(stardist_beads)
    other = _round_xy_for_merge(voronoi_beads[["x", "y", "cy0", "cy1"]])
    merged = base.merge(other, on=["x", "y"], suffixes=("", "_other"), how="left")
    mask_main_255 = merged["cy0"] == 255
    mask_other_valid = merged["cy0_other"].notna() & (merged["cy0_other"] != 255)
    mask_update = mask_main_255 & mask_other_valid
    if mask_update.any():
        merged.loc[mask_update, "cy0"] = merged.loc[mask_update, "cy0_other"]
        merged.loc[mask_update, "cy1"] = merged.loc[mask_update, "cy1_other"]
    merged = merged.drop(columns=["cy0_other", "cy1_other"], errors="ignore")
    for col in [c for c in merged.columns if c.startswith("cy")]:
        merged[col] = (
            pd.to_numeric(merged[col], errors="coerce").fillna(255).astype(np.uint16)
        )
    merged["x"] = merged["x"].astype(np.float32)
    merged["y"] = merged["y"].astype(np.float32)
    return merged


def build_ensembled_beads_from_cache(
    pre_ensemble_beads: pd.DataFrame,
    ensemble_cache: dict,
    ratio: float,
) -> pd.DataFrame:
    if pre_ensemble_beads is None:
        raise ValueError("Missing pre-ensemble beads for applying ensemble ratio.")
    if ensemble_cache is None:
        raise ValueError("Missing ensemble cache for applying ensemble ratio.")
    voronoi_resolved = _build_voronoi_resolved_from_cache(
        pre_ensemble_beads=pre_ensemble_beads,
        ensemble_cache=ensemble_cache,
        ratio=float(ratio),
    )
    return _merge_with_voronoi_candidates(pre_ensemble_beads, voronoi_resolved)


def compute_ensemble_sweep_stats(
    pre_ensemble_beads: pd.DataFrame,
    ensemble_cache: dict,
    start: float,
    end: float,
    step: float,
    progress_units_callback=None,
) -> pd.DataFrame:
    if pre_ensemble_beads is None:
        raise ValueError("Missing pre-ensemble beads for ensemble sweep.")
    if ensemble_cache is None:
        raise ValueError("Missing ensemble cache for ensemble sweep.")
    ratios = _build_ratio_values(float(start), float(end), float(step))
    rows = []
    total_ratios = len(ratios)
    for index, ratio in enumerate(ratios):
        ensembled_df = build_ensembled_beads_from_cache(
            pre_ensemble_beads=pre_ensemble_beads,
            ensemble_cache=ensemble_cache,
            ratio=ratio,
        )
        metrics = _compute_ensemble_preview_metrics(ensembled_df)
        rows.append(
            {
                "ratio": float(ratio),
                "total": metrics["total"],
                "valid": metrics["valid"],
                "valid_pct": metrics["valid_pct"],
                "invalid_count": metrics["invalid_count"],
                "invalid_pct": metrics["invalid_pct"],
                "filtered_count": metrics["filtered_count"],
                "filtered_pct": metrics["filtered_pct"],
                "mean_beads_per_protein": metrics["mean_beads_per_protein"],
                "invalid_error_percentage": metrics["invalid_error_percentage"],
            }
        )
        if progress_units_callback:
            progress_units_callback("voronoi_sweep", index + 1, total_ratios)
    return pd.DataFrame(rows)


def _compute_voronoi_ensemble_cache(
    bead_df: pd.DataFrame,
    cycles: list[np.ndarray],
    metadata_list: List[MetaData],
    max_size: int,
    min_assigned_value: float = DEFAULT_VORONOI_MIN_ASSIGNED_VALUE,
    border_erosion: int = 0,
    progress_units_callback=None,
) -> dict:
    n_beads = len(bead_df)
    num_cycles = len(cycles)
    if n_beads == 0:
        if progress_units_callback:
            progress_units_callback("voronoi_cache", 1, 1)
        return {
            "assigned_layers": {
                f"cy{i}": np.array([], dtype=np.uint8) for i in range(num_cycles)
            },
            "assigned_values": {
                f"cy{i}": np.array([], dtype=np.float32) for i in range(num_cycles)
            },
            "assigned_margins": {
                f"cy{i}": np.array([], dtype=np.float32) for i in range(num_cycles)
            },
            "min_assigned_value": float(min_assigned_value),
            "num_cycles": num_cycles,
        }

    total_layer_units = 0
    for metadata in metadata_list:
        if metadata.flors_layers is not None:
            total_layer_units += len(metadata.flors_layers)
    total_units = 1 + total_layer_units
    done_units = 0

    shape = (int(max_size), int(max_size))
    xy = bead_df[["x", "y"]].to_numpy(dtype=np.float64)
    vor_lbl = voronoi_from_centers_tiled(
        xy,
        shape,
        tile=2048,
        show_progress=False,
    )
    done_units += 1
    if progress_units_callback:
        progress_units_callback("voronoi_cache", done_units, total_units)
    h0 = min(int(max_size), vor_lbl.shape[0])
    w0 = min(int(max_size), vor_lbl.shape[1])
    vor = vor_lbl[:h0, :w0]
    if border_erosion > 0:
        boundary = np.zeros(vor.shape, dtype=bool)
        boundary[:-1, :] |= vor[:-1, :] != vor[1:, :]
        boundary[1:, :] |= vor[:-1, :] != vor[1:, :]
        boundary[:, :-1] |= vor[:, :-1] != vor[:, 1:]
        boundary[:, 1:] |= vor[:, :-1] != vor[:, 1:]
        dist = ndi.distance_transform_edt(~boundary)
        vor = vor.copy()
        vor[dist < int(border_erosion)] = -1

    raw_cols: Dict[str, np.ndarray] = {}
    col_names: list[str] = []
    for ci, (cycle, md) in enumerate(zip(cycles, metadata_list)):
        h = min(int(max_size), cycle.shape[-2])
        w = min(int(max_size), cycle.shape[-1])
        v = vor[:h, :w]
        rf = v.ravel()
        flors_layers = md.flors_layers
        if flors_layers is None or len(flors_layers) == 0:
            raise ValueError(
                f"Cycle {ci} has no fluorescence layers for Voronoi ensemble."
            )
        for li, ch_idx in enumerate(flors_layers):
            col = f"cy{ci}_{li}"
            col_names.append(col)
            img = cycle[ch_idx, :h, :w].astype(np.float32)
            raw_cols[col] = _median_per_region(img.ravel(), rf, n_beads)
            done_units += 1
            if progress_units_callback:
                progress_units_callback("voronoi_cache", done_units, total_units)

    norm_cols: Dict[str, np.ndarray] = {}
    for col, vals in raw_cols.items():
        lo = float(vals.min()) if len(vals) > 0 else 0.0
        hi = float(vals.max()) if len(vals) > 0 else 0.0
        if hi > lo:
            norm_cols[col] = (vals - lo) / (hi - lo + 1e-8)
        else:
            norm_cols[col] = np.zeros_like(vals, dtype=np.float32)

    assigned_layers: Dict[str, np.ndarray] = {}
    assigned_values: Dict[str, np.ndarray] = {}
    assigned_margins: Dict[str, np.ndarray] = {}
    for ci in range(num_cycles):
        cy_cols = [c for c in col_names if c.startswith(f"cy{ci}_")]
        if len(cy_cols) == 0:
            assigned_layers[f"cy{ci}"] = np.zeros(n_beads, dtype=np.uint8)
            assigned_values[f"cy{ci}"] = np.zeros(n_beads, dtype=np.float32)
            assigned_margins[f"cy{ci}"] = np.full(n_beads, np.inf, dtype=np.float32)
            continue
        mat = np.stack([norm_cols[c] for c in cy_cols], axis=1)
        best_layer = np.argmax(mat, axis=1).astype(np.uint8)
        best_val = mat[np.arange(n_beads), best_layer].astype(np.float32)
        assigned_layers[f"cy{ci}"] = best_layer
        assigned_values[f"cy{ci}"] = best_val
        if mat.shape[1] >= 2:
            sorted_mat = np.sort(mat, axis=1)
            second_best = sorted_mat[:, -2]
            margin_ratio = best_val / (second_best + 1e-8)
            assigned_margins[f"cy{ci}"] = margin_ratio.astype(np.float32)
        else:
            assigned_margins[f"cy{ci}"] = np.full(n_beads, np.inf, dtype=np.float32)

    return {
        "assigned_layers": assigned_layers,
        "assigned_values": assigned_values,
        "assigned_margins": assigned_margins,
        "min_assigned_value": float(min_assigned_value),
        "num_cycles": num_cycles,
    }


def _process_beads_notebook_stardist(
    brightfield,
    tifs,
    max_size,
    model_name,
    update_progress,
    is_running,
    progress_units_callback=None,
    stardist_use_guess_tiles=True,
    stardist_n_tiles=1,
    ensemble_ratio_start=DEFAULT_ENSEMBLE_RATIO_START,
    ensemble_ratio_end=DEFAULT_ENSEMBLE_RATIO_END,
    ensemble_ratio_step=DEFAULT_ENSEMBLE_RATIO_STEP,
):
    bead_detection_scale = 2
    stardist_prob_thresh = 0.15
    stardist_nms_thresh = 0.1
    if stardist_use_guess_tiles:
        stardist_n_tiles_value = 0
    else:
        stardist_n_tiles_value = max(int(stardist_n_tiles), 1)
    stardist_block_size = 700 if max_size <= 2000 else 2000
    bead_detection_block_size = 700 if max_size <= 2000 else 2000
    initial_detection_progress_message = lambda msg: update_progress(10, msg)
    activation_regions_progress_message = lambda msg: update_progress(30, msg)

    update_progress(10, "Initial bead detection...")
    bf = brightfield[:max_size, :max_size]
    beads = beadfinding_notebook_stardist(
        bf,
        scale=bead_detection_scale,
        block_size=bead_detection_block_size,
        n_tiles=stardist_n_tiles_value,
        progress_units_callback=progress_units_callback,
        progress_callback=initial_detection_progress_message,
    )
    beads = np.unique(beads, axis=0)
    bead_df = pd.DataFrame(beads[:, :2], columns=["x", "y"], dtype=np.float32)
    bead_df = bead_df.query("10 <= x < @max_size-10 and 10 <= y < @max_size-10")
    if not is_running():
        return None

    update_progress(30, "Getting activation regions from cycles")
    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = [np.ascontiguousarray(img)[:, :max_size, :max_size] for img, _ in tifs]
    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)
        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j > int(md.reference_channel)
        ]
    num_cycles = len(tif_metadata)
    num_layers = _validate_fluorescence_layers(tif_metadata)
    model_dir = _resolve_model_dir(model_name)
    custom_model = load_custom_model(model_dir)
    cycle_labels_kwargs = {
        "block_size": stardist_block_size,
        "prob_thresh": stardist_prob_thresh,
        "nms_thresh": stardist_nms_thresh,
        "n_tiles": stardist_n_tiles_value,
    }
    if progress_units_callback:
        cycle_labels_kwargs["progress_units_callback"] = progress_units_callback
    cycle_labels_kwargs["progress_callback"] = activation_regions_progress_message
    cycle_labels = get_labels_from_cycles_with_prob(
        tif_images,
        tif_metadata,
        max_size,
        custom_model,
        **cycle_labels_kwargs,
    )
    if not is_running():
        return None

    update_progress(60, "Assigning beads labels")
    bead_df = assign_beads_labels_with_prob_patch3x3_fallback(
        bead_df,
        cycle_labels,
        invalid_value=0,
        min_patch_prob=None,
        center_vs_patch_margin=0.01,
        min_center_prob=0.35,
        min_patch_support=21,
        min_prob_margin=0,
    )
    bead_df = enforce_single_layer_per_cycle(
        bead_df,
        num_cycles=num_cycles,
        num_layers=num_layers,
        min_prob=0,
        invalid_value=0,
    )
    pre_ensemble_beads = resolve_layers_to_cycles(
        bead_df,
        num_cycles=num_cycles,
        num_layers=num_layers,
    )

    update_progress(72, "Running Voronoi ensemble analysis")
    if not is_running():
        return None

    update_progress(76, "Voronoi median: building ensemble cache")
    ensemble_cache_kwargs = {
        "bead_df": pre_ensemble_beads,
        "cycles": tif_images,
        "metadata_list": tif_metadata,
        "max_size": max_size,
        "min_assigned_value": DEFAULT_VORONOI_MIN_ASSIGNED_VALUE,
    }
    if progress_units_callback:
        ensemble_cache_kwargs["progress_units_callback"] = progress_units_callback
    ensemble_cache = _compute_voronoi_ensemble_cache(
        **ensemble_cache_kwargs,
    )
    if not is_running():
        return None

    update_progress(84, "Voronoi median: sweeping ensemble ratios")
    ensemble_sweep_kwargs = {
        "pre_ensemble_beads": pre_ensemble_beads,
        "ensemble_cache": ensemble_cache,
        "start": float(ensemble_ratio_start),
        "end": float(ensemble_ratio_end),
        "step": float(ensemble_ratio_step),
    }
    if progress_units_callback:
        ensemble_sweep_kwargs["progress_units_callback"] = progress_units_callback
    ensemble_sweep_stats = compute_ensemble_sweep_stats(**ensemble_sweep_kwargs)
    if not is_running():
        return None

    ratio_applied = float(ensemble_ratio_start)
    if not ensemble_sweep_stats.empty:
        ratio_applied = float(ensemble_sweep_stats.iloc[0]["ratio"])
    update_progress(90, "Voronoi median: applying selected ratio")
    results_df = build_ensembled_beads_from_cache(
        pre_ensemble_beads=pre_ensemble_beads,
        ensemble_cache=ensemble_cache,
        ratio=ratio_applied,
    )

    cycles = {}
    for i in range(len(tifs)):
        cycles[f"cy{i}"] = tifs[i][0]
    labeled_image = np.zeros(bf.shape, dtype=np.uint16)
    return {
        "beads": results_df,
        "post_resolution_beads": results_df,
        "pre_ensemble_beads": pre_ensemble_beads,
        "ensemble_cache": ensemble_cache,
        "ensemble_sweep_stats": ensemble_sweep_stats,
        "ensemble_ratio_applied": ratio_applied,
        "cycles": cycles,
        "labeled_image": labeled_image,
    }


def prepare_data_for_resolution(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    layer_threshold_dict=defaultdict(int),
    progress_callback=None,
    is_running_callback=None,
    roi_coords=None,
    n_workers=10,
    radius=2,
    low_percentile=50,
    high_percentile=65,
    adaptive_thresholding=False,
    tile_size=100,
    border_erode=1,
    fill=True,
    use_stardist=False,  # Toggle between watershed and StarDist
    model_name="model_5_400epoch",  # Model path for StarDist
    normalize_method="none",  # Exposure normalization method
    progress_units_callback=None,
):
    """

    Prepares data for bead assignment resolution.

    Returns all necessary data structures for testing different resolution methods.

    """

    def update_progress(value, message):
        if progress_callback:
            overall_progress = 40 + (value / 100) * 50

            progress_callback(int(overall_progress), message)

        log(message)

    def is_running():
        return is_running_callback() if is_running_callback else True

    ColorThreshold = signal_to_noise_cutoff
    export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

    tif_metadata = [f.metadata for _, f in tifs]

    tif_images = [np.array(img) for img, _ in tifs]
    resized_tif_images = []
    for img in tif_images:
        if img.shape[-2:] != (
            max_size,
            max_size,
        ):  # Check last 2 dimensions (height, width)
            # Resize each channel separately if it's a multi-channel image
            if img.ndim == 3:
                resized_channels = []
                for channel in img:
                    resized_channel = cv2.resize(
                        channel, (max_size, max_size), interpolation=cv2.INTER_LINEAR
                    )
                    resized_channels.append(resized_channel)
                resized_img = np.stack(resized_channels, axis=0)
            else:  # 2D image
                resized_img = cv2.resize(
                    img, (max_size, max_size), interpolation=cv2.INTER_LINEAR
                )
            resized_tif_images.append(resized_img)
        else:
            resized_tif_images.append(img)

    tif_images = resized_tif_images

    # Setup flors_layers for each metadata object

    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)

        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j > int(md.reference_channel)
        ]

    total_beads = len(beads)

    total_cycles = len(tif_metadata)

    total_layers = len(tif_metadata[0].flors_layers) if total_cycles > 0 else 0

    update_progress(10, "Getting activation regions from cycles")
    # Conditionally load StarDist model
    if use_stardist:
        model = StarDist2D(
            None, name=model_name, basedir=resource_path("assets/" + model_name)
        )
    else:
        model = None  # Not needed for watershed

    label_kwargs = {
        "fill": fill,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "adaptive_thresholding": adaptive_thresholding,
        "tile_size": tile_size,
        "border_erode": border_erode,
        "use_stardist": use_stardist,
        "model": model,
        "normalize_method": normalize_method,
    }
    if progress_units_callback:
        label_kwargs["progress_units_callback"] = progress_units_callback
    labels = get_labels_from_cycles(
        tif_images,
        tif_metadata,
        max_size,
        **label_kwargs,
    )

    update_progress(50, "Assigning beads labels")

    df = assign_beads_labels(beads, labels)

    cycle_layer_columns = {}

    for i in range(total_cycles):
        cols = []

        for j in range(total_layers):
            cycle_layer_columns[(i, j)] = f"cy{i}_{j}"

            cols.append(cycle_layer_columns[(i, j)])

        df[f"cy{i}_count"] = (df[cols] > 0).sum(axis=1)

    count_cols = [f"cy{i}_count" for i in range(total_cycles)]

    both_assignment = (df[count_cols] == 1).all(axis=1)

    no_assignment = (df[count_cols] == 0).any(axis=1)

    counts = {
        "no_assignment": no_assignment.sum(),
        "both_single": both_assignment.sum(),
    }

    print(counts)

    # Zero-pad cycles for patch extraction

    pad = 2

    # tif_images_padded = np.pad(np.array(tif_images), ((0,), (0,), (2,), (2,)))

    columns = [f"cy{i}" for i in range(len(tif_images))]

    # Prepare final_df structure

    final_df = df[["x", "y"]].copy()

    final_df[columns] = 255  # Initialize with default value

    num_flor_layers = len(tif_metadata[0].flors_layers)

    # 2. Filter the source DataFrame once to work on the relevant subset

    subset_df = df.loc[both_assignment]

    update_progress(80, "Processing single assignments")

    # 3. Loop through cycles (this loop is short and efficient)

    for i in range(total_cycles):
        # Define the prefix for the current cycle (e.g., 'cy0')

        col_prefix = f"cy{i}"

        # List the related data columns for this cycle (e.g., ['cy0_0', 'cy0_1', ...])

        related_cols = [f"{col_prefix}_{j}" for j in range(num_flor_layers)]

        # Use idxmax() to find the column with the max value for ALL rows at once

        # This is the core vectorized operation that replaces get_assignment for each row

        assignments = subset_df[related_cols].idxmax(axis=1)

        # Extract the layer number from the column name (e.g., 'cy0_2' -> '2')

        final_values = assignments.str.split("_").str[-1].astype(np.uint8)

        # Assign the entire series of results to the final DataFrame

        final_df.loc[both_assignment, col_prefix] = final_values

    single_distr = (
        final_df.loc[both_assignment].groupby(columns).size().reset_index(name="count")
    )

    # print(single_distr.head(30))

    # Return all data needed for resolution methods

    return {
        "df": df,
        "final_df": final_df,
        "tif_metadata": tif_metadata,
        "ColorThreshold": ColorThreshold,
        "total_cycles": total_cycles,
        "both_assignment": both_assignment,
        "no_assignment": no_assignment,
        "columns": columns,
        "pad": pad,
    }


def _create_patch_indices(coords, patch_radius, img_shape):
    """Helper to create broadcastable indices for patch extraction for all coordinates."""
    # Coords is an (N, 2) array of (x, y)
    # Create a grid of offsets, e.g., [-1, 0, 1] for radius 1
    d = np.arange(-patch_radius, patch_radius + 1)
    # Shape: (1, 2*radius+1, 1) and (1, 1, 2*radius+1) for broadcasting
    dx, dy = d[np.newaxis, :, np.newaxis], d[np.newaxis, np.newaxis, :]

    # Add offsets to each coordinate
    # Coords shape: (N, 1, 1) for broadcasting with dx, dy
    x_indices = coords[:, 0, np.newaxis, np.newaxis] + dx
    y_indices = coords[:, 1, np.newaxis, np.newaxis] + dy

    # Ensure indices are within image bounds
    np.clip(x_indices, 0, img_shape[1] - 1, out=x_indices)
    np.clip(y_indices, 0, img_shape[0] - 1, out=y_indices)

    return y_indices.astype(int), x_indices.astype(
        int
    )  # Return as (row, col) for NumPy indexing


def combine_results(data_dict, cycle_values, correction_indices):
    """
    Combine the resolution results with the prepared data to get final_df.
    """
    final_df = data_dict["final_df"].copy()
    columns = data_dict["columns"]

    if len(cycle_values) > 0:
        final_df.loc[correction_indices, columns] = cycle_values

    return final_df


def _medianroi_method_optimized(
    x_coords,
    y_coords,
    tif_images_padded,
    tif_metadata,
    ColorThreshold,
    correction_indices,
    background_method="local_ring",
):
    """
    Vectorized method using median intensity in a 3x3 ROI with background-dependent filtering.
    """
    num_beads = len(x_coords)
    num_cycles = len(tif_images_padded)
    coords = np.array([x_coords, y_coords]).T  # Shape: (num_beads, 2)

    img_height, img_width = tif_images_padded.shape[-2:]

    # 1. Pre-calculate all indices for ROI and Background patches
    patch_radius = 4
    roi_rows, roi_cols = _create_patch_indices(
        coords, patch_radius, (img_height, img_width)
    )  # 3x3 ROI patch
    if background_method in ["psnr"]:
        bg_radius = patch_radius + 1
    elif background_method in ["local_ring", "adaptive_snr"]:
        bg_radius = 3  # 7x7 patch
    else:  # local_patch or percentile_diff
        bg_radius = 4  # 9x9 patch

    bg_rows, bg_cols = _create_patch_indices(coords, bg_radius, (img_height, img_width))

    # Create a boolean mask to exclude the center ROI from the background patch
    # This effectively creates a "ring" of pixels for the background calculation
    if background_method in ["local_ring", "local_patch"]:
        mask_size = 2 * bg_radius + 1
        center_start = bg_radius - 1  # Center 3x3 starts at radius-1
        center_end = bg_radius + 2  # and ends at radius+2
        bg_mask = np.ones((mask_size, mask_size), dtype=bool)
        bg_mask[center_start:center_end, center_start:center_end] = False

    # Initialize the final results array with the "all filtered" value
    cycle_values = np.full((num_beads, num_cycles), 255, dtype=np.uint8)

    # Main loop over cycles
    for cycle_idx in tqdm(range(num_cycles), desc="Processing cycles"):
        num_channels = tif_images_padded[cycle_idx].shape[0]
        all_scores = np.zeros((num_beads, num_channels), dtype=np.float32)

        # Inner loop over fluorescence channels
        for layer_idx in range(num_channels):
            img_layer = tif_images_padded[cycle_idx][layer_idx]

            # 2. Vectorized data extraction for ALL beads at once
            roi_patches = img_layer[roi_rows, roi_cols]
            signal_medians = np.median(roi_patches, axis=(1, 2))

            # bg_patches = img_layer[bg_rows, bg_cols]
            bg_patches = img_layer[bg_rows, bg_cols].astype(np.float32)

            # 3. Vectorized score calculation for ALL beads
            scores = np.zeros(num_beads, dtype=np.float32)
            # Replace zeros with NaN to ignore them in calculations
            # bg_patches[bg_patches == 0] = np.nan
            # Suppress RuntimeWarning for patches that are all NaNs
            with np.errstate(all="ignore"):
                if (
                    background_method == "local_ring"
                    or background_method == "local_patch"
                ):
                    valid_bg = bg_patches[:, bg_mask]
                    background_medians = np.nanmedian(valid_bg, axis=1)
                    background_medians = np.nan_to_num(background_medians)
                    scores = signal_medians / (background_medians + 1e-8)
                elif background_method == "psnr":
                    valid_bg = bg_patches
                    background_percentile = np.percentile(valid_bg, 50, axis=(1, 2))
                    scores = (signal_medians - background_percentile) / (
                        background_percentile + 1e-8
                    )
                elif background_method == "percentile_diff":
                    # Reshape from (N, 9, 9) to (N, 81) for percentile calculation
                    bg_flat = bg_patches.reshape(num_beads, -1)
                    background_75th = np.nanpercentile(bg_flat, 75, axis=1)
                    background_75th = np.nan_to_num(background_75th)
                    scores = signal_medians - background_75th

            all_scores[:, layer_idx] = scores

        # 4. Vectorized Filtering and Argmax
        if background_method in ["local_ring", "local_patch"]:
            threshold = 1.0 + ColorThreshold
        else:  # adaptive_snr, percentile_diff
            threshold = ColorThreshold

        passed_mask = all_scores > threshold
        any_passed = np.any(passed_mask, axis=1)
        num_passed = sum(passed_mask)
        print(f"corrected :{num_passed}")

        if np.any(any_passed):
            # Set scores of failing channels to -inf to exclude them from argmax
            scores_for_argmax = np.where(passed_mask, all_scores, -np.inf)
            winner_indices = np.argmax(scores_for_argmax, axis=1)

            # Update only the beads where at least one channel passed the filter
            cycle_values[any_passed, cycle_idx] = winner_indices[any_passed]

    return cycle_values, correction_indices


import itertools


def optimize_parameters(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    param_grid,
    progress_units_callback=None,
):
    """
    Iteratively runs bead processing with different parameters to find the optimal set.
    """
    if tqdm is None:
        raise ImportError(
            "tqdm is required for the optimization progress bar. "
            "Please install it using 'pip install tqdm'"
        )

    # Create all parameter combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    param_combinations = [
        dict(zip(param_keys, v)) for v in itertools.product(*param_values)
    ]

    # Filter combinations where high_percentile is not greater than low_percentile
    valid_combinations = [
        p
        for p in param_combinations
        if p.get("high_percentile", 0) > p.get("low_percentile", -1)
    ]

    results = []
    print(
        f"\nStarting optimization for {len(valid_combinations)} parameter combinations..."
    )

    total_combinations = len(valid_combinations)
    for index, params in enumerate(
        tqdm(valid_combinations, desc="Optimizing Parameters"), start=1
    ):
        try:
            # Run the main processing function with the current set of parameters
            data_dict = prepare_data_for_resolution(
                beads,
                signal_to_noise_cutoff,
                tifs,
                max_size,
                adaptive_thresholding=True,
                tile_size=1000,
                **params,
            )

            # Run the analysis function (in automatic mode, without a protein profile)
            error_metrics = get_error(data_dict["final_df"])

            # Store the parameters and their results
            results.append(
                {
                    **params,
                    "error_percent": error_metrics["invalid_error_percentage"],
                    "filtered_count": error_metrics["filtered_count"],
                }
            )

        except Exception as e:
            print(f"Error with params {params}: {e}")
            results.append(
                {**params, "error_percent": np.nan, "filtered_count": np.nan}
            )
        if progress_units_callback:
            progress_units_callback("optimize_params", index, total_combinations)

    # Analyze and display best results
    if not results:
        print("Optimization did not produce any results.")
        return

    results_df = pd.DataFrame(results).dropna()
    if results_df.empty:
        print("All parameter combinations resulted in errors.")
        return

    # Normalize the two metrics to a 0-1 scale (lower is better)
    results_df["error_norm"] = (
        results_df["error_percent"] - results_df["error_percent"].min()
    ) / (results_df["error_percent"].max() - results_df["error_percent"].min())
    results_df["filtered_norm"] = (
        results_df["filtered_count"] - results_df["filtered_count"].min()
    ) / (results_df["filtered_count"].max() - results_df["filtered_count"].min())
    results_df["score"] = (results_df["error_norm"] + results_df["filtered_norm"]) / 2
    results_df_sorted = results_df.sort_values("score", ascending=True)

    print("\n--- ✅ Optimization Complete ---")
    print("\nTop 5 Parameter Sets (Lower Score is Better):")
    print(
        results_df_sorted[
            [
                "border_erode",
                "low_percentile",
                "high_percentile",
                "error_percent",
                "filtered_count",
                "score",
            ]
        ]
        .head(5)
        .round(2)
    )

    best_params = results_df_sorted.iloc[0]
    print("\n--- 🏆 Best Configuration Found ---")
    print(f"  Border Erode:    {int(best_params['border_erode'])}")
    print(f"  Low Percentile:  {int(best_params['low_percentile'])}")
    print(f"  High Percentile: {int(best_params['high_percentile'])}")
    print("------------------------------------")
    print(f"  Resulting Error:       {best_params['error_percent']:.2f}%")
    print(f"  Resulting Filtered:    {int(best_params['filtered_count'])}")
    print(f"  Combined Score:        {best_params['score']:.3f}")
    return best_params.to_dict()


def get_excel(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    layer_threshold_dict=defaultdict(int),
    progress_callback=None,
    is_running_callback=None,
    roi_coords=None,
    n_workers=10,
    radius=2,
    use_stardist=False,  # Toggle between watershed and StarDist
    model_name="model_5_400epoch",  # Model path for StarDist
    progress_units_callback=None,
):
    def update_progress(value, message):
        if progress_callback:
            overall_progress = 40 + (value / 100) * 50
            progress_callback(int(overall_progress), message)

    def is_running():
        return is_running_callback() if is_running_callback else True

    ColorThreshold = signal_to_noise_cutoff
    export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = [np.ascontiguousarray(img)[:, :max_size, :max_size] for img, _ in tifs]

    # Setup flors_layers for each metadata object
    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)
        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j > int(md.reference_channel)
        ]

    total_cycles = len(tif_metadata)
    beads = pd.DataFrame(beads[:, :2], columns=["x", "y"], dtype=np.float32)

    # Only optimize parameters for watershed
    if not use_stardist:
        # find optimal parameters
        update_progress(10, "Finding optimal parameters...")
        param_grid = {
            "border_erode": [0, 1, 2],
            "low_percentile": [45, 50, 55, 60],
            "high_percentile": [65, 70, 75, 80],
        }
        sample_size = 1000
        cropped_tifs = [
            (np.ascontiguousarray(img[:, :sample_size, :sample_size]), meta)
            for img, meta in tifs
        ]
        bead_positions = beads.query(f"x < {sample_size} and y < {sample_size}")
        optimization_kwargs = {}
        if progress_units_callback:
            optimization_kwargs["progress_units_callback"] = progress_units_callback
        optimization_results = optimize_parameters(
            bead_positions,
            signal_to_noise_cutoff,
            cropped_tifs,
            sample_size,
            param_grid,
            **optimization_kwargs,
        )
        if optimization_results is None:
            log("Optimization failed, using default parameters.")
            optimization_results = {}
        border_erode = int(optimization_results.get("border_erode", 1))
        low_percentile = int(optimization_results.get("low_percentile", 50))
        high_percentile = int(optimization_results.get("high_percentile", 70))
        update_progress(30, "Getting activation regions from cycles")
    else:
        # StarDist doesn't need parameter optimization
        update_progress(10, "Loading StarDist model...")
        border_erode = 1  # Unused but keep for consistency
        low_percentile = 50  # Unused but keep for consistency
        high_percentile = 70  # Unused but keep for consistency
        update_progress(30, "Getting activation regions from cycles")

    prepare_kwargs = {
        "border_erode": border_erode,
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "use_stardist": use_stardist,
        "model_name": model_name,
        "progress_callback": update_progress,
    }
    if progress_units_callback:
        prepare_kwargs["progress_units_callback"] = progress_units_callback
    data_dict = prepare_data_for_resolution(
        beads,
        signal_to_noise_cutoff,
        tifs,
        max_size,
        **prepare_kwargs,
    )
    final_df = data_dict["final_df"]
    tif_metadata = data_dict["tif_metadata"]
    ColorThreshold = data_dict["ColorThreshold"]
    total_cycles = data_dict["total_cycles"]
    both_assignment = data_dict["both_assignment"]
    no_assignment = data_dict["no_assignment"]
    update_progress(50, "Assigning beads labels")
    df = data_dict["df"]

    count_cols = [f"cy{i}_count" for i in range(total_cycles)]

    both_assignment = (df[count_cols] == 1).all(axis=1)
    no_assignment = (df[count_cols] == 0).any(axis=1)

    counts = {
        "no_assignment": no_assignment.sum(),
        "both_single": both_assignment.sum(),
    }
    print(counts)
    # tif_images = np.pad(np.array(tif_images), ((0,), (0,), (2,), (2,)))
    columns = [f"cy{i}" for i in range(len(tif_images))]

    pad = data_dict["pad"]

    # Identify beads that need correction
    need_correction = df[~both_assignment]

    print(f"Beads needing correction: {len(need_correction)}")
    update_progress(70, "Resolving Bead Labels")

    print("Using median ROI method for correction")
    # Zero-pad cycles for patch extraction
    ColorThreshold = 0.1
    tif_images_padded = np.pad(np.array(tif_images), ((0,), (0,), (pad,), (pad,)))
    x_coords = np.rint(need_correction["x"].to_numpy() + pad).astype(np.int32)
    y_coords = np.rint(need_correction["y"].to_numpy() + pad).astype(np.int32)
    cycle_values, correction_indices = _medianroi_method_optimized(
        x_coords,
        y_coords,
        tif_images_padded[:, 1:, :, :],
        tif_metadata,
        ColorThreshold,
        need_correction.index,
        background_method="psnr",
    )
    new_dd = combine_results(data_dict, cycle_values, correction_indices)
    ff = {"final_df": new_dd, "columns": data_dict["columns"]}
    post_resolution_final_df = ff["final_df"]

    return final_df, post_resolution_final_df


import gc


def process_beads(
    brightfield,
    tifs,
    max_size,
    signal_to_noise_cutoff,
    progress_callback=None,
    is_running_callback=None,
    use_stardist=False,  # Toggle between watershed and StarDist
    model_name="model_5_400epoch",  # Model path for StarDist
    stardist_use_guess_tiles=True,
    stardist_n_tiles=1,
    progress_units_callback=None,
    ensemble_ratio_start=DEFAULT_ENSEMBLE_RATIO_START,
    ensemble_ratio_end=DEFAULT_ENSEMBLE_RATIO_END,
    ensemble_ratio_step=DEFAULT_ENSEMBLE_RATIO_STEP,
):
    trace_started_at = datetime.now().isoformat()
    trace_start = time.perf_counter()
    trace_status = "running"
    trace_error = None
    trace_events = []
    trace_counts = {}
    trace_path = os.path.abspath("t.json")

    def record_event(value, message):
        now = time.perf_counter()
        event = {
            "progress": int(value) if isinstance(value, (int, np.integer)) else value,
            "message": str(message),
            "offset_s": round(now - trace_start, 6),
        }
        if trace_events:
            prev = trace_events[-1]
            prev["duration_s"] = round(event["offset_s"] - prev["offset_s"], 6)
        trace_events.append(event)

    def finalize_trace():
        total_elapsed = round(time.perf_counter() - trace_start, 6)
        if trace_events:
            trace_events[-1]["duration_s"] = round(
                total_elapsed - trace_events[-1]["offset_s"], 6
            )

        ordered_messages = []
        durations_by_message = {}
        for event in trace_events:
            message = event["message"]
            if message not in durations_by_message:
                durations_by_message[message] = 0.0
                ordered_messages.append(message)
            durations_by_message[message] += float(event.get("duration_s", 0.0))

        step_durations = [
            {
                "message": message,
                "duration_s": round(durations_by_message[message], 6),
            }
            for message in ordered_messages
        ]

        trace_payload = {
            "started_at": trace_started_at,
            "ended_at": datetime.now().isoformat(),
            "total_duration_s": total_elapsed,
            "status": trace_status,
            "error": trace_error,
            "use_stardist": bool(use_stardist),
            "model_name": model_name,
            "stardist_use_guess_tiles": bool(stardist_use_guess_tiles),
            "stardist_n_tiles": int(stardist_n_tiles),
            "max_size": int(max_size),
            "signal_to_noise_cutoff": float(signal_to_noise_cutoff),
            "ensemble_ratio_start": float(ensemble_ratio_start),
            "ensemble_ratio_end": float(ensemble_ratio_end),
            "ensemble_ratio_step": float(ensemble_ratio_step),
            "counts": trace_counts,
            "events": trace_events,
            "step_durations": step_durations,
        }

        try:
            with open(trace_path, "w", encoding="utf-8") as trace_file:
                json.dump(trace_payload, trace_file, indent=2)
        except Exception:
            pass

    def update_progress(value, message):
        record_event(value, message)
        if progress_callback:
            progress_callback(value, message)

    def is_running():
        if is_running_callback:
            return is_running_callback()
        return True

    try:
        update_progress(0, "Preprocessing brightfield image...")
        if not is_running():
            trace_status = "cancelled"
            return None
        log(f"Preprocessed to shape: {brightfield.shape}")

        if use_stardist:
            try:
                notebook_kwargs = {
                    "brightfield": brightfield,
                    "tifs": tifs,
                    "max_size": max_size,
                    "model_name": model_name,
                    "update_progress": update_progress,
                    "is_running": is_running,
                    "stardist_use_guess_tiles": stardist_use_guess_tiles,
                    "stardist_n_tiles": stardist_n_tiles,
                    "ensemble_ratio_start": ensemble_ratio_start,
                    "ensemble_ratio_end": ensemble_ratio_end,
                    "ensemble_ratio_step": ensemble_ratio_step,
                }
                if progress_units_callback:
                    notebook_kwargs["progress_units_callback"] = progress_units_callback
                notebook_results = _process_beads_notebook_stardist(**notebook_kwargs)
            except Exception as e:
                log(f"Error during notebook StarDist bead processing: {e}")
                trace_status = "error"
                trace_error = str(e)
                update_progress(-1, f"Error during bead processing., {e}")
                return None
            if notebook_results is None:
                trace_status = "cancelled"
                return None
            beads_df = notebook_results.get("beads")
            if isinstance(beads_df, pd.DataFrame):
                trace_counts["beads"] = int(len(beads_df))
            pre_ensemble_df = notebook_results.get("pre_ensemble_beads")
            if isinstance(pre_ensemble_df, pd.DataFrame):
                trace_counts["pre_ensemble_beads"] = int(len(pre_ensemble_df))
            sweep_df = notebook_results.get("ensemble_sweep_stats")
            if isinstance(sweep_df, pd.DataFrame):
                trace_counts["ensemble_sweep_rows"] = int(len(sweep_df))
            ratio_applied = notebook_results.get("ensemble_ratio_applied")
            if ratio_applied is not None:
                trace_counts["ensemble_ratio_applied"] = float(ratio_applied)
            update_progress(95, "Bead generation complete.")
            trace_status = "success"
            return notebook_results

        update_progress(10, "Initial bead detection...")
        if not is_running():
            trace_status = "cancelled"
            return None
        brightfield = brightfield[:max_size, :max_size]
        log("Initial bead detection...")
        try:
            beads = beadfinding(brightfield, is_running_callback=is_running)
        except Exception as e:
            log(f"Error during beadfinding: {e}")
            trace_status = "error"
            trace_error = str(e)
            return None
        if beads is None:
            trace_status = "cancelled"
            return None
        initial_bead_count = len(beads)
        trace_counts["initial_beads"] = int(initial_bead_count)
        log(f"Initial bead detection found {initial_bead_count} beads")
        update_progress(20, "Removing duplicate beads...")
        if not is_running():
            trace_status = "cancelled"
            return None
        beads = np.unique(beads, axis=0)
        unique_bead_count = len(beads)
        trace_counts["unique_beads"] = int(unique_bead_count)
        log(
            f"After deduplication: {unique_bead_count} unique beads (removed {initial_bead_count - unique_bead_count} duplicates)"
        )
        update_progress(30, "Performing second pass bead detection...")
        if not is_running():
            trace_status = "cancelled"
            return None
        final_bead_count = len(beads)
        trace_counts["final_detected_beads"] = int(final_bead_count)
        log(f"Total beads detected from brightfield layer: {final_bead_count}")
        update_progress(40, "Calculating signal-to-noise ratios...")
        if not is_running():
            trace_status = "cancelled"
            return None
        log(f"Signal-to-noise cutoff: {signal_to_noise_cutoff}")
        gc.collect()
        try:
            excel_kwargs = {
                "progress_callback": update_progress,
                "is_running_callback": is_running,
                "use_stardist": False,
                "model_name": model_name,
            }
            if progress_units_callback:
                excel_kwargs["progress_units_callback"] = progress_units_callback
            df, post_resoluton_df = get_excel(
                beads,
                signal_to_noise_cutoff,
                tifs,
                max_size,
                **excel_kwargs,
            )
        except Exception as e:
            log(f"Error during get_excel: {e}")
            trace_status = "error"
            trace_error = str(e)
            update_progress(-1, f"Error during bead processing., {e}")
            return None
        if isinstance(df, pd.DataFrame):
            trace_counts["pre_resolution_rows"] = int(len(df))
        if isinstance(post_resoluton_df, pd.DataFrame):
            trace_counts["post_resolution_rows"] = int(len(post_resoluton_df))
        update_progress(90, "Filtering out rows with all zeros...")
        labeled_image = np.zeros(brightfield.shape, dtype=np.uint16)
        update_progress(95, "Bead generation complete.")
        results = {}
        try:
            bboxs = df.pop("bbox")
        except:
            bboxs = None
        cycles = {}
        for i in range(len(tifs)):
            cycles[f"cy{i}"] = tifs[i][0]
        results["beads"] = df
        results["post_resolution_beads"] = post_resoluton_df
        results["cycles"] = cycles
        results["labeled_image"] = labeled_image
        trace_status = "success"
        return results
    finally:
        finalize_trace()


def get_error(bead_df: pd.DataFrame, protein_profile_df: pd.DataFrame = None) -> dict:
    """
    Analyzes bead data to calculate invalid and filtered bead counts and error rates.

    This function is self-contained and operates in two modes:
    1. With a protein profile: It merges bead data with the provided profile,
       identifying beads as 'Invalid' if they don't match the profile.
    2. Without a protein profile (protein_profile_df is None): It automatically
       separates beads into two groups by clustering their counts. The group with
       a significantly lower number of beads is considered 'Invalid'. This requires
       'scikit-learn' to be installed.

    In both modes, it also identifies and counts 'Filtered' beads with saturation values.

    Args:
        bead_df: DataFrame containing processed bead data with 'cy0' and 'cy1' columns.
        protein_profile_df: Optional DataFrame mapping 'cy0' and 'cy1' to 'Protein name'.
                            If None, automatic separation is performed.

    Returns:
        A dictionary containing key analysis results.

    Raises:
        ImportError: If protein_profile_df is None and scikit-learn is not installed.
    """
    # Mode 2: Automatic separation using clustering when no profile is given
    if protein_profile_df is None:
        if KMeans is None:
            raise ImportError(
                "scikit-learn is required for automatic bead separation. "
                "Please install it using 'pip install scikit-learn'"
            )

        # Identify and count filtered beads first, then remove them for clustering
        filter_mask = (bead_df["cy0"] >= 254) | (bead_df["cy1"] >= 254)
        filtered_count = int(filter_mask.sum())
        remaining_beads_df = bead_df[~filter_mask]

        if remaining_beads_df.empty:
            return {
                "invalid_count": 0,
                "filtered_count": filtered_count,
                "mean_beads_per_protein": 0.0,
                "invalid_error_percentage": 0.0,
            }

        # Count occurrences of each unique (cy0, cy1) bead type
        counts_df = (
            remaining_beads_df.groupby(["cy0", "cy1"]).size().reset_index(name="count")
        )
        counts_array = counts_df["count"].values.reshape(-1, 1)

        # Only cluster if there are at least two distinct count values to separate
        if len(np.unique(counts_array)) < 2:
            invalid_count = 0  # Assume all are valid if no clear separation is possible
            valid_protein_counts = counts_df["count"]
        else:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(
                counts_array
            )

            # The cluster with the smaller center is the 'low count' (invalid) group
            low_count_cluster_label = np.argmin(kmeans.cluster_centers_)
            counts_df["cluster"] = kmeans.labels_

            is_invalid_mask = counts_df["cluster"] == low_count_cluster_label

            invalid_count = int(counts_df.loc[is_invalid_mask, "count"].sum())
            valid_protein_counts = counts_df.loc[~is_invalid_mask, "count"]

        # Calculate final metrics based on the clustered groups
        mean_beads_per_protein = (
            valid_protein_counts.mean() if not valid_protein_counts.empty else 0.0
        )

        if mean_beads_per_protein > 0:
            invalid_error_percentage = (invalid_count / mean_beads_per_protein) * 100
        else:
            invalid_error_percentage = float("inf") if invalid_count > 0 else 0.0

        return {
            "invalid_count": invalid_count,
            "filtered_count": filtered_count,
            "mean_beads_per_protein": round(mean_beads_per_protein, 2),
            "invalid_error_percentage": round(invalid_error_percentage, 4),
            "filtered_percentage": (
                filtered_count / len(bead_df) * 100 if len(bead_df) > 0 else 0.0
            ),
        }

    metrics = compute_bead_profile_metrics(bead_df, protein_profile_df)
    invalid_count = int(metrics["invalid"])
    filtered_count = int(metrics["filtered"])
    cycle_cols = metrics["cycle_cols"]

    if len(cycle_cols) == 0 or metrics["valid"] == 0:
        valid_protein_counts = pd.Series(dtype=float)
    else:
        valid_combo_df = bead_df.loc[metrics["valid_mask"], cycle_cols]
        valid_combo_df = valid_combo_df.apply(pd.to_numeric, errors="coerce")
        valid_combo_df = valid_combo_df.dropna()
        if valid_combo_df.empty:
            valid_protein_counts = pd.Series(dtype=float)
        else:
            valid_protein_counts = valid_combo_df.groupby(cycle_cols).size()

    mean_beads_per_protein = (
        valid_protein_counts.mean() if not valid_protein_counts.empty else 0.0
    )

    # Calculate the invalid error rate as a percentage relative to the mean bead count
    if mean_beads_per_protein > 0:
        invalid_error_percentage = (invalid_count / mean_beads_per_protein) * 100
    else:
        # Handle case with no valid beads to avoid division by zero
        invalid_error_percentage = float("inf") if invalid_count > 0 else 0.0

    return {
        "invalid_count": invalid_count,
        "filtered_count": filtered_count,
        "mean_beads_per_protein": round(mean_beads_per_protein, 2),
        "invalid_error_percentage": round(invalid_error_percentage, 4),
    }


def analyze(
    raw_bead_data: list, protein_profile_df: pd.DataFrame, output_csv_path: str = None
) -> dict:
    """
    Performs a full, self-contained analysis of raw bead data.

    This function converts raw bead data into a DataFrame, calculates error metrics,
    and optionally saves the processed DataFrame to a CSV file.

    Args:
        raw_bead_data: A list of lists, where each inner list represents a bead
                       with format [x, y, cy0, cy1, bbox].
        protein_profile_df: DataFrame mapping 'cy0' and 'cy1' to 'Protein name'.
        output_csv_path: Optional file path to save the processed bead DataFrame.

    Returns:
        A dictionary containing the total bead count and a nested dictionary of error metrics.
    """
    # Convert raw list data into a structured DataFrame
    headers = ["x", "y", "cy0", "cy1", "bbox"]
    df = pd.DataFrame(raw_bead_data, columns=headers)

    # Clean and type-cast data, ensuring robust conversion
    for col in ["x", "y", "cy0", "cy1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)  # Drop rows where conversion failed

    df = df.astype({"x": int, "y": int, "cy0": int, "cy1": int})
    df.pop("bbox")

    total_beads = len(df)

    # Optionally save the cleaned DataFrame
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Processed bead data saved to {output_csv_path}")

    # Calculate error metrics using the self-contained helper function
    error_metrics = get_error(df, protein_profile_df)

    return {"total_beads": total_beads, "error_metrics": error_metrics}
