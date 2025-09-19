import concurrent.futures
import itertools
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import gauss
from typing import List

import cv2

# import diplib as dip
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, cKDTree
from skimage.color import label2rgb
from skimage.exposure import match_histograms
from skimage.filters import threshold_isodata, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border, expand_labels
from tqdm import tqdm

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


import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---- Helper: tile processing ----
def process_tile(idx, bounds, px_overlap, tile_size, brightfield_path):
    # Load memory-mapped brightfield in subprocess
    bf = np.load(brightfield_path, mmap_mode="r")

    ymin = int(round(bounds["center"][1] - tile_size))
    ymax = int(round(bounds["center"][1] + tile_size))
    xmin = int(round(bounds["center"][0] - tile_size))
    xmax = int(round(bounds["center"][0] + tile_size))

    # clip coordinates inside image bounds
    crop_ymin = max(0, ymin)
    crop_ymax = min(bf.shape[0], ymax)
    crop_xmin = max(0, xmin)
    crop_xmax = min(bf.shape[1], xmax)

    tile = bf[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # pad if needed
    # pad_top = max(0, 0 - ymin)
    # pad_bottom = max(0, ymax - bf.shape[0])
    # pad_left = max(0, 0 - xmin)
    # pad_right = max(0, xmax - bf.shape[1])

    # tile_padded = cv2.copyMakeBorder(
    #     tile,
    #     pad_top,
    #     pad_bottom,
    #     pad_left,
    #     pad_right,
    #     borderType=cv2.BORDER_CONSTANT,
    #     value=0,
    # )

    # if tile_padded.size == 0 or tile_padded.shape[0] == 0 or tile_padded.shape[1] == 0:
    #     return idx, np.empty((0, 2)), [], np.empty((0, 2)), []
    beads, roi_coords = find_beads(tile)  # user-provided

    # Adjust coordinates to global image space
    beads[:, 0] += xmin
    beads[:, 1] += ymin

    roi_coords_adjusted = [region + np.array([ymin, xmin]) for region in roi_coords]

    # Mark edge beads near overlap
    mask_edge = (
        (beads[:, 0] - xmin <= px_overlap)
        | (xmax - beads[:, 0] <= px_overlap)
        | (beads[:, 1] - ymin <= px_overlap)
        | (ymax - beads[:, 1] <= px_overlap)
    )

    return (
        idx,
        beads[~mask_edge],
        [roi_coords_adjusted[i] for i in np.where(~mask_edge)[0]],
        beads[mask_edge],
        [roi_coords_adjusted[i] for i in np.where(mask_edge)[0]],
    )


def merge_edge_pairs(beads1, rois1, beads2, rois2, radius=1):
    if len(beads1) == 0 or len(beads2) == 0:
        return beads1, rois1, beads2, rois2

    all_beads = np.vstack([beads1, beads2])
    all_rois = rois1 + rois2
    tree = cKDTree(all_beads)

    visited = np.zeros(len(all_beads), dtype=bool)
    merged_beads, merged_rois = [], []

    for i in range(len(all_beads)):
        if visited[i]:
            continue
        cluster_idx = set()
        to_visit = [i]
        while to_visit:
            idx = to_visit.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster_idx.add(idx)
            neighbors = tree.query_ball_point(all_beads[idx], r=radius)
            for n in neighbors:
                if not visited[n]:
                    to_visit.append(n)

        merged_roi = np.vstack([all_rois[k] for k in cluster_idx])
        new_center_yx = merged_roi.mean(axis=0)
        merged_beads.append(np.array([new_center_yx[1], new_center_yx[0]]))
        merged_rois.append(merged_roi)

    # After merging, no double-counting: all merged beads belong to first tile
    return merged_beads, merged_rois, [], []


import tempfile


# ---- Main beadfinding ----
def beadfinding(
    brightfield,
    num_tiles=10,
    px_overlap=100,
    workers=10,
    is_running_callback=None,
):
    # New implementation: preprocess tiles with quadtree_threshold, stitch masks, then find beads from the stitched mask.
    brightfield_path = "./tmp/brightfield_memmap.npy"
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    np.save(brightfield_path, brightfield)

    try:
        tileset = TileMap("tm", brightfield, px_overlap, num_tiles)
        stitched_mask = np.zeros_like(brightfield, dtype=bool)

        def get_mask_from_tile(tile_and_bounds):
            if is_running_callback and not is_running_callback():
                return None

            tile, bounds = tile_and_bounds
            mask = quadtree_threshold(tile)

            center = bounds["center"]
            tile_half_width = tileset.tile_size
            tile_size_with_overlap = round(tile_half_width) + px_overlap

            ymin = int(round(center[1] - tile_size_with_overlap))
            ymax = int(round(center[1] + tile_size_with_overlap))
            xmin = int(round(center[0] - tile_size_with_overlap))
            xmax = int(round(center[0] + tile_size_with_overlap))

            img_h, img_w = brightfield.shape[:2]

            crop_ymin = max(0, ymin)
            crop_ymax = min(img_h, ymax)
            crop_xmin = max(0, xmin)
            crop_xmax = min(img_w, xmax)

            pad_top = max(0, -ymin)
            pad_bottom = max(0, ymax - img_h)
            pad_left = max(0, -xmin)
            pad_right = max(0, xmax - img_w)

            h, w = mask.shape
            unpadded_mask = mask[pad_top : h - pad_bottom, pad_left : w - pad_right]

            return unpadded_mask, (crop_ymin, crop_ymax, crop_xmin, crop_xmax)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(get_mask_from_tile, tb) for tb in tileset]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing tiles"
            ):
                if is_running_callback and not is_running_callback():
                    executor.shutdown(wait=False)
                    return None, None

                result = future.result()
                if result is None:
                    continue

                unpadded_mask, (ymin, ymax, xmin, xmax) = result

                # Stitching with logical OR for overlap regions
                stitched_mask[ymin:ymax, xmin:xmax] |= unpadded_mask

        # Now, run the rest of find_beads logic on the stitched mask.
        brighter_regions = np.where(brightfield, stitched_mask, 0)

        bw = closing(brighter_regions, square(1))
        cleared = clear_border(bw)  # optional

        label_image = label(cleared)
        centers = []
        coords = []
        for region in tqdm(
            regionprops(label_image, brightfield), desc="Computing centroids"
        ):
            y, x = region.centroid_weighted
            coords.append(region.coords)
            centers.append([x, y])

        centers = np.array(centers)
        if centers.size > 0:
            centers = np.rint(centers).astype(np.uint16)

        print(f"Final bead count: {len(centers)}")
        return centers, coords, label_image
    finally:
        if os.path.exists(brightfield_path):
            os.remove(brightfield_path)
            log(f"Cleaned up temporary file: {brightfield_path}")

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
        if (x - 2 * radius) > 0 and (x + 2 * radius+1) < max_size:
            if (y - 2 * radius) > 0 and (y + 2 * radius+1) < max_size:
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
#     for i in range(len(tif_images)):
#         tif_images[i] = process_cycle(tif_images[i], tif_metadata[i])[:max_size,:max_size]
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
#             beads = list(beads)
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
#                     y - radius : y + radius+1,
#                     x - radius : x + radius+1,
#                 ]
#                 brightness = np.median(roi)

#                 if (x - 2 * 20) > 0 and (x + 2 * 20) < max_size:
#                     if (y - 2 * 20) > 0 and (y + 2 * 20) < max_size:
#                         local_flor_layer = flor_layer[
#                             y - radius : y + radius+1,
#                             x - radius : x + radius+1,
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
    
#     cy_columns = ["x","y"]
#     for i in range(len(tifs)):
#         cy_columns.append(f"cy{i}")
    
#     export_to_excel = pd.DataFrame(export_to_excel,columns=cy_columns)
#     return export_to_excel

import numpy as np
from scipy.signal import convolve2d, correlate2d


def gaussian_kernel(size: int, sigma: float = None) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel.
    Args:
        size (int): Odd kernel size
        sigma (float): Standard deviation. Defaults to size/6.
    """
    if size % 2 == 0:
        raise ValueError("Size must be odd")
    if sigma is None:
        sigma = size / 6.0

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def extract_bead_rois(flor_layer: np.ndarray, beads_batch: np.ndarray, radius: int):
    """
    Extract same-shaped, padded ROIs for beads.
    Returns shape: (N, 2r+1, 2r+1)
    """
    H, W = flor_layer.shape
    pad = radius
    padded = np.pad(flor_layer, pad_width=pad, mode="constant", constant_values=0)

    beads_batch = np.rint(beads_batch).astype(int)
    beads_batch_padded = beads_batch + pad

    roi_size = 2 * radius + 1
    bead_rois = np.empty((len(beads_batch), roi_size, roi_size), dtype=flor_layer.dtype)

    for i, (x, y) in enumerate(beads_batch_padded):
        bead_rois[i] = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]

    return bead_rois


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


import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import utils


def print_timing_summary(timings):
    """Print a formatted summary of timing results."""
    print("\n=== Watershed Segmentation Timing Summary ===")
    print(f"{'Step':<25} {'Time (ms)':<12} {'Percentage':<12}")
    print("-" * 50)

    total_time = timings["total"]
    for step, time_val in timings.items():
        if step != "total":
            percentage = (time_val / total_time) * 100
            print(f"{step:<25} {time_val*1000:>8.2f} ms {percentage:>8.1f}%")

    print("-" * 50)
    print(f"{'TOTAL':<25} {total_time*1000:>8.2f} ms {100.0:>8.1f}%")
    print("=" * 50)

def watershed_segmentation_cv2(
    img,
    border=1,
    marker_low=30 / 255.0,
    marker_high=40 / 255.0,
    use_numexpr=True,
    bg=None,
    visualize=False,
    bg_percentile_low=20,
    bg_percentile_high=80,
    histogram_cutoffs=None,
    background_mask=None,
):
    timings = {}

    # Step 1: Convert to float
    t0 = time.time()
    img_float = img.astype(np.float64)
    timings["convert_to_float"] = time.time() - t0

    # Step 2: Optional normalization using numexpr
    t1 = time.time()
    timings["numexpr_preprocessing"] = time.time() - t1

    # Step 3: Compute elevation map using Sobel
    t2 = time.time()
    elevation_map = filters.sobel(
        img_float,
        mask=~bg[: img.shape[-2], : img.shape[-1]] if bg is not None else None,
    )
    timings["sobel"] = time.time() - t2

    # Step 4: Create markers
    t3 = time.time()
    markers = np.zeros_like(img, dtype=np.int32)

    # Original threshold-based approach
    if histogram_cutoffs is not None:
        if background_mask is None:
            markers = np.digitize(img_float, histogram_cutoffs).astype(np.int32)
        else:
            markers = np.digitize(img_float, histogram_cutoffs).astype(np.int32)
            markers[background_mask] = 1
            # markers[img_float>histogram_cutoffs[1]] = 2
            # markers = np.digitize(img_float, [0]+histogram_cutoffs).astype(np.int32)
        print(markers.dtype, markers.shape)
    else:
        markers[img_float <= marker_low] = 1  # Background
        markers[img_float > marker_high] = 2  # Background
        if bg is not None:
            markers[bg[: img.shape[-2], : img.shape[-1]]] = 1
        # markers[img_float > ski.filters.threshold_li(img_float)] = 2  # Foreground
        # foreground = (img_float > ski.filters.threshold_li(img_float)).astype(np.uint8)*2

    # Add border markers to ensure proper segmentation
    # markers[:border, :] = 1
    # markers[-border:, :] = 1
    # markers[:, :border] = 1
    # markers[:, -border:] = 1

    timings["marker_creation"] = time.time() - t3

    # Step 5: Watershed segmentation
    t4 = time.time()
    # Convert elevation map to CV2 format (3-channel uint8)
    elevation_map_cv = cv2.cvtColor(
        (elevation_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    # Apply watershed
    labeled_markers = cv2.watershed(elevation_map_cv, markers.copy())
    timings["watershed"] = time.time() - t4

    # Step 6: Post-process results
    t5 = time.time()
    # Create binary segmentation (foreground regions)
    segmentation_coins = (labeled_markers >= 2).astype(np.uint8)

    if border > 0:
        segmentation_coins = ndi.binary_erosion(segmentation_coins, iterations=border)
    # segmentation_coins = ndi.binary_fill_holes(segmentation_coins, structure=ndi.generate_binary_structure(2,1))

    # Create connected component labels for individual objects
    labeled_coins = measure.label(segmentation_coins)

    # Remove small components (noise reduction)
    # min_size = 10
    # labeled_coins = ski.morphology.remove_small_objects(labeled_coins, min_size=min_size)

    # Update binary mask to match cleaned labels
    segmentation_coins = (labeled_coins > 0).astype(np.uint8)

    timings["post_processing"] = time.time() - t5
    timings["total"] = time.time() - t0

    return segmentation_coins, labeled_coins, timings


from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters import gaussian
from skimage.restoration import rolling_ball


def get_labels_from_cycles(cycles, cycles_metadata: List[MetaData],max_size):
    cycle_labels = []
    
    for i, cycle in enumerate(cycles):
        flor_layers = cycles_metadata[i].flors_layers
        assert flor_layers is not None, "forgot to initialize flors layer idx"
        max_size = cycles_metadata[i].max_size
        labels = []
        bead_radius= 5
        for layer_idx in flor_layers:
            img = cycle[layer_idx]
            print(img.shape)
            img = (img - img.min()) / (img.max() - img.min())
            seg, label, timing = watershed_segmentation_cv2(
                img, use_numexpr=False, border=0, marker_low=0.1, marker_high=0.15
            )
            print_timing_summary(timing)
            labels.append(label)
        cycle_labels.append(labels)
    return cycle_labels, cycles

import numpy as np
import cv2
from skimage.exposure import match_histograms, adjust_sigmoid, equalize_adapthist
from skimage.morphology import erosion
from skimage import img_as_float, img_as_uint
from utils import background_subtraction_with_histogram
def process_cycle(cycle, metadata: MetaData):
    """
    Process one imaging cycle:
    - Leave channels before reference untouched
    - Use (reference_channel+1) as reference, equalize it
    - Match all later channels to the reference, then adjust sigmoid
    """
    num_layers = cycle.shape[0]
    processed = []

    ref_idx = metadata.reference_channel + 1

    # keep all channels before reference
    processed.extend(cycle[:ref_idx])

    # reference channel
    ref16 = cycle[ref_idx]
    ref_float = img_as_float(ref16)
    ref_adj = adjust_sigmoid(ref_float, cutoff=0.1, gain=1)
    processed.append(img_as_uint(ref_adj))

    # channels after reference
    for j in range(ref_idx + 1, num_layers):
        img16 = cycle[j]
        matched_float = img_as_float(img16)
        matched = match_histograms(matched_float, ref_float)
        matched = adjust_sigmoid(matched, cutoff=0.1, gain=1)
        processed.append(img_as_uint(matched))

    return np.stack(processed, axis=0)
import numpy as np
from tqdm import tqdm


def assign_beads_labels(bead_data, cycle_labels):
    bead_data = bead_data.copy()
    for i, cycle_label in enumerate(cycle_labels):
        # Build one big array of shape (n_flour, H, W)
        labs = np.array(cycle_label)  # shape (n_flour, H, W)
        print(labs.shape)
        # Round coords and clip to bounds
        xs = np.rint(bead_data["x"].to_numpy()).astype(int)
        ys = np.rint(bead_data["y"].to_numpy()).astype(int)
        # xs = np.clip(xs, 0, labs.shape[-1]-1)
        # ys = np.clip(ys, 0, labs.shape[-2]-1)

        # Fancy index: take values at (flour, y, x)
        # shape (n_points, n_flour)
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
):
    """
    Prepares data for bead assignment resolution.
    Returns all necessary data structures for testing different resolution methods.
    """
    def update_progress(value, message):
        if progress_callback:
            overall_progress = 40 + (value / 100) * 30
            progress_callback(int(overall_progress), message)

    def is_running():
        return is_running_callback() if is_running_callback else True

    ColorThreshold = signal_to_noise_cutoff

    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = np.array([np.array(img) for img, _ in tifs]).copy()
    
    # Setup flors_layers for each metadata object
    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)
        md.flors_layers = [
            j for j in range(len(tif_images[i])) if j > int(md.reference_channel)
        ]
    for i in range(len(tif_images)):
        res = process_cycle(tif_images[i], tif_metadata[i])
        tif_images[i] = res
    total_beads = len(beads)
    total_cycles = len(tif_metadata)
    total_layers = len(tif_metadata[0].flors_layers) if total_cycles > 0 else 0
    beads = pd.DataFrame(beads[:, :2], columns=["x", "y"], dtype=np.float32)
    
    update_progress(10, "Getting activation regions from cycles")
    print(tif_images.shape, max_size)
    labels, cycles = get_labels_from_cycles(tif_images, tif_metadata, max_size)
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
    tif_images_padded = np.pad(np.array(tif_images), ((0,), (0,), (2,), (2,)))
    columns = [f"cy{i}" for i in range(len(tif_images))]
    
    # Prepare final_df structure
    final_df = df[["x", "y"]].copy()
    final_df[columns] = 255

    def assignment_row(row):
        return [
            get_assignment(
                row, [f"{col}_{i}" for i in range(len(tif_metadata[0].flors_layers))]
            )
            for col in columns
        ]

    # Handle beads with single assignments (no correction needed)
    singles_vals = df.loc[both_assignment].apply(
        assignment_row, axis=1, result_type="expand"
    )
    final_df.loc[both_assignment, columns] = singles_vals.values
    
    single_distr = final_df.loc[both_assignment].groupby(columns).size().reset_index(name='count')
    print(single_distr.head(30))
    
    # Return all data needed for resolution methods
    return {
        'df': df,
        'final_df': final_df,
        'tif_images': tif_images,
        'tif_images_padded': tif_images_padded,
        'tif_metadata': tif_metadata,
        'ColorThreshold': ColorThreshold,
        'total_cycles': total_cycles,
        'both_assignment': both_assignment,
        "no_assignment":no_assignment,
        'columns': columns,
        'pad': pad
    }
import numpy as np
from tqdm import tqdm

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
    
    return y_indices.astype(int), x_indices.astype(int) # Return as (row, col) for NumPy indexing
def combine_results(data_dict, cycle_values, correction_indices):
    """
    Combine the resolution results with the prepared data to get final_df.
    """
    final_df = data_dict['final_df'].copy()
    columns = data_dict['columns']
    
    if len(cycle_values) > 0:
        final_df.loc[correction_indices, columns] = cycle_values
    
    return final_df
def _medianroi_method_optimized(x_coords, y_coords, tif_images_padded, tif_metadata, ColorThreshold, correction_indices, background_method='local_ring'):
    """
    Vectorized method using median intensity in a 3x3 ROI with background-dependent filtering.
    """
    num_beads = len(x_coords)
    num_cycles = len(tif_images_padded)
    coords = np.array([x_coords, y_coords]).T  # Shape: (num_beads, 2)
    
    img_height, img_width = tif_images_padded.shape[-2:]
    
    # 1. Pre-calculate all indices for ROI and Background patches
    patch_radius = 4
    roi_rows, roi_cols = _create_patch_indices(coords, patch_radius, (img_height, img_width)) # 3x3 ROI patch
    if background_method in ['psnr']:
        bg_radius = patch_radius+1
    elif background_method in ['local_ring', 'adaptive_snr']:
        bg_radius = 3 # 7x7 patch
    else: # local_patch or percentile_diff
        bg_radius = 4 # 9x9 patch
        
    bg_rows, bg_cols = _create_patch_indices(coords, bg_radius, (img_height, img_width))
    
    # Create a boolean mask to exclude the center ROI from the background patch
    # This effectively creates a "ring" of pixels for the background calculation
    if background_method in ['local_ring', 'local_patch']:
        mask_size = 2 * bg_radius + 1
        center_start = bg_radius - 1 # Center 3x3 starts at radius-1
        center_end = bg_radius + 2   # and ends at radius+2
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
            with np.errstate(all='ignore'):
                if background_method == 'local_ring' or background_method == 'local_patch':
                    valid_bg = bg_patches[:, bg_mask]
                    background_medians = np.nanmedian(valid_bg, axis=1)
                    background_medians = np.nan_to_num(background_medians)
                    scores = signal_medians / (background_medians + 1e-8)
                elif background_method == 'psnr':
                    valid_bg = bg_patches
                    background_percentile = np.percentile(valid_bg, 50,axis=(1,2))
                    scores = (signal_medians - background_percentile) / (background_percentile + 1e-8)
                elif background_method == 'percentile_diff':
                    # Reshape from (N, 9, 9) to (N, 81) for percentile calculation
                    bg_flat = bg_patches.reshape(num_beads, -1)
                    background_75th = np.nanpercentile(bg_flat, 75, axis=1)
                    background_75th = np.nan_to_num(background_75th)
                    scores = signal_medians - background_75th

            all_scores[:, layer_idx] = scores

        # 4. Vectorized Filtering and Argmax
        if background_method in ['local_ring', 'local_patch']:
            threshold = 1.0 + ColorThreshold
        else: # adaptive_snr, percentile_diff
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

def _template_matching_method(x_coords, y_coords, tif_images_padded, tif_metadata, ColorThreshold, correction_indices):
    """Template matching method with bounds checking for large templates"""
    template = gaussian_kernel(5).astype(np.float32)
    template_radius = 4  # radius for template (4 for 9x9)
    cycle_values = []
    
    # Get image dimensions
    img_height, img_width = tif_images_padded.shape[-2:]
    
    for i, (x, y) in tqdm(enumerate(zip(x_coords, y_coords))):
        row_cycles_val = []
        for cycle_idx in range(len(tif_images_padded)):
            flors_layers = tif_metadata[cycle_idx].flors_layers
            
            # Calculate patch bounds
            patch_radius = template_radius  # Use same radius as template
            y_start = y - patch_radius
            y_end = y + patch_radius + 1
            x_start = x - patch_radius  
            x_end = x + patch_radius + 1
            
            # Extract patches with bounds checking
            patches = []
            for layer in flors_layers:
                # Create patch with bounds checking - fill out-of-bounds with black (0)
                patch = np.zeros((template_radius * 2 + 1, template_radius * 2 + 1), dtype=np.float32)
                
                # Calculate valid region within the patch
                valid_y_start = max(0, y_start)
                valid_y_end = min(img_height, y_end)
                valid_x_start = max(0, x_start)
                valid_x_end = min(img_width, x_end)
                
                # Calculate corresponding indices in the patch
                patch_y_start = max(0, -y_start)
                patch_y_end = patch_y_start + (valid_y_end - valid_y_start)
                patch_x_start = max(0, -x_start)
                patch_x_end = patch_x_start + (valid_x_end - valid_x_start)
                
                # Copy valid region from image to patch
                if valid_y_end > valid_y_start and valid_x_end > valid_x_start:
                    patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
                        tif_images_padded[cycle_idx][layer, valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                    patch = np.array(patch)
                    # patch = utils.adjust_contrast(patch,2, 70)
                patches.append(patch)
            
            patches = np.array(patches)
            patches = utils.adjust_contrast(patches, 30, 70,axis=(1,2)).astype(np.float32)
            
            flor_assignments = np.array([
                np.median(correlate2d(patch, template, mode="valid"))
                for patch in patches
            ])
            max_flor_layer = np.argmax(flor_assignments)
            if flor_assignments[max_flor_layer] > ColorThreshold:
                row_cycles_val.append(max_flor_layer)
            else:
                row_cycles_val.append(255)
        cycle_values.append(row_cycles_val)
    
    return np.array(cycle_values), correction_indices

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
    tif_images = [np.array(img)[:,:max_size,:max_size] for img, _ in tifs]
    

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
    data_dict = prepare_data_for_resolution(beads, signal_to_noise_cutoff, tifs, max_size, progress_callback=update_progress)
    update_progress(50, "Assigning beads labels")
    # df = assign_beads_labels(beads, labels)
    # cycle_layer_columns = {}
    df = data_dict['df']

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
    
    

    
    # final_df = df[["x", "y"]].copy()
    # final_df[columns] = 255

    # def assignment_row(row):
    #     return [
    #         get_assignment(
    #             row, [f"{col}_{i}" for i in range(len(tif_metadata[0].flors_layers))]
    #         )
    #         for col in columns
    #     ]

    # singles_vals = df.loc[both_assignment].apply(
        # assignment_row, axis=1, result_type="expand"
    # )
    # final_df.loc[both_assignment, columns] = singles_vals.values
    
    # single_distr = final_df.loc[both_assignment].groupby(columns).size().reset_index(name='count')
    # print(single_distr.head(30))
    tif_images_padded = data_dict['tif_images_padded']
    for i in range(len(tif_images_padded)):
        ref_idx = tif_metadata[i].reference_channel+1
        for j in range(ref_idx+1,len(tif_images_padded[i])):
            tif_images_padded[i][j] = match_histograms(tif_images_padded[i][j], tif_images_padded[i][ref_idx])
            
    tif_metadata = data_dict['tif_metadata']
    ColorThreshold = data_dict['ColorThreshold']
    total_cycles = data_dict['total_cycles']
    both_assignment = data_dict['both_assignment']
    no_assignment = data_dict["no_assignment"]
    pad = data_dict['pad']
    
    # Identify beads that need correction
    need_correction = df[~both_assignment & ~no_assignment]
    
    print(f"Beads needing correction: {len(need_correction)}")
    update_progress(70, "Resolving Bead Labels")
    ColorThreshold = 0.005
    # Convert coordinates to integer indices (adjusted for padding)
    x_coords = np.rint(need_correction["x"].to_numpy() + pad).astype(np.int32)
    y_coords = np.rint(need_correction["y"].to_numpy() + pad).astype(np.int32)
    # You'd need to modify resolve_bead_assignments to accept background_method parameter
    cycle_values, correction_indices =_medianroi_method_optimized(
            x_coords, y_coords, tif_images_padded[:,1:,:,:], tif_metadata, 
            ColorThreshold, need_correction.index, background_method='psnr'
        )
    new_dd= combine_results(data_dict, cycle_values, correction_indices)
    ff = {"final_df":new_dd,"columns":data_dict['columns']}
    final_df = ff['final_df']
    need_correction = df[no_assignment]
    x_coords = np.rint(need_correction["x"].to_numpy() + pad).astype(np.int32)
    y_coords = np.rint(need_correction["y"].to_numpy() + pad).astype(np.int32)
    ColorThreshold = 0.85
    
    cycle_values, correction_indices = _template_matching_method(x_coords,y_coords,tif_images_padded, tif_metadata, 
            ColorThreshold, need_correction.index)
    final_df = combine_results(ff, cycle_values, correction_indices)


    return final_df

    # Histogram matching to first cycle's first flor layer
    # for i, md in enumerate(tif_metadata):
    #     reference = tif_images[i][md.flors_layers[0]]
    #     for layer in md.flors_layers[1:]:
    #         img16 = match_histograms(tif_images[i][layer], reference)
    #         img8 = np.zeros_like(img16, dtype=np.uint8)
    #         img8 = cv2.normalize(img16, img8, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #         img_eq = cv2.equalizeHist(img8)
    #         tif_images[i][layer] = img_eq

    # bounding_boxes = np.zeros((total_beads, 4), dtype=int)
    # if roi_coords:
    #     coords_array = [np.array(region) for region in roi_coords]
    #     # Ensure bounding boxes are square (max width/height)
    #     bounding_boxes = np.array(
    #         [(*region.max(axis=0), *region.max(axis=0)) for region in coords_array]
    #     )

    # Calculate batch size and number of batches reliably
    # batch_size = max(1, total_beads // max(1, n_workers))

    # with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
    #     for cycle_idx, md in enumerate(tif_metadata):
    #         if not is_running():
    #             return None

    #         cycle_data = np.zeros((total_beads, len(md.flors_layers)), dtype="uint16")
    # cycle_snr = np.zeros((total_beads, len(md.flors_layers)), dtype="float")
    # cycle_threshold = np.zeros(
    #     (total_beads, len(md.flors_layers)), dtype="float"
    # )

    # for layer_idx, layer in enumerate(md.flors_layers):
    #     if not is_running():
    #         return None

    #     flor_layer = tif_images[cycle_idx][layer]
    #     layer_threshold = layer_threshold_dict.get(layer, 0)

    #     n_batches = (total_beads + batch_size - 1) // batch_size
    #     batch_args = []

    #     for batch_idx in range(n_batches):
    #         start = batch_idx * batch_size
    #         end = min(start + batch_size, total_beads)
    #         beads_batch = beads[start:end]
    #         roi_batch = roi_coords[start:end] if roi_coords else None
    #         batch_args.append(
    #             (
    #                 flor_layer,
    #                 beads_batch,
    #                 radius,
    #                 layer_threshold,
    #                 roi_batch,
    #                 start,
    #                 end,
    #             )
    #         )

    #     # Map batches to worker function
    #     batch_results = list(executor.map(process_bead_batch, batch_args))

    #     # Collect batch results into cycle arrays
    #     for batch_idx, (data, snr_data, threshold_data) in enumerate(
    #         batch_results
    #     ):
    #         start = batch_idx * batch_size
    #         end = min(start + batch_size, total_beads)
    #         cycle_data[start:end, layer_idx] = data
    #         # cycle_snr[start:end, layer_idx] = snr_data
    #         # cycle_threshold[start:end, layer_idx] = threshold_data

    #     # Update progress after each layer
    #     progress = (
    #         ((cycle_idx * total_layers) + (layer_idx + 1))
    #         / (total_cycles * total_layers)
    #         * 100
    #     )
    #     update_progress(
    #         progress,
    #         f"Processing cycle {cycle_idx + 1}/{total_cycles}, layer {layer_idx + 1}/{total_layers}",
    #     )

    # # Find brightest layers per bead
    # brightest_layers = np.argmax(cycle_data, axis=1)
    # export_to_excel[:, cycle_idx] = brightest_layers

    # # Apply filters
    # for i, snr_row in enumerate(cycle_snr):
    #     if np.min(snr_row) > 1 - ColorThreshold:
    #         export_to_excel[i, cycle_idx] = 254
    #         # print("Sig noise filtered:", beads[i])

    # for i, thresh_row in enumerate(cycle_threshold):
    #     if np.all(thresh_row == 0):
    #         export_to_excel[i, cycle_idx] = 255
    # print("Threshold filtered:", beads[i])

    # bounding_boxes_str = np.array(
    #     [
    #         f"({x1}, {y1}, {max(x2 - x1, y2 - y1) + x1}, {max(x2 - x1, y2 - y1) + y1})"
    #         for x1, y1, x2, y2 in bounding_boxes
    #     ]
    # ).reshape(-1, 1)

    export_to_excel = np.hstack((beads[:, :2], export_to_excel))
    columns = ["x", "y"] + [f"cy{i}" for i in range(len(tifs))]
    df = pd.DataFrame(export_to_excel, columns=columns)
    count_table = (
        df.groupby([f"cy{i}" for i in range(len(tifs))])
        .size()
        .reset_index(name="counts")
    )
    (invalid_beads, valid_beads), best_gap = best_split_df_max_avg_gap(
        count_table, "counts"
    )  # type: ignore
    invalid_idx = invalid_beads.merge(df, how="left").index

    # Get indices of invalid beads
    invalid_idx = invalid_beads.merge(df, how="left").index
    coords = (
        df.loc[invalid_idx, ["x", "y"]].astype("float").round().astype(int).to_numpy()
    )

    # Vectorized cycle loop
    for cycle_idx, md in enumerate(tif_metadata):
        # Extract all channels for this cycle
        channels = md.flors_layers
        img = tif_images[cycle_idx]  # shape: (num_channels, H, W)

        # Build padded ROIs
        rois = extract_padded_rois(
            img, coords, channels, radius=2
        )  # shape: (num_beads, num_channels, 5, 5)

        # Flatten for manual vectorization
        num_beads, num_channels = rois.shape[:2]
        flat_rois = rois.reshape(-1, 5, 5)

        # Apply template match to each ROI
        scores_flat = np.apply_along_axis(
            lambda r: calculate_template_match(r.reshape(5, 5)),
            axis=1,
            arr=flat_rois.reshape(flat_rois.shape[0], -1),
        )

        # Reshape back and get best channel per bead
        scores = scores_flat.reshape(num_beads, num_channels)
        best_channels = np.argmax(scores, axis=1)

        # Write back into df
        df.loc[invalid_idx, f"cy{cycle_idx}"] = best_channels

    print(f"Final get_excel length: {len(df)}")
    return df, tif_images


def blur_layer(layer, image_stack, blur_percentage=1):
    layer_4 = image_stack[layer]  # Modify the 4th layer
    blurred_mask = cv2.GaussianBlur(layer_4, (101, 101), 0)
    blurred_mask_adjusted = (blurred_mask * blur_percentage).astype(np.uint16)
    corrected_layer_4 = cv2.subtract(layer_4, blurred_mask_adjusted)
    corrected_layer_4 = np.clip(corrected_layer_4, 0, 65535).astype(np.uint16)
    image_stack[layer] = corrected_layer_4


class CellIntensity:
    def __init__(self):
        """Initializes the CellIntensity analysis tool."""
        self.params = {
            "max_size": 23000,
            "num_decoding_cycles": 3,
            "num_decoding_colors": 3,
            "radius_fg": 2,
            "radius_bg": 7,
        }

        self.color_code = None
        self.stardist_labels = np.array([], dtype=np.uint16)
        self.df_cell_data = None
        self.bead_data = None
        self.protein_signal_array = None
        self.same_cells = False
        self.cell_centroids = {}

    def use_same_cell_centroids(self, same_cells: bool):
        assert (
            len(self.cell_centroids) > 0 or not same_cells
        ), "Cell centroids must be computed before setting same_cells."
        self.same_cells = same_cells

    def load_protein_signal_array(self, arr: np.ndarray):
        """Loads the protein signal image array."""
        log("Loading protein signal array.")
        self.protein_signal_array = arr

    def load_stardist_labels(self, stardist_labels_array: np.ndarray):
        """Loads the segmented cell labels array."""
        log("Loading stardist labels array.")
        log("Stardist label dtype:", stardist_labels_array.dtype)
        log(
            "Stardist label max and min:",
            np.max(stardist_labels_array),
            np.min(stardist_labels_array),
        )
        self.stardist_labels = stardist_labels_array.astype(np.uint16)

    def load_bead_data(self, bead_data: np.ndarray):
        """Loads the bead data numpy array."""
        if isinstance(bead_data, np.ndarray):
            self.bead_data = bead_data
        else:
            raise TypeError("Bead data must be a NumPy array.")

    def load_color_code(self, color_code: pd.DataFrame):
        """Loads the color code pandas DataFrame."""
        if isinstance(color_code, pd.DataFrame):
            self.color_code = color_code
            try:
                self.color_code = self.color_code.dropna(how="all", axis=1).dropna(
                    how="all", axis=0
                )
            except Exception as e:
                log(e)
                self.color_code = pd.DataFrame(self.color_code)
                self.color_code = self.color_code.dropna(how="all", axis=1).dropna(
                    how="all", axis=0
                )
        else:
            raise TypeError("Color code must be a pandas DataFrame.")

    def generate_cell_intensity_table(self):
        """
        Starts the cell intensity calculation.

        Returns:
            pd.DataFrame: The computed cell intensity table.
        """
        log("Starting Cell Intensity Calculation...")
        return self.run()

    def critical_error(self, msg: str):
        """Handles critical errors by printing a message and raising an exception."""
        error_message = f"CRITICAL ERROR: {msg}"
        log(error_message)
        raise ValueError(error_message)

    def compute_all_centroids(self):
        """
        Compute centroids for all unique labels in the mask (excluding 0).
        Uses skimage.measure.regionprops for efficiency.

        Returns:
            dict: A dictionary mapping each label to its (cx, cy) centroid.
        """
        log("Finding centroids for all cells...")
        centroids = {}
        regions = regionprops(self.stardist_labels)
        for region in tqdm(regions, desc="Finding Centroids"):
            cy, cx = region.centroid
            centroids[region.label] = (int(cx), int(cy))
        return centroids

    def infer_params(self):
        """Infers parameters from the provided color code and bead data."""
        if not isinstance(self.color_code, pd.DataFrame):
            self.critical_error("Color code is not a pandas DataFrame.")

        # Infer the number of decoding cycles from the color code columns
        self.params["num_decoding_cycles"] = self.color_code.shape[1] - 1

        # Infer the number of colors from the values in the color code
        color_code_np = self.color_code.iloc[:, 1:].to_numpy()
        max_color_value = np.max(color_code_np)
        self.params["num_decoding_colors"] = int(max_color_value) + 1
        log("Inferred params:", self.params)

    def run(self):
        """The main execution logic for calculating cell intensities."""
        # 1. VALIDATE INPUTS
        if self.stardist_labels is None or self.stardist_labels.size == 0:
            self.critical_error("Stardist labels are not loaded.")
        if self.bead_data is None:
            self.critical_error("Bead data is not loaded.")
        if self.color_code is None:
            self.critical_error("Color code is not loaded.")
        if self.protein_signal_array is None:
            self.critical_error("Protein signal array is not loaded.")

        # 2. SETUP & PARAMETER INFERENCE
        self.infer_params()

        # Create mappings for protein identification
        possible_values = list(range(self.params["num_decoding_colors"]))
        all_perms = [
            "".join(map(str, p))
            for p in itertools.product(
                possible_values, repeat=self.params["num_decoding_cycles"]
            )
        ]
        color_code_to_index = {int(k): i for i, k in enumerate(all_perms)}
        index_to_color_code = {v: k for k, v in color_code_to_index.items()}

        # Initialize data structure to hold intensities
        num_proteins = len(color_code_to_index)
        max_cell_id = np.max(self.stardist_labels)
        cell_data_dict = {
            cell_id: [[] for _ in range(num_proteins)]
            for cell_id in range(1, max_cell_id + 1)
        }
        # 3. PROCESS BEADS
        log("Processing beads...")
        # Prepare bead data by combining color codes into a single string identifier
        data_modified = np.zeros((len(self.bead_data), 3))
        data_modified[:, 0:2] = self.bead_data[:, 0:2].astype("uint16")

        # Correctly slice and join color codes
        cycle_cols = self.bead_data[:, 2 : 2 + self.params["num_decoding_cycles"]]
        data_modified[:, 2] = np.array(
            [int("".join(map(str, map(int, bead)))) for bead in cycle_cols]
        )
        radius_bg = self.params["radius_bg"]
        max_size = self.params["max_size"]

        # Ensure they are integers for indexing
        bead_xs = data_modified[:, 0].astype(int)
        bead_ys = data_modified[:, 1].astype(int)

        # Get the cell ID for every bead in a single, fast operation
        cell_ids_for_beads = self.stardist_labels[bead_ys, bead_xs]

        # --- 2. Create Boolean Masks for All Conditions ---

        # Mask 1: Beads that are inside any cell (ID > 0)
        in_cell_mask = cell_ids_for_beads > 0

        # Mask 2: Beads that are within the processing boundaries
        # (This prevents errors in get_adjusted_median_intensity)
        in_bounds_mask = (
            (bead_xs > radius_bg)
            & (bead_ys > radius_bg)
            & (bead_xs < (max_size - radius_bg))
            & (bead_ys < (max_size - radius_bg))
        )

        # --- 3. Combine Masks ---
        # The final mask identifies beads that satisfy ALL conditions
        valid_bead_mask = in_cell_mask & in_bounds_mask

        # --- 4. Filter the Data ---
        # Create a much smaller array containing only the beads we need to process
        valid_beads = data_modified[valid_bead_mask]
        valid_cell_ids = cell_ids_for_beads[valid_bead_mask]

        # --- 5. Loop Over the SMALLER Filtered Dataset ---

        for i, bead in enumerate(valid_beads):
            bead_x, bead_y, color_code = int(bead[0]), int(bead[1]), bead[2]

            # We already know this bead is in a cell, so we get its ID
            cell_associated_id = valid_cell_ids[i]

            # The expensive calculation is only called for valid beads
            adjusted_median_intensity = self.get_adjusted_median_intensity(
                bead_x, bead_y
            )

            protein_idx = color_code_to_index.get(color_code)
            if protein_idx is not None and adjusted_median_intensity is not None:
                cell_data_dict[cell_associated_id][protein_idx].append(
                    adjusted_median_intensity
                )

        # 4. IMPUTE MISSING PROTEINS
        log("Imputing values for cells with incomplete protein profiles...")
        log("Building K-D trees for fast nearest-neighbor search...")
        protein_kdtree_map = {}
        for i in tqdm(range(num_proteins), desc="Building KD-Trees"):
            protein_code = index_to_color_code.get(i)
            if protein_code is not None:
                protein_beads = data_modified[data_modified[:, 2] == protein_code][
                    :, 0:2
                ].astype(int)
                if len(protein_beads) > 0:
                    protein_kdtree_map[i] = KDTree(protein_beads)
        # Group bead locations by protein for fast nearest-neighbor search

        # Find cell centroids
        if not self.same_cells:
            cell_centroids = self.compute_all_centroids()
            self.cell_centroids = cell_centroids
        else:
            cell_centroids = self.cell_centroids

        for cell_id in tqdm(cell_data_dict.keys(), desc="Imputing Missing Proteins"):
            cell_center = cell_centroids[cell_id]
            for protein_idx, intensities in enumerate(cell_data_dict[cell_id]):
                if (
                    not intensities
                ):  # If no beads were found for this protein of cell_id
                    kdtree = protein_kdtree_map.get(protein_idx)
                    if kdtree:  # Check if a tree was successfully built
                        _, index = kdtree.query(cell_center)
                        nn_x, nn_y = kdtree.data[index]
                        if (
                            nn_x > radius_bg
                            and nn_y > radius_bg
                            and nn_x < (max_size - radius_bg)
                            and nn_y < (max_size - radius_bg)
                        ):
                            adjusted_intensity = self.get_adjusted_median_intensity(
                                int(nn_x), int(nn_y)
                            )
                            if adjusted_intensity is not None:
                                cell_data_dict[cell_id][protein_idx].append(
                                    adjusted_intensity
                                )

        # 5. AGGREGATE RESULTS & CREATE DATAFRAME
        log("Aggregating results and creating DataFrame...")
        median_values = {}
        for cell_id, protein_lists in cell_data_dict.items():
            # Calculate median for each protein, use np.nan if list is empty
            medians = [
                np.nanmedian(p_list) if p_list else np.nan for p_list in protein_lists
            ]
            median_values[cell_id] = medians

        # Create human-readable column headers
        color_code_map = {
            int("".join(map(str, map(int, row[1:])))): row[0]
            for _, row in self.color_code.iterrows()
        }
        header = ["Global X", "Global Y"] + [
            color_code_map.get(index_to_color_code.get(i), f"Protein_{i}")
            for i in range(num_proteins)
        ]

        # Combine centroid and intensity data
        final_data = []
        for cell_id, centroid in sorted(cell_centroids.items()):
            if cell_id in median_values:
                row = list(centroid) + median_values[cell_id]
                final_data.append(row)

        self.df_cell_data = pd.DataFrame(final_data, columns=header).set_index(
            pd.Index(sorted(cell_centroids.keys()), name="Cell ID")
        )
        log("Cell data table generated successfully.")
        return self.df_cell_data

    def save_cell_data(self, file_path: str):
        """
        Saves the generated cell data table to a file.

        Args:
            file_path (str): The path to save the file (e.g., 'cell_data.csv').
                             The format is inferred from the extension (.csv or .xlsx).
        """
        log(f"Saving cell data to {file_path}")
        if self.df_cell_data is None:
            self.critical_error("Cannot save. No cell data available.")
            return

        if file_path.endswith(".csv"):
            self.df_cell_data.to_csv(file_path)
        elif file_path.endswith(".xlsx"):
            self.df_cell_data.to_excel(file_path)
        else:
            log(f"Warning: Unknown file extension. Saving as CSV to {file_path}.csv")
            self.df_cell_data.to_csv(f"{file_path}.csv")

    def get_adjusted_median_intensity(self, bead_x, bead_y, bead_median_threshold=5000):
        """
        Calculate the adjusted median intensity given the bead coordinates

        :param bead_x: The x-coordinate of the bead
        :param bead_y: The y-coordinate of the bead
        :param bead_median_threshold: the threshold needed to apply median intensity correction
        :type bead_x: int
        :type bead_y: int
        :type bead_median_threshold: int

        :returns: The adjusted median intensity value of the bead
        :rtype: float
        """

        if self.protein_signal_array is None:
            return

        radius_bg = self.params["radius_bg"]
        radius_fg = self.params["radius_fg"]

        # Extract the 5x5 region around the bead
        bead_region = self.protein_signal_array[
            bead_y - radius_fg : bead_y + radius_fg + 1,
            bead_x - radius_fg : bead_x + radius_fg + 1,
        ]

        # Calculate the mean and median intensity of the 5x5 bead region
        mean_5x5 = np.mean(bead_region)
        bead_median_org = np.median(bead_region)
        bead_median = bead_median_org.copy()

        # Extract the 15x15 surrounding region
        surrounding_region = self.protein_signal_array[
            bead_y - radius_bg : bead_y + radius_bg + 1,
            bead_x - radius_bg : bead_x + radius_bg + 1,
        ]  # Convert to float to handle NaN values

        # Ensure the 15x15 region is valid
        if surrounding_region.shape != (15, 15):
            return bead_median_org  # Return unadjusted median if the 15x15 region is invalid

        # Mask out the 5x5 region from the 15x15 region
        surrounding_region = surrounding_region.astype(
            float
        )  # Convert to float to handle NaN values
        surrounding_region[
            bead_y - radius_fg : bead_y + radius_fg + 1,
            bead_x - radius_fg : bead_x + radius_fg + 1,
        ] = np.nan

        # Calculate the mean intensity of the surrounding 15x15 area, excluding the 5x5 region
        surrounding_mean_15x15 = np.nanmean(surrounding_region)

        # Apply correction only if 15x15 mean is 1.5x greater than 5x5 mean, and bead median > threshold
        if (
            surrounding_mean_15x15 > 1.5 * mean_5x5
            and bead_median > bead_median_threshold
        ):
            # Calculate the correction factor and apply linear correction
            correction_factor = mean_5x5 * (mean_5x5 / surrounding_mean_15x15)
            y = self.linear_correction(correction_factor)

            # Apply the correction to the bead median
            bead_median = bead_median - y + 2000

        # Ensure no negative values
        if bead_median < 1:
            bead_median = 1

        # Return the final adjusted bead median
        return bead_median

    def linear_correction(self, x):
        """A linear function for intensity correction."""
        return 0.8266 * x + 3970.1

    def set_param(self, key, value):
        """Sets a single parameter."""
        if key in self.params:
            log(f"Setting parameter '{key}' to {value}")
            self.params[key] = value
        else:
            log(f"Warning: Parameter '{key}' not found.")


def best_split_df_max_avg_gap(df, count_col):
    # Sort by counts
    df_sorted = df.sort_values(by=count_col).reset_index(drop=True)

    best_gap = float("-inf")
    best_split = None

    for i in range(1, len(df_sorted)):
        left = df_sorted.iloc[:i]
        right = df_sorted.iloc[i:]
        avg_left = left[count_col].mean()
        avg_right = right[count_col].mean()
        gap = avg_right - avg_left

        if gap > best_gap:
            best_gap = gap
            best_split = (left, right)

    return best_split, best_gap
from utils import find_min_std_partition

def process_beads(
    brightfield,
    tifs,
    max_size,
    signal_to_noise_cutoff,
    progress_callback=None,
    is_running_callback=None,
):
    def update_progress(value, message):
        if progress_callback:
            progress_callback(value, message)

    def is_running():
        if is_running_callback:
            return is_running_callback()
        return True

    update_progress(0, "Preprocessing brightfield image...")
    if not is_running():
        return None
    # log(f"Preprocessing brightfield image (max_size: {max_size})")
    # brightfield = preprocess_brightfield(brightfield, max_size)
    log(f"Preprocessed to shape: {brightfield.shape}")
    update_progress(10, "Initial bead detection...")
    if not is_running():
        return None
    brightfield = brightfield[:max_size,:max_size]
    log("Initial bead detection...")
    beads, roi_coords, _ = beadfinding(brightfield, is_running_callback=is_running)
    if beads is None:
        return None
    initial_bead_count = len(beads)
    log(f"Initial bead detection found {initial_bead_count} beads")
    update_progress(20, "Removing duplicate beads...")
    if not is_running():
        return None

    log("Removing duplicate beads...")
    beads = np.unique(beads, axis=0)
    unique_bead_count = len(beads)
    log(
        f"After deduplication: {unique_bead_count} unique beads (removed {initial_bead_count - unique_bead_count} duplicates)"
    )
    update_progress(30, "Performing second pass bead detection...")
    if not is_running():
        return None

    # log("Performing second pass bead detection...")
    # beads = second_pass_beadfinding(brightfield, beads)
    final_bead_count = len(beads)
    # log(f"Second pass found {final_bead_count - unique_bead_count} additional beads")
    log(f"Total beads detected from brightfield layer: {final_bead_count}")
    update_progress(40, "Calculating signal-to-noise ratios...")
    if not is_running():
        return None
    log(f"Signal-to-noise cutoff: {signal_to_noise_cutoff}")
    df = get_excel(
        beads,
        signal_to_noise_cutoff,
        tifs,
        max_size,
        progress_callback=progress_callback,
        is_running_callback=is_running,
        # roi_coords=roi_coords,
    )
    log("Done getting excel")
    if df is None:
        return None

    # if not is_running():
        # return None

    # update_progress(90, "Filtering out rows with all zeros...")
    labeled_image = np.zeros(brightfield.shape, dtype=np.uint16)
    
    # group by cy0,cy1... and find min-std partition
    log(f"Dataframe created with shape: {df.shape}")

    update_progress(95, "Bead generation complete.")

    results = {}
    try:
        bboxs = df.pop("bbox")
    except:
        bboxs = None
    cycles = {}
    for i in range(len(tifs)):
        cycles[f"cy{i}"] = tifs[i][0]
        print(f"Cycle {i} image shape: {tifs[i][0].shape}")
    results["beads"] = df
    results["cycles"] = cycles
    results["labeled_image"] = labeled_image
    

    return results
