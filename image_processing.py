import itertools
import time
from collections import defaultdict
from functools import reduce

import cv2
import diplib as dip
import numpy as np
import pandas as pd
import tifffile as tiff
import tqdm
from scipy.spatial import KDTree
from skimage.exposure import match_histograms
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.filters import threshold_isodata, threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
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
        y = round(y)
        x = round(x)
        tile_size = round(self.tile_size) + self.overlap

        return image[
            self.keep_in_bounds(y - tile_size) : self.keep_in_bounds(y + tile_size),
            self.keep_in_bounds(x - tile_size) : self.keep_in_bounds(x + tile_size),
        ]

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


def beadfinding(brightfield, preferred_thresholding, num_tiles=12, px_overlap=20):
    # tilize brightfield 0 -> tilemap
    # tileset = TileMap("tm", brightfield, px_overlap, num_tiles)

    # build tilaziable list
    # all_beads = np.array([[0, 0]])

    # # filtering information, min and max to exclude
    # filter_min = px_overlap
    # filter_max = px_overlap + brightfield.shape[0] // num_tiles

    # # for tile t in tilemap
    # i = 0
    # for tile, bounds in tileset:
    #     # thresholded_image = preferred_thresholding(tile)
    #     # contours, _ = cv2.findContours(
    #     #     thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #     # )

    #     beads = find_beads(tile)

    #     tile_top_left_x = bounds["center"][0] - tileset.tile_size - bounds["xmin"] / 2
    #     tile_top_left_y = bounds["center"][1] - tileset.tile_size - bounds["ymin"] / 2

    #     beads[:, 0] += round(tile_top_left_x)
    #     beads[:, 1] += round(tile_top_left_y)

    #     all_beads = np.concatenate((all_beads, beads), axis=0)

    #     i += 1
    # old method missed around 5% of beads, even after second pass
    # new method gets more beads and is like 2 passes in one
    # since second pass doesn't rarely adds new beads
    # (2688242-2548509)/2688242 = 0.0519793233
    
    # in addition, new method centroids are more centered than before
    # think before they were skewed top left due to always rounding down
    
    # new method also gives regions that can be displayed
    all_beads,roi_coords = find_beads(brightfield)
    
    return all_beads,roi_coords

def find_beads(brightfield):
    mask = threshold_otsu(brightfield)
    bw = closing(brightfield > mask, square(1))

    cleared = clear_border(bw)
    label_image = label(cleared)
    centers =[]
    coords = []
    for region in regionprops(label_image, brightfield):
        y, x = region.centroid_weighted
        coords.append(region.coords)
        centers.append([x,y])
    centers = np.array(centers)
    centers = np.rint(centers)
    centers = centers.astype(np.uint16)
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
from collections import defaultdict
import concurrent.futures
from functools import partial

def process_bead_batch(args):
    """Process a batch of beads for a single layer"""
    (flor_layer, beads_batch, radius, layer_threshold, roi_coords_batch, 
    start_idx, end_idx, reference_for_hist_match) = args
    
    batch_size = end_idx - start_idx
    layer_specific_data = np.zeros(batch_size, dtype="uint16")
    sig_noise_data = np.zeros(batch_size, dtype="float")
    layer_threshold_data = np.zeros(batch_size, dtype="float")
    
    # Apply histogram matching if needed
    if reference_for_hist_match is not None:
        flor_layer = match_histograms(flor_layer, reference_for_hist_match)
    
    # Process each bead in the batch
    if roi_coords_batch is not None:
        # Use provided ROI coordinates
        bead_rois = [
            flor_layer[coords[:, 0], coords[:, 1]]
            for coords in roi_coords_batch
        ]
    else:
        # Use circular ROI around bead coordinates
        bead_rois = [
            flor_layer[y - radius : y + radius, x - radius : x + radius]
            for x, y in beads_batch
        ]
    
    percentile_map = [np.percentile(region_vals, 10) for region_vals in bead_rois]
    
    for b_i, bead in enumerate(beads_batch):
        roi = bead_rois[b_i]
        brightness = np.median(roi)
        flor_layer_background_intensity_local = percentile_map[b_i]
        
        if flor_layer_background_intensity_local > 0:
            signal_noise_ratio = (
                brightness - flor_layer_background_intensity_local
            ) / flor_layer_background_intensity_local
        else:
            signal_noise_ratio = 0

        layer_specific_data[b_i] = brightness
        sig_noise_data[b_i] = signal_noise_ratio
        layer_threshold_data[b_i] = brightness > layer_threshold
    
    return layer_specific_data, sig_noise_data, layer_threshold_data

def get_excel(
    beads,
    signal_to_noise_cutoff,
    tifs,
    max_size,
    layer_threshold_dict=defaultdict(int),
    progress_callback=None,
    is_running_callback=None,
    roi_coords=None,
    batch_size=50000,  # Process in batches
    n_workers=6  # Number of parallel workers
):
    def update_progress_internal(value, message):
        if progress_callback:
            overall_progress = 40 + (value / 100) * 50
            progress_callback(int(overall_progress), message)

    def is_running():
        if is_running_callback:
            return is_running_callback()
        return True

    # beads = np.array(list(filter(edge_bead_filtering(20, max_size), beads)))
    # print(f"Processing {beads.shape[0]} beads")
    
    ColorThreshold = signal_to_noise_cutoff
    export_to_excel = np.zeros((len(beads), len(tifs)), dtype="uint8")

    results_from_cycles = []
    results_from_cycles_SNR = []
    results_from_cycles_Sig_absolute_threshold = []
    
    tif_metadata = [f.metadata for _, f in tifs]
    tif_images = [np.array(img) for img, _ in tifs]
    
    # Setup flors layers
    for i, md in enumerate(tif_metadata):
        assert isinstance(md, MetaData)
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
    radius = 2
    # roi_coords = np.array(roi_coords)
    bounding_boxes = np.array(len(beads))
    if roi_coords is not None:
        coords_array = [np.array(region) for region in roi_coords]

        bounding_boxes = [
            (*region.min(axis=0), *region.max(axis=0)) for region in coords_array
        ]
    # Process each cycle
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
                hist_match_ref = None  # Don't match first layer to itself
            else:
                hist_match_ref = reference_for_hist_match

            layer_threshold = layer_threshold_dict[layer]
            
            # Create batches for parallel processing
            n_batches = (total_beads + batch_size - 1) // batch_size
            batch_args = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_beads)
                
                beads_batch = beads[start_idx:end_idx]
                roi_coords_batch = roi_coords[start_idx:end_idx] if roi_coords else None
                
                batch_args.append((
                    flor_layer, beads_batch, radius, layer_threshold, 
                    roi_coords_batch, start_idx, end_idx, hist_match_ref
                ))
            
            # Process batches in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                batch_results = list(executor.map(process_bead_batch, batch_args))
            
            # Combine results from all batches
            for batch_idx, (layer_data, snr_data, threshold_data) in enumerate(batch_results):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_beads)
                
                cycle_specific_data[start_idx:end_idx, i] = layer_data
                cycle_specific_sig_noise_data[start_idx:end_idx, i] = snr_data
                cycle_specific_Sig_absolute_threshold_data[start_idx:end_idx, i] = threshold_data
            
            # Update progress
            progress = ((tif_count * total_layers_per_cycle + i + 1) / 
                       (total_cycles * total_layers_per_cycle)) * 100
            update_progress_internal(
                progress,
                f"Processing cycle {tif_count + 1}/{total_cycles}, layer {i + 1}/{total_layers_per_cycle}"
            )

        results_from_cycles.append(cycle_specific_data)
        results_from_cycles_SNR.append(cycle_specific_sig_noise_data)
        results_from_cycles_Sig_absolute_threshold.append(
            cycle_specific_Sig_absolute_threshold_data
        )

        brightest_layers = np.argmax(cycle_specific_data, axis=1)
        export_to_excel[:, tif_count] = brightest_layers

        for i, row in enumerate(cycle_specific_sig_noise_data):
            # if np.max(row) < ColorThreshold
            # ^ Previoys if statement, it filtered out rows
            # if any one of the channels were noisy
            # but this led to a lot of good/clear beads being filtered 
            # There were cases where there is a clear channel with a clear bead in the roi
            # but was filtered out because other channels were noisy 
            
            # so changed to if the channels are all noisy
            if np.min(row) > 1-ColorThreshold:
                export_to_excel[i, tif_count] = 255
                print("Sig noise filtered:", beads[i])

        for i, row in enumerate(cycle_specific_Sig_absolute_threshold_data):
            if np.all(row) == 0:
                export_to_excel[i, tif_count] = 255
                print("threshold filtered: ", beads[i])

    # Add bead coordinates as first two columns
    # bounding_boxes: shape (N, 4)
    bounding_boxes_str = np.array([
        f"({x1}, {y1}, {x2}, {y2})" for x1, y1, x2, y2 in bounding_boxes
    ]).reshape(-1, 1)  # Make it a column vector
    export_to_excel = np.hstack((beads[:, :2], export_to_excel, bounding_boxes_str))
    print(f"Final get excel length: {len(export_to_excel)}")
    return export_to_excel

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
        for region in tqdm.tqdm(regions, desc="Finding Centroids"):
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
        for i in tqdm.tqdm(range(num_proteins), desc="Building KD-Trees"):
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

        for cell_id in tqdm.tqdm(
            cell_data_dict.keys(), desc="Imputing Missing Proteins"
        ):
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

    log("Initial bead detection...")
    beads,roi_coords = beadfinding(brightfield, thresholding)
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

    bead_data = get_excel(
        beads,
        signal_to_noise_cutoff,
        tifs,
        max_size,
        progress_callback=progress_callback,
        is_running_callback=is_running,
        roi_coords = roi_coords
    )
    if not is_running():
        return None

    update_progress(90, "Filtering out rows with all zeros...")
    log(f"Beads with valid protein data: {len(bead_data)}")
    filtered_rows = []
    log("Filtering out rows set to 255 in protein channels...")
    for i, row in enumerate(tqdm.tqdm(bead_data)):
        if not is_running():
            return None
        if not np.any(bead_data[i, 2:-1] == 255):
            filtered_rows.append(row)
    filtered_rows = np.array(filtered_rows)
    bead_data = filtered_rows
    headers = ["x", "y"]
    for i in range(len(tifs)):
        headers.append(f"cy{i}")
    headers.append("bbox")
    df = pd.DataFrame(bead_data, columns=headers)
    log(f"Dataframe created with shape: {df.shape}")
    update_progress(100, "Bead generation complete.")
    return df
