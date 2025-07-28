import heapq
import math
import re
import time

import astroalign as aa
import cv2
import diplib as dip
import numpy as np
import pystackreg.util
import SimpleITK as sitk
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from image_processing import adjust_contrast
from utils import calculate_ncc, to_uint8


class Register(QThread):
    image_ready = pyqtSignal(bool)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    alignment_complete = pyqtSignal(list)

    def __init__(
        self,
        reference_image: np.ndarray,
        params: dict,
        to_be_aligned: list,
        template_size=None,
        max_points=5000,
    ):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 99999999999

        # initialize variables
        self._is_running = False

        self.params = params
        self.tifs = to_be_aligned
        self.image = reference_image
        self.min_circularity = 0.5
        self.template_size = template_size
        self.max_points = max_points

    def _fatal_error_message(self, msg):
        self.error.emit(msg)
        self.progress.emit(100, "Retry Maybe")

    def _handle_cancel(self):
        if self._is_running == False:
            self._fatal_error_message("Cancelled registration")
            return True
        else:
            return False

    def run_registration(self):
        self._is_running = True
        print("Running registration on device:", "CPU")  # or device_name
        if self.isRunning() and self._is_running:
            self.error.emit("Registration is already running")
            return
        elif self.isRunning():
            self.error.emit("Registration is cancelling")
            return
        self.start()
        self.finished.connect(lambda: setattr(self, "_is_running", False))
        self.finished.connect(self.deleteLater)

    def run(self):
        self.progress.emit(0, "preparing alignment")  # update progress bar
        m = self.params["max_size"]
        self.overlap = self.params["overlap"]  # overlap between each tile
        self.num_tiles = self.params["num_tiles"]  # how many tiles we want
        reference_tif_index = self.params["alignment_layer"]
        print("Reference channel is: ", reference_tif_index)
        print(f"Aligning {len(self.image)} channels with max size {m}")
        reference_bf = self.image[reference_tif_index][:m, :m]
        alignment_layers = []

        for i, tif in enumerate(self.tifs):
            if self._handle_cancel():
                return
            else:
                bf_channel = tif.get("alignment_layer", 0)
                if bf_channel >= len(tif["image"]):
                    self._fatal_error_message(
                        f"Alignment layer {bf_channel} out of bounds for image with {len(tif['image'])} channels"
                    )
                    return
                alignment_layers.append(
                    adjust_contrast(tif["image"][bf_channel][:m, :m], 50, 99)
                )

        fixed_map = TileMap(
            "fixed", reference_bf, int(self.overlap), int(self.num_tiles)
        )
        aligned_outputs = []
        moving_maps = []
        for tif_n, alignable_brightfield in enumerate(alignment_layers):
            if self._handle_cancel():
                return
            moving_map = TileMap(
                "moving", alignable_brightfield, self.overlap, self.num_tiles
            )
            moving_maps.append(moving_map)

            inputs = []
            radius = int(fixed_map.tile_size)
            for mov_data, fix_data in list(zip(moving_map, fixed_map)):

                (moving_img, moving_bounds) = mov_data
                (fixed_img, _) = fix_data

                x, y = moving_bounds["center"]
                ymin = moving_bounds["ymin"]
                xmin = moving_bounds["xmin"]

                radius = int(fixed_map.tile_size)
                inputs.append((fixed_img, moving_img, ymin, xmin, radius, x, y))

            # Select the inputs number
            outputs = []

            for tile_n, tile_set in enumerate(inputs):
                # update progress bar
                if self._handle_cancel():
                    return
                progress_update = int(((tile_n + 1) / len(inputs)) * 100)
                self.progress.emit(
                    progress_update,
                    str(f"aligning tile {tile_n+1}/{len(inputs)}"),
                )

                result = self.align_two_img_robust(*tile_set)  # align

                if result is None:
                    continue
                outputs.append(result)
            aligned_outputs.append(outputs)

        #########################################################
        # move the other layers
        total = 0
        aligned_tifs = []
        for i, bf in enumerate(alignment_layers):
            if self._handle_cancel():
                return
            progress_update = int(((i + 1) / len(alignment_layers)) * 100)
            dest = Image.fromarray(
                np.zeros((m, m), dtype="float")
            )  # need to determine the final bit size
            current_tif = self.tifs[i]
            aligned_channels = []
            for channel_image in current_tif["image"]:
                prev_transf = None
                prev_itk_transf = None
                moving_map = moving_maps[i]
                for result in aligned_outputs[i]:
                    transforms, ymin, xmin, radius, x, y, best_ncc = result
                    if best_ncc < 0.3:  # use previous
                        transforms[0] = prev_transf
                        transforms[1] = prev_itk_transf
                    else:
                        prev_transf = transforms[0]
                        prev_itk_transf = transforms[1]
                    source = moving_map.get_tile_by_center(channel_image, x, y)
                    target = moving_map.get_tile_by_center(reference_bf, x, y).astype(
                        float
                    )
                    total += 1
                    transf = transforms[0]
                    itk_transf = transforms[1]

                    if transf is not None:
                        if isinstance(transf, np.ndarray):
                            registered = cv2.warpAffine(
                                source, transf, (target.shape[1], target.shape[0])
                            )
                        else:
                            registered, _ = aa.apply_transform(transf, source, target)
                    else:
                        registered = source
                    # Apply additional ITK transformation if available
                    if itk_transf is not None:
                        registered = sitk.GetArrayFromImage(
                            sitk.Resample(
                                sitk.GetImageFromArray(registered),
                                sitk.GetImageFromArray(target),
                                itk_transf,
                                sitk.sitkLinear,
                                0.0,
                                sitk.sitkUInt16,
                            )
                        )

                    corresponding_tile = registered[
                        ymin : ymin + radius * 2, xmin : xmin + radius * 2
                    ]

                    dest.paste(
                        Image.fromarray(pystackreg.util.to_uint16(corresponding_tile)),
                        (int(x - radius), int(y - radius)),
                    )

                dest_arr = np.array(dest)
                aligned_channels.append(dest_arr)

            new_aligned_arrays = [x.astype("uint16") for x in aligned_channels]
            new_aligned_arrays = np.stack(new_aligned_arrays)
            aligned_tifs.append(new_aligned_arrays)
        # insert reference_bf

        self.alignment_complete.emit(aligned_tifs)
        self.progress.emit(100, "Alignment Done")

    def find_points(self, image, min_circularity=0.5, top_k=500):
        image = to_uint8(image.copy())
        image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduce noise
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Use a min-heap to keep top_k largest area centers
        heap = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity >= min_circularity:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Push to heap as (area, (cx, cy))
                    if len(heap) < top_k:
                        heapq.heappush(heap, (area, (cx, cy)))
                    else:
                        heapq.heappushpop(heap, (area, (cx, cy)))
        # Extract centers from heap, sorted by area descending
        top_centers = [center for _, center in sorted(heap, reverse=True)]
        return np.array(top_centers)

    def find_points_robust(self, img, top_k=500):
        """Try multiple feature detectors in order of preference"""
        detectors = [
            ("ORB", cv2.ORB_create(nfeatures=top_k or 500)),
            (
                "CUSTOM",
                self.find_points(
                    img, min_circularity=self.min_circularity, top_k=top_k
                ),
            ),
        ]
        result = {}
        for name, detector in detectors:
            if name == "CUSTOM":
                points = detector
                result[name] = points
                continue
            kp = detector.detect(img, None)
            if len(kp) >= 10:  # Minimum threshold
                points = np.array([p.pt for p in kp])
                result[name] = points
        return result

    def align_two_img_robust(
        self,
        fixed_img,
        moving_img,
        ymin,
        xmin,
        radius,
        x,
        y,
        tile_id=None,
        try_astro_align=False,
    ):
        source = moving_img.copy()
        target = fixed_img.copy()
        source = self.adjust_contrast(source, 50, 99)
        target = self.adjust_contrast(target, 50, 99)
        start_time = time.time()
        # Strategy 1: Optical flow with multiple scales
        moving_points = self.find_points_robust(source, top_k=self.max_points)
        fixed_points = self.find_points_robust(target, top_k=self.max_points)
        # ncc before
        ncc_before = calculate_ncc(source, target)
        # Try optical flow first
        best_ncc = 0
        transf = None
        name = ""
        ncc_after = -1
        best_registered = source
        itk_transf = None
        for key in moving_points:
            if len(moving_points[key]) < 4 or len(fixed_points[key]) < 4:
                continue
            moving_points[key] = moving_points[key][: self.max_points]
            fixed_points[key] = fixed_points[key][: self.max_points]
            M, success = self.try_optical_flow_alignment(
                source, target, moving_points[key], fixed_points[key]
            )
            if success:
                assert M is not None, "Optical flow transformation matrix is None"
                registered = cv2.warpAffine(
                    source, M, (target.shape[1], target.shape[0])
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after alignment is None"
                if ncc_after > best_ncc:
                    name = key
                    best_registered = registered
                    best_ncc = ncc_after
                    transf = M
                print(
                    f"Optical flow alignment successful with {key} points, NCC before: {ncc_before}, after: {ncc_after}"
                )
        if try_astro_align:
            try:
                t, _ = aa.find_transform(
                    moving_points["CUSTOM"],
                    fixed_points["CUSTOM"],
                    detection_sigma=3,
                    min_area=5,
                    max_control_points=50,
                )
                registered, _ = aa.apply_transform(t, source, target)
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after astro alignment is None"

                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                    transf = t
                    best_registered = registered
                    name = "AstroAlign"
            except Exception as e:
                print(f"AstroAlign failed: {e}")

        if (
            best_ncc < 0.5
        ):  # Only try feature matching if optical flow didn't yield good results
            M, success = self.try_feature_matching_alignment(source, target)
            if M is not None and success:
                registered = cv2.warpAffine(
                    source, M, (target.shape[1], target.shape[0])
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after feature matching is None"
                print(
                    f"Feature matching alignment successful, NCC before: {ncc_before}, after: {ncc_after}"
                )
                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                    transf = M
                    name = "SIFT"
                    best_registered = registered
        if best_ncc < 0.6:  # If still not good, improve with ITK
            itk_transf, s, t = self.try_itk_alignment(best_registered, target)
            if itk_transf is not None:
                registered = sitk.GetArrayFromImage(
                    sitk.Resample(
                        s, t, itk_transf, sitk.sitkLinear, 0.0, s.GetPixelID()
                    )
                )
                ncc_after = calculate_ncc(registered, target)
                assert ncc_after is not None, "NCC after ITK alignment is None"
                print(
                    f"ITK alignment successful, NCC before: {best_ncc}, after: {ncc_after}"
                )
                if ncc_after > best_ncc:
                    best_ncc = ncc_after
                else:
                    itk_transf = None
        return [transf, itk_transf], ymin, xmin, radius, x, y, best_ncc

    def try_optical_flow_alignment(self, source, target, moving_points, fixed_points):
        """Try optical flow with multiple pyramid levels"""
        if len(moving_points) < 4:
            return None, False

        # Parameters for different scales
        lk_params = [
            dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            ),
            dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            ),
            dict(
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
            ),
        ]
        best_inliers = 0
        best_M = None
        moving_points_cv = moving_points.astype(np.float32).reshape(-1, 1, 2)
        fixed_points_cv = fixed_points.astype(np.float32).reshape(-1, 1, 2)
        if len(moving_points_cv) > len(fixed_points_cv):
            moving_points_cv = moving_points_cv[: len(fixed_points_cv)]
        else:
            fixed_points_cv = fixed_points_cv[: len(moving_points_cv)]
        for params in lk_params:
            try:
                nextPts, status, err = cv2.calcOpticalFlowPyrLK(
                    source, target, moving_points_cv, fixed_points_cv, **params
                )
            except cv2.error as e:
                print(f"Could not compute optical flow for this level: {params}")
                continue
            good_indices = (status.flatten() == 1) & (
                err.flatten() < 50
            )  # Error threshold

            if np.sum(good_indices) >= 4:
                good_moving = moving_points_cv[good_indices][:, 0, :]
                good_next = nextPts[good_indices][:, 0, :]

                M, inliers = cv2.estimateAffine2D(
                    good_moving,
                    good_next,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=100000,
                    confidence=0.99,
                )
                if np.sum(inliers) > best_inliers:
                    best_inliers = np.sum(inliers)
                    best_M = M

        return best_M, best_inliers > 0

    def try_feature_matching_alignment(self, source, target):
        """Use SIFT/ORB feature matching instead of optical flow"""
        sift = cv2.SIFT_create()

        # Find keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(source, None)
        kp2, des2 = sift.detectAndCompute(target, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, False

        # Match features
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        matches = matcher.knnMatch(des1, des2, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return None, False

        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Estimate transformation with RANSAC
        M, mask = cv2.estimateAffine2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=100000,
            confidence=0.99,
        )

        return M, M is not None and np.sum(mask) >= 4

    def try_itk_alignment(self, source, target):
        """Use ITK for image registration"""

        # Convert images to SimpleITK format
        source_itk = sitk.GetImageFromArray(source)
        target_itk = sitk.GetImageFromArray(target)

        # Set pixel type to float
        source_itk = sitk.Cast(source_itk, sitk.sitkFloat32)
        target_itk = sitk.Cast(target_itk, sitk.sitkFloat32)

        # Initialize registration method
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        # Set initial transform
        transform_domain_mesh_size = [
            8,
            8,
        ]  # More control points = more flexible deformation
        bspline_transform = sitk.BSplineTransformInitializer(
            target_itk, transform_domain_mesh_size
        )

        registration_method.SetInitialTransform(bspline_transform, inPlace=False)
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-4,
            numberOfIterations=30,
            maximumNumberOfCorrections=5,
            costFunctionConvergenceFactor=1e6,
            maximumNumberOfFunctionEvaluations=200,
        )
        registration_method.SetInterpolator(sitk.sitkLinear)
        # Perform registration
        final_transform = registration_method.Execute(target_itk, source_itk)

        return final_transform, source_itk, target_itk

    def set_alignment_layer(self, channel):
        match = re.search(r"\d+", channel)
        if match:
            number = int(match.group())
            result = number - 1  # 0 index
            self.params["alignment_layer"] = result
            print("alignment layer is: ", self.params["alignment_layer"])

    def set_cell_layer(self, channel):
        match = re.search(r"\d+", channel)
        if match:
            number = int(match.group())
            result = number - 1  # 0 index
            self.params["cell_layer"] = result
            print("cell layer is: ", self.params["cell_layer"])

    def set_protein_detection_layer(self, channel):
        match = re.search(r"\d+", channel)
        if match:
            number = int(match.group())
            result = number - 1  # 0 index
            self.params["protein_detection_layer"] = result
        print("protein_detection_layer is: ", self.params["protein_detection_layer"])

    def set_max_size(self, value):
        self.params["max_size"] = value

    def set_num_tiles(self, value):
        self.params["num_tiles"] = value

    def set_overlap(self, value):
        self.params["overlap"] = value

    def on_skip(self, param):
        _, _, ymin, xmin, radius, x, y = param
        return (None, x, y, (None, ymin, xmin, radius, x, y))

    def adjust_contrast(self, img, contrast_min=2, contrast_max=98):
        # pixvals = np.array(img)
        minval = np.percentile(img, contrast_min)  # room for experimentation
        maxval = np.percentile(img, contrast_max)  # room for experimentation
        img = np.clip(img, minval, maxval)
        img = ((img - minval) / (maxval - minval)) * 255
        return img.astype(np.uint8)

    def equalize_shape(self, cy1_rescale, cy2_rescale):
        [cy1x, cy1y] = cy1_rescale.shape
        [cy2x, cy2y] = cy2_rescale.shape

        def relu(x):
            return x if x > 0 else 0

        pos = relu

        # print(pos(cy1x-cy2x), pos(cy1y-cy2y))
        cy2_rescale = np.pad(
            cy2_rescale,
            (
                (
                    int(math.floor(pos(cy1x - cy2x) / 2)),
                    int(math.ceil(pos(cy1x - cy2x) / 2)),
                ),
                (math.floor((pos(cy1y - cy2y) / 2)), math.ceil((pos(cy1y - cy2y) / 2))),
            ),
            "empty",
        )
        # Sometimes "edge" might work better

        cy2_rescale = cy2_rescale[0:cy1x, 0:cy1y]

        return cy1_rescale, cy2_rescale

    def update_moving_image(self, channels) -> None:
        self.protein_channels = channels
        self.tifs[1]["image_dict"] = channels
        if not self.reference_channels is None:
            self.image_ready.emit(True)
            print("moving/protein signal image updated")

    def update_reference_channels(self, reference_channels) -> None:
        self.reference_channels = reference_channels
        self.tifs[0]["image_dict"] = reference_channels
        if not self.protein_channels is None:
            self.image_ready.emit(True)
            print("reference image updated")

    def cancel(self):

        # self.exit?
        # self.quit?
        self._is_running = False


############################
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
            tuple((np.array(moving_array.shape) / 10).astype(int)),
            interpolation=cv2.INTER_LINEAR,
        )

        im = np.invert(threshold(blur(small), 20))

        out = dip.AreaOpening(im, filterSize=150, connectivity=2)  # type: ignore
        out = np.array(out)

        big = cv2.resize(
            out,
            tuple((np.array(moving_array.shape)).astype(int)),
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
        # print("Got ", x, y)
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
                # print("THIS IS THE TILE WE TALKIGN ABOUT", j)
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
                # print((i + 1), cuts, (j + 1), cuts)
                row.append(
                    np.array([(2 * i + 1) / (cuts * 2), (2 * j + 1) / (cuts * 2)])
                )
                # print((2*i + 1) / (cuts *2))

            centerpoints.append(np.array(row))

        return np.array(centerpoints)
