import heapq
import logging

import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QThread, pyqtSignal
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from utils import calculate_ncc, to_uint8

logger = logging.getLogger(__name__)

_MIN_INLIERS = 5
_EMPTY_PTS = np.empty((0, 2), dtype=np.float32)
_BLOB_MATCH_RADIUS = 3.0
_QUAD_MIN_SIZE = 32
_QUAD_MAX_STD = 10.0


def scale_adjust(arr: np.ndarray) -> NDArray[np.uint8]:
    if arr.dtype == np.uint16:
        return cv2.convertScaleAbs(arr, alpha=(255.0 / 65535.0))
    elif arr.dtype == np.uint8:
        return np.clip(arr, 0, 255)
    elif arr.dtype in [np.uint32, np.int32, np.int64, np.uint64]:
        arr_clipped = np.clip(arr, 0, None)
        max_val = arr_clipped.max()
        if max_val == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr_clipped / max_val) * 255).astype(np.uint8)
    elif arr.dtype in [np.float32, np.float64]:
        arr_clipped = np.clip(arr, 0, None)
        max_val = arr_clipped.max()
        if max_val > 0:
            arr_scaled = (arr_clipped / max_val) * 255
        else:
            arr_scaled = arr_clipped
        return np.clip(arr_scaled, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported array type: {arr.dtype}")


def adjustContrast(img, alpha=5, beta=15):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def auto_contrast(img):
    return adjustContrast(scale_adjust(img))


def find_points(image, min_circularity=0.5, top_k=500):
    image = to_uint8(image.copy())
    image = cv2.GaussianBlur(image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                if len(heap) < top_k:
                    heapq.heappush(heap, (area, (cx, cy)))
                else:
                    heapq.heappushpop(heap, (area, (cx, cy)))
    top_centers = [center for _, center in sorted(heap, reverse=True)]
    return np.array(top_centers)


def find_points_robust(img, top_k=500, min_circularity=0.5):
    detectors = [
        ("ORB", cv2.ORB_create(nfeatures=top_k or 500)),
        ("CUSTOM", None),
    ]
    result = {}
    for name, detector in detectors:
        if name == "CUSTOM":
            points = find_points(img, min_circularity=min_circularity, top_k=top_k)
            result[name] = points
            continue
        kp = detector.detect(img, None)
        if len(kp) >= 10:
            points = np.array([p.pt for p in kp])
            result[name] = points
    return result


def try_optical_flow_alignment(source, target, moving_points, fixed_points):
    if len(moving_points) < 4:
        return None, False

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
        except cv2.error:
            logger.warning("Could not compute optical flow for this level: %s", params)
            continue
        good_indices = (status.flatten() == 1) & (err.flatten() < 50)

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
            if M is not None and np.sum(inliers) > best_inliers:
                best_inliers = np.sum(inliers)
                best_M = M

    return best_M, best_inliers


def quadtree_threshold(
    img: np.ndarray, min_size: int = _QUAD_MIN_SIZE, max_std: float = _QUAD_MAX_STD
) -> np.ndarray:
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    def process_tile(x0, y0, width, height):
        if width <= 0 or height <= 0:
            return
        tile = img[y0 : y0 + height, x0 : x0 + width]
        if tile.std() <= max_std or width <= min_size or height <= min_size:
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


def try_auto_contrast(img: np.ndarray) -> np.ndarray:
    return auto_contrast(img)


def brightfield_binarize(
    img: np.ndarray, min_size: int = _QUAD_MIN_SIZE, max_std: float = _QUAD_MAX_STD
) -> np.ndarray:
    return to_uint8(img)


def _blob_centroids(gray_u8: np.ndarray, max_pts: int = 4000) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 0:
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    if not pts:
        return _EMPTY_PTS
    arr = np.asarray(pts, dtype=np.float32)
    if len(arr) > max_pts:
        idx = np.random.default_rng(0).choice(len(arr), max_pts, replace=False)
        arr = arr[idx]
    return arr


def _matched_blob_fraction(
    ref_pts: np.ndarray, region_pts: np.ndarray, radius: float = _BLOB_MATCH_RADIUS
) -> float:
    if len(ref_pts) == 0 or len(region_pts) == 0:
        return 0.0
    tree = cKDTree(region_pts)
    dist, _ = tree.query(ref_pts, k=1, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(dist)))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] in (3, 4):
            return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        return img[..., 0]
    raise ValueError(f"Unsupported image shape {img.shape}")


def infer_tile_size(img: np.ndarray) -> int:
    try:
        gray = _to_gray(img)
        h, w = gray.shape[:2]

        profiles = []
        if w >= 256:
            profiles.append((np.mean(gray, axis=0), w))
        if h >= 256:
            profiles.append((np.mean(gray, axis=1), h))

        if not profiles:
            return 2000

        detected_sizes = []

        for profile, n in profiles:
            x = np.arange(n)
            poly = np.polyfit(x, profile, deg=2)
            detrended = profile - np.polyval(poly, x)

            min_lag = max(128, n // 20)
            max_lag = min(4096, n // 2)
            if max_lag <= min_lag:
                continue

            lags = np.arange(min_lag, max_lag)
            corrs = []

            d_mean = detrended.mean()
            d_std = detrended.std()
            if d_std <= 1e-5:
                continue
            norm_sig = (detrended - d_mean) / d_std

            for lag in lags:
                sig1 = norm_sig[:-lag]
                sig2 = norm_sig[lag:]
                c = np.mean(sig1 * sig2)
                corrs.append(c)

            corrs = np.array(corrs)
            if len(corrs) < 3:
                continue

            peaks = []
            for i in range(1, len(corrs) - 1):
                val = corrs[i]
                if val > corrs[i - 1] and val > corrs[i + 1]:
                    left_trough = np.min(corrs[max(0, i - 50) : i])
                    right_trough = np.min(corrs[i : min(len(corrs), i + 50)])
                    prominence = val - max(left_trough, right_trough)

                    if val > 0.05 and prominence > 0.02:
                        lag = lags[i]
                        peaks.append((lag, val, prominence))

            if peaks:
                peaks.sort(key=lambda x: (x[2], x[1]), reverse=True)
                detected_sizes.append(peaks[0][0])

        if not detected_sizes:
            return 2000

        best_size = int(np.median(detected_sizes))

        common_sizes = [256, 512, 1024, 2048, 2560, 3072, 4096]
        for standard in common_sizes:
            if abs(best_size - standard) <= max(16, int(standard * 0.05)):
                logger.info(
                    "Snapping inferred tile size %d to standard microscope size %d",
                    best_size,
                    standard,
                )
                return standard

        return best_size
    except Exception as exc:
        logger.warning(
            "Failed to infer tile size from shading: %s. Using default 2000.", exc
        )
        return 2000


class CropAnchorFinder(QThread):
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    candidates_ready = pyqtSignal(list)

    def __init__(
        self,
        reference_img: np.ndarray,
        moving_img: np.ndarray,
        patch_size: int = 0,
        num_candidates: int = 5,
        n_features: int = 50000,
        ratio: float = 0.75,
        ransac_thresh: float = 8.0,
        quad_min_size: int = _QUAD_MIN_SIZE,
        quad_max_std: float = _QUAD_MAX_STD,
        assume_no_transform: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.reference_img = reference_img
        self.moving_img = moving_img
        self.assume_no_transform = bool(assume_no_transform)
        if patch_size is None or int(patch_size) <= 0:
            self.patch_size = infer_tile_size(reference_img)
        else:
            self.patch_size = int(patch_size)
        self.num_candidates = max(1, int(num_candidates))
        self.n_features = int(n_features)
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.quad_min_size = int(quad_min_size)
        self.quad_max_std = float(quad_max_std)

    def run(self):
        try:
            candidates = self.find_candidates()
            self.candidates_ready.emit(candidates)
        except Exception as exc:
            logger.error("CropAnchorFinder failed: %s", exc, exc_info=True)
            self.error.emit(str(exc))

    def find_candidates(self) -> list[dict]:
        ref = _to_gray(self.reference_img)
        mov = _to_gray(self.moving_img)

        s = min(self.patch_size, ref.shape[0], ref.shape[1])
        self.progress.emit(10, "Extracting reference patch")
        patch = to_uint8(ref[:s, :s])
        mov_u8 = to_uint8(mov)

        downscale = 8
        self.progress.emit(30, "Downscaling images for global template match")

        ref_h, ref_w = patch.shape[:2]
        mov_h, mov_w = mov_u8.shape[:2]

        s_w, s_h = ref_w // downscale, ref_h // downscale
        m_w, m_h = mov_w // downscale, mov_h // downscale

        if s_w < 4 or s_h < 4 or m_w < s_w or m_h < s_h:
            downscale = 1
            ref_small = patch
            mov_small = mov_u8
        else:
            ref_small = cv2.resize(patch, (s_w, s_h), interpolation=cv2.INTER_AREA)
            mov_small = cv2.resize(mov_u8, (m_w, m_h), interpolation=cv2.INTER_AREA)

        self.progress.emit(50, "Performing template matching")
        try:
            res = cv2.matchTemplate(mov_small, ref_small, cv2.TM_CCOEFF_NORMED)
        except Exception as exc:
            logger.error("Template matching failed: %s", exc)
            return []

        template_h, template_w = ref_small.shape[:2]
        suppress_h = max(1, template_h // 2)
        suppress_w = max(1, template_w // 2)

        peaks = []
        res_copy = res.copy()

        for _ in range(self.num_candidates):
            _, max_val, _, max_loc = cv2.minMaxLoc(res_copy)
            if max_val < 0.05:
                break

            peaks.append((max_loc[0], max_loc[1], float(max_val)))

            x, y = max_loc
            y_start = max(0, y - suppress_h)
            y_end = min(res_copy.shape[0], y + suppress_h)
            x_start = max(0, x - suppress_w)
            x_end = min(res_copy.shape[1], x + suppress_w)

            res_copy[y_start:y_end, x_start:x_end] = -1.0

        if not peaks:
            logger.warning("No template match peaks found.")
            return []

        candidates = []
        self.progress.emit(
            70,
            f"Found {len(peaks)} candidate locations; starting optical flow refinement",
        )

        for idx, (x_small, y_small, coeff) in enumerate(peaks):
            x_full = float(x_small * downscale)
            y_full = float(y_small * downscale)

            T0 = np.array([[1.0, 0.0, x_full], [0.0, 1.0, y_full]], dtype=np.float64)
            warped = self._derotate(T0, mov_u8, s)
            best_flow_M = None
            best_flow_inliers = 0

            if warped is not None:
                moving_pts_dict = find_points_robust(warped, top_k=5000)
                fixed_pts_dict = find_points_robust(patch, top_k=5000)

                for key in moving_pts_dict:
                    if key not in fixed_pts_dict:
                        continue
                    mv_pts = moving_pts_dict[key]
                    fx_pts = fixed_pts_dict[key]
                    if len(mv_pts) < 4 or len(fx_pts) < 4:
                        continue

                    M_flow, flow_inliers = try_optical_flow_alignment(
                        warped, patch, mv_pts, fx_pts
                    )
                    if M_flow is not None and flow_inliers > best_flow_inliers:
                        best_flow_inliers = flow_inliers
                        best_flow_M = M_flow

            T_refined = T0.copy()
            if best_flow_M is not None:
                if self.assume_no_transform:
                    tx = float(best_flow_M[0, 2])
                    ty = float(best_flow_M[1, 2])
                    best_flow_M = np.array(
                        [[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64
                    )
                try:
                    invT = self._invert(T0)
                    A = invT[:, :2]
                    b = invT[:, 2]
                    C = best_flow_M[:, :2]
                    d = best_flow_M[:, 2]

                    refined_invT = np.hstack([C @ A, (C @ b + d).reshape(2, 1)])
                    T_refined = self._invert(refined_invT).astype(np.float64)
                    warped = self._derotate(T_refined, mov_u8, s)
                except Exception as e:
                    logger.warning(
                        "Failed combining local optical flow refinement: %s", e
                    )

            ncc = calculate_ncc(warped, patch) if warped is not None else None
            score = float(ncc) if ncc is not None else -1.0

            ref_pts = _blob_centroids(patch)
            region_pts = _blob_centroids(warped) if warped is not None else _EMPTY_PTS
            blob_frac = _matched_blob_fraction(ref_pts, region_pts)

            candidates.append(
                {
                    "anchor": (float(T_refined[0, 2]), float(T_refined[1, 2])),
                    "angle": float(
                        np.degrees(np.arctan2(T_refined[1, 0], T_refined[0, 0]))
                    ),
                    "inliers": int(best_flow_inliers),
                    "inlier_ratio": 1.0,
                    "residual": 0.0,
                    "spread": 1.0,
                    "blob_fraction": blob_frac,
                    "score": score,
                    "flow_inliers": int(best_flow_inliers),
                    "T": T_refined,
                }
            )

        candidates.sort(key=lambda c: (c["score"], c["flow_inliers"]), reverse=True)
        self.progress.emit(100, "Done")
        return candidates

    def _derotate(self, T, mov_u8, s):
        try:
            inv = self._invert(T).astype(np.float32)
            return cv2.warpAffine(mov_u8, inv, (s, s))
        except Exception:
            return None

    @staticmethod
    def _invert(T: np.ndarray) -> np.ndarray:
        A = np.asarray(T, dtype=np.float64)[:, :2]
        t = np.asarray(T, dtype=np.float64)[:, 2]
        invA = np.linalg.inv(A)
        return np.hstack([invA, (-invA @ t).reshape(2, 1)])
