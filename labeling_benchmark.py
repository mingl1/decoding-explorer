"""
Labeling Benchmark — 5 segmentation strategies for fluorescent bead decoding.

Usage:
    python labeling_benchmark.py \
        --cycle1 path/to/cycle1.tif \
        --cycle2 path/to/cycle2.tif \
        --protein-csvs prot1.csv prot2.csv \
        --crop-size 1000 \
        --stardist-model path/to/model \
        --methods all
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json as json_mod
import os
import time
from datetime import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import filters, measure
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, square, white_tophat
from skimage.segmentation import clear_border
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class MetaData:
    flors_layers: list[int] = field(default_factory=list)
    max_size: int = 10000


# ---------------------------------------------------------------------------
# Margin ratio helpers
# ---------------------------------------------------------------------------
def _parse_margin_ratios(s: str) -> list[float]:
    """Parse '1.2' → [1.2]  or  '1.0-1.5' → [1.0, 1.05, 1.1, …, 1.5] (0.05 steps)."""
    s = str(s)
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = float(lo_s), float(hi_s)
        ratios, r = [], lo
        while r <= hi + 1e-9:
            ratios.append(round(r, 4))
            r += 0.05
        return ratios
    return [float(s)]


# ---------------------------------------------------------------------------
# Optimization cache
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".benchmark_cache")


def _cache_key(method: str, cycle1_path: str, cycle2_path: str, flors_layers: list, opt_crop_size: int) -> str:
    raw = json_mod.dumps((method, cycle1_path, cycle2_path, flors_layers, opt_crop_size), sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return os.path.join(_CACHE_DIR, f"{method}_{h}.json")


def _load_cache(path: str):
    if os.path.isfile(path):
        with open(path) as f:
            return json_mod.load(f)
    return None


def _save_cache(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json_mod.dump(data, f)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_tiff(path: str, crop_size: int | None) -> np.ndarray:
    """Load a TIFF and return (C, H, W) cropped to crop_size. If None, returns full image."""
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = img[np.newaxis]
    if crop_size is None:
        return img
    return img[:, :crop_size, :crop_size]


def load_protein_profiles(file_paths: list[str]) -> pd.DataFrame:
    """Load protein profiles from CSV/Excel files."""
    dfs = []
    for path in file_paths:
        if path.endswith(".csv"):
            dfs.append(pd.read_csv(path))
        else:
            dfs.append(pd.read_excel(path))
    merged = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    cols = merged.columns.tolist()
    renamed = {cols[0]: "Protein name"}
    for i in range(1, len(cols)):
        renamed[cols[i]] = f"cy{i - 1}"
    merged.rename(columns=renamed, inplace=True)
    print(f"Loaded {len(merged)} proteins, {len(cols) - 1} cycles")
    return merged


# ---------------------------------------------------------------------------
# Bead detection
# ---------------------------------------------------------------------------
def _quadtree_threshold(
    img: np.ndarray, min_size: int = 32, max_std: int = 10
) -> np.ndarray:
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


def find_beads(bf_image: np.ndarray) -> pd.DataFrame:
    """Detect beads from brightfield image (no StarDist). Returns DataFrame with x, y columns."""
    mask_value = _quadtree_threshold(bf_image)
    brighter_regions = np.where(bf_image, mask_value, 0)
    bw = closing(brighter_regions, square(1))
    cleared = clear_border(bw)
    label_image = label(cleared)
    centers = []
    for region in regionprops(label_image, bf_image):
        y, x = region.centroid_weighted
        centers.append([x, y])
    if not centers:
        return pd.DataFrame(columns=["x", "y"])
    arr = np.rint(np.array(centers)).astype(np.uint16)
    return pd.DataFrame(arr, columns=["x", "y"])


def find_beads_stardist(bf_image: np.ndarray) -> pd.DataFrame:
    """Detect beads from brightfield using pretrained StarDist 2D_versatile_fluo."""
    from csbdeep.utils import normalize
    from skimage.transform import rescale
    from stardist.models import StarDist2D

    scale = 2.0
    block_size = 700 if min(bf_image.shape[:2]) < 1000 else 1500

    print("  Loading pretrained StarDist model (2D_versatile_fluo)...")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    print(f"  Scaling image by {scale}...")
    img = rescale(bf_image, scale, preserve_range=True).astype(bf_image.dtype)

    print(f"  Running StarDist segmentation (block_size={block_size})...")
    img_norm = normalize(img, 1, 75)
    labels_img, _ = model.predict_instances_big(
        img_norm, axes="YX", block_size=block_size,
        min_overlap=24, show_progress=True, n_tiles=(1, 1),
    )
    num_labels = int(labels_img.max())
    if num_labels == 0:
        return pd.DataFrame(columns=["x", "y"])

    centroids = ndi.center_of_mass(labels_img, labels_img, range(1, num_labels + 1))
    centers = np.array([(c[1] / scale, c[0] / scale) for c in centroids if not np.isnan(c[0])])
    arr = np.rint(centers).astype(np.uint16)
    return pd.DataFrame(arr, columns=["x", "y"])


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def _normalize_01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return img


def _csbdeep_normalize(
    img: np.ndarray, pmin: float = 1.0, pmax: float = 99.8
) -> np.ndarray:
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)
    if hi > lo:
        return (img.astype(np.float32) - lo) / (hi - lo)
    return img.astype(np.float32)


def _preprocess_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement."""
    img_u8 = np.clip(img / max(img.max(), 1) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_u8).astype(np.float32)


def _preprocess_denoise(img: np.ndarray) -> np.ndarray:
    """Gaussian denoise + white tophat background subtraction."""
    img_f = img.astype(np.float32)
    img_smooth = gaussian_filter(img_f, sigma=1)
    return white_tophat(_normalize_01(img_smooth), disk(15))


def _preprocess_histmatch(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Histogram-match to reference."""
    return match_histograms(img.astype(np.float32), reference.astype(np.float32))


def _preprocess_equalize_median(img: np.ndarray) -> np.ndarray:
    """Histogram equalization + 3x3 median filter. Handles uint16 input."""
    img_u16 = img.astype(np.uint16)
    # cv2.equalizeHist requires uint8; rescale uint16 → uint16 range first
    # then use CLAHE-like manual equalization for 16-bit, or downscale to 8-bit
    # For uint16: normalize to full uint16 range, equalize via LUT
    lo, hi = img_u16.min(), img_u16.max()
    if hi > lo:
        scaled = ((img_u16.astype(np.float32) - lo) / (hi - lo) * 65535).astype(np.uint16)
    else:
        scaled = img_u16
    # Build 16-bit histogram equalization
    hist, _ = np.histogram(scaled.ravel(), bins=65536, range=(0, 65535))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total = scaled.size
    lut = ((cdf - cdf_min) / max(total - cdf_min, 1) * 65535).astype(np.uint16)
    eq = lut[scaled]
    return cv2.medianBlur(eq, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Adaptive watershed (from image_processing.py)
# ---------------------------------------------------------------------------
def _create_adaptive_threshold_maps(
    img: np.ndarray, tile_size: int, low_percentile: int, high_percentile: int
):
    img_h, img_w = img.shape
    tiles_h = int(np.ceil(img_h / tile_size))
    tiles_w = int(np.ceil(img_w / tile_size))
    low_res_low = np.zeros((tiles_h, tiles_w), dtype=np.float32)
    low_res_high = np.zeros((tiles_h, tiles_w), dtype=np.float32)

    for i in range(tiles_h):
        for j in range(tiles_w):
            y1, y2 = i * tile_size, min((i + 1) * tile_size, img_h)
            x1, x2 = j * tile_size, min((j + 1) * tile_size, img_w)
            tile = img[y1:y2, x1:x2]
            if tile.size > 0:
                low_res_low[i, j] = np.percentile(tile, low_percentile)
                low_res_high[i, j] = np.percentile(tile, high_percentile)

    full_low = cv2.resize(low_res_low, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    full_high = cv2.resize(low_res_high, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    return full_low, full_high


def _watershed_segmentation_adaptive(
    img: np.ndarray,
    border: int,
    tile_size: int = 64,
    low_percentile: int = 50,
    high_percentile: int = 65,
    fill: bool = True,
) -> np.ndarray:
    """Adaptive watershed segmentation. Returns labeled image."""
    img_float = img.astype(np.float32)
    adaptive_low, adaptive_high = _create_adaptive_threshold_maps(
        img_float, tile_size, low_percentile, high_percentile
    )
    elevation = filters.sobel(img_float)
    markers = np.zeros_like(img_float, dtype=np.int32)
    markers[img_float <= adaptive_low] = 1
    markers[img_float > adaptive_high] = 2
    elev_bgr = cv2.cvtColor((elevation * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    labeled_markers = cv2.watershed(elev_bgr, markers.copy())
    seg_mask = (labeled_markers >= 2).astype(np.uint8)
    if border > 0:
        seg_mask = ndi.binary_erosion(seg_mask, iterations=border)
    if fill:
        seg_mask = ndi.binary_fill_holes(
            seg_mask, structure=ndi.generate_binary_structure(2, 1)
        )
    return measure.label(seg_mask)


def _compute_prob_lut(labeled: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Compute mean normalized intensity per label as prob_lut."""
    n_labels = int(labeled.max()) + 1
    prob_lut = np.zeros(n_labels, dtype=np.float32)
    img_norm = _normalize_01(img)
    if n_labels <= 1:
        return prob_lut
    sums = ndi.sum(img_norm, labeled, index=np.arange(1, n_labels))
    counts = ndi.sum(np.ones_like(img_norm), labeled, index=np.arange(1, n_labels))
    counts = np.maximum(counts, 1)
    prob_lut[1:] = sums / counts
    return prob_lut


# ---------------------------------------------------------------------------
# SA optimization for StarDist (from notebook)
# ---------------------------------------------------------------------------
def sa_optimize_params(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    preprocess_fn,
    *,
    crop_size: int = 1000,
    target_min: int = 6500,
    target_max: int = 7500,
    prob_bounds: tuple[float, float] = (0.02, 0.40),
    nms_bounds: tuple[float, float] = (0.10, 0.50),
    block_size: int = 700,
    n_tiles: int = 3,
    min_overlap: int = 24,
    steps: int = 10,
    T0: float = 1.0,
    Tend: float = 0.05,
    prob_step: float = 0.03,
    nms_step: float = 0.05,
    seed: int = 0,
    tracer=None,
) -> list[list[dict]]:
    """
    SA-optimize prob_thresh and nms_thresh per (cycle, layer) for StarDist.
    Returns list[list[dict]] with keys: prob_thresh, nms_thresh, n_instances.
    """
    rng = np.random.RandomState(seed)
    params_all = []

    if tracer:
        tracer.init_sa_optimization(
            {
                "target_min": target_min,
                "target_max": target_max,
                "prob_bounds": list(prob_bounds),
                "nms_bounds": list(nms_bounds),
                "steps": steps,
                "T0": T0,
                "Tend": Tend,
            }
        )

    for ci, cycle in enumerate(cycles):
        flors = metadata_list[ci].flors_layers
        params_cycle = []
        for li, layer_idx in enumerate(flors):
            img_raw = cycle[layer_idx, :crop_size, :crop_size]
            img = _csbdeep_normalize(preprocess_fn(img_raw))

            # Score function: distance from target range
            def score_fn(prob, nms, _cache={}):
                key = (ci, li, round(prob, 4), round(nms, 3))
                if key in _cache:
                    return _cache[key]
                labels, _ = model.predict_instances_big(
                    img,
                    axes="YX",
                    block_size=block_size,
                    min_overlap=min_overlap,
                    n_tiles=(n_tiles, n_tiles),
                    prob_thresh=prob,
                    nms_thresh=nms,
                    show_progress=False,
                )
                n = int(labels.max())
                if target_min <= n <= target_max:
                    s = 0.0
                else:
                    s = min(abs(n - target_min), abs(n - target_max)) / max(
                        target_max, 1
                    )
                _cache[key] = (s, n)
                return s, n

            # SA
            if tracer:
                tracer.begin_sa_layer(ci, li)
            prob = 0.15
            nms = 0.40
            best_score, best_n = score_fn(prob, nms)
            best_prob, best_nms = prob, nms

            for step in range(steps):
                T = T0 * (Tend / T0) ** (step / max(steps - 1, 1))
                new_prob = np.clip(
                    prob + rng.uniform(-prob_step, prob_step), *prob_bounds
                )
                new_nms = np.clip(nms + rng.uniform(-nms_step, nms_step), *nms_bounds)
                new_score, new_n = score_fn(new_prob, new_nms)

                delta = new_score - best_score
                accepted = delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-8))
                if accepted:
                    prob, nms = new_prob, new_nms
                    if new_score < best_score or (new_score == best_score == 0.0):
                        best_score, best_n = new_score, new_n
                        best_prob, best_nms = prob, nms

                if tracer:
                    tracer.record_sa_step(
                        step, new_prob, new_nms, new_score, new_n, T, accepted
                    )

                if best_score == 0.0:
                    break

            params_cycle.append(
                {
                    "prob_thresh": best_prob,
                    "nms_thresh": best_nms,
                    "n_instances": best_n,
                }
            )
            if tracer:
                tracer.record_sa_best(ci, li, best_prob, best_nms, best_n)
            print(
                f"  cy{ci}_layer{li}: prob={best_prob:.3f} nms={best_nms:.3f} n={best_n}"
            )

        params_all.append(params_cycle)
    return params_all


# ---------------------------------------------------------------------------
# Grid search optimization for watershed
# ---------------------------------------------------------------------------
def grid_search_watershed(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    bead_df: pd.DataFrame,
    crop_size: int,
    protein_df: pd.DataFrame,
    param_grid: dict | None = None,
    tracer=None,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
) -> dict:
    """Grid search over watershed params. Returns best (border_erode, low_pct, high_pct).

    Scoring: invalid_pct = n_invalid / (n_valid / num_proteins) * 100
    Target: under 7%, preferably 5%.
    """
    if param_grid is None:
        param_grid = {
            "border_erode": [0, 1, 2],
            "low_percentile": [40, 50, 60],
            "high_percentile": [60, 65, 70, 75],
        }

    combos = [
        dict(zip(param_grid.keys(), v))
        for v in itertools.product(*param_grid.values())
        if dict(zip(param_grid.keys(), v)).get("high_percentile", 0)
        > dict(zip(param_grid.keys(), v)).get("low_percentile", -1)
    ]

    if tracer:
        tracer.init_grid_search(param_grid)

    best_score = float("inf")
    best_params = combos[0] if combos else {}

    num_cycles = len(cycles)

    for params in tqdm(combos, desc="Watershed grid search"):
        # Run watershed with these params on all cycles/layers
        cycle_labels = _run_watershed_all(cycles, metadata_list, crop_size, params)
        # Evaluate full pipeline (pre-merge stats to avoid double-counting)
        df = assign_labels_to_beads(bead_df, cycle_labels)
        num_layers = len(metadata_list[0].flors_layers)
        df = enforce_single_layer(df, num_cycles, num_layers, min_prob=min_prob, min_prob_ratio=min_prob_ratio)
        df = resolve_to_cycles(df, num_cycles, num_layers)
        stats = compute_stats(df, protein_df, num_cycles)
        score = stats["invalid_pct"]
        if tracer:
            tracer.record_grid_result(params, score)
        if score < best_score:
            best_score = score
            best_params = params

    if tracer:
        tracer.record_grid_best(best_params, best_score)
    print(f"Best watershed params: {best_params} (invalid%={best_score:.2f}%)")
    return best_params


def _run_watershed_all(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    crop_size: int,
    params: dict,
    tracer=None,
) -> list[list[dict]]:
    """Run adaptive watershed on all cycle/layers with given params."""
    cycle_labels = []
    for ci, cycle in enumerate(cycles):
        layer_labels = []
        for li, layer_idx in enumerate(metadata_list[ci].flors_layers):
            img = _normalize_01(cycle[layer_idx, :crop_size, :crop_size])
            labeled = _watershed_segmentation_adaptive(
                img,
                border=params["border_erode"],
                low_percentile=params["low_percentile"],
                high_percentile=params["high_percentile"],
            )
            prob_lut = _compute_prob_lut(labeled, img)
            if tracer:
                tracer.record_layer_segmentation(
                    ci, li, int(labeled.max()), params, labeled
                )
            layer_labels.append({"lbl": labeled.astype(np.int32), "prob_lut": prob_lut})
        cycle_labels.append(layer_labels)
    return cycle_labels


# ---------------------------------------------------------------------------
# StarDist label generation (shared by stardist methods)
# ---------------------------------------------------------------------------
def _run_stardist_all(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    crop_size: int,
    preprocess_fn,
    sa_params: list[list[dict]],
    tracer=None,
) -> list[list[dict]]:
    """Run StarDist with SA-optimized params on all cycle/layers."""
    cycle_labels = []
    for ci, cycle in enumerate(cycles):
        layer_labels = []
        for li, layer_idx in enumerate(metadata_list[ci].flors_layers):
            img_raw = cycle[layer_idx, :crop_size, :crop_size]
            img = _csbdeep_normalize(preprocess_fn(img_raw))
            p = sa_params[ci][li]
            labels, _ = model.predict_instances_big(
                img,
                axes="YX",
                block_size=700,
                min_overlap=24,
                n_tiles=(3, 3),
                prob_thresh=p["prob_thresh"],
                nms_thresh=p["nms_thresh"],
                show_progress=False,
            )
            labels = labels.astype(np.int32)
            prob_lut = _compute_prob_lut(labels, img)
            if tracer:
                tracer.record_layer_segmentation(
                    ci, li, int(labels.max()), dict(p), labels
                )
            layer_labels.append({"lbl": labels, "prob_lut": prob_lut})
        cycle_labels.append(layer_labels)
    return cycle_labels


# ---------------------------------------------------------------------------
# 6 labeling methods
# ---------------------------------------------------------------------------
def labels_stardist_clahe(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    crop_size: int,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
) -> list[list[dict]]:
    """Method 1: CLAHE + SA-optimized StarDist."""
    if opt_crop_size is None:
        opt_crop_size = crop_size
    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        print(f"\n[stardist_clahe] Loaded cached params from {cache_path}")
        sa_params = cached
    else:
        print("\n[stardist_clahe] SA optimizing...")
        sa_params = sa_optimize_params(
            cycles,
            metadata_list,
            model,
            _preprocess_clahe,
            crop_size=opt_crop_size,
            tracer=tracer,
        )
        if cache_path:
            _save_cache(cache_path, sa_params)
    print("[stardist_clahe] Running StarDist...")
    return _run_stardist_all(
        cycles,
        metadata_list,
        model,
        crop_size,
        _preprocess_clahe,
        sa_params,
        tracer=tracer,
    )


def labels_stardist_denoise(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    crop_size: int,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
) -> list[list[dict]]:
    """Method 2: Denoise + background subtract + SA-optimized StarDist."""
    if opt_crop_size is None:
        opt_crop_size = crop_size
    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        print(f"\n[stardist_denoise] Loaded cached params from {cache_path}")
        sa_params = cached
    else:
        print("\n[stardist_denoise] SA optimizing...")
        sa_params = sa_optimize_params(
            cycles,
            metadata_list,
            model,
            _preprocess_denoise,
            crop_size=opt_crop_size,
            tracer=tracer,
        )
        if cache_path:
            _save_cache(cache_path, sa_params)
    print("[stardist_denoise] Running StarDist...")
    return _run_stardist_all(
        cycles,
        metadata_list,
        model,
        crop_size,
        _preprocess_denoise,
        sa_params,
        tracer=tracer,
    )


def labels_watershed_optimized(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    crop_size: int,
    bead_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
) -> list[list[dict]]:
    """Method 3: Grid-search optimized adaptive watershed."""
    if opt_crop_size is None:
        opt_crop_size = crop_size
    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        print(f"\n[watershed_optimized] Loaded cached params from {cache_path}")
        best_params = cached
    else:
        print("\n[watershed_optimized] Grid searching...")
        # Filter beads to opt region for grid search evaluation
        opt_bead_df = bead_df[(bead_df["x"] < opt_crop_size) & (bead_df["y"] < opt_crop_size)].copy()
        best_params = grid_search_watershed(
            cycles, metadata_list, opt_bead_df, opt_crop_size, protein_df, tracer=tracer,
            min_prob=min_prob, min_prob_ratio=min_prob_ratio,
        )
        if cache_path:
            _save_cache(cache_path, best_params)
    print("[watershed_optimized] Running with best params...")
    return _run_watershed_all(
        cycles, metadata_list, crop_size, best_params, tracer=tracer
    )


def labels_stardist_histmatch(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    crop_size: int,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
) -> list[list[dict]]:
    """Method 4: Histogram match to reference + SA-optimized StarDist."""
    if opt_crop_size is None:
        opt_crop_size = crop_size
    # Reference = first fluorescent layer of cycle 1
    ref_layer_idx = metadata_list[0].flors_layers[0]
    reference = cycles[0][ref_layer_idx, :crop_size, :crop_size].astype(np.float32)

    def preprocess_histmatch(img):
        return _preprocess_histmatch(img, reference)

    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        print(f"\n[stardist_histmatch] Loaded cached params from {cache_path}")
        sa_params = cached
    else:
        print("\n[stardist_histmatch] SA optimizing...")
        sa_params = sa_optimize_params(
            cycles,
            metadata_list,
            model,
            preprocess_histmatch,
            crop_size=opt_crop_size,
            tracer=tracer,
        )
        if cache_path:
            _save_cache(cache_path, sa_params)
    print("[stardist_histmatch] Running StarDist...")
    return _run_stardist_all(
        cycles,
        metadata_list,
        model,
        crop_size,
        preprocess_histmatch,
        sa_params,
        tracer=tracer,
    )


def labels_stardist_raw(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    model,
    crop_size: int,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
) -> list[list[dict]]:
    """Method 6: No preprocessing — only CSBDeep normalize (applied by runner)."""
    def _identity(img):
        return img.astype(np.float32)

    if opt_crop_size is None:
        opt_crop_size = crop_size
    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        print(f"\n[stardist_raw] Loaded cached params from {cache_path}")
        sa_params = cached
    else:
        print("\n[stardist_raw] SA optimizing...")
        sa_params = sa_optimize_params(
            cycles,
            metadata_list,
            model,
            _identity,
            crop_size=opt_crop_size,
            tracer=tracer,
        )
        if cache_path:
            _save_cache(cache_path, sa_params)
    print("[stardist_raw] Running StarDist...")
    return _run_stardist_all(
        cycles,
        metadata_list,
        model,
        crop_size,
        _identity,
        sa_params,
        tracer=tracer,
    )


def labels_threshold_connected(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    crop_size: int,
    tracer=None,
) -> list[list[dict]]:
    """Method 5: Threshold + connected components (simplest baseline)."""
    print("\n[threshold_connected] Running...")
    cycle_labels = []
    for ci, cycle in enumerate(cycles):
        layer_labels = []
        for li, layer_idx in enumerate(metadata_list[ci].flors_layers):
            img = cycle[layer_idx, :crop_size, :crop_size].astype(np.float32)
            img_norm = _normalize_01(img)
            thresh = np.percentile(img_norm, 85)
            binary = img_norm > thresh
            binary = ndi.binary_fill_holes(binary)
            binary = closing(binary.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
            labeled = measure.label(binary).astype(np.int32)
            prob_lut = _compute_prob_lut(labeled, img)
            if tracer:
                tracer.record_layer_segmentation(
                    ci, li, int(labeled.max()), {"percentile": 85}, labeled
                )
            layer_labels.append({"lbl": labeled, "prob_lut": prob_lut})
        cycle_labels.append(layer_labels)
    return cycle_labels


def _equalize_median_segmentation(
    img: np.ndarray,
    threshold_percentile: int = 80,
    border_erode: int = 1,
) -> np.ndarray:
    """Segment using equalize_hist + median preprocessing, then threshold."""
    preprocessed = _preprocess_equalize_median(img)
    normed = _normalize_01(preprocessed)
    thresh = np.percentile(normed, threshold_percentile)
    binary = normed > thresh
    binary = ndi.binary_fill_holes(binary)
    binary = closing(binary.astype(np.uint8), np.ones((3, 3), dtype=np.uint8))
    if border_erode > 0:
        binary = ndi.binary_erosion(binary, iterations=border_erode)
    return measure.label(binary).astype(np.int32)


def _run_equalize_median_all(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    crop_size: int,
    params: dict,
    tracer=None,
) -> list[list[dict]]:
    """Run equalize+median segmentation on all cycle/layers.

    Activation is assigned to the layer with the highest median intensity
    per segmented instance, implemented via prob_lut (median intensity per label).
    enforce_single_layer downstream picks the highest-prob layer per cycle.
    """
    cycle_labels = []
    for ci, cycle in enumerate(cycles):
        layer_labels = []
        for li, layer_idx in enumerate(metadata_list[ci].flors_layers):
            img = cycle[layer_idx, :crop_size, :crop_size].astype(np.float32)
            labeled = _equalize_median_segmentation(
                img,
                threshold_percentile=params["threshold_percentile"],
                border_erode=params["border_erode"],
            )
            # prob_lut = median intensity per label (activation = highest median)
            n_labels = int(labeled.max()) + 1
            prob_lut = np.zeros(n_labels, dtype=np.float32)
            img_norm = _normalize_01(img)
            for lbl_id in range(1, n_labels):
                pixels = img_norm[labeled == lbl_id]
                if len(pixels) > 0:
                    prob_lut[lbl_id] = np.median(pixels)
            if tracer:
                tracer.record_layer_segmentation(
                    ci, li, int(labeled.max()), params, labeled
                )
                # Record median activation stats per layer
                valid_medians = prob_lut[1:]  # skip background
                if len(valid_medians) > 0:
                    tracer.record_pipeline_stage(
                        f"median_activation_cy{ci}_layer{li}",
                        {
                            "n_instances": int(labeled.max()),
                            "median_min": round(float(valid_medians.min()), 4),
                            "median_max": round(float(valid_medians.max()), 4),
                            "median_mean": round(float(valid_medians.mean()), 4),
                        },
                    )
            layer_labels.append({"lbl": labeled, "prob_lut": prob_lut})
        cycle_labels.append(layer_labels)
    return cycle_labels


def grid_search_equalize_median(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    bead_df: pd.DataFrame,
    crop_size: int,
    protein_df: pd.DataFrame,
    param_grid: dict | None = None,
    tracer=None,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
) -> dict:
    """Grid search over equalize+median params. Target: invalid < 5%."""
    if param_grid is None:
        param_grid = {
            "threshold_percentile": [70, 75, 80, 85, 90],
            "border_erode": [0, 1, 2],
        }

    combos = [
        dict(zip(param_grid.keys(), v))
        for v in itertools.product(*param_grid.values())
    ]

    if tracer:
        tracer.init_grid_search(param_grid)

    best_score = float("inf")
    best_params = combos[0] if combos else {}
    num_cycles = len(cycles)

    for params in tqdm(combos, desc="Equalize+median grid search"):
        cycle_labels = _run_equalize_median_all(cycles, metadata_list, crop_size, params)
        df = assign_labels_to_beads(bead_df, cycle_labels)
        num_layers = len(metadata_list[0].flors_layers)
        df = enforce_single_layer(df, num_cycles, num_layers, min_prob=min_prob, min_prob_ratio=min_prob_ratio)
        df = resolve_to_cycles(df, num_cycles, num_layers)
        stats = compute_stats(df, protein_df, num_cycles)
        score = stats["invalid_pct"]
        if tracer:
            tracer.record_grid_result(params, score)
        if score < best_score:
            best_score = score
            best_params = params

    if tracer:
        tracer.record_grid_best(best_params, best_score)
    return best_params


def labels_equalize_median(
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    crop_size: int,
    bead_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    opt_crop_size: int | None = None,
    cache_paths: tuple[str, ...] | None = None,
    tracer=None,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
) -> list[list[dict]]:
    """Equalize hist + median 3x3, activation = highest median, grid-search optimized."""
    if opt_crop_size is None:
        opt_crop_size = crop_size
    cache_path = cache_paths[0] if cache_paths else None
    cached = _load_cache(cache_path) if cache_path else None
    if cached is not None:
        best_params = cached
        if tracer:
            tracer.record_pipeline_stage("params_source", {"source": "cache", "path": cache_path, **best_params})
    else:
        opt_bead_df = bead_df[(bead_df["x"] < opt_crop_size) & (bead_df["y"] < opt_crop_size)].copy()
        best_params = grid_search_equalize_median(
            cycles, metadata_list, opt_bead_df, opt_crop_size, protein_df, tracer=tracer,
            min_prob=min_prob, min_prob_ratio=min_prob_ratio,
        )
        if cache_path:
            _save_cache(cache_path, best_params)
        if tracer:
            tracer.record_pipeline_stage("params_source", {"source": "grid_search", "opt_crop_size": opt_crop_size, **best_params})
    return _run_equalize_median_all(
        cycles, metadata_list, crop_size, best_params, tracer=tracer
    )


# ---------------------------------------------------------------------------
# Downstream pipeline
# ---------------------------------------------------------------------------
def assign_labels_to_beads(
    bead_df: pd.DataFrame,
    cycle_labels: list[list[dict]],
    min_patch_support: int = 2,
    invalid_value: int = 0,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Assign bead labels with center-pixel + 5x5 patch fallback."""
    bead_df = bead_df.copy()
    xs0 = np.rint(bead_df["x"].to_numpy()).astype(np.int32)
    ys0 = np.rint(bead_df["y"].to_numpy()).astype(np.int32)
    N = xs0.size

    # 5x5 offsets
    dx = np.array([-2, -1, 0, 1, 2] * 5, dtype=np.int32)
    dy = np.array([-2] * 5 + [-1] * 5 + [0] * 5 + [1] * 5 + [2] * 5, dtype=np.int32)

    for i, cycle_layers in enumerate(cycle_labels):
        for j, layer in enumerate(cycle_layers):
            lbl = layer["lbl"]
            prob_lut = layer["prob_lut"]
            H, W = lbl.shape[:2]

            out_id = np.empty(N, dtype=np.int32)
            out_prob = np.empty(N, dtype=np.float32)

            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                xs = np.clip(xs0[s:e], 0, W - 1)
                ys = np.clip(ys0[s:e], 0, H - 1)

                # Center assignment
                center_id = lbl[ys, xs].astype(np.int32)
                center_prob = prob_lut[center_id].astype(np.float32)
                trust_center = center_id > 0

                best_id = np.where(trust_center, center_id, invalid_value).astype(
                    np.int32
                )
                best_prob = np.where(trust_center, center_prob, 0.0).astype(np.float32)

                # 5x5 patch fallback
                need_patch = ~trust_center
                if np.any(need_patch):
                    xs_np = xs[need_patch]
                    ys_np = ys[need_patch]
                    xs_n = np.clip(xs_np[:, None] + dx[None, :], 0, W - 1)
                    ys_n = np.clip(ys_np[:, None] + dy[None, :], 0, H - 1)

                    ids = lbl[ys_n, xs_n].astype(np.int32)
                    probs = prob_lut[ids].astype(np.float32)
                    probs_sel = np.where(ids > 0, probs, -np.inf)
                    best_k = np.argmax(probs_sel, axis=1)
                    row = np.arange(ids.shape[0])
                    patch_best_id = ids[row, best_k]
                    patch_best_prob = prob_lut[patch_best_id].astype(np.float32)

                    # Support check
                    if min_patch_support > 1:
                        support = (ids == patch_best_id[:, None]).sum(axis=1)
                        ok = (support >= min_patch_support) & (patch_best_id > 0)
                    else:
                        ok = patch_best_id > 0

                    idxs = np.flatnonzero(need_patch)
                    good_idxs = idxs[ok]
                    best_id[good_idxs] = patch_best_id[ok]
                    best_prob[good_idxs] = patch_best_prob[ok]

                out_id[s:e] = best_id
                out_prob[s:e] = best_prob

            bead_df[f"cy{i}_{j}"] = out_id
            bead_df[f"cy{i}_{j}_prob"] = out_prob

    return bead_df


def enforce_single_layer(
    df: pd.DataFrame,
    num_cycles: int,
    num_layers: int,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
    invalid_value: int = 255,
) -> pd.DataFrame:
    """Keep only highest-prob layer per cycle for ambiguous beads.

    min_prob_ratio: if > 0, only resolve ambiguity when best_prob / second_prob
    >= ratio.  Otherwise invalidate ALL layers for that bead (filter rather
    than guess).
    """
    df = df.copy()
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        prob_cols = [f"cy{i}_{j}_prob" for j in range(num_layers)]

        layers = df[layer_cols].to_numpy()
        probs = df[prob_cols].to_numpy()

        valid_mask = (layers != invalid_value) & (layers > 0) & (probs >= min_prob)
        valid_counts = valid_mask.sum(axis=1)
        ambiguous_idx = np.where(valid_counts > 1)[0]

        if len(ambiguous_idx) > 0:
            masked_probs = probs.copy()
            masked_probs[~valid_mask] = -np.inf
            best_layers = np.argmax(masked_probs, axis=1)

            # If min_prob_ratio is active, check confidence gap
            ratio_fail = None
            if min_prob_ratio > 0:
                sorted_probs = np.sort(masked_probs, axis=1)[:, ::-1]
                best_prob = sorted_probs[:, 0]
                second_prob = sorted_probs[:, 1]
                # Ratio check only for ambiguous beads
                ratio_ok = np.ones(len(df), dtype=bool)
                amb_mask = valid_counts > 1
                safe_second = np.where(second_prob > 0, second_prob, 1e-8)
                ratio_ok[amb_mask] = (best_prob[amb_mask] / safe_second[amb_mask]) >= min_prob_ratio
                # Invalidate ALL layers for ambiguous beads that fail ratio check
                ratio_fail = amb_mask & ~ratio_ok
                if ratio_fail.any():
                    layers[ratio_fail] = invalid_value
                    probs[ratio_fail] = 0.0

            keep_mask = np.zeros_like(valid_mask, dtype=bool)
            keep_mask[np.arange(len(df)), best_layers] = True

            invalidate_mask = valid_mask & ~keep_mask
            invalidate_mask[valid_counts <= 1] = False
            # Don't touch beads already fully invalidated by ratio check
            if ratio_fail is not None:
                invalidate_mask[ratio_fail] = False

            layers[invalidate_mask] = invalid_value
            probs[invalidate_mask] = 0.0

            df[layer_cols] = layers
            df[prob_cols] = probs

        # Invalidate all below-threshold layers so resolve_to_cycles sees clean data
        below_thresh = (layers != invalid_value) & (layers > 0) & (probs < min_prob)
        if below_thresh.any():
            layers[below_thresh] = invalid_value
            probs[below_thresh] = 0.0
            df[layer_cols] = layers
            df[prob_cols] = probs
    return df


def resolve_to_cycles(
    df: pd.DataFrame,
    num_cycles: int,
    num_layers: int,
    invalid_value: int = 255,
) -> pd.DataFrame:
    """Convert per-layer columns to single cy{i} assignment per cycle."""
    final_df = df[["x", "y"]].copy()
    out_cols = [f"cy{i}" for i in range(num_cycles)]
    final_df[out_cols] = np.uint16(invalid_value)

    valid_layer_masks = []
    counts_list = []
    for i in range(num_cycles):
        layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
        labs = df[layer_cols].to_numpy()
        valid = (labs != invalid_value) & (labs > 0)
        valid_layer_masks.append(valid)
        counts_list.append(valid.sum(axis=1))

    counts = np.stack(counts_list, axis=1)
    valid_mask = (counts == 1).all(axis=1)

    if valid_mask.any():
        for i in range(num_cycles):
            layer_idx = valid_layer_masks[i].argmax(axis=1).astype(np.uint16)
            final_df.loc[valid_mask, f"cy{i}"] = layer_idx[valid_mask]

    return final_df


def label_with_proteins(
    bead_df: pd.DataFrame, protein_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge bead cycle assignments with protein profiles."""
    cycle_cols = [c for c in protein_df.columns if c.startswith("cy")]
    for col in cycle_cols:
        if col in bead_df.columns:
            bead_df[col] = bead_df[col].astype(int)
            protein_df[col] = protein_df[col].astype(int)
    result = bead_df.merge(protein_df, on=cycle_cols, how="left")
    result["Protein name"] = result["Protein name"].fillna("Invalid")
    filter_mask = pd.Series(False, index=result.index)
    for col in cycle_cols:
        filter_mask |= result[col] >= 254
    result.loc[filter_mask, "Protein name"] = "Filtered"
    return result


def compute_stats(
    bead_df: pd.DataFrame, protein_df: pd.DataFrame, num_cycles: int,
) -> dict:
    """Compute Valid/Invalid/Filtered counts from pre-merge bead df.

    num_proteins = number of unique cycle-column combos in protein_df.
    invalid_pct = n_invalid / (n_valid / num_proteins) * 100
    """
    cycle_cols = [f"cy{i}" for i in range(num_cycles)]
    total = len(bead_df)
    if total == 0:
        return {"valid_pct": 0, "invalid_pct": 0, "filtered_pct": 0,
                "n_valid": 0, "n_invalid": 0, "n_filtered": 0, "total": 0}

    # Filtered: any cycle == 255
    filtered_mask = (bead_df[cycle_cols] >= 254).any(axis=1)
    n_filtered = int(filtered_mask.sum())

    # Valid: cycle combo exists in protein_df (among non-filtered beads)
    protein_combos = protein_df[cycle_cols].drop_duplicates()
    num_proteins = len(protein_combos)
    not_filtered = bead_df.loc[~filtered_mask, cycle_cols].astype(int)
    valid_mask_nf = not_filtered.merge(
        protein_combos.assign(_match=1), on=cycle_cols, how="left"
    )["_match"].fillna(0).astype(bool)
    n_valid = int(valid_mask_nf.sum())
    n_invalid = total - n_filtered - n_valid

    expected_per_protein = max(n_valid, 1) / max(num_proteins, 1)
    return {
        "valid_pct": 100 * n_valid / total,
        "invalid_pct": 100 * n_invalid / expected_per_protein,
        "filtered_pct": 100 * n_filtered / total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_filtered": n_filtered,
        "total": total,
        "num_proteins": num_proteins,
    }


def print_stats(stats: dict) -> None:
    """Print stats dict."""
    total = stats["total"]
    if total == 0:
        print("  No beads.")
        return
    n_valid = stats["n_valid"]
    n_invalid = stats["n_invalid"]
    n_filtered = stats["n_filtered"]
    num_proteins = stats["num_proteins"]
    print(f"  Valid:    {stats['valid_pct']:.1f}%  ({n_valid}/{total})")
    print(f"  Invalid:  {stats['invalid_pct']:.1f}%  ({n_invalid} / ({n_valid}/{num_proteins}))")
    print(f"  Filtered: {stats['filtered_pct']:.1f}%  ({n_filtered}/{total})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def benchmark(
    name: str,
    method_fn,
    bead_df: pd.DataFrame,
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    protein_df: pd.DataFrame,
    tracer=None,
    output_folder: str | None = None,
    enforce_single_layer_on: bool = True,
    min_prob: float = 0.02,
    min_prob_ratio: float = 0.0,
) -> dict:
    """Run one labeling method end-to-end and print stats."""
    num_cycles = len(cycles)
    num_layers = len(metadata_list[0].flors_layers)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    if tracer:
        tracer.begin_method(name)

    t0 = time.time()
    cycle_labels = method_fn()
    elapsed = time.time() - t0

    df = assign_labels_to_beads(bead_df, cycle_labels)
    if tracer:
        nonzero = (
            (
                df[
                    [
                        c
                        for c in df.columns
                        if c.startswith("cy") and "_prob" not in c and "_" in c
                    ]
                ]
                > 0
            )
            .any(axis=1)
            .sum()
        )
        tracer.record_pipeline_stage(
            "assign_labels_to_beads",
            {
                "total_beads": len(df),
                "assigned_nonzero": int(nonzero),
            },
        )
        # Per-layer: how many segmented instances were never assigned to a bead
        layer_coverage = {}
        for i, cycle_layers in enumerate(cycle_labels):
            for j, layer in enumerate(cycle_layers):
                n_detected = int(layer["lbl"].max())
                col = f"cy{i}_{j}"
                assigned_ids = df[col].to_numpy()
                n_assigned = len(np.unique(assigned_ids[(assigned_ids > 0) & (assigned_ids != 255)]))
                layer_coverage[col] = {
                    "detected": n_detected,
                    "assigned_to_bead": n_assigned,
                    "unassigned": n_detected - n_assigned,
                }
        tracer.record_pipeline_stage("layer_assignment_coverage", layer_coverage)

    if tracer:
        diag = {}
        for i in range(num_cycles):
            layer_cols = [f"cy{i}_{j}" for j in range(num_layers)]
            prob_cols = [f"cy{i}_{j}_prob" for j in range(num_layers)]
            layers = df[layer_cols].to_numpy()
            probs = df[prob_cols].to_numpy()
            assigned = (layers > 0) & (layers != 255)
            counts = assigned.sum(axis=1)
            cycle_diag = {
                "0_layers": int((counts == 0).sum()),
                "1_layer": int((counts == 1).sum()),
                "2plus_layers": int((counts >= 2).sum()),
            }
            if (counts >= 2).any():
                amb_probs = probs[counts >= 2]
                amb_valid = amb_probs[amb_probs > 0]
                if len(amb_valid):
                    cycle_diag["ambiguous_prob_min"] = round(float(amb_valid.min()), 4)
                    cycle_diag["ambiguous_prob_max"] = round(float(amb_valid.max()), 4)
                    cycle_diag["ambiguous_prob_median"] = round(float(np.median(amb_valid)), 4)
            diag[f"cy{i}"] = cycle_diag
        tracer.record_pipeline_stage("pre_enforce_diagnostics", diag)

    if enforce_single_layer_on:
        df = enforce_single_layer(df, num_cycles, num_layers, min_prob=min_prob, min_prob_ratio=min_prob_ratio)
        if tracer:
            tracer.record_pipeline_stage("enforce_single_layer", {"rows": len(df)})
    else:
        if tracer:
            tracer.record_pipeline_stage("enforce_single_layer", {"skipped": True})

    df = resolve_to_cycles(df, num_cycles, num_layers)
    if tracer:
        cycle_cols = [f"cy{i}" for i in range(num_cycles)]
        valid = (df[cycle_cols] != 255).all(axis=1).sum()
        invalid = len(df) - valid
        tracer.record_pipeline_stage(
            "resolve_to_cycles",
            {
                "valid_beads": int(valid),
                "invalid_beads": int(invalid),
            },
        )

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_cols = ["x", "y"] + [f"cy{i}" for i in range(num_cycles)]
        csv_path = os.path.join(output_folder, f"{name}.csv")
        df[out_cols].to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

    # Compute stats on pre-merge df to avoid double-counting
    stats = compute_stats(df, protein_df, num_cycles)
    print_stats(stats)
    stats["elapsed_s"] = elapsed

    if tracer:
        tracer.record_pipeline_stage(
            "final_stats",
            {
                "total": stats["total"],
                "valid": stats["n_valid"],
                "invalid": stats["n_invalid"],
                "filtered": stats["n_filtered"],
                "valid_pct": round(stats["valid_pct"], 1),
                "invalid_pct": round(stats["invalid_pct"], 1),
                "filtered_pct": round(stats["filtered_pct"], 1),
                "num_proteins": stats["num_proteins"],
            },
        )
        tracer.record_final_stats(stats)
        tracer.end_method()

    print(f"  Time:     {elapsed:.1f}s")
    return stats


def _build_voronoi_resolved_df(
    bead_df, assigned_layers, assigned_values, threshold,
    assigned_margins=None, min_margin_ratio=1.0,
):
    """Build a resolved bead_df filtering beads below threshold and margin."""
    N = len(bead_df)
    surviving = np.ones(N, dtype=bool)
    for vals in assigned_values.values():
        surviving &= vals >= threshold
    if assigned_margins and min_margin_ratio > 1.0:
        for margins in assigned_margins.values():
            surviving &= margins >= min_margin_ratio

    resolved = bead_df[["x", "y"]].copy()
    for cy_key, layers in assigned_layers.items():
        col = np.full(N, 255, dtype=np.uint8)
        col[surviving] = layers[surviving].astype(np.uint8)
        resolved[cy_key] = col
    return resolved


def benchmark_voronoi_median(
    name: str,
    bead_df: pd.DataFrame,
    cycles: list[np.ndarray],
    metadata_list: list[MetaData],
    protein_df: pd.DataFrame,
    crop_size: int,
    tracer=None,
    output_folder: str | None = None,
    min_assigned_value: float = 0.1,
    min_margin_ratio: float | list[float] = 1.0,
    border_erosion: int = 0,
    voronoi_cache_path: str | None = None,
) -> dict:
    """Run voronoi median labeling end-to-end and print stats."""
    num_cycles = len(cycles)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    if tracer:
        tracer.begin_method(name)

    t0 = time.time()

    run_tracer = tracer
    if run_tracer is None:
        from benchmark_tracer import BenchmarkTracer
        run_tracer = BenchmarkTracer()
        run_tracer.begin_method(name)

    resolved_df, assigned_layers, assigned_values, assigned_margins = (
        run_tracer.record_voronoi_median_analysis(
            bead_df, cycles, metadata_list,
            shape=(crop_size, crop_size),
            min_assigned_value=min_assigned_value,
            border_erosion=border_erosion,
            voronoi_cache_path=voronoi_cache_path,
        )
    )
    elapsed = time.time() - t0

    # --- Margin ratio sweep ---
    # If a list of ratios is given, sweep them and auto-select best valid%-invalid%.
    # Otherwise fall back to the fixed diagnostic set.
    if isinstance(min_margin_ratio, (list, tuple)):
        sweep_ratios = list(min_margin_ratio)
        auto_select = len(sweep_ratios) > 1
    else:
        sweep_ratios = [float(min_margin_ratio)]
        auto_select = False

    display_ratios = sweep_ratios if auto_select else [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

    print(f"\n  Margin ratio sweep (min_assigned_value={min_assigned_value}"
          + (" — auto-selecting best" if auto_select else "") + "):")
    print(f"  {'Ratio':>6}  {'Filt%':>7}  {'Valid%':>7}  {'Invld%':>7}  {'Score':>7}  {'Kept':>6}")
    print(f"  {'-'*52}")

    margin_sweep = {}
    for ratio in display_ratios:
        rdf = _build_voronoi_resolved_df(
            bead_df, assigned_layers, assigned_values, min_assigned_value,
            assigned_margins, ratio,
        )
        s = compute_stats(rdf, protein_df, num_cycles)
        _inv = s["invalid_pct"]
        _penalty = (_inv - 0.4) / 4 if _inv > 0.4 else 0
        score = round(s["valid_pct"] - _penalty, 2)
        margin_sweep[f"r{ratio}"] = {
            "min_margin_ratio": ratio,
            "valid_pct": round(s["valid_pct"], 1),
            "invalid_pct": round(s["invalid_pct"], 1),
            "filtered_pct": round(s["filtered_pct"], 1),
            "score": score,
            "n_valid": s["n_valid"],
            "n_invalid": s["n_invalid"],
            "n_filtered": s["n_filtered"],
        }
        print(
            f"  {ratio:>5.2f}x  {s['filtered_pct']:>6.1f}%  "
            f"{s['valid_pct']:>6.1f}%  {s['invalid_pct']:>6.1f}%  "
            f"{score:>7.2f}  {s['n_valid'] + s['n_invalid']:>6}"
        )

    if auto_select:
        # Walk from highest ratio (most filtered) down; stop when
        # the invalid% gain exceeds the valid% gain at that step.
        desc = sorted(sweep_ratios, reverse=True)
        selected_margin_ratio = desc[0]  # fallback: most filtered
        for i in range(1, len(desc)):
            prev_r, curr_r = desc[i - 1], desc[i]
            d_valid   = margin_sweep[f"r{curr_r}"]["valid_pct"]   - margin_sweep[f"r{prev_r}"]["valid_pct"]
            d_invalid = margin_sweep[f"r{curr_r}"]["invalid_pct"] - margin_sweep[f"r{prev_r}"]["invalid_pct"]
            if 0.9 * d_invalid > d_valid:
                break  # this step costs more than it gains — keep prev
            selected_margin_ratio = curr_r  # gain still worth it, keep going
        print(f"\n  Auto-selected min_margin_ratio={selected_margin_ratio} "
              f"(next step: Δvalid={d_valid:.1f}% Δinvalid={d_invalid:.1f}%)")
    else:
        selected_margin_ratio = sweep_ratios[0]

    if tracer:
        tracer.record_pipeline_stage("margin_sweep", margin_sweep)

    # --- Percentile sweep on assigned values ---
    all_vals = np.concatenate(list(assigned_values.values()))
    percentiles = [0, 5, 10, 25, 50, 75, 90]
    thresholds = {p: float(np.percentile(all_vals, p)) for p in percentiles}

    print(f"\n  Threshold sweep (no margin filter):")
    print(f"  {'Pctl':>5}  {'Thresh':>7}  {'Filt%':>7}  {'Valid%':>7}  {'Invld%':>7}  {'Kept':>6}")
    print(f"  {'-'*47}")

    percentile_sweep = {}
    for p, thresh in thresholds.items():
        rdf = _build_voronoi_resolved_df(bead_df, assigned_layers, assigned_values, thresh)
        s = compute_stats(rdf, protein_df, num_cycles)
        percentile_sweep[f"p{p}"] = {
            "threshold": round(thresh, 4),
            "valid_pct": round(s["valid_pct"], 1),
            "invalid_pct": round(s["invalid_pct"], 1),
            "filtered_pct": round(s["filtered_pct"], 1),
            "n_valid": s["n_valid"],
            "n_invalid": s["n_invalid"],
            "n_filtered": s["n_filtered"],
        }
        print(
            f"  p{p:<4}  {thresh:>7.4f}  {s['filtered_pct']:>6.1f}%  "
            f"{s['valid_pct']:>6.1f}%  {s['invalid_pct']:>6.1f}%  "
            f"{s['n_valid'] + s['n_invalid']:>6}"
        )

    if tracer:
        tracer.record_pipeline_stage("percentile_sweep", percentile_sweep)

    # --- Primary stats (using selected ratio) ---
    resolved_df = _build_voronoi_resolved_df(
        bead_df, assigned_layers, assigned_values, min_assigned_value,
        assigned_margins, selected_margin_ratio,
    )
    print(f"\n  Using min_assigned_value={min_assigned_value}, min_margin_ratio={selected_margin_ratio}:")
    stats = compute_stats(resolved_df, protein_df, num_cycles)
    print_stats(stats)
    stats["elapsed_s"] = elapsed

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_cols = ["x", "y"] + [f"cy{i}" for i in range(num_cycles)]
        csv_path = os.path.join(output_folder, f"{name}.csv")
        resolved_df[out_cols].to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        if auto_select:
            sweep_path = os.path.join(output_folder, "margin_sweep.json")
            with open(sweep_path, "w") as _sf:
                json_mod.dump({
                    "selected_margin_ratio": selected_margin_ratio,
                    "sweep": margin_sweep,
                }, _sf, indent=2)
            print(f"  Saved margin sweep → {sweep_path}")

    if tracer:
        tracer.record_pipeline_stage(
            "final_stats",
            {
                "total": stats["total"],
                "valid": stats["n_valid"],
                "invalid": stats["n_invalid"],
                "filtered": stats["n_filtered"],
                "valid_pct": round(stats["valid_pct"], 1),
                "invalid_pct": round(stats["invalid_pct"], 1),
                "filtered_pct": round(stats["filtered_pct"], 1),
                "num_proteins": stats["num_proteins"],
            },
        )
        tracer.record_final_stats(stats)
        tracer.end_method()

    print(f"  Time:     {elapsed:.1f}s")
    return stats


def _infer_flors_layers(cycle: np.ndarray) -> list[int]:
    """Channel 0 = brightfield, rest = fluorescent."""
    n_channels = cycle.shape[0]
    return list(range(1, n_channels))


def main():
    parser = argparse.ArgumentParser(description="Labeling Benchmark — 5 strategies")
    parser.add_argument("--cycle1", required=True, help="Path to cycle 1 TIFF")
    parser.add_argument("--cycle2", required=True, help="Path to cycle 2 TIFF")
    parser.add_argument(
        "--protein-csvs", nargs="+", required=True, help="Protein profile CSV(s)"
    )
    parser.add_argument(
        "--crop-size", type=int, default=None,
        help="Crop size (default: min of image width and height)"
    )
    parser.add_argument(
        "--stardist-model", default=None, help="Path to StarDist model dir"
    )
    parser.add_argument(
        "--bead-csv",
        default=None,
        help="Path to CSV with bead centers (must have x,y columns; floats OK). "
        "Skips bead detection if provided.",
    )
    parser.add_argument(
        "--bead-finding",
        choices=["stardist", "no-stardist"],
        default="stardist",
        help="Bead detection method: stardist (default, pretrained 2D_versatile_fluo) "
        "or no-stardist (quadtree threshold + regionprops). Ignored if --bead-csv is set.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to run. Names: stardist_clahe stardist_denoise stardist_raw "
        "watershed_optimized stardist_histmatch threshold_connected equalize_median "
        "voronoi_median. Groups: all, stardist, no_stardist. "
        "Prefix with - to exclude: 'all -stardist' or '-stardist_clahe'.",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help="Path to save trace JSON (default: no tracing)",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder to save per-method CSV output (default: no CSV)",
    )
    parser.add_argument(
        "--enforce-single-layer",
        choices=["on", "off", "both"],
        default="on",
        help="Toggle enforce_single_layer: on (default), off, or both",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.02,
        help="min_prob threshold for enforce_single_layer (default: 0.02)",
    )
    parser.add_argument(
        "--min-prob-ratio",
        type=float,
        default=0.0,
        help="Min best/second prob ratio for ambiguity resolution (default: 0.0 = disabled). "
        "E.g. 1.5 means only pick a winner if it's 50%% more confident than the runner-up.",
    )
    parser.add_argument(
        "--min-margin-ratio",
        type=str,
        default="1.0",
        help="Voronoi: min best/second-best ratio to keep a bead (default: 1.0 = disabled). "
        "Single value (e.g. 1.5) or range (e.g. 1.0-1.5) at 0.05 steps; "
        "range auto-selects the ratio with highest valid%% - invalid%%.",
    )
    parser.add_argument(
        "--border-erosion",
        type=int,
        default=0,
        help="Voronoi: pixels to erode from region borders before computing medians (default: 0). "
        "E.g. 3 excludes pixels within 3px of any Voronoi boundary.",
    )
    args = parser.parse_args()

    # Load data
    print("Loading TIFFs...")
    cy1 = load_tiff(args.cycle1, None)
    if args.crop_size is None:
        args.crop_size = min(cy1.shape[-2], cy1.shape[-1])
        print(f"Auto crop-size: {args.crop_size} (min of image dims)")
    cy1 = cy1[:, :args.crop_size, :args.crop_size]
    cy2 = load_tiff(args.cycle2, args.crop_size)
    cycles = [cy1, cy2]

    # Infer metadata
    flors1 = _infer_flors_layers(cy1)
    flors2 = _infer_flors_layers(cy2)
    metadata_list = [
        MetaData(flors_layers=flors1, max_size=args.crop_size),
        MetaData(flors_layers=flors2, max_size=args.crop_size),
    ]

    print(f"Cycle 1: {cy1.shape} — flors layers: {flors1}")
    print(f"Cycle 2: {cy2.shape} — flors layers: {flors2}")

    # Load proteins
    protein_df = load_protein_profiles(args.protein_csvs)

    # Detect beads from brightfield (channel 0) of cycle 1
    if args.bead_csv:
        print(f"Loading bead centers from {args.bead_csv}...")
        bead_df = pd.read_csv(args.bead_csv)
        # Normalize column names to lowercase
        bead_df.columns = [c.lower() for c in bead_df.columns]
        if "x" not in bead_df.columns or "y" not in bead_df.columns:
            raise ValueError(f"Bead CSV must have x,y columns. Found: {list(bead_df.columns)}")
        bead_df = bead_df[["x", "y"]].astype(np.float32)
    else:
        bf = cy1[0, : args.crop_size, : args.crop_size]
        if args.bead_finding == "stardist":
            print("Detecting beads (StarDist pretrained)...")
            bead_df = find_beads_stardist(bf)
        else:
            print("Detecting beads (quadtree threshold)...")
            bead_df = find_beads(bf)
    print(f"Found {len(bead_df)} beads")

    # Resolve methods — supports groups and -prefix exclusions
    ALL_METHODS = [
        "stardist_clahe",
        "stardist_denoise",
        "stardist_raw",
        "watershed_optimized",
        "stardist_histmatch",
        "threshold_connected",
        "equalize_median",
        "voronoi_median",
    ]
    METHOD_GROUPS = {
        "all": ALL_METHODS,
        "stardist": [m for m in ALL_METHODS if m.startswith("stardist")],
        "no_stardist": [m for m in ALL_METHODS if not m.startswith("stardist")],
    }
    methods = []
    excludes = set()
    for tok in args.methods:
        if tok.startswith("-"):
            key = tok[1:]
            excludes.update(METHOD_GROUPS.get(key, [key]))
        else:
            methods.extend(METHOD_GROUPS.get(tok, [tok]))
    if not methods:
        methods = list(ALL_METHODS)
    methods = [m for m in methods if m not in excludes]

    # Auto-generate run dir if output-folder and trace-output not specified
    if args.output_folder is None and args.trace_output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        methods_tag = "_".join(methods[:3])
        if len(methods) > 3:
            methods_tag += f"_+{len(methods) - 3}"
        run_dir = os.path.join("runs", f"{stamp}_{methods_tag}")
        os.makedirs(run_dir, exist_ok=True)
        args.output_folder = run_dir
        args.trace_output = os.path.join(run_dir, "trace.json")
        print(f"Auto-created run dir: {run_dir}")

    # Set up tracer
    tracer = None
    if args.trace_output:
        from benchmark_tracer import BenchmarkTracer

        tracer = BenchmarkTracer()
        tracer.set_config(
            crop_size=args.crop_size,
            num_beads=len(bead_df),
            num_cycles=len(cycles),
            cycle1=args.cycle1,
            cycle2=args.cycle2,
            min_prob=args.min_prob,
            min_prob_ratio=args.min_prob_ratio,
            enforce_single_layer=args.enforce_single_layer,
        )

    # Load StarDist model if needed
    model = None

    needs_stardist = any(m.startswith("stardist") for m in methods)
    if needs_stardist:
        if args.stardist_model is None:
            parser.error("--stardist-model is required for StarDist methods")
        from stardist.models import StarDist2D

        print(f"Loading StarDist model from {args.stardist_model}...")
        model = StarDist2D(None, name=args.stardist_model)

    # Optimization crop size
    opt_crop_size = min(args.crop_size, 1000)

    # Cache key helper
    flors_all = [metadata_list[i].flors_layers for i in range(len(metadata_list))]
    def _mk_cache(method):
        return (_cache_key(method, args.cycle1, args.cycle2, flors_all, opt_crop_size),)

    # Method registry
    method_fns = {
        "stardist_clahe": lambda: labels_stardist_clahe(
            cycles, metadata_list, model, args.crop_size,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("stardist_clahe"), tracer=tracer,
        ),
        "stardist_denoise": lambda: labels_stardist_denoise(
            cycles, metadata_list, model, args.crop_size,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("stardist_denoise"), tracer=tracer,
        ),
        "stardist_raw": lambda: labels_stardist_raw(
            cycles, metadata_list, model, args.crop_size,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("stardist_raw"), tracer=tracer,
        ),
        "watershed_optimized": lambda: labels_watershed_optimized(
            cycles, metadata_list, args.crop_size, bead_df, protein_df,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("watershed_optimized"), tracer=tracer,
            min_prob=args.min_prob, min_prob_ratio=args.min_prob_ratio,
        ),
        "stardist_histmatch": lambda: labels_stardist_histmatch(
            cycles, metadata_list, model, args.crop_size,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("stardist_histmatch"), tracer=tracer,
        ),
        "threshold_connected": lambda: labels_threshold_connected(
            cycles, metadata_list, args.crop_size, tracer=tracer
        ),
        "equalize_median": lambda: labels_equalize_median(
            cycles, metadata_list, args.crop_size, bead_df, protein_df,
            opt_crop_size=opt_crop_size, cache_paths=_mk_cache("equalize_median"), tracer=tracer,
            min_prob=args.min_prob, min_prob_ratio=args.min_prob_ratio,
        ),
    }

    # Run benchmarks
    all_stats = {}
    for method_name in methods:
        if method_name == "voronoi_median":
            cy1_stem = os.path.splitext(os.path.basename(args.cycle1))[0]
            _vor_cache = os.path.join(_CACHE_DIR, f"voronoi_{cy1_stem}_{args.crop_size}.npy")
            stats = benchmark_voronoi_median(
                method_name,
                bead_df,
                cycles,
                metadata_list,
                protein_df,
                crop_size=args.crop_size,
                tracer=tracer,
                output_folder=args.output_folder,
                min_margin_ratio=_parse_margin_ratios(args.min_margin_ratio),
                border_erosion=args.border_erosion,
                voronoi_cache_path=_vor_cache,
            )
            all_stats[method_name] = stats
            continue

        if method_name not in method_fns:
            print(f"Unknown method: {method_name}")
            continue

        # Determine enforce_single_layer runs
        if args.enforce_single_layer == "off":
            runs = [("", False)]
        elif args.enforce_single_layer == "both":
            runs = [("", True), ("_no_enforce", False)]
        else:
            runs = [("", True)]

        for suffix, enforce_on in runs:
            run_name = method_name + suffix
            stats = benchmark(
                run_name,
                method_fns[method_name],
                bead_df,
                cycles,
                metadata_list,
                protein_df,
                tracer=tracer,
                output_folder=args.output_folder,
                enforce_single_layer_on=enforce_on,
                min_prob=args.min_prob,
                min_prob_ratio=args.min_prob_ratio,
            )
            all_stats[run_name] = stats

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Method':<25} {'Valid%':>8} {'Invalid%':>9} {'Filtered%':>10} {'Time':>8}")
    print("-" * 60)
    for name, s in all_stats.items():
        print(
            f"{name:<25} {s['valid_pct']:>7.1f}% {s['invalid_pct']:>8.1f}% "
            f"{s['filtered_pct']:>9.1f}% {s['elapsed_s']:>7.1f}s"
        )

    if tracer:
        tracer.save(args.trace_output)


if __name__ == "__main__":
    main()
