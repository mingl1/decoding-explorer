"""BenchmarkTracer — accumulates tracing data and writes JSON."""

from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime

import numpy as np
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm


def _median_per_region(values, regions, n_regions):
    """
    Exact median of `values` grouped by integer `regions` in [0..n_regions-1].
    Returns (n_regions,) float32; regions with no pixels get 0.
    Filters out negative region ids (sentinel for eroded/excluded pixels).
    """
    # Filter out sentinel pixels (region == -1) before sorting
    mask = regions >= 0
    if not mask.all():
        values = values[mask]
        regions = regions[mask]

    values = values.astype(np.float32, copy=False)
    regions = regions.astype(np.int32, copy=False)

    # Single argsort on region*offset + value is faster than lexsort
    # Pack region into high bits so sort groups by region, then by value within
    n_pixels = len(values)
    if n_pixels == 0:
        return np.zeros(n_regions, dtype=np.float32)

    order = np.argsort(regions, kind="mergesort")  # stable sort groups by region
    r_sorted = regions[order]
    v_sorted = values[order]

    # Within each region group, sort by value
    change = np.empty(n_pixels, dtype=bool)
    change[0] = True
    change[1:] = r_sorted[1:] != r_sorted[:-1]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], n_pixels]
    r_ids = r_sorted[starts]

    # Sort values within each group for median
    for i in range(len(starts)):
        s, e = starts[i], ends[i]
        if e - s > 1:
            v_sorted[s:e] = np.partition(v_sorted[s:e], (e - s) // 2)

    out = np.zeros(n_regions, dtype=np.float32)
    counts = ends - starts
    mid_lo = starts + (counts - 1) // 2
    mid_hi = starts + counts // 2
    med = 0.5 * (v_sorted[mid_lo] + v_sorted[mid_hi])
    out[r_ids] = med
    return out


def voronoi_from_centers_tiled(
    xy,  # (N,2) float/int in (x,y) order
    shape,  # (H,W)
    *,
    tile=2048,
    leaf_size=40,
    dtype=np.int32,
    show_progress=True,
):
    """
    Returns:
      labels: (H,W) int32 where each pixel = index of nearest center in [0..N-1]
    """
    H, W = shape
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2) in (x,y) order")

    tree = KDTree(xy, leaf_size=leaf_size)
    labels = np.empty((H, W), dtype=dtype)

    y_starts = list(range(0, H, tile))
    x_starts = list(range(0, W, tile))
    total_tiles = len(y_starts) * len(x_starts)

    pbar = tqdm(total=total_tiles, desc="Voronoi tiling", disable=not show_progress)

    for y0 in y_starts:
        y1 = min(H, y0 + tile)
        for x0 in x_starts:
            x1 = min(W, x0 + tile)

            th, tw = y1 - y0, x1 - x0
            # Build query points — x,y order to match xy
            xs = np.arange(x0, x1, dtype=np.float64)
            ys = np.arange(y0, y1, dtype=np.float64)
            pts = np.empty((th * tw, 2), dtype=np.float64)
            pts[:, 0] = np.tile(xs, th)
            pts[:, 1] = np.repeat(ys, tw)

            _, nn = tree.query(pts, k=1)
            labels[y0:y1, x0:x1] = nn.ravel().astype(dtype).reshape(th, tw)

            pbar.update(1)

    pbar.close()
    return labels


def _label_to_base64_png(labeled: np.ndarray, max_dim: int = 256) -> str:
    """Downsample label map, colorize deterministically, return data URI."""
    from PIL import Image

    h, w = labeled.shape
    scale = min(max_dim / max(h, w), 1.0)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    # Nearest-neighbor downsample to preserve label IDs
    small = np.array(
        Image.fromarray(labeled.astype(np.int32)).resize((new_w, new_h), Image.NEAREST)
    )

    # Deterministic colorization via hash
    unique_ids = np.unique(small)
    rgb = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for uid in unique_ids:
        if uid == 0:
            continue
        rng = np.random.RandomState(int(uid) * 7 + 13)
        color = rng.randint(60, 256, size=3)
        rgb[small == uid] = color

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class BenchmarkTracer:
    """Accumulates benchmark trace data and writes JSON."""

    def __init__(self):
        self.data = {
            "config": {},
            "methods": {},
            "timestamp": datetime.now().isoformat(),
        }
        self._current_method = None
        self._method_start = None

    # -- Config --
    def set_config(self, **kwargs):
        self.data["config"].update(kwargs)

    # -- Method timing --
    def begin_method(self, name: str):
        self._current_method = name
        self._method_start = time.time()
        self.data["methods"][name] = {
            "optimization": {},
            "segmentation": [],
            "pipeline": [],
            "final_stats": {},
        }

    def end_method(self):
        if self._current_method and self._method_start:
            m = self.data["methods"][self._current_method]
            m["elapsed_s"] = round(time.time() - self._method_start, 2)
        self._current_method = None
        self._method_start = None

    # -- SA tracing --
    def init_sa_optimization(self, config: dict):
        m = self.data["methods"][self._current_method]
        m["optimization"] = {"type": "sa", "config": config, "layers": {}}

    def begin_sa_layer(self, cycle: int, layer: int):
        key = f"cy{cycle}_layer{layer}"
        opt = self.data["methods"][self._current_method]["optimization"]
        opt["layers"][key] = {"steps": []}

    def record_sa_step(self, step, prob, nms, score, n_instances, T, accepted):
        key = list(self.data["methods"][self._current_method]["optimization"]["layers"].keys())[-1]
        layer_data = self.data["methods"][self._current_method]["optimization"]["layers"][key]
        layer_data["steps"].append({
            "step": step,
            "prob": round(float(prob), 4),
            "nms": round(float(nms), 3),
            "score": round(float(score), 4),
            "n_instances": int(n_instances),
            "T": round(float(T), 4),
            "accepted": bool(accepted),
        })

    def record_sa_best(self, cycle, layer, prob, nms, n_instances):
        key = f"cy{cycle}_layer{layer}"
        opt = self.data["methods"][self._current_method]["optimization"]
        opt["layers"][key]["best"] = {
            "prob": round(float(prob), 4),
            "nms": round(float(nms), 3),
            "n_instances": int(n_instances),
        }

    # -- Grid search --
    def init_grid_search(self, param_grid: dict):
        m = self.data["methods"][self._current_method]
        m["optimization"] = {
            "type": "grid",
            "param_grid": {k: [_jsonable(v) for v in vs] for k, vs in param_grid.items()},
            "results": [],
        }

    def record_grid_result(self, params: dict, score: float):
        opt = self.data["methods"][self._current_method]["optimization"]
        opt["results"].append({
            "params": {k: _jsonable(v) for k, v in params.items()},
            "score": round(float(score), 6),
        })

    def record_grid_best(self, params: dict, score: float):
        opt = self.data["methods"][self._current_method]["optimization"]
        opt["best"] = {
            "params": {k: _jsonable(v) for k, v in params.items()},
            "score": round(float(score), 6),
        }

    # -- Segmentation --
    def record_layer_segmentation(self, cycle, layer, n_instances, params_used, labeled_img):
        m = self.data["methods"][self._current_method]
        entry = {
            "cycle": int(cycle),
            "layer": int(layer),
            "n_instances": int(n_instances),
            "params": {k: _jsonable(v) for k, v in params_used.items()} if params_used else {},
        }
        try:
            entry["preview"] = _label_to_base64_png(labeled_img)
        except Exception:
            pass
        m["segmentation"].append(entry)

    # -- Pipeline --
    def record_pipeline_stage(self, stage_name: str, counts: dict):
        m = self.data["methods"][self._current_method]
        m["pipeline"].append({
            "stage": stage_name,
            "counts": {k: _jsonable(v) for k, v in counts.items()},
        })

    def record_final_stats(self, stats: dict):
        m = self.data["methods"][self._current_method]
        m["final_stats"] = {k: _jsonable(v) for k, v in stats.items()}

    # -- Voronoi median analysis --
    def record_voronoi_median_analysis(
        self,
        bead_df,
        cycle_images,
        cycle_metadata,
        shape,
        max_size=10000,
        min_assigned_value=0.01,
        border_erosion=0,
        tile=2048,
    ):
        from scipy.ndimage import distance_transform_edt

        xy = bead_df[["x", "y"]].values
        N = len(xy)

        # a. Build voronoi regions
        vor_lbl = voronoi_from_centers_tiled(xy, shape, tile=tile)
        H0 = min(max_size, shape[0])
        W0 = min(max_size, shape[1])
        vor = vor_lbl[:H0, :W0]

        # a2. Erode region borders: mask out pixels within border_erosion of any boundary
        if border_erosion > 0:
            boundary = np.zeros(vor.shape, dtype=bool)
            boundary[:-1, :] |= vor[:-1, :] != vor[1:, :]
            boundary[1:, :]  |= vor[:-1, :] != vor[1:, :]
            boundary[:, :-1] |= vor[:, :-1] != vor[:, 1:]
            boundary[:, 1:]  |= vor[:, :-1] != vor[:, 1:]
            dist = distance_transform_edt(~boundary)
            vor = vor.copy()
            vor[dist < border_erosion] = -1
            n_eroded = int((dist < border_erosion).sum())
            n_total = vor.size
            print(f"  Border erosion={border_erosion}: excluded {n_eroded}/{n_total} pixels ({100*n_eroded/n_total:.1f}%)")

        # b. Compute median intensity per bead per cycle_layer
        n_cycles = len(cycle_images)
        col_names = []
        raw_cols = {}

        for ci, (cyc, md) in enumerate(zip(cycle_images, cycle_metadata)):
            H = min(max_size, cyc.shape[-2])
            W = min(max_size, cyc.shape[-1])
            v = vor[:H, :W]
            rf = v.ravel()

            flayers = md.flors_layers
            for lj, ch_idx in enumerate(flayers):
                col = f"cy{ci}_{lj}"
                col_names.append(col)
                img = cyc[ch_idx, :H, :W].astype(np.float32)
                raw_cols[col] = _median_per_region(img.ravel(), rf, N)

        # c. Normalize by layer (column): min-max per column
        norm_cols = {}
        for col, vals in raw_cols.items():
            lo, hi = vals.min(), vals.max()
            norm_cols[col] = (vals - lo) / (hi - lo + 1e-8)

        # d. Assign best layer per cycle and record assigned value + margin
        assigned_layers = {}
        assigned_values = {}
        assigned_margins = {}  # best / second_best ratio per bead per cycle

        for ci in range(n_cycles):
            cy_cols = [c for c in col_names if c.startswith(f"cy{ci}_")]
            if not cy_cols:
                continue
            mat = np.stack([norm_cols[c] for c in cy_cols], axis=1)
            best_layer = np.argmax(mat, axis=1)
            best_val = mat[np.arange(N), best_layer]
            assigned_layers[f"cy{ci}"] = best_layer
            assigned_values[f"cy{ci}"] = best_val

            # Margin: ratio of best to second-best normalized value
            if mat.shape[1] >= 2:
                sorted_mat = np.sort(mat, axis=1)
                second_best = sorted_mat[:, -2]
                margin_ratio = best_val / (second_best + 1e-8)
                assigned_margins[f"cy{ci}"] = margin_ratio
            else:
                assigned_margins[f"cy{ci}"] = np.full(N, np.inf)

        # e. Filter beads with assigned value < min_assigned_value
        # A bead survives if ALL its cycle assigned values pass the threshold
        surviving = np.ones(N, dtype=bool)
        for cy_key, vals in assigned_values.items():
            surviving &= vals >= min_assigned_value
        n_surviving = int(surviving.sum())

        # f. Percentile statistics on surviving beads' assigned values
        all_assigned = np.concatenate([v[surviving] for v in assigned_values.values()])
        percentile_counts = {}
        if len(all_assigned) > 0:
            for p in [10, 25, 50, 75, 90]:
                threshold = float(np.percentile(all_assigned, p))
                count = int((all_assigned >= threshold).sum())
                percentile_counts[f"p{p}"] = {
                    "threshold": round(threshold, 4),
                    "count": count,
                }

        # Per-cycle stats
        per_cycle_stats = {}
        for cy_key, vals in assigned_values.items():
            sv = vals[surviving]
            per_cycle_stats[cy_key] = {
                "mean_assigned": round(float(sv.mean()), 4) if len(sv) > 0 else 0.0,
                "median_assigned": round(float(np.median(sv)), 4) if len(sv) > 0 else 0.0,
                "n_surviving": int(surviving.sum()),
            }

        # g. Build resolved bead_df with cy0, cy1, ... columns
        import pandas as pd

        resolved = bead_df[["x", "y"]].copy()
        for cy_key, layers in assigned_layers.items():
            col = np.full(N, 255, dtype=np.uint8)
            col[surviving] = layers[surviving].astype(np.uint8)
            resolved[cy_key] = col

        # h. Store results
        m = self.data["methods"][self._current_method]
        m["voronoi_analysis"] = {
            "n_beads_total": N,
            "n_beads_after_filter": n_surviving,
            "min_assigned_value": min_assigned_value,
            "border_erosion": border_erosion,
            "percentile_counts": percentile_counts,
            "per_cycle_stats": per_cycle_stats,
        }

        return resolved, assigned_layers, assigned_values, assigned_margins

    # -- Save --
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2)
        print(f"Trace saved to {path}")


def _jsonable(v):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

