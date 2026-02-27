from types import SimpleNamespace

import numpy as np
import pandas as pd

import image_processing
from model.file_item import MetaData


def _build_tifs(max_size=50, channels=3, cycles=2):
    out = []
    for _ in range(cycles):
        img = np.zeros((channels, max_size, max_size), dtype=np.uint16)
        md = MetaData(max_size=max_size, reference_channel=0)
        out.append((img, SimpleNamespace(metadata=md)))
    return out


def test_process_beads_uses_notebook_stardist_path(monkeypatch):
    calls = {}
    tifs = _build_tifs(max_size=50, channels=3, cycles=2)
    brightfield = np.zeros((50, 50), dtype=np.uint16)

    def fail_get_excel(*args, **kwargs):
        raise AssertionError("get_excel should not be used in use_stardist=True mode")

    def fake_beadfinding_notebook_stardist(
        brightfield,
        scale=1.0,
        block_size=700,
        n_tiles=1,
        progress_units_callback=None,
        progress_callback=None,
    ):
        calls["beadfinding"] = {
            "scale": scale,
            "block_size": block_size,
            "n_tiles": n_tiles,
        }
        return np.array([[20, 20], [30, 30]], dtype=np.float32)

    def fake_load_custom_model(model_dir):
        calls["model_dir"] = model_dir
        return object()

    def fake_get_labels_from_cycles_with_prob(
        cycles,
        metadata_list,
        max_size,
        model,
        block_size=700,
        prob_thresh=0.1,
        n_tiles=1,
        nms_thresh=0.1,
        progress_callback=None,
    ):
        calls["labels"] = {
            "max_size": max_size,
            "block_size": block_size,
            "prob_thresh": prob_thresh,
            "n_tiles": n_tiles,
            "nms_thresh": nms_thresh,
            "num_cycles": len(cycles),
        }
        out = []
        for _ in cycles:
            layer0 = {
                "lbl": np.ones((50, 50), dtype=np.int32),
                "prob_lut": np.array([0.0, 0.8], dtype=np.float32),
            }
            layer1 = {
                "lbl": np.ones((50, 50), dtype=np.int32),
                "prob_lut": np.array([0.0, 0.7], dtype=np.float32),
            }
            out.append([layer0, layer1])
        return out

    def fake_assign(bead_df, cycle_labels, **kwargs):
        calls["assign"] = kwargs
        out = bead_df.copy()
        out["cy0_0"] = np.array([1, 0], dtype=np.int32)
        out["cy0_1"] = np.array([0, 1], dtype=np.int32)
        out["cy1_0"] = np.array([0, 1], dtype=np.int32)
        out["cy1_1"] = np.array([1, 0], dtype=np.int32)
        out["cy0_0_prob"] = np.array([0.9, 0.0], dtype=np.float32)
        out["cy0_1_prob"] = np.array([0.0, 0.8], dtype=np.float32)
        out["cy1_0_prob"] = np.array([0.0, 0.7], dtype=np.float32)
        out["cy1_1_prob"] = np.array([0.85, 0.0], dtype=np.float32)
        return out

    def fake_enforce(df, **kwargs):
        calls["enforce"] = kwargs
        return df

    def fake_resolve(df, num_cycles, num_layers, invalid_value=255):
        return pd.DataFrame({"x": df["x"], "y": df["y"], "cy0": [0, 1], "cy1": [1, 0]})

    def fake_compute_cache(
        bead_df,
        cycles,
        metadata_list,
        max_size,
        min_assigned_value=0.1,
        border_erosion=0,
    ):
        calls["cache"] = {
            "max_size": max_size,
            "min_assigned_value": min_assigned_value,
            "num_cycles": len(cycles),
        }
        return {"cached": True}

    def fake_sweep_stats(pre_ensemble_beads, ensemble_cache, start, end, step):
        calls["sweep"] = {"start": start, "end": end, "step": step}
        return pd.DataFrame(
            [
                {
                    "ratio": 1.0,
                    "valid_pct": 20.0,
                    "invalid_pct": 10.0,
                    "filtered_pct": 70.0,
                    "invalid_count": 1,
                    "filtered_count": 1,
                },
                {
                    "ratio": 1.05,
                    "valid_pct": 21.0,
                    "invalid_pct": 9.0,
                    "filtered_pct": 70.0,
                    "invalid_count": 1,
                    "filtered_count": 1,
                },
            ]
        )

    def fake_build_ensembled(pre_ensemble_beads, ensemble_cache, ratio):
        calls["applied_ratio"] = ratio
        return pd.DataFrame(
            {
                "x": pre_ensemble_beads["x"],
                "y": pre_ensemble_beads["y"],
                "cy0": [3, 4],
                "cy1": [5, 6],
            }
        )

    monkeypatch.setattr(image_processing, "get_excel", fail_get_excel)
    monkeypatch.setattr(
        image_processing,
        "beadfinding_notebook_stardist",
        fake_beadfinding_notebook_stardist,
    )
    monkeypatch.setattr(image_processing, "load_custom_model", fake_load_custom_model)
    monkeypatch.setattr(
        image_processing,
        "get_labels_from_cycles_with_prob",
        fake_get_labels_from_cycles_with_prob,
    )
    monkeypatch.setattr(
        image_processing, "assign_beads_labels_with_prob_patch3x3_fallback", fake_assign
    )
    monkeypatch.setattr(
        image_processing, "enforce_single_layer_per_cycle", fake_enforce
    )
    monkeypatch.setattr(image_processing, "resolve_layers_to_cycles", fake_resolve)
    monkeypatch.setattr(
        image_processing, "_compute_voronoi_ensemble_cache", fake_compute_cache
    )
    monkeypatch.setattr(
        image_processing, "compute_ensemble_sweep_stats", fake_sweep_stats
    )
    monkeypatch.setattr(
        image_processing, "build_ensembled_beads_from_cache", fake_build_ensembled
    )

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=50,
        signal_to_noise_cutoff=0.1,
        use_stardist=True,
        model_name="model_5_400epoch",
    )

    assert results is not None
    assert set(results.keys()) >= {
        "beads",
        "post_resolution_beads",
        "pre_ensemble_beads",
        "ensemble_cache",
        "ensemble_sweep_stats",
        "ensemble_ratio_applied",
        "cycles",
        "labeled_image",
    }
    assert set(results["beads"].columns) == {"x", "y", "cy0", "cy1"}
    assert results["beads"].equals(results["post_resolution_beads"])
    assert set(results["pre_ensemble_beads"].columns) == {"x", "y", "cy0", "cy1"}
    assert isinstance(results["ensemble_sweep_stats"], pd.DataFrame)
    assert calls["cache"]["min_assigned_value"] == 0.1
    assert calls["sweep"] == {"start": 1.0, "end": 1.5, "step": 0.05}
    assert calls["applied_ratio"] == 1.0
    assert set(results["cycles"].keys()) == {"cy0", "cy1"}
    assert calls["beadfinding"]["scale"] == 2
    assert calls["beadfinding"]["block_size"] == 700
    assert calls["beadfinding"]["n_tiles"] == 0
    assert calls["labels"]["prob_thresh"] == 0.1
    assert calls["labels"]["nms_thresh"] == 0.1
    assert calls["labels"]["n_tiles"] == 0
    assert calls["labels"]["block_size"] == 700
    assert "assets/model_5_400epoch" in calls["model_dir"].replace("\\", "/")


def test_process_beads_off_mode_delegates_to_legacy_get_excel(monkeypatch):
    calls = {}
    tifs = _build_tifs(max_size=32, channels=3, cycles=2)
    brightfield = np.zeros((32, 32), dtype=np.uint16)

    legacy_df = pd.DataFrame({"x": [12.0], "y": [13.0], "cy0": [1], "cy1": [2]})
    post_df = pd.DataFrame({"x": [12.0], "y": [13.0], "cy0": [1], "cy1": [2]})

    def fake_beadfinding(
        brightfield, num_tiles=10, px_overlap=100, workers=10, is_running_callback=None
    ):
        return np.array([[12, 13]], dtype=np.uint16)

    def fake_get_excel(
        beads,
        signal_to_noise_cutoff,
        tifs,
        max_size,
        layer_threshold_dict=None,
        progress_callback=None,
        is_running_callback=None,
        roi_coords=None,
        n_workers=10,
        radius=2,
        use_stardist=False,
        model_name="model_5_400epoch",
    ):
        calls["get_excel"] = {"use_stardist": use_stardist, "model_name": model_name}
        return legacy_df.copy(), post_df.copy()

    def fail_notebook_path(*args, **kwargs):
        raise AssertionError("Notebook path should not run in use_stardist=False mode")

    monkeypatch.setattr(image_processing, "beadfinding", fake_beadfinding)
    monkeypatch.setattr(image_processing, "get_excel", fake_get_excel)
    monkeypatch.setattr(
        image_processing, "_process_beads_notebook_stardist", fail_notebook_path
    )

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=32,
        signal_to_noise_cutoff=0.1,
        use_stardist=False,
        model_name="model_5_400epoch",
    )

    assert results is not None
    assert calls["get_excel"]["use_stardist"] is False
    assert results["beads"].equals(legacy_df)
    assert results["post_resolution_beads"].equals(post_df)
    assert set(results["cycles"].keys()) == {"cy0", "cy1"}


def test_process_beads_stardist_forwards_custom_ensemble_sweep(monkeypatch):
    calls = {}
    tifs = _build_tifs(max_size=24, channels=3, cycles=2)
    brightfield = np.zeros((24, 24), dtype=np.uint16)

    def fake_notebook_stardist(
        brightfield,
        tifs,
        max_size,
        model_name,
        update_progress,
        is_running,
        progress_units_callback=None,
        stardist_use_guess_tiles=True,
        stardist_n_tiles=1,
        ensemble_ratio_start=1.0,
        ensemble_ratio_end=1.5,
        ensemble_ratio_step=0.05,
    ):
        calls["sweep"] = {
            "start": ensemble_ratio_start,
            "end": ensemble_ratio_end,
            "step": ensemble_ratio_step,
        }
        return {
            "beads": pd.DataFrame({"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}),
            "post_resolution_beads": pd.DataFrame(
                {"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}
            ),
            "pre_ensemble_beads": pd.DataFrame(
                {"x": [1.0], "y": [1.0], "cy0": [0], "cy1": [1]}
            ),
            "ensemble_cache": {"cached": True},
            "ensemble_sweep_stats": pd.DataFrame(
                [{"ratio": 1.2, "valid_pct": 10.0, "invalid_pct": 20.0, "filtered_pct": 70.0}]
            ),
            "ensemble_ratio_applied": 1.2,
            "cycles": {"cy0": tifs[0][0], "cy1": tifs[1][0]},
            "labeled_image": np.zeros((24, 24), dtype=np.uint16),
        }

    monkeypatch.setattr(
        image_processing, "_process_beads_notebook_stardist", fake_notebook_stardist
    )

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=24,
        signal_to_noise_cutoff=0.1,
        use_stardist=True,
        ensemble_ratio_start=1.2,
        ensemble_ratio_end=1.6,
        ensemble_ratio_step=0.1,
    )

    assert results is not None
    assert calls["sweep"] == {"start": 1.2, "end": 1.6, "step": 0.1}


def test_process_beads_stardist_forwards_progress_units_callback(monkeypatch):
    calls = {"progress_units": []}
    tifs = _build_tifs(max_size=50, channels=3, cycles=2)
    brightfield = np.zeros((50, 50), dtype=np.uint16)

    def fake_beadfinding_notebook_stardist(
        brightfield,
        scale=1.0,
        block_size=700,
        n_tiles=1,
        progress_units_callback=None,
        progress_callback=None,
    ):
        return np.array([[20, 20], [30, 30]], dtype=np.float32)

    def fake_load_custom_model(model_dir):
        return object()

    def fake_get_labels_from_cycles_with_prob(
        cycles,
        metadata_list,
        max_size,
        model,
        block_size=700,
        prob_thresh=0.1,
        n_tiles=1,
        nms_thresh=0.1,
        progress_units_callback=None,
        progress_callback=None,
    ):
        if progress_units_callback:
            progress_units_callback("activation_regions", 1, 4)
            progress_units_callback("activation_regions", 4, 4)
        out = []
        for _ in cycles:
            layer0 = {
                "lbl": np.ones((50, 50), dtype=np.int32),
                "prob_lut": np.array([0.0, 0.8], dtype=np.float32),
            }
            layer1 = {
                "lbl": np.ones((50, 50), dtype=np.int32),
                "prob_lut": np.array([0.0, 0.7], dtype=np.float32),
            }
            out.append([layer0, layer1])
        return out

    def fake_assign(bead_df, cycle_labels, **kwargs):
        out = bead_df.copy()
        out["cy0_0"] = np.array([1, 0], dtype=np.int32)
        out["cy0_1"] = np.array([0, 1], dtype=np.int32)
        out["cy1_0"] = np.array([0, 1], dtype=np.int32)
        out["cy1_1"] = np.array([1, 0], dtype=np.int32)
        out["cy0_0_prob"] = np.array([0.9, 0.0], dtype=np.float32)
        out["cy0_1_prob"] = np.array([0.0, 0.8], dtype=np.float32)
        out["cy1_0_prob"] = np.array([0.0, 0.7], dtype=np.float32)
        out["cy1_1_prob"] = np.array([0.85, 0.0], dtype=np.float32)
        return out

    def fake_enforce(df, **kwargs):
        return df

    def fake_resolve(df, num_cycles, num_layers, invalid_value=255):
        return pd.DataFrame({"x": df["x"], "y": df["y"], "cy0": [0, 1], "cy1": [1, 0]})

    def fake_compute_cache(
        bead_df,
        cycles,
        metadata_list,
        max_size,
        min_assigned_value=0.1,
        border_erosion=0,
        progress_units_callback=None,
    ):
        if progress_units_callback:
            progress_units_callback("voronoi_cache", 1, 3)
            progress_units_callback("voronoi_cache", 3, 3)
        return {"cached": True}

    def fake_sweep_stats(
        pre_ensemble_beads,
        ensemble_cache,
        start,
        end,
        step,
        progress_units_callback=None,
    ):
        if progress_units_callback:
            progress_units_callback("voronoi_sweep", 1, 2)
            progress_units_callback("voronoi_sweep", 2, 2)
        return pd.DataFrame(
            [
                {
                    "ratio": 1.0,
                    "valid_pct": 20.0,
                    "invalid_pct": 10.0,
                    "filtered_pct": 70.0,
                },
                {
                    "ratio": 1.05,
                    "valid_pct": 21.0,
                    "invalid_pct": 9.0,
                    "filtered_pct": 70.0,
                },
            ]
        )

    def fake_build_ensembled(pre_ensemble_beads, ensemble_cache, ratio):
        return pd.DataFrame(
            {
                "x": pre_ensemble_beads["x"],
                "y": pre_ensemble_beads["y"],
                "cy0": [3, 4],
                "cy1": [5, 6],
            }
        )

    monkeypatch.setattr(
        image_processing,
        "beadfinding_notebook_stardist",
        fake_beadfinding_notebook_stardist,
    )
    monkeypatch.setattr(image_processing, "load_custom_model", fake_load_custom_model)
    monkeypatch.setattr(
        image_processing,
        "get_labels_from_cycles_with_prob",
        fake_get_labels_from_cycles_with_prob,
    )
    monkeypatch.setattr(
        image_processing, "assign_beads_labels_with_prob_patch3x3_fallback", fake_assign
    )
    monkeypatch.setattr(
        image_processing, "enforce_single_layer_per_cycle", fake_enforce
    )
    monkeypatch.setattr(image_processing, "resolve_layers_to_cycles", fake_resolve)
    monkeypatch.setattr(
        image_processing, "_compute_voronoi_ensemble_cache", fake_compute_cache
    )
    monkeypatch.setattr(
        image_processing, "compute_ensemble_sweep_stats", fake_sweep_stats
    )
    monkeypatch.setattr(
        image_processing, "build_ensembled_beads_from_cache", fake_build_ensembled
    )

    def progress_units_callback(stage, done, total):
        calls["progress_units"].append((stage, done, total))

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=50,
        signal_to_noise_cutoff=0.1,
        use_stardist=True,
        model_name="model_5_400epoch",
        progress_units_callback=progress_units_callback,
    )

    assert results is not None
    assert ("activation_regions", 4, 4) in calls["progress_units"]
    assert ("voronoi_cache", 3, 3) in calls["progress_units"]
    assert ("voronoi_sweep", 2, 2) in calls["progress_units"]


def test_process_beads_stardist_uses_stricter_prob_thresh_for_large_images(monkeypatch):
    calls = {}
    tifs = _build_tifs(max_size=64, channels=3, cycles=2)
    brightfield = np.zeros((64, 64), dtype=np.uint16)

    def fake_beadfinding_notebook_stardist(
        brightfield,
        scale=1.0,
        block_size=700,
        n_tiles=1,
        progress_units_callback=None,
        progress_callback=None,
    ):
        return np.array([[20, 20], [30, 30]], dtype=np.float32)

    def fake_load_custom_model(model_dir):
        return object()

    def fake_get_labels_from_cycles_with_prob(
        cycles,
        metadata_list,
        max_size,
        model,
        block_size=700,
        prob_thresh=0.1,
        n_tiles=1,
        nms_thresh=0.1,
        progress_units_callback=None,
        progress_callback=None,
    ):
        calls["labels"] = {"prob_thresh": prob_thresh, "block_size": block_size}
        out = []
        for _ in cycles:
            out.append(
                [
                    {
                        "lbl": np.ones((64, 64), dtype=np.int32),
                        "prob_lut": np.array([0.0, 0.8], dtype=np.float32),
                    },
                    {
                        "lbl": np.ones((64, 64), dtype=np.int32),
                        "prob_lut": np.array([0.0, 0.7], dtype=np.float32),
                    },
                ]
            )
        return out

    def fake_assign(bead_df, cycle_labels, **kwargs):
        out = bead_df.copy()
        out["cy0_0"] = np.array([1, 0], dtype=np.int32)
        out["cy0_1"] = np.array([0, 1], dtype=np.int32)
        out["cy1_0"] = np.array([0, 1], dtype=np.int32)
        out["cy1_1"] = np.array([1, 0], dtype=np.int32)
        out["cy0_0_prob"] = np.array([0.9, 0.0], dtype=np.float32)
        out["cy0_1_prob"] = np.array([0.0, 0.8], dtype=np.float32)
        out["cy1_0_prob"] = np.array([0.0, 0.7], dtype=np.float32)
        out["cy1_1_prob"] = np.array([0.85, 0.0], dtype=np.float32)
        return out

    def fake_enforce(df, **kwargs):
        return df

    def fake_resolve(df, num_cycles, num_layers, invalid_value=255):
        return pd.DataFrame({"x": df["x"], "y": df["y"], "cy0": [0, 1], "cy1": [1, 0]})

    def fake_compute_cache(
        bead_df,
        cycles,
        metadata_list,
        max_size,
        min_assigned_value=0.1,
        border_erosion=0,
        progress_units_callback=None,
    ):
        return {"cached": True}

    def fake_sweep_stats(
        pre_ensemble_beads,
        ensemble_cache,
        start,
        end,
        step,
        progress_units_callback=None,
    ):
        return pd.DataFrame(
            [
                {
                    "ratio": 1.0,
                    "valid_pct": 20.0,
                    "invalid_pct": 10.0,
                    "filtered_pct": 70.0,
                }
            ]
        )

    def fake_build_ensembled(pre_ensemble_beads, ensemble_cache, ratio):
        return pd.DataFrame(
            {
                "x": pre_ensemble_beads["x"],
                "y": pre_ensemble_beads["y"],
                "cy0": [3, 4],
                "cy1": [5, 6],
            }
        )

    monkeypatch.setattr(
        image_processing,
        "beadfinding_notebook_stardist",
        fake_beadfinding_notebook_stardist,
    )
    monkeypatch.setattr(image_processing, "load_custom_model", fake_load_custom_model)
    monkeypatch.setattr(
        image_processing,
        "get_labels_from_cycles_with_prob",
        fake_get_labels_from_cycles_with_prob,
    )
    monkeypatch.setattr(
        image_processing, "assign_beads_labels_with_prob_patch3x3_fallback", fake_assign
    )
    monkeypatch.setattr(
        image_processing, "enforce_single_layer_per_cycle", fake_enforce
    )
    monkeypatch.setattr(image_processing, "resolve_layers_to_cycles", fake_resolve)
    monkeypatch.setattr(
        image_processing, "_compute_voronoi_ensemble_cache", fake_compute_cache
    )
    monkeypatch.setattr(
        image_processing, "compute_ensemble_sweep_stats", fake_sweep_stats
    )
    monkeypatch.setattr(
        image_processing, "build_ensembled_beads_from_cache", fake_build_ensembled
    )

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=5000,
        signal_to_noise_cutoff=0.1,
        use_stardist=True,
        model_name="model_5_400epoch",
    )

    assert results is not None
    assert calls["labels"]["prob_thresh"] == 0.25
    assert calls["labels"]["block_size"] == 2000


def test_process_beads_legacy_forwards_progress_units_callback(monkeypatch):
    calls = {"progress_units": []}
    tifs = _build_tifs(max_size=32, channels=3, cycles=2)
    brightfield = np.zeros((32, 32), dtype=np.uint16)

    legacy_df = pd.DataFrame({"x": [12.0], "y": [13.0], "cy0": [1], "cy1": [2]})

    def fake_beadfinding(
        brightfield,
        num_tiles=10,
        px_overlap=100,
        workers=10,
        is_running_callback=None,
    ):
        return np.array([[12, 13]], dtype=np.uint16)

    def fake_get_excel(
        beads,
        signal_to_noise_cutoff,
        tifs,
        max_size,
        layer_threshold_dict=None,
        progress_callback=None,
        is_running_callback=None,
        roi_coords=None,
        n_workers=10,
        radius=2,
        use_stardist=False,
        model_name="model_5_400epoch",
        progress_units_callback=None,
    ):
        if progress_units_callback:
            progress_units_callback("activation_regions", 1, 2)
            progress_units_callback("activation_regions", 2, 2)
        return legacy_df.copy(), legacy_df.copy()

    monkeypatch.setattr(image_processing, "beadfinding", fake_beadfinding)
    monkeypatch.setattr(image_processing, "get_excel", fake_get_excel)

    def progress_units_callback(stage, done, total):
        calls["progress_units"].append((stage, done, total))

    results = image_processing.process_beads(
        brightfield=brightfield,
        tifs=tifs,
        max_size=32,
        signal_to_noise_cutoff=0.1,
        use_stardist=False,
        model_name="model_5_400epoch",
        progress_units_callback=progress_units_callback,
    )

    assert results is not None
    assert ("activation_regions", 2, 2) in calls["progress_units"]
