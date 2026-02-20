import numpy as np
import pandas as pd

import image_processing


def _cache_for_rows(n_rows, *, min_assigned_value=0.1):
    return {
        "assigned_layers": {
            "cy0": np.zeros(n_rows, dtype=np.uint8),
            "cy1": np.zeros(n_rows, dtype=np.uint8),
        },
        "assigned_values": {
            "cy0": np.ones(n_rows, dtype=np.float32),
            "cy1": np.ones(n_rows, dtype=np.float32),
        },
        "assigned_margins": {
            "cy0": np.ones(n_rows, dtype=np.float32),
            "cy1": np.ones(n_rows, dtype=np.float32),
        },
        "min_assigned_value": float(min_assigned_value),
        "num_cycles": 2,
    }


def test_build_ensembled_beads_uses_literal_batch_merge_rule():
    pre_df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 30.0],
            "y": [10.0, 20.0, 30.0],
            "cy0": [255, 2, 255],
            "cy1": [255, 3, 255],
        }
    )
    cache = _cache_for_rows(3)
    cache["assigned_layers"]["cy0"] = np.array([5, 7, 9], dtype=np.uint8)
    cache["assigned_layers"]["cy1"] = np.array([6, 8, 10], dtype=np.uint8)
    cache["assigned_values"]["cy0"] = np.array([0.3, 0.3, 0.05], dtype=np.float32)
    cache["assigned_values"]["cy1"] = np.array([0.3, 0.3, 0.05], dtype=np.float32)

    out = image_processing.build_ensembled_beads_from_cache(
        pre_ensemble_beads=pre_df,
        ensemble_cache=cache,
        ratio=1.0,
    )

    assert out["cy0"].tolist() == [5, 2, 255]
    assert out["cy1"].tolist() == [6, 3, 255]


def test_compute_ensemble_sweep_stats_metrics_are_post_merge(monkeypatch):
    pre_df = pd.DataFrame(
        {
            "x": [11.0, 22.0],
            "y": [11.0, 22.0],
            "cy0": [255, 1],
            "cy1": [255, 1],
        }
    )
    cache = _cache_for_rows(2)
    cache["assigned_layers"]["cy0"] = np.array([4, 2], dtype=np.uint8)
    cache["assigned_layers"]["cy1"] = np.array([4, 2], dtype=np.uint8)
    cache["assigned_values"]["cy0"] = np.array([0.4, 0.4], dtype=np.float32)
    cache["assigned_values"]["cy1"] = np.array([0.4, 0.4], dtype=np.float32)
    cache["assigned_margins"]["cy0"] = np.array([1.0, 1.5], dtype=np.float32)
    cache["assigned_margins"]["cy1"] = np.array([1.0, 1.5], dtype=np.float32)

    seen_cy0 = []

    def fake_get_error(bead_df, protein_profile_df=None):
        seen_cy0.append(bead_df["cy0"].astype(int).tolist())
        invalid = int((bead_df["cy0"] == 255).sum())
        filtered = int((bead_df["cy0"] >= 254).sum())
        return {
            "invalid_count": invalid,
            "filtered_count": filtered,
            "mean_beads_per_protein": 1.0,
            "invalid_error_percentage": float(invalid),
            "filtered_percentage": (filtered / len(bead_df)) * 100,
        }

    monkeypatch.setattr(image_processing, "get_error", fake_get_error)

    sweep_df = image_processing.compute_ensemble_sweep_stats(
        pre_ensemble_beads=pre_df,
        ensemble_cache=cache,
        start=1.0,
        end=1.1,
        step=0.1,
    )

    assert sweep_df["ratio"].tolist() == [1.0, 1.1]
    assert seen_cy0[0] == [4, 1]
    assert seen_cy0[1] == [255, 1]
    assert sweep_df.loc[0, "invalid_count"] == 0
    assert sweep_df.loc[1, "invalid_count"] == 1
