import numpy as np
import pandas as pd

from image_processing import (
    assign_beads_labels_with_prob_patch3x3_fallback,
    enforce_single_layer_per_cycle,
    resolve_layers_to_cycles,
)


def test_assign_patch_fallback_center_override_and_support_gate():
    lbl = np.zeros((25, 25), dtype=np.int32)
    lbl[10:15, 10:15] = 2
    lbl[12, 12] = 1
    lbl[6, 6] = 1
    patch_coords = [(r, c) for r in range(18, 23) for c in range(18, 23)]
    for r, c in patch_coords[:10]:
        lbl[r, c] = 2

    prob_lut = np.zeros(3, dtype=np.float32)
    prob_lut[1] = 0.4
    prob_lut[2] = 0.9

    cycle_labels = [[{"lbl": lbl, "prob_lut": prob_lut}]]
    bead_df = pd.DataFrame({"x": [12, 6, 20], "y": [12, 6, 20]}, dtype=np.float32)

    out = assign_beads_labels_with_prob_patch3x3_fallback(
        bead_df,
        cycle_labels,
        min_center_prob=0.35,
        min_patch_support=21,
        min_prob_margin=0,
        center_vs_patch_margin=0.01,
        invalid_value=0,
    )

    assert out["cy0_0"].tolist() == [2, 1, 0]
    assert np.allclose(out["cy0_0_prob"].to_numpy(), np.array([0.9, 0.4, 0.0]))


def test_enforce_then_resolve_layers():
    df = pd.DataFrame(
        {
            "x": [10, 20, 30],
            "y": [11, 21, 31],
            "cy0_0": [5, 5, 0],
            "cy0_1": [7, 6, 0],
            "cy0_2": [0, 9, 0],
            "cy0_3": [0, 0, 4],
            "cy0_0_prob": [0.8, 0.8, 0.0],
            "cy0_1_prob": [0.9, 0.6, 0.0],
            "cy0_2_prob": [0.0, 0.7, 0.0],
            "cy0_3_prob": [0.0, 0.0, 0.9],
            "cy1_0": [0, 0, 1],
            "cy1_1": [0, 2, 0],
            "cy1_2": [3, 0, 0],
            "cy1_3": [0, 0, 0],
            "cy1_0_prob": [0.0, 0.0, 0.9],
            "cy1_1_prob": [0.0, 0.5, 0.0],
            "cy1_2_prob": [0.7, 0.0, 0.0],
            "cy1_3_prob": [0.0, 0.0, 0.0],
        }
    )

    enforced = enforce_single_layer_per_cycle(
        df,
        num_cycles=2,
        num_layers=4,
        min_prob=0,
        invalid_value=0,
    )

    assert enforced.loc[0, "cy0_0"] == 0
    assert enforced.loc[0, "cy0_0_prob"] == 0.0
    assert enforced.loc[0, "cy0_1"] == 7

    assert enforced.loc[1, "cy0_0"] == 5
    assert enforced.loc[1, "cy0_1"] == 6
    assert enforced.loc[1, "cy0_2"] == 9

    resolved = resolve_layers_to_cycles(enforced, num_cycles=2, num_layers=4)
    assert resolved.loc[0, "cy0"] == 1
    assert resolved.loc[0, "cy1"] == 2
    assert resolved.loc[1, "cy0"] == 255
    assert resolved.loc[1, "cy1"] == 255
    assert resolved.loc[2, "cy0"] == 3
    assert resolved.loc[2, "cy1"] == 0
