import pandas as pd

from ensemble import compute_bead_profile_metrics
from image_processing import get_error


def test_compute_bead_profile_metrics_uses_bead_level_counts():
    bead_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "cy0": [1, 1, 9, 255],
            "cy1": [2, 2, 9, 1],
        }
    )
    protein_df = pd.DataFrame(
        {
            "Protein name": ["A", "B", "C"],
            "cy0": [1, 1, 3],
            "cy1": [2, 2, 4],
        }
    )

    metrics = compute_bead_profile_metrics(bead_df, protein_df)

    assert metrics["total"] == 4
    assert metrics["valid"] == 2
    assert metrics["invalid"] == 1
    assert metrics["filtered"] == 1
    assert metrics["unique_cycle_combos"] == 2
    assert metrics["invalid_pct_formula"] == 100.0
    assert metrics["filtered_pct"] == 25.0


def test_compute_bead_profile_metrics_prints_invalid_cycle_combinations(capsys):
    bead_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "cy0": [1, 9, 8],
            "cy1": [2, 9, 8],
        }
    )
    protein_df = pd.DataFrame(
        {
            "Protein name": ["A"],
            "cy0": [1],
            "cy1": [2],
        }
    )

    compute_bead_profile_metrics(bead_df, protein_df)
    captured = capsys.readouterr().out

    assert "Invalid cycle combinations:" in captured
    assert "cy0" in captured
    assert "cy1" in captured
    assert "9" in captured
    assert "8" in captured


def test_compute_bead_profile_metrics_filtered_not_inflated_by_profile_duplicates():
    bead_df = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "cy0": [255, 255],
            "cy1": [1, 1],
        }
    )
    protein_df = pd.DataFrame(
        {
            "Protein name": ["F1", "F2", "A"],
            "cy0": [255, 255, 1],
            "cy1": [1, 1, 2],
        }
    )

    metrics = compute_bead_profile_metrics(bead_df, protein_df)

    assert metrics["total"] == 2
    assert metrics["filtered"] == 2
    assert metrics["invalid"] == 0
    assert metrics["valid"] == 0
    assert metrics["invalid_pct_formula"] == 0.0
    assert metrics["filtered_pct"] == 100.0


def test_compute_bead_profile_metrics_zero_unique_combos_returns_zero_invalid_pct():
    bead_df = pd.DataFrame(
        {
            "x": [1.0],
            "y": [1.0],
            "cy0": [9],
            "cy1": [9],
        }
    )
    protein_df = pd.DataFrame({"Protein name": [], "cy0": [], "cy1": []})

    metrics = compute_bead_profile_metrics(bead_df, protein_df)

    assert metrics["unique_cycle_combos"] == 0
    assert metrics["valid"] == 0
    assert metrics["invalid"] == 1
    assert metrics["invalid_pct_formula"] == 0.0


def test_get_error_profile_mode_uses_bead_level_filtered_count():
    bead_df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "cy0": [1, 9, 255],
            "cy1": [2, 9, 1],
        }
    )
    protein_df = pd.DataFrame(
        {
            "Protein name": ["A", "B", "F1", "F2"],
            "cy0": [1, 1, 255, 255],
            "cy1": [2, 2, 1, 1],
        }
    )

    error = get_error(bead_df, protein_df)

    assert error["invalid_count"] == 1
    assert error["filtered_count"] == 1
    assert error["mean_beads_per_protein"] == 1.0
    assert error["invalid_error_percentage"] == 100.0
