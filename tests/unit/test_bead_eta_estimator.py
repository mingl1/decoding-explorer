import pytest

from viewmodel.bead_eta_estimator import (
    BeadEtaEstimator,
    EtaRange,
    LEGACY_STAGE_ORDER,
    LEGACY_PRIORS_PER_PIXEL_SQ,
    STARDIST_PRIORS_PER_PIXEL_SQ,
    STARDIST_STAGE_ORDER,
    WARMUP_PROGRESS_GATE,
)
from viewmodel.eta_profile_store import EtaProfileStore


LEGACY_MESSAGES = {
    "load_images": "Loading images (1/1)",
    "preprocess": "Preprocessing brightfield image",
    "initial_detection": "Initial bead detection...",
    "deduplicate": "Removing duplicate beads",
    "second_pass": "Performing second pass bead detection",
    "optimize_params": "Finding optimal parameters",
    "activation_regions": "Getting activation regions from cycles",
    "assign_labels": "Assigning beads labels",
    "resolve_labels": "Resolving Bead Labels",
    "finalize": "Bead generation complete.",
}

STARDIST_MESSAGES = {
    "load_images": "Loading images (1/1)",
    "preprocess": "Preprocessing brightfield image",
    "init_det_tiles": "Initial bead detection (tile 1/10)",
    "init_det_morphology": "Initial bead detection - finalizing",
    "activation_regions": "Processing fluorescence channels (1/6)",
    "assign_labels": "Assigning beads labels",
    "voronoi_cache": "Voronoi median: building ensemble cache",
    "voronoi_sweep": "Voronoi median: sweeping ensemble ratios",
    "voronoi_apply": "Voronoi median: applying selected ratio",
    "finalize": "Bead generation complete.",
}


def make_estimator(profile_path, mode="legacy", channels=1, max_size=10000):
    store = EtaProfileStore(path=str(profile_path))
    est = BeadEtaEstimator(mode=mode, profile_store=store)
    est.set_workload(total_channels=channels, max_size_pixels=max_size)
    return est


def test_legacy_walkthrough_produces_monotonic_progress(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 1000.0
    progresses = []
    elapsed = 0.0
    for i, stage in enumerate(LEGACY_STAGE_ORDER):
        elapsed += 1.0
        progress, _, _, _ = est.update_from_message(
            LEGACY_MESSAGES[stage], now=t0 + elapsed
        )
        progresses.append(progress)
    assert progresses == sorted(progresses)
    assert max(progresses) <= 99


def test_whole_run_scale_doubles_future_when_observed_doubles(tmp_path):
    est_baseline = make_estimator(tmp_path / "a.json")
    t0 = 1000.0
    est_baseline.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    expected_load = LEGACY_PRIORS_PER_PIXEL_SQ["load_images"] * (10000 ** 2)
    expected_pre = LEGACY_PRIORS_PER_PIXEL_SQ["preprocess"] * (10000 ** 2)
    est_baseline.update_from_message(
        LEGACY_MESSAGES["preprocess"], now=t0 + expected_load
    )
    est_baseline.update_from_message(
        LEGACY_MESSAGES["initial_detection"], now=t0 + expected_load + expected_pre
    )
    est_baseline.heartbeat(now=t0 + expected_load + expected_pre + 0.001)
    total_baseline = est_baseline._smoothed_total_eta

    est_scaled = make_estimator(tmp_path / "b.json")
    est_scaled.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    est_scaled.update_from_message(
        LEGACY_MESSAGES["preprocess"], now=t0 + expected_load * 2.0
    )
    est_scaled.update_from_message(
        LEGACY_MESSAGES["initial_detection"],
        now=t0 + expected_load * 2.0 + expected_pre * 2.0,
    )
    est_scaled.heartbeat(now=t0 + expected_load * 2.0 + expected_pre * 2.0 + 0.001)
    total_scaled = est_scaled._smoothed_total_eta

    assert total_baseline > 0
    assert total_scaled > 1.5 * total_baseline


def test_finish_failure_does_not_persist_profile(tmp_path):
    profile_path = tmp_path / "p.json"
    est = make_estimator(profile_path)
    t0 = 1000.0
    est.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    est.update_from_message(LEGACY_MESSAGES["preprocess"], now=t0 + 1.0)
    est.finish(success=False, now=t0 + 2.0)
    assert not profile_path.exists()


def test_finish_success_persists_profile(tmp_path):
    profile_path = tmp_path / "p.json"
    est = make_estimator(profile_path)
    t0 = 1000.0
    est.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    est.update_from_message(LEGACY_MESSAGES["preprocess"], now=t0 + 5.0)
    est.finish(success=True, now=t0 + 6.0)
    assert profile_path.exists()
    reloaded = EtaProfileStore(path=str(profile_path))
    loaded = reloaded.load("legacy")
    assert "load_images" in loaded
    assert reloaded.runs("legacy") == 1


def test_stardist_walkthrough_produces_monotonic_progress(tmp_path):
    est = make_estimator(tmp_path / "stardist.json", mode="stardist", channels=6, max_size=10000)
    t0 = 2000.0
    progresses = []

    progress, _, _, _ = est.update_stage_units(
        "load_images", 1, 1, message=STARDIST_MESSAGES["load_images"], now=t0 + 1.0
    )
    progresses.append(progress)
    progress, _, _, _ = est.update_from_message(
        STARDIST_MESSAGES["preprocess"], now=t0 + 2.0
    )
    progresses.append(progress)
    progress, _, _, _ = est.update_stage_units(
        "init_det_tiles",
        1,
        10,
        message=STARDIST_MESSAGES["init_det_tiles"],
        now=t0 + 3.0,
    )
    progresses.append(progress)
    progress, _, _, _ = est.update_from_message(
        STARDIST_MESSAGES["init_det_morphology"], now=t0 + 4.0
    )
    progresses.append(progress)
    progress, _, _, _ = est.update_stage_units(
        "activation_regions",
        1,
        6,
        message=STARDIST_MESSAGES["activation_regions"],
        now=t0 + 5.0,
    )
    progresses.append(progress)
    for i, stage in enumerate(
        ["assign_labels", "voronoi_cache", "voronoi_sweep", "voronoi_apply", "finalize"]
    ):
        progress, _, _, _ = est.update_from_message(
            STARDIST_MESSAGES[stage], now=t0 + 6.0 + i
        )
        progresses.append(progress)

    assert len(progresses) == len(STARDIST_STAGE_ORDER)
    assert progresses == sorted(progresses)
    assert max(progresses) <= 99


def test_stardist_total_eta_increases_after_set_workload_by_activation_prior():
    t0 = 6000.0
    channels = 7
    max_size = 10000
    est_before = BeadEtaEstimator(mode="stardist")
    est_before.update_stage_units(
        "init_det_tiles",
        1,
        10,
        message=STARDIST_MESSAGES["init_det_tiles"],
        now=t0,
    )
    est_before.heartbeat(now=t0 + 1.0)
    total_before = est_before._smoothed_total_eta

    est_after = BeadEtaEstimator(mode="stardist")
    est_after.set_workload(total_channels=channels, max_size_pixels=max_size)
    est_after.update_stage_units(
        "init_det_tiles",
        1,
        10,
        message=STARDIST_MESSAGES["init_det_tiles"],
        now=t0,
    )
    est_after.heartbeat(now=t0 + 1.0)
    total_after = est_after._smoothed_total_eta

    expected_delta = (
        channels
        * STARDIST_PRIORS_PER_PIXEL_SQ["activation_regions"]
        * (max_size ** 2)
    )
    observed_delta = float(total_after) - float(total_before)
    assert observed_delta == pytest.approx(expected_delta, rel=0.12)


def test_stardist_finish_failure_does_not_persist_profile(tmp_path):
    profile_path = tmp_path / "stardist_profile.json"
    est = make_estimator(profile_path, mode="stardist", channels=4, max_size=10000)
    t0 = 7000.0
    est.update_stage_units(
        "init_det_tiles", 3, 10, message=STARDIST_MESSAGES["init_det_tiles"], now=t0
    )
    est.finish(success=False, now=t0 + 1.0)
    assert not profile_path.exists()


def test_rate_info_units_per_second_when_stage_has_units(tmp_path):
    est = make_estimator(tmp_path / "p.json", mode="stardist", channels=6, max_size=10000)
    t0 = 8000.0
    est.update_stage_units(
        "activation_regions",
        1,
        6,
        message=STARDIST_MESSAGES["activation_regions"],
        now=t0,
    )
    _, _, _, rate_info = est.update_stage_units(
        "activation_regions",
        2,
        6,
        message=STARDIST_MESSAGES["activation_regions"],
        now=t0 + 4.0,
    )
    assert rate_info is not None
    assert not rate_info.is_progress_pct
    assert rate_info.units_done == 2
    assert rate_info.units_total == 6
    assert rate_info.stage == "activation_regions"
    assert rate_info.rate == pytest.approx(0.5, rel=0.01)


def test_rate_info_falls_back_to_percent_per_second_without_units(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 9000.0
    est.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    expected_load = LEGACY_PRIORS_PER_PIXEL_SQ["load_images"] * (10000 ** 2)
    expected_pre = LEGACY_PRIORS_PER_PIXEL_SQ["preprocess"] * (10000 ** 2)
    est.update_from_message(
        LEGACY_MESSAGES["preprocess"], now=t0 + expected_load
    )
    est.update_from_message(
        LEGACY_MESSAGES["initial_detection"],
        now=t0 + expected_load + expected_pre,
    )
    _, _, _, rate_info = est.heartbeat(
        now=t0 + expected_load + expected_pre + 4.0
    )
    if rate_info is not None:
        assert rate_info.is_progress_pct
        assert rate_info.units_done is None
        assert rate_info.units_total is None
        assert rate_info.rate > 0


def test_rate_info_suppressed_below_min_elapsed(tmp_path):
    est = make_estimator(tmp_path / "p.json", mode="stardist", channels=6, max_size=10000)
    t0 = 10000.0
    _, _, _, rate_info = est.update_stage_units(
        "init_det_tiles", 1, 10, message=STARDIST_MESSAGES["init_det_tiles"], now=t0
    )
    assert rate_info is None


def _walk_legacy_past_warmup(est, t0):
    e_load = LEGACY_PRIORS_PER_PIXEL_SQ["load_images"] * (10000 ** 2)
    e_pre = LEGACY_PRIORS_PER_PIXEL_SQ["preprocess"] * (10000 ** 2)
    e_init = LEGACY_PRIORS_PER_PIXEL_SQ["initial_detection"] * (10000 ** 2)
    e_dedup = LEGACY_PRIORS_PER_PIXEL_SQ["deduplicate"] * (10000 ** 2)
    est.update_from_message(LEGACY_MESSAGES["load_images"], now=t0)
    est.update_from_message(
        LEGACY_MESSAGES["preprocess"], now=t0 + e_load
    )
    est.update_from_message(
        LEGACY_MESSAGES["initial_detection"], now=t0 + e_load + e_pre
    )
    est.update_from_message(
        LEGACY_MESSAGES["deduplicate"], now=t0 + e_load + e_pre + e_init
    )
    est.update_from_message(
        LEGACY_MESSAGES["second_pass"],
        now=t0 + e_load + e_pre + e_init + e_dedup,
    )
    return t0 + e_load + e_pre + e_init + e_dedup


def test_eta_range_warmup_gate_then_visible_after_progress(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 11000.0
    _, _, range_warm, _ = est.update_from_message(
        LEGACY_MESSAGES["load_images"], now=t0
    )
    assert range_warm is None
    after = _walk_legacy_past_warmup(est, t0)
    progress, _, eta_range, _ = est.heartbeat(now=after + 0.5)
    assert progress >= WARMUP_PROGRESS_GATE
    assert eta_range is not None
    assert eta_range.lo <= eta_range.hi
    assert eta_range.lo > 0


def test_eta_range_upper_bound_non_increasing_within_stage(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 12000.0
    after = _walk_legacy_past_warmup(est, t0)
    his = []
    elapsed_extra = 0.0
    for _ in range(15):
        elapsed_extra += 0.5
        _, _, eta_range, _ = est.heartbeat(now=after + elapsed_extra)
        if eta_range is not None:
            his.append(eta_range.hi)
    assert len(his) >= 5
    for i in range(1, len(his)):
        assert his[i] <= his[i - 1] + 1e-6


def test_eta_range_upper_bound_resets_after_stage_transition(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 13000.0
    after = _walk_legacy_past_warmup(est, t0)
    elapsed_extra = 0.0
    range_before = None
    for _ in range(20):
        elapsed_extra += 0.5
        _, _, candidate_range, _ = est.heartbeat(now=after + elapsed_extra)
        if candidate_range is not None:
            range_before = candidate_range
    assert range_before is not None
    e_second = LEGACY_PRIORS_PER_PIXEL_SQ["second_pass"] * (10000 ** 2)
    transition_now = after + e_second * 3.0
    est.update_from_message(
        LEGACY_MESSAGES["optimize_params"], now=transition_now
    )
    _, _, range_after_transition, _ = est.heartbeat(now=transition_now + 0.5)
    if range_after_transition is not None:
        assert range_after_transition.hi != pytest.approx(range_before.hi, abs=1e-6)


def test_format_eta_range_collapses_when_bounds_equal():
    assert BeadEtaEstimator.format_eta_range(EtaRange(lo=30, hi=30)) == "~30s"
    assert BeadEtaEstimator.format_eta_range(EtaRange(lo=180, hi=300)) == "~3m–5m"
    out = BeadEtaEstimator.format_eta_range(EtaRange(lo=45, hi=90))
    assert out.startswith("~") and "–" in out
