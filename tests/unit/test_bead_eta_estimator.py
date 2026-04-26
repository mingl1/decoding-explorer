from viewmodel.bead_eta_estimator import (
    BeadEtaEstimator,
    LEGACY_STAGE_ORDER,
    LEGACY_PRIORS_PER_PIXEL_SQ,
    OVERRUN_FLOOR_SECONDS,
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


def make_estimator(profile_path, mode="legacy", channels=1, max_size=10000):
    store = EtaProfileStore(path=str(profile_path))
    est = BeadEtaEstimator(mode=mode, profile_store=store)
    est.set_workload(total_channels=channels, max_size_pixels=max_size)
    return est


def test_format_eta_zero_padded():
    assert BeadEtaEstimator.format_eta(125) == "02:05"
    assert BeadEtaEstimator.format_eta(0) == "00:00"
    assert BeadEtaEstimator.format_eta(59) == "00:59"
    assert BeadEtaEstimator.format_eta(3600) == "60:00"


def test_legacy_walkthrough_produces_monotonic_progress(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 1000.0
    progresses = []
    elapsed = 0.0
    for i, stage in enumerate(LEGACY_STAGE_ORDER):
        elapsed += 1.0
        progress, _, _, _ = est.update_from_message_with_eta_details(
            LEGACY_MESSAGES[stage], now=t0 + elapsed
        )
        progresses.append(progress)
    assert progresses == sorted(progresses)
    assert max(progresses) <= 99


def test_overrun_grace_step_eta_floors_and_total_eta_does_not_shrink(tmp_path):
    est = make_estimator(tmp_path / "p.json")
    t0 = 5000.0
    est.update_from_message_with_eta_details(
        LEGACY_MESSAGES["activation_regions"], now=t0
    )
    expected = LEGACY_PRIORS_PER_PIXEL_SQ["activation_regions"] * (10000 ** 2)
    _, _, total_before, step_before = est.heartbeat_with_eta_details(now=t0 + expected * 0.5)
    assert step_before is not None and step_before > 0
    assert not est.step_overrun
    _, _, total_overrun, step_overrun = est.heartbeat_with_eta_details(
        now=t0 + expected + 50.0
    )
    assert est.step_overrun is True
    assert step_overrun == OVERRUN_FLOOR_SECONDS
    future = sum(
        LEGACY_PRIORS_PER_PIXEL_SQ[s] * (10000 ** 2)
        for s in ("assign_labels", "resolve_labels", "finalize")
    )
    floor = OVERRUN_FLOOR_SECONDS + future
    assert total_overrun >= floor - 1e-6
    _, _, total_overrun_later, step_overrun_later = est.heartbeat_with_eta_details(
        now=t0 + expected + 200.0
    )
    assert est.step_overrun is True
    assert step_overrun_later == OVERRUN_FLOOR_SECONDS
    assert total_overrun_later >= floor - 1e-6


def test_whole_run_scale_doubles_future_when_observed_doubles(tmp_path):
    est_baseline = make_estimator(tmp_path / "a.json")
    t0 = 1000.0
    est_baseline.update_from_message_with_eta_details(LEGACY_MESSAGES["load_images"], now=t0)
    expected_load = LEGACY_PRIORS_PER_PIXEL_SQ["load_images"] * (10000 ** 2)
    expected_pre = LEGACY_PRIORS_PER_PIXEL_SQ["preprocess"] * (10000 ** 2)
    est_baseline.update_from_message_with_eta_details(
        LEGACY_MESSAGES["preprocess"], now=t0 + expected_load
    )
    est_baseline.update_from_message_with_eta_details(
        LEGACY_MESSAGES["initial_detection"], now=t0 + expected_load + expected_pre
    )
    _, _, total_baseline, _ = est_baseline.heartbeat_with_eta_details(
        now=t0 + expected_load + expected_pre + 0.001
    )

    est_scaled = make_estimator(tmp_path / "b.json")
    est_scaled.update_from_message_with_eta_details(LEGACY_MESSAGES["load_images"], now=t0)
    est_scaled.update_from_message_with_eta_details(
        LEGACY_MESSAGES["preprocess"], now=t0 + expected_load * 2.0
    )
    est_scaled.update_from_message_with_eta_details(
        LEGACY_MESSAGES["initial_detection"],
        now=t0 + expected_load * 2.0 + expected_pre * 2.0,
    )
    _, _, total_scaled, _ = est_scaled.heartbeat_with_eta_details(
        now=t0 + expected_load * 2.0 + expected_pre * 2.0 + 0.001
    )
    assert total_scaled > 1.5 * total_baseline


def test_finish_failure_does_not_persist_profile(tmp_path):
    profile_path = tmp_path / "p.json"
    est = make_estimator(profile_path)
    t0 = 1000.0
    est.update_from_message_with_eta_details(LEGACY_MESSAGES["load_images"], now=t0)
    est.update_from_message_with_eta_details(LEGACY_MESSAGES["preprocess"], now=t0 + 1.0)
    est.finish(success=False, now=t0 + 2.0)
    assert not profile_path.exists()


def test_finish_success_persists_profile(tmp_path):
    profile_path = tmp_path / "p.json"
    est = make_estimator(profile_path)
    t0 = 1000.0
    est.update_from_message_with_eta_details(LEGACY_MESSAGES["load_images"], now=t0)
    est.update_from_message_with_eta_details(LEGACY_MESSAGES["preprocess"], now=t0 + 5.0)
    est.finish(success=True, now=t0 + 6.0)
    assert profile_path.exists()
    reloaded = EtaProfileStore(path=str(profile_path))
    loaded = reloaded.load("legacy")
    assert "load_images" in loaded
    assert reloaded.runs("legacy") == 1
