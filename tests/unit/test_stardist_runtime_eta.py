import pytest

from viewmodel.stardist_runtime_eta import StarDistRuntimeEtaTracker


def test_activation_eta_uses_two_x_initial_detection_prior():
    tracker = StarDistRuntimeEtaTracker()
    t0 = 1000.0

    tracker.update_from_message("Initial bead detection...", now=t0)
    tracker.update_stage_units("initial_detection", 1, 4, now=t0 + 1.0)
    tracker.update_stage_units("initial_detection", 4, 4, now=t0 + 5.0)
    tracker.update_from_message("Getting activation regions from cycles", now=t0 + 5.0)
    _, _, _, step_eta = tracker.update_stage_units_with_eta_details(
        "activation_regions", 0, 8, now=t0 + 5.0
    )

    assert step_eta is not None
    assert step_eta == pytest.approx(10.0, abs=0.5)


def test_activation_eta_adapts_with_tile_observations():
    tracker = StarDistRuntimeEtaTracker()
    t0 = 2000.0

    tracker.update_from_message("Initial bead detection...", now=t0)
    tracker.update_stage_units("initial_detection", 4, 4, now=t0 + 4.0)
    tracker.update_from_message("Getting activation regions from cycles", now=t0 + 4.0)
    _, _, _, step_eta_before = tracker.update_stage_units_with_eta_details(
        "activation_regions", 0, 4, now=t0 + 4.0
    )
    _, _, _, step_eta_after_first = tracker.update_stage_units_with_eta_details(
        "activation_regions", 1, 4, now=t0 + 5.0
    )
    _, _, _, step_eta_after_second = tracker.update_stage_units_with_eta_details(
        "activation_regions", 2, 4, now=t0 + 7.0
    )

    assert step_eta_before is not None
    assert step_eta_after_first is not None
    assert step_eta_after_second is not None
    assert step_eta_after_first != pytest.approx(step_eta_before)
    assert step_eta_after_second != pytest.approx(step_eta_after_first)


def test_initial_detection_progress_waits_for_first_unit():
    tracker = StarDistRuntimeEtaTracker()
    t0 = 3000.0

    progress_start, _, _ = tracker.update_from_message(
        "Initial bead detection...", now=t0
    )
    progress_before_first_tile, _, _ = tracker.heartbeat(now=t0 + 30.0)
    progress_after_first_tile, _, _ = tracker.update_stage_units(
        "initial_detection", 1, 4, now=t0 + 31.0
    )

    assert progress_start == 10
    assert progress_before_first_tile == progress_start
    assert progress_after_first_tile > progress_before_first_tile


def test_initial_detection_eta_updates_before_first_tile():
    tracker = StarDistRuntimeEtaTracker()
    t0 = 4000.0

    tracker.update_from_message("Initial bead detection...", now=t0)
    (
        progress_early,
        _,
        total_eta_early,
        step_eta_early,
    ) = tracker.heartbeat_with_eta_details(now=t0 + 1.0)
    (
        progress_late,
        _,
        total_eta_late,
        step_eta_late,
    ) = tracker.heartbeat_with_eta_details(now=t0 + 8.0)

    assert progress_early == 10
    assert progress_late == 10
    assert step_eta_early is not None and step_eta_late is not None
    assert total_eta_early is not None and total_eta_late is not None
    assert step_eta_late < step_eta_early
    assert total_eta_late < total_eta_early
