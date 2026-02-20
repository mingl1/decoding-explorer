import json

from viewmodel.bead_progress_estimator import BeadProgressEstimator


def test_estimator_progress_is_monotonic_and_capped_before_finish(tmp_path):
    profile_path = tmp_path / "profile.json"
    trace_path = tmp_path / "t.json"
    estimator = BeadProgressEstimator(
        mode="stardist",
        profile_path=str(profile_path),
        trace_path=str(trace_path),
    )
    t0 = 1000.0
    values = []
    values.append(estimator.update_from_message("Preprocessing brightfield image...", now=t0)[0])
    values.append(estimator.update_from_message("Initial bead detection...", now=t0 + 0.2)[0])
    values.append(estimator.heartbeat(now=t0 + 2.0)[0])
    values.append(estimator.heartbeat(now=t0 + 8.0)[0])
    values.append(estimator.update_from_message("Getting activation regions from cycles", now=t0 + 9.0)[0])
    values.append(estimator.heartbeat(now=t0 + 20.0)[0])

    assert values == sorted(values)
    assert max(values) <= 99

    estimator.finish(success=True, now=t0 + 25.0)
    final_value, _, _ = estimator.heartbeat(now=t0 + 25.1)
    assert final_value == 100


def test_estimator_eta_is_stable_and_non_negative(tmp_path):
    profile_path = tmp_path / "profile.json"
    estimator = BeadProgressEstimator(mode="stardist", profile_path=str(profile_path))
    t0 = 2000.0
    estimator.update_from_message("Getting activation regions from cycles", now=t0)
    _, _, eta1 = estimator.heartbeat(now=t0 + 1.0)
    _, _, eta2 = estimator.heartbeat(now=t0 + 8.0)
    _, _, eta3 = estimator.heartbeat(now=t0 + 16.0)

    assert eta1 is not None and eta2 is not None and eta3 is not None
    assert eta1 >= 0 and eta2 >= 0 and eta3 >= 0
    assert eta3 <= eta2 <= eta1


def test_stardist_workload_tuning_scales_activation_from_cycle_layers(tmp_path):
    profile_path = tmp_path / "profile.json"
    estimator = BeadProgressEstimator(mode="stardist", profile_path=str(profile_path))
    baseline_activation = estimator._expected_stage_seconds["activation_regions"]
    estimator.configure_stardist_layer_workload(
        total_cycle_layers=40,
        initial_detection_equivalent_layers=4.0,
    )
    tuned_activation = estimator._expected_stage_seconds["activation_regions"]
    assert tuned_activation > baseline_activation


def test_stardist_initial_detection_observation_retunes_activation_eta(tmp_path):
    profile_path = tmp_path / "profile.json"
    estimator = BeadProgressEstimator(mode="stardist", profile_path=str(profile_path))
    estimator.configure_stardist_layer_workload(
        total_cycle_layers=12,
        initial_detection_equivalent_layers=4.0,
    )
    before = estimator._expected_stage_seconds["activation_regions"]
    t0 = 5000.0
    estimator.update_from_message("Initial bead detection...", now=t0)
    estimator.update_from_message("Getting activation regions from cycles", now=t0 + 80.0)
    after = estimator._expected_stage_seconds["activation_regions"]
    assert after > before


def test_estimator_profile_persists_and_reloads(tmp_path):
    profile_path = tmp_path / "bead_profile.json"
    estimator = BeadProgressEstimator(mode="legacy", profile_path=str(profile_path))
    t0 = 3000.0
    estimator.update_from_message("Preprocessing brightfield image...", now=t0)
    estimator.update_from_message("Initial bead detection...", now=t0 + 1.5)
    estimator.update_from_message("Removing duplicate beads...", now=t0 + 5.0)
    estimator.finish(success=True, now=t0 + 7.0)

    assert profile_path.exists()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert "legacy" in payload["modes"]
    assert payload["modes"]["legacy"]["runs"] >= 1

    estimator_reloaded = BeadProgressEstimator(mode="legacy", profile_path=str(profile_path))
    progress_value, _, _ = estimator_reloaded.update_from_message(
        "Initial bead detection...", now=t0 + 10.0
    )
    assert progress_value >= 0


def test_message_normalization_maps_stage_with_eta_suffix(tmp_path):
    estimator = BeadProgressEstimator(mode="stardist", profile_path=str(tmp_path / "p.json"))
    stage = estimator.map_message_to_stage("Getting activation regions from cycles (ETA 00:42)")
    assert stage == "activation_regions"
    stage_with_extended_suffix = estimator.map_message_to_stage(
        "Getting activation regions from cycles (Elapsed 00:12, Step ETA 00:31, Total ETA 00:45)"
    )
    assert stage_with_extended_suffix == "activation_regions"
    progress_value, message, _ = estimator.update_from_message(
        "Getting activation regions from cycles (ETA 00:42)", now=4000.0
    )
    assert progress_value >= 0
    assert message == "Getting activation regions from cycles"
