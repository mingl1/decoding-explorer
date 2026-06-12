import json

from viewmodel.eta_profile_store import PROFILE_VERSION, EtaProfileStore


def test_round_trip_record_and_load(tmp_path):
    profile_path = tmp_path / "profile.json"
    store = EtaProfileStore(path=str(profile_path))
    store.record_run("legacy", {"load_images": 1.0e-7, "preprocess": 5.0e-9})
    store.save()

    reloaded = EtaProfileStore(path=str(profile_path))
    loaded = reloaded.load("legacy")
    assert loaded["load_images"] == 1.0e-7
    assert loaded["preprocess"] == 5.0e-9
    assert reloaded.runs("legacy") == 1


def test_ema_blends_two_runs_strictly_between(tmp_path):
    profile_path = tmp_path / "profile.json"
    store = EtaProfileStore(path=str(profile_path), alpha=0.35)
    store.record_run("legacy", {"load_images": 2.0e-7})
    store.save()

    store2 = EtaProfileStore(path=str(profile_path), alpha=0.35)
    store2.record_run("legacy", {"load_images": 1.0e-7})
    store2.save()

    final = EtaProfileStore(path=str(profile_path)).load("legacy")["load_images"]
    assert min(2.0e-7, 1.0e-7) < final < max(2.0e-7, 1.0e-7)
    expected = 0.35 * 1.0e-7 + 0.65 * 2.0e-7
    assert abs(final - expected) < 1e-12


def test_per_mode_isolation(tmp_path):
    profile_path = tmp_path / "profile.json"
    store = EtaProfileStore(path=str(profile_path))
    store.record_run("stardist", {"activation_regions": 1.6e-6})
    store.save()
    store2 = EtaProfileStore(path=str(profile_path))
    store2.record_run("legacy", {"activation_regions": 1.8e-7})
    store2.save()

    final = EtaProfileStore(path=str(profile_path))
    assert final.load("stardist")["activation_regions"] == 1.6e-6
    assert final.load("legacy")["activation_regions"] == 1.8e-7


def test_version_mismatch_resets_to_defaults(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {"version": 1, "modes": {"legacy": {"per_pixel_sq_seconds": {"x": 9.9}}}}
        ),
        encoding="utf-8",
    )
    store = EtaProfileStore(path=str(profile_path))
    assert store.load("legacy") == {}
    assert store.runs("legacy") == 0


def test_corrupt_json_returns_defaults(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{not valid json", encoding="utf-8")
    store = EtaProfileStore(path=str(profile_path))
    assert store.load("legacy") == {}
    assert store.load("stardist") == {}
    store.record_run("legacy", {"preprocess": 1.0e-9})
    store.save()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["version"] == PROFILE_VERSION


def test_atomic_write_uses_tmp_then_rename(tmp_path):
    profile_path = tmp_path / "subdir" / "profile.json"
    store = EtaProfileStore(path=str(profile_path))
    store.record_run("legacy", {"finalize": 3.0e-9})
    store.save()
    assert profile_path.exists()
    assert not (profile_path.with_suffix(profile_path.suffix + ".tmp")).exists()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["version"] == PROFILE_VERSION
    assert payload["modes"]["legacy"]["per_pixel_sq_seconds"]["finalize"] == 3.0e-9
