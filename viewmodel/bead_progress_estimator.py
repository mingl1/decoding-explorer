import json
import re
import time
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple


PROFILE_VERSION = 1
DEFAULT_PROFILE_PATH = Path.home() / ".decoding-explorer" / "bead_progress_profile.json"


STAGE_ORDER = {
    "stardist": [
        "load_images",
        "preprocess",
        "initial_detection",
        "activation_regions",
        "assign_labels",
        "voronoi_cache",
        "voronoi_sweep",
        "voronoi_apply",
        "finalize",
    ],
    "legacy": [
        "load_images",
        "preprocess",
        "initial_detection",
        "deduplicate",
        "second_pass",
        "optimize_params",
        "activation_regions",
        "assign_labels",
        "resolve_labels",
        "finalize",
    ],
}


DEFAULT_STAGE_SECONDS = {
    "stardist": {
        "load_images": 6.0,
        "preprocess": 0.1,
        "initial_detection": 14.0,
        "activation_regions": 40.0,
        "assign_labels": 1.0,
        "voronoi_cache": 1.5,
        "voronoi_sweep": 0.8,
        "voronoi_apply": 0.2,
        "finalize": 0.3,
    },
    "legacy": {
        "load_images": 6.0,
        "preprocess": 0.1,
        "initial_detection": 10.0,
        "deduplicate": 1.0,
        "second_pass": 2.0,
        "optimize_params": 15.0,
        "activation_regions": 18.0,
        "assign_labels": 2.0,
        "resolve_labels": 5.0,
        "finalize": 0.3,
    },
}


ETA_SUFFIX_RE = re.compile(r"\s+\((?:[^()]*)ETA[^()]*\)\s*$")

MESSAGE_STAGE_RULES = [
    (re.compile(r"^Loading images"), "load_images"),
    (re.compile(r"^Preprocessing brightfield image"), "preprocess"),
    (re.compile(r"^Initial bead detection"), "initial_detection"),
    (re.compile(r"^Removing duplicate beads"), "deduplicate"),
    (re.compile(r"^Performing second pass bead detection"), "second_pass"),
    (re.compile(r"^Finding optimal parameters"), "optimize_params"),
    (re.compile(r"^Getting activation regions from cycles"), "activation_regions"),
    (re.compile(r"^Assigning beads labels"), "assign_labels"),
    (re.compile(r"^Processing single assignments"), "resolve_labels"),
    (re.compile(r"^Resolving Bead Labels"), "resolve_labels"),
    (re.compile(r"^Running Voronoi ensemble analysis"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: building ensemble cache"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: sweeping ensemble ratios"), "voronoi_sweep"),
    (re.compile(r"^Voronoi median: applying selected ratio"), "voronoi_apply"),
    (re.compile(r"^Filtering out rows with all zeros"), "finalize"),
    (re.compile(r"^Bead generation complete"), "finalize"),
    (re.compile(r"^Done generating beads"), "finalize"),
]


class BeadProgressEstimator:
    def __init__(
        self,
        mode: str,
        profile_path: Optional[str] = None,
        trace_path: Optional[str] = None,
    ):
        self.mode = "stardist" if mode == "stardist" else "legacy"
        self.profile_path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        self.trace_path = Path(trace_path) if trace_path else Path.cwd() / "t.json"
        self._lock = Lock()
        self._stage_order = STAGE_ORDER[self.mode]
        self._stage_index = {stage: i for i, stage in enumerate(self._stage_order)}
        self._expected_stage_seconds = self._load_expected_stage_seconds()
        self._runs = self._load_runs()
        self._current_stage: Optional[str] = None
        self._current_stage_started_at: Optional[float] = None
        self._stage_units_done: Optional[float] = None
        self._stage_units_total: Optional[float] = None
        self._stage_elapsed_observed: Dict[str, float] = {}
        self._current_message = "Starting bead generation..."
        self._finished = False
        self._last_progress = 0
        self._smoothed_total_eta_seconds: Optional[float] = None
        self._smoothed_step_eta_seconds: Optional[float] = None
        self._stardist_total_cycle_layers: Optional[int] = None
        self._stardist_initial_detection_equivalent_layers = 4.0
        self._stardist_per_layer_seconds: Optional[float] = None

    @staticmethod
    def strip_eta_suffix(message: str) -> str:
        return ETA_SUFFIX_RE.sub("", str(message or "")).strip()

    @staticmethod
    def format_eta(eta_seconds: float) -> str:
        seconds = max(0, int(round(float(eta_seconds))))
        minutes = seconds // 60
        seconds_part = seconds % 60
        return f"{minutes:02d}:{seconds_part:02d}"

    def map_message_to_stage(self, message: str) -> Optional[str]:
        normalized = self.strip_eta_suffix(message)
        for pattern, stage in MESSAGE_STAGE_RULES:
            if pattern.search(normalized):
                return stage
        return None

    def update_from_message(
        self,
        message: str,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float]]:
        progress, normalized_message, total_eta, _ = self.update_from_message_with_eta_details(
            message,
            now=now,
        )
        return progress, normalized_message, total_eta

    def update_from_message_with_eta_details(
        self,
        message: str,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        with self._lock:
            ts = now if now is not None else time.monotonic()
            normalized = self.strip_eta_suffix(message)
            self._current_message = normalized
            stage = self.map_message_to_stage(normalized)
            if stage:
                self._transition_to_stage(stage, ts)
            return self._snapshot_with_eta_details_unlocked(ts)

    def update_stage_units(
        self,
        stage: str,
        done: float,
        total: float,
        message: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float]]:
        progress, progress_message, total_eta, _ = self.update_stage_units_with_eta_details(
            stage=stage,
            done=done,
            total=total,
            message=message,
            now=now,
        )
        return progress, progress_message, total_eta

    def update_stage_units_with_eta_details(
        self,
        stage: str,
        done: float,
        total: float,
        message: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        with self._lock:
            ts = now if now is not None else time.monotonic()
            if message:
                self._current_message = self.strip_eta_suffix(message)
            self._transition_to_stage(stage, ts)
            self._stage_units_total = max(float(total), 1.0)
            self._stage_units_done = min(max(float(done), 0.0), self._stage_units_total)
            return self._snapshot_with_eta_details_unlocked(ts)

    def heartbeat(self, now: Optional[float] = None) -> Tuple[int, str, Optional[float]]:
        progress, progress_message, total_eta, _ = self.heartbeat_with_eta_details(now=now)
        return progress, progress_message, total_eta

    def heartbeat_with_eta_details(
        self,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        with self._lock:
            ts = now if now is not None else time.monotonic()
            return self._snapshot_with_eta_details_unlocked(ts)

    def finish(self, success: bool = True, now: Optional[float] = None):
        with self._lock:
            ts = now if now is not None else time.monotonic()
            if self._current_stage:
                self._close_current_stage(ts)
            self._finished = True
            if success:
                self._last_progress = 100
            self._persist_profile()

    def configure_stardist_layer_workload(
        self,
        total_cycle_layers: int,
        initial_detection_equivalent_layers: float = 4.0,
    ):
        if self.mode != "stardist":
            return
        with self._lock:
            layers = int(total_cycle_layers)
            if layers <= 0:
                return
            eq_layers = float(initial_detection_equivalent_layers)
            if eq_layers <= 0:
                eq_layers = 4.0
            self._stardist_total_cycle_layers = layers
            self._stardist_initial_detection_equivalent_layers = eq_layers
            initial_expected = max(
                self._expected_stage_seconds.get("initial_detection", 0.1),
                0.1,
            )
            per_layer = initial_expected / eq_layers
            self._stardist_per_layer_seconds = per_layer
            self._apply_stardist_workload_tuning(per_layer)

    def _load_runs(self) -> int:
        data = self._read_profile_data()
        mode_data = data.get("modes", {}).get(self.mode, {})
        runs = mode_data.get("runs", 0)
        if isinstance(runs, int) and runs >= 0:
            return runs
        return 0

    def _load_expected_stage_seconds(self) -> Dict[str, float]:
        expected = dict(DEFAULT_STAGE_SECONDS[self.mode])
        data = self._read_profile_data()
        mode_data = data.get("modes", {}).get(self.mode, {})
        stage_data = mode_data.get("stage_ema_seconds", {})
        if isinstance(stage_data, dict):
            for stage in self._stage_order:
                value = stage_data.get(stage)
                if isinstance(value, (int, float)) and value > 0:
                    expected[stage] = float(value)
        if mode_data:
            return expected
        bootstrapped = self._bootstrap_from_trace()
        if bootstrapped:
            for stage, value in bootstrapped.items():
                if stage in expected and value > 0:
                    expected[stage] = value
        return expected

    def _bootstrap_from_trace(self) -> Dict[str, float]:
        if not self.trace_path.exists():
            return {}
        try:
            payload = json.loads(self.trace_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        step_durations = payload.get("step_durations", [])
        if not isinstance(step_durations, list):
            return {}
        out: Dict[str, float] = {}
        for row in step_durations:
            if not isinstance(row, dict):
                continue
            message = row.get("message")
            duration = row.get("duration_s")
            if not isinstance(message, str) or not isinstance(duration, (int, float)):
                continue
            stage = self.map_message_to_stage(message)
            if not stage:
                continue
            out[stage] = out.get(stage, 0.0) + float(duration)
        return out

    def _read_profile_data(self) -> dict:
        if not self.profile_path.exists():
            return {}
        try:
            raw = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        if raw.get("version") != PROFILE_VERSION:
            return {}
        return raw

    def _transition_to_stage(self, stage: str, now: float):
        if stage not in self._stage_index:
            return
        if self._current_stage == stage:
            return
        if self._current_stage is not None:
            self._close_current_stage(now)
        self._current_stage = stage
        self._current_stage_started_at = now
        self._stage_units_done = None
        self._stage_units_total = None

    def _close_current_stage(self, now: float):
        closed_stage = self._current_stage
        if closed_stage is None or self._current_stage_started_at is None:
            return
        elapsed = max(0.0, now - self._current_stage_started_at)
        self._stage_elapsed_observed[closed_stage] = (
            self._stage_elapsed_observed.get(closed_stage, 0.0) + elapsed
        )
        self._current_stage_started_at = None
        self._stage_units_done = None
        self._stage_units_total = None
        self._update_stardist_model_from_observed(closed_stage, elapsed)

    def _apply_stardist_workload_tuning(self, per_layer_seconds: float):
        if self.mode != "stardist":
            return
        if self._stardist_total_cycle_layers is None:
            return
        layers = max(1, int(self._stardist_total_cycle_layers))
        per_layer = max(float(per_layer_seconds), 0.01)
        eq_layers = max(float(self._stardist_initial_detection_equivalent_layers), 1.0)
        activation_expected = max(0.2, per_layer * layers)
        default_activation = max(DEFAULT_STAGE_SECONDS["stardist"]["activation_regions"], 0.2)
        default_cache = max(DEFAULT_STAGE_SECONDS["stardist"]["voronoi_cache"], 0.05)
        cache_ratio = default_cache / default_activation
        self._expected_stage_seconds["initial_detection"] = max(0.2, per_layer * eq_layers)
        self._expected_stage_seconds["activation_regions"] = activation_expected
        self._expected_stage_seconds["voronoi_cache"] = max(
            0.05,
            activation_expected * cache_ratio,
        )

    def _update_stardist_model_from_observed(self, stage: str, elapsed: float):
        if self.mode != "stardist":
            return
        if self._stardist_total_cycle_layers is None:
            return
        if stage != "initial_detection":
            return
        eq_layers = max(float(self._stardist_initial_detection_equivalent_layers), 1.0)
        observed_per_layer = max(float(elapsed), 0.01) / eq_layers
        previous = (
            self._stardist_per_layer_seconds
            if self._stardist_per_layer_seconds is not None
            else observed_per_layer
        )
        blended = 0.6 * observed_per_layer + 0.4 * max(float(previous), 0.01)
        self._stardist_per_layer_seconds = blended
        self._apply_stardist_workload_tuning(blended)

    def _snapshot_unlocked(self, now: float) -> Tuple[int, str, Optional[float]]:
        progress, message, total_eta, _ = self._snapshot_with_eta_details_unlocked(now)
        return progress, message, total_eta

    def _snapshot_with_eta_details_unlocked(
        self, now: float
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        if self._finished:
            return 100, self._current_message, 0.0, 0.0

        expected_total = sum(self._expected_stage_seconds.values())
        if expected_total <= 0:
            return self._last_progress, self._current_message, None, None

        current_stage_index = -1
        stage_fraction = 0.0
        elapsed_current = 0.0
        expected_current = 0.0

        if self._current_stage:
            current_stage_index = self._stage_index[self._current_stage]
            if self._current_stage_started_at is not None:
                elapsed_current = max(0.0, now - self._current_stage_started_at)

            expected_current = self._expected_stage_seconds.get(self._current_stage, 0.1)
            if (
                self._stage_units_total is not None
                and self._stage_units_total > 0
                and self._stage_units_done is not None
            ):
                stage_fraction = min(max(self._stage_units_done / self._stage_units_total, 0.0), 1.0)
            else:
                stage_fraction = min(max(elapsed_current / max(expected_current, 0.1), 0.0), 0.99)

        completed_expected = 0.0
        completed_observed = 0.0
        for stage in self._stage_order:
            idx = self._stage_index[stage]
            if idx < current_stage_index:
                completed_expected += self._expected_stage_seconds.get(stage, 0.0)
                completed_observed += self._stage_elapsed_observed.get(stage, 0.0)

        if self._current_stage:
            completed_progress_expected = completed_expected + expected_current * stage_fraction
        else:
            completed_progress_expected = completed_expected

        scale = 1.0
        if completed_expected > 0 and completed_observed > 0:
            scale = completed_observed / completed_expected
            scale = min(max(scale, 0.5), 3.0)

        progress_fraction = min(max(completed_progress_expected / expected_total, 0.0), 0.999)
        raw_progress = int(progress_fraction * 100)
        raw_progress = min(max(raw_progress, self._last_progress), 99)
        self._last_progress = raw_progress

        remaining_expected = max(expected_total - completed_progress_expected, 0.0)
        raw_total_eta = remaining_expected * scale
        if raw_total_eta <= 0:
            total_eta = 0.0
            self._smoothed_total_eta_seconds = 0.0
        else:
            if self._smoothed_total_eta_seconds is None:
                self._smoothed_total_eta_seconds = raw_total_eta
            else:
                self._smoothed_total_eta_seconds = (
                    0.25 * raw_total_eta + 0.75 * self._smoothed_total_eta_seconds
                )
            total_eta = self._smoothed_total_eta_seconds

        step_eta: Optional[float] = None
        if self._current_stage:
            stage_remaining_expected = max(expected_current * (1.0 - stage_fraction), 0.0)
            stage_scale = scale
            expected_elapsed_current = expected_current * stage_fraction
            if elapsed_current > 0 and expected_elapsed_current > 0:
                stage_scale = elapsed_current / expected_elapsed_current
                stage_scale = min(max(stage_scale, 0.5), 3.0)
            raw_step_eta = stage_remaining_expected * stage_scale
            if raw_step_eta <= 0:
                step_eta = 0.0
                self._smoothed_step_eta_seconds = 0.0
            else:
                if self._smoothed_step_eta_seconds is None:
                    self._smoothed_step_eta_seconds = raw_step_eta
                else:
                    self._smoothed_step_eta_seconds = (
                        0.25 * raw_step_eta + 0.75 * self._smoothed_step_eta_seconds
                    )
                step_eta = self._smoothed_step_eta_seconds

        return raw_progress, self._current_message, total_eta, step_eta

    def _persist_profile(self):
        existing = self._read_profile_data()
        if not existing:
            existing = {"version": PROFILE_VERSION, "modes": {}}
        modes = existing.setdefault("modes", {})
        mode_data = modes.setdefault(self.mode, {})
        ema = dict(mode_data.get("stage_ema_seconds", {}))
        alpha = 0.35
        for stage in self._stage_order:
            observed = self._stage_elapsed_observed.get(stage)
            if observed is None or observed <= 0:
                continue
            previous = ema.get(stage, self._expected_stage_seconds.get(stage, observed))
            if not isinstance(previous, (int, float)) or previous <= 0:
                previous = observed
            ema[stage] = round(alpha * observed + (1.0 - alpha) * float(previous), 6)
        mode_data["stage_ema_seconds"] = ema
        mode_data["runs"] = int(mode_data.get("runs", self._runs)) + 1
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.profile_path.with_suffix(self.profile_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp_path.replace(self.profile_path)
