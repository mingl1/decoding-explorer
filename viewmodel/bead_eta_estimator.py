import re
import time
from threading import Lock
from typing import Dict, Optional, Tuple

from viewmodel.eta_profile_store import EtaProfileStore


ETA_SUFFIX_RE = re.compile(r"\s+\((?:[^()]*)ETA[^()]*\)\s*$")

DEFAULT_MAX_SIZE = 10000

LEGACY_STAGE_ORDER = [
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
]

STARDIST_STAGE_ORDER = [
    "load_images",
    "preprocess",
    "init_det_tiles",
    "init_det_morphology",
    "activation_regions",
    "assign_labels",
    "voronoi_cache",
    "voronoi_sweep",
    "voronoi_apply",
    "finalize",
]

LEGACY_PRIORS_PER_PIXEL_SQ = {
    "load_images": 6.0e-8,
    "preprocess": 1.0e-9,
    "initial_detection": 1.0e-7,
    "deduplicate": 1.0e-8,
    "second_pass": 2.0e-8,
    "optimize_params": 1.5e-7,
    "activation_regions": 1.8e-7,
    "assign_labels": 2.0e-8,
    "resolve_labels": 5.0e-8,
    "finalize": 3.0e-9,
}

STARDIST_PRIORS_PER_PIXEL_SQ = {
    "load_images": 6.0e-8,
    "preprocess": 1.08e-8,
    "init_det_tiles": 9.5e-7,
    "init_det_morphology": 2.4e-7,
    "activation_regions": 1.63e-6,
    "assign_labels": 1.44e-7,
    "voronoi_cache": 2.69e-6,
    "voronoi_sweep": 4.99e-7,
    "voronoi_apply": 4.01e-8,
    "finalize": 6.45e-9,
}

LEGACY_MESSAGE_STAGE_RULES = [
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
    (re.compile(r"^Filtering out rows with all zeros"), "finalize"),
    (re.compile(r"^Bead generation complete"), "finalize"),
    (re.compile(r"^Done generating beads"), "finalize"),
]

STARDIST_MESSAGE_STAGE_RULES = [
    (re.compile(r"^Loading images"), "load_images"),
    (re.compile(r"^Preprocessing brightfield image"), "preprocess"),
    (re.compile(r"^Initial bead detection - finalizing"), "init_det_morphology"),
    (re.compile(r"^Initial bead detection"), "init_det_tiles"),
    (re.compile(r"^Getting activation regions from cycles"), "activation_regions"),
    (re.compile(r"^Processing fluorescence channels"), "activation_regions"),
    (re.compile(r"^Assigning beads labels"), "assign_labels"),
    (re.compile(r"^Running Voronoi ensemble analysis"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: building ensemble cache"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: sweeping ensemble ratios"), "voronoi_sweep"),
    (re.compile(r"^Voronoi median: applying selected ratio"), "voronoi_apply"),
    (re.compile(r"^Bead generation complete"), "finalize"),
    (re.compile(r"^Done generating beads"), "finalize"),
]

OVERRUN_FLOOR_SECONDS = 10.0
SCALE_MIN = 0.5
SCALE_MAX = 3.0
SCALE_GATE_SECONDS = 5.0
TOTAL_ETA_ALPHA = 0.25
STEP_ETA_HIDE_THRESHOLD = 2.0


class BeadEtaEstimator:
    def __init__(
        self,
        mode: str,
        profile_store: Optional[EtaProfileStore] = None,
    ):
        self.mode = "stardist" if mode == "stardist" else "legacy"
        self._lock = Lock()
        if self.mode == "stardist":
            self._stage_order = list(STARDIST_STAGE_ORDER)
            self._priors = dict(STARDIST_PRIORS_PER_PIXEL_SQ)
            self._message_rules = STARDIST_MESSAGE_STAGE_RULES
        else:
            self._stage_order = list(LEGACY_STAGE_ORDER)
            self._priors = dict(LEGACY_PRIORS_PER_PIXEL_SQ)
            self._message_rules = LEGACY_MESSAGE_STAGE_RULES
        self._stage_index = {stage: i for i, stage in enumerate(self._stage_order)}
        self._profile_store = profile_store
        if profile_store is not None:
            for stage, value in profile_store.load(self.mode).items():
                if stage in self._priors and value > 0:
                    self._priors[stage] = float(value)
        self._max_size: Optional[int] = None
        self._total_channels: Optional[int] = None
        self._expected_seconds: Dict[str, float] = self._compute_expected_seconds(
            DEFAULT_MAX_SIZE, 1
        )
        self._current_stage: Optional[str] = None
        self._current_stage_started_at: Optional[float] = None
        self._stage_units_done: Optional[float] = None
        self._stage_units_total: Optional[float] = None
        self._stage_elapsed_observed: Dict[str, float] = {}
        self._current_message = "Starting bead generation..."
        self._finished = False
        self._success = False
        self._last_progress = 0
        self._smoothed_total_eta: Optional[float] = None
        self._step_overrun = False
        self._total_eta_floor: Optional[float] = None

    @staticmethod
    def format_eta(eta_seconds: float) -> str:
        seconds = max(0, int(round(float(eta_seconds))))
        minutes = seconds // 60
        seconds_part = seconds % 60
        return f"{minutes:02d}:{seconds_part:02d}"

    @staticmethod
    def strip_eta_suffix(message: str) -> str:
        return ETA_SUFFIX_RE.sub("", str(message or "")).strip()

    @property
    def step_overrun(self) -> bool:
        return self._step_overrun

    def map_message_to_stage(self, message: str) -> Optional[str]:
        normalized = self.strip_eta_suffix(message)
        for pattern, stage in self._message_rules:
            if pattern.search(normalized):
                return stage
        return None

    def set_workload(self, total_channels: int, max_size_pixels: int):
        with self._lock:
            try:
                channels = max(int(total_channels), 1)
            except (TypeError, ValueError):
                channels = 1
            try:
                max_size = max(int(max_size_pixels), 1)
            except (TypeError, ValueError):
                max_size = DEFAULT_MAX_SIZE
            self._total_channels = channels
            self._max_size = max_size
            self._expected_seconds = self._compute_expected_seconds(max_size, channels)

    def _compute_expected_seconds(
        self, max_size: int, total_channels: int
    ) -> Dict[str, float]:
        pixels_sq = float(max_size) * float(max_size)
        out: Dict[str, float] = {}
        for stage in self._stage_order:
            prior = self._priors.get(stage, 0.0)
            if stage == "activation_regions" and self.mode == "stardist":
                out[stage] = prior * pixels_sq * float(total_channels)
            else:
                out[stage] = prior * pixels_sq
        return out

    def update_from_message_with_eta_details(
        self,
        message: str,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        with self._lock:
            ts = now if now is not None else time.monotonic()
            normalized = self.strip_eta_suffix(message)
            self._current_message = normalized
            stage = self._map_message_to_stage_unlocked(normalized)
            if stage:
                self._transition_to_stage(stage, ts)
            return self._snapshot_unlocked(ts)

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
            if stage in self._stage_index:
                self._transition_to_stage(stage, ts)
                self._stage_units_total = max(float(total), 1.0)
                self._stage_units_done = min(
                    max(float(done), 0.0), self._stage_units_total
                )
            return self._snapshot_unlocked(ts)

    def heartbeat_with_eta_details(
        self,
        now: Optional[float] = None,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        with self._lock:
            ts = now if now is not None else time.monotonic()
            return self._snapshot_unlocked(ts)

    def update_from_message(self, message: str, now: Optional[float] = None):
        progress, msg, total_eta, _ = self.update_from_message_with_eta_details(
            message, now=now
        )
        return progress, msg, total_eta

    def update_stage_units(
        self,
        stage: str,
        done: float,
        total: float,
        message: Optional[str] = None,
        now: Optional[float] = None,
    ):
        progress, msg, total_eta, _ = self.update_stage_units_with_eta_details(
            stage=stage, done=done, total=total, message=message, now=now
        )
        return progress, msg, total_eta

    def heartbeat(self, now: Optional[float] = None):
        progress, msg, total_eta, _ = self.heartbeat_with_eta_details(now=now)
        return progress, msg, total_eta

    def finish(self, success: bool = True, now: Optional[float] = None):
        with self._lock:
            ts = now if now is not None else time.monotonic()
            if self._current_stage is not None:
                self._close_current_stage(ts)
            self._finished = True
            self._success = bool(success)
            if success:
                self._last_progress = 100
            if success and self._profile_store is not None:
                observed = self._observed_per_pixel_sq()
                if observed:
                    self._profile_store.record_run(self.mode, observed)
                    self._profile_store.save()

    def _observed_per_pixel_sq(self) -> Dict[str, float]:
        max_size = self._max_size if self._max_size else DEFAULT_MAX_SIZE
        pixels_sq = float(max_size) * float(max_size)
        if pixels_sq <= 0:
            return {}
        out: Dict[str, float] = {}
        for stage, elapsed in self._stage_elapsed_observed.items():
            if elapsed <= 0:
                continue
            if stage == "activation_regions" and self.mode == "stardist":
                channels = max(self._total_channels or 1, 1)
                out[stage] = elapsed / (pixels_sq * channels)
            else:
                out[stage] = elapsed / pixels_sq
        return out

    def _map_message_to_stage_unlocked(self, normalized: str) -> Optional[str]:
        for pattern, stage in self._message_rules:
            if pattern.search(normalized):
                return stage
        return None

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
        self._step_overrun = False
        self._total_eta_floor = None

    def _close_current_stage(self, now: float):
        closed_stage = self._current_stage
        if closed_stage is None or self._current_stage_started_at is None:
            return
        elapsed = max(0.0, now - self._current_stage_started_at)
        self._stage_elapsed_observed[closed_stage] = (
            self._stage_elapsed_observed.get(closed_stage, 0.0) + elapsed
        )
        self._current_stage = None
        self._current_stage_started_at = None
        self._stage_units_done = None
        self._stage_units_total = None
        self._step_overrun = False
        self._total_eta_floor = None

    def _whole_run_scale(self) -> float:
        sum_expected = 0.0
        sum_observed = 0.0
        for stage, observed in self._stage_elapsed_observed.items():
            expected = self._expected_seconds.get(stage, 0.0)
            sum_expected += expected
            sum_observed += observed
        if sum_expected < SCALE_GATE_SECONDS:
            return 1.0
        if sum_observed <= 0:
            return 1.0
        scale = sum_observed / sum_expected
        return min(max(scale, SCALE_MIN), SCALE_MAX)

    def _compute_progress(self, scale: float, now: float) -> int:
        expected_total = sum(
            max(self._expected_seconds.get(stage, 0.0), 0.0) * scale
            for stage in self._stage_order
        )
        if expected_total <= 0:
            return self._last_progress
        completed = 0.0
        for stage in self._stage_order:
            if stage == self._current_stage:
                break
            completed += self._expected_seconds.get(stage, 0.0) * scale
        if self._current_stage is not None and self._current_stage_started_at is not None:
            elapsed = max(0.0, now - self._current_stage_started_at)
            expected_current = self._expected_seconds.get(self._current_stage, 0.0) * scale
            stage_fraction = 0.0
            if (
                self._stage_units_total is not None
                and self._stage_units_total > 0
                and self._stage_units_done is not None
            ):
                stage_fraction = min(
                    max(self._stage_units_done / self._stage_units_total, 0.0),
                    1.0,
                )
            elif expected_current > 0:
                stage_fraction = min(max(elapsed / expected_current, 0.0), 0.99)
            completed += expected_current * stage_fraction
        fraction = min(max(completed / expected_total, 0.0), 0.999)
        raw_progress = int(fraction * 100)
        raw_progress = min(max(raw_progress, self._last_progress), 99)
        self._last_progress = raw_progress
        return raw_progress

    def _compute_step_eta(self, scale: float, now: float) -> Optional[float]:
        if self._current_stage is None or self._current_stage_started_at is None:
            return None
        expected_current = self._expected_seconds.get(self._current_stage, 0.0) * scale
        elapsed = max(0.0, now - self._current_stage_started_at)
        if (
            self._stage_units_total is not None
            and self._stage_units_total > 0
            and self._stage_units_done is not None
            and self._stage_units_done > 0
        ):
            per_unit_observed = elapsed / self._stage_units_done
            per_unit_prior = (
                expected_current / self._stage_units_total
                if self._stage_units_total > 0
                else per_unit_observed
            )
            per_unit = (
                per_unit_observed if per_unit_observed > 0 else max(per_unit_prior, 0.01)
            )
            remaining_units = max(self._stage_units_total - self._stage_units_done, 0.0)
            self._step_overrun = False
            return max(remaining_units * per_unit, 0.0)
        if (
            self._stage_units_total is not None
            and self._stage_units_total > 0
            and (self._stage_units_done is None or self._stage_units_done <= 0)
        ):
            self._step_overrun = False
            return max(expected_current, 0.0)
        if expected_current > 0 and elapsed > expected_current:
            self._step_overrun = True
            return OVERRUN_FLOOR_SECONDS
        self._step_overrun = False
        return max(expected_current - elapsed, 0.0)

    def _compute_total_eta(
        self, scale: float, step_eta: Optional[float]
    ) -> Optional[float]:
        if self._current_stage is None:
            return None
        idx = self._stage_index[self._current_stage]
        future = sum(
            self._expected_seconds.get(stage, 0.0) * scale
            for stage in self._stage_order[idx + 1 :]
        )
        step_component = step_eta if step_eta is not None else 0.0
        raw_total = max(step_component + future, 0.0)
        if self._step_overrun:
            floor = OVERRUN_FLOOR_SECONDS + future
            if self._total_eta_floor is None or floor < self._total_eta_floor:
                self._total_eta_floor = floor
            raw_total = max(raw_total, self._total_eta_floor)
        else:
            self._total_eta_floor = None
        if raw_total <= 0:
            self._smoothed_total_eta = 0.0
            return 0.0
        if self._smoothed_total_eta is None:
            self._smoothed_total_eta = raw_total
        else:
            self._smoothed_total_eta = (
                TOTAL_ETA_ALPHA * raw_total
                + (1.0 - TOTAL_ETA_ALPHA) * self._smoothed_total_eta
            )
        if self._step_overrun and self._smoothed_total_eta < raw_total:
            self._smoothed_total_eta = raw_total
        return self._smoothed_total_eta

    def _snapshot_unlocked(
        self, now: float
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        if self._finished:
            self._step_overrun = False
            if self._success:
                return 100, self._current_message, 0.0, 0.0
            return self._last_progress, self._current_message, None, None
        scale = self._whole_run_scale()
        progress = self._compute_progress(scale, now)
        step_eta = self._compute_step_eta(scale, now)
        total_eta = self._compute_total_eta(scale, step_eta)
        if self._current_stage is not None:
            expected_current = self._expected_seconds.get(self._current_stage, 0.0) * scale
            if expected_current < STEP_ETA_HIDE_THRESHOLD and not self._step_overrun:
                step_eta = None
        if progress <= 0 or progress >= 100:
            step_eta = None
            total_eta = None
        return progress, self._current_message, total_eta, step_eta
