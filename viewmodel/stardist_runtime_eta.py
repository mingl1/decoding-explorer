import re
import time
from threading import Lock
from typing import Dict, Optional, Tuple


ETA_SUFFIX_RE = re.compile(r"\s+\((?:[^()]*)ETA[^()]*\)\s*$")

STAGE_ORDER = [
    "load_images",
    "preprocess",
    "initial_detection",
    "activation_regions",
    "assign_labels",
    "voronoi_cache",
    "voronoi_sweep",
    "voronoi_apply",
    "finalize",
]

STAGE_PROGRESS_BOUNDS = {
    "load_images": (0, 8),
    "preprocess": (8, 10),
    "initial_detection": (10, 30),
    "activation_regions": (30, 60),
    "assign_labels": (60, 72),
    "voronoi_cache": (72, 84),
    "voronoi_sweep": (84, 90),
    "voronoi_apply": (90, 95),
    "finalize": (95, 99),
}

DEFAULT_STAGE_SECONDS = {
    "load_images": 3.0,
    "preprocess": 0.4,
    "initial_detection": 14.0,
    "activation_regions": 28.0,
    "assign_labels": 1.5,
    "voronoi_cache": 2.0,
    "voronoi_sweep": 1.0,
    "voronoi_apply": 0.4,
    "finalize": 0.4,
}

MESSAGE_STAGE_RULES = [
    (re.compile(r"^Loading images"), "load_images"),
    (re.compile(r"^Preprocessing brightfield image"), "preprocess"),
    (re.compile(r"^Initial bead detection"), "initial_detection"),
    (re.compile(r"^Getting activation regions from cycles"), "activation_regions"),
    (re.compile(r"^Assigning beads labels"), "assign_labels"),
    (re.compile(r"^Running Voronoi ensemble analysis"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: building ensemble cache"), "voronoi_cache"),
    (re.compile(r"^Voronoi median: sweeping ensemble ratios"), "voronoi_sweep"),
    (re.compile(r"^Voronoi median: applying selected ratio"), "voronoi_apply"),
    (re.compile(r"^Bead generation complete"), "finalize"),
    (re.compile(r"^Done generating beads"), "finalize"),
]

class StarDistRuntimeEtaTracker:
    def __init__(
        self,
        tile_alpha: float = 0.2,
        eta_alpha: float = 0.25,
    ):
        self._lock = Lock()
        self._current_stage: Optional[str] = None
        self._current_message = "Starting bead generation..."
        self._current_stage_started_at: Optional[float] = None
        self._current_stage_units_done: Optional[float] = None
        self._current_stage_units_total: Optional[float] = None
        self._current_stage_last_unit_done: Optional[float] = None
        self._current_stage_last_unit_time: Optional[float] = None
        self._finished = False
        self._last_progress = 0
        self._stage_observed_seconds: Dict[str, float] = {}
        self._stage_unit_avg_seconds: Dict[str, float] = {}
        self._smoothed_total_eta_seconds: Optional[float] = None
        self._smoothed_step_eta_seconds: Optional[float] = None
        self._tile_alpha = max(min(float(tile_alpha), 1.0), 0.01)
        self._eta_alpha = max(min(float(eta_alpha), 1.0), 0.01)
        self._initial_detection_elapsed: Optional[float] = None
        self._activation_tile_seconds: Optional[float] = None
        self._activation_prior_tile_seconds: Optional[float] = None

    @staticmethod
    def strip_eta_suffix(message: str) -> str:
        return ETA_SUFFIX_RE.sub("", str(message or "")).strip()

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
            message=message,
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
            if stage not in STAGE_PROGRESS_BOUNDS:
                return self._snapshot_with_eta_details_unlocked(ts)
            self._transition_to_stage(stage, ts)
            self._update_stage_units_unlocked(
                stage=stage,
                done=max(float(done), 0.0),
                total=max(float(total), 1.0),
                now=ts,
            )
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

    def _transition_to_stage(self, stage: str, now: float):
        if self._current_stage == stage:
            return
        if self._current_stage is not None:
            self._close_current_stage(now)
        self._current_stage = stage
        self._current_stage_started_at = now
        self._current_stage_units_done = None
        self._current_stage_units_total = None
        self._current_stage_last_unit_done = None
        self._current_stage_last_unit_time = None
        self._smoothed_step_eta_seconds = None
        if stage == "activation_regions":
            self._seed_activation_prior_unlocked()

    def _close_current_stage(self, now: float):
        closed_stage = self._current_stage
        started_at = self._current_stage_started_at
        if closed_stage is None or started_at is None:
            return
        elapsed = max(0.0, now - started_at)
        self._stage_observed_seconds[closed_stage] = elapsed
        if closed_stage == "initial_detection":
            self._initial_detection_elapsed = elapsed
        self._current_stage = None
        self._current_stage_started_at = None
        self._current_stage_units_done = None
        self._current_stage_units_total = None
        self._current_stage_last_unit_done = None
        self._current_stage_last_unit_time = None

    def _seed_activation_prior_unlocked(self):
        if self._initial_detection_elapsed is None:
            return
        total_units = self._current_stage_units_total
        if total_units is None or total_units <= 0:
            return
        prior_total = max(self._initial_detection_elapsed * 2.0, 0.1)
        prior_tile = prior_total / total_units
        self._activation_prior_tile_seconds = max(prior_tile, 0.01)
        if self._activation_tile_seconds is None:
            self._activation_tile_seconds = self._activation_prior_tile_seconds

    def _update_stage_units_unlocked(
        self,
        stage: str,
        done: float,
        total: float,
        now: float,
    ):
        clipped_done = min(done, total)
        if self._current_stage != stage:
            return
        previous_done = (
            self._current_stage_last_unit_done
            if self._current_stage_last_unit_done is not None
            else clipped_done
        )
        previous_time = self._current_stage_last_unit_time
        self._current_stage_units_done = clipped_done
        self._current_stage_units_total = total
        if stage == "activation_regions" and self._activation_tile_seconds is None:
            self._seed_activation_prior_unlocked()
        if (
            stage == "activation_regions"
            and clipped_done <= 0
            and self._activation_tile_seconds is not None
        ):
            self._smoothed_step_eta_seconds = None
        if previous_time is not None and clipped_done > previous_done:
            delta_done = clipped_done - previous_done
            delta_time = max(now - previous_time, 1e-6)
            observed_per_unit = delta_time / delta_done
            current_avg = self._stage_unit_avg_seconds.get(stage)
            if current_avg is None:
                blended = observed_per_unit
            else:
                blended = (1.0 - self._tile_alpha) * current_avg + self._tile_alpha * observed_per_unit
            blended = max(blended, 0.01)
            self._stage_unit_avg_seconds[stage] = blended
            if stage == "activation_regions":
                prior = self._activation_tile_seconds
                if prior is None:
                    self._activation_tile_seconds = blended
                else:
                    self._activation_tile_seconds = (
                        (1.0 - self._tile_alpha) * prior + self._tile_alpha * observed_per_unit
                    )
        self._current_stage_last_unit_done = clipped_done
        self._current_stage_last_unit_time = now

    def _snapshot_with_eta_details_unlocked(
        self,
        now: float,
    ) -> Tuple[int, str, Optional[float], Optional[float]]:
        if self._finished:
            return 100, self._current_message, 0.0, 0.0

        progress_value = self._compute_progress_unlocked(now)
        step_eta = self._compute_step_eta_unlocked(now)
        total_eta = self._compute_total_eta_unlocked(step_eta, now)

        return progress_value, self._current_message, total_eta, step_eta

    def _compute_progress_unlocked(self, now: float) -> int:
        if self._current_stage is None:
            return self._last_progress
        stage = self._current_stage
        start, end = STAGE_PROGRESS_BOUNDS[stage]
        span = max(end - start, 1)
        fraction = self._compute_stage_fraction_unlocked(stage, now)
        raw_progress = int(start + span * fraction)
        raw_progress = min(max(raw_progress, self._last_progress), 99)
        self._last_progress = raw_progress
        return raw_progress

    def _compute_stage_fraction_unlocked(self, stage: str, now: float) -> float:
        done = self._current_stage_units_done
        total = self._current_stage_units_total
        if done is not None and total is not None and total > 0:
            return min(max(done / total, 0.0), 1.0)
        started_at = self._current_stage_started_at
        if started_at is None:
            return 0.0
        elapsed = max(now - started_at, 0.0)
        expected = self._estimate_stage_total_seconds_unlocked(stage, now)
        if expected <= 0:
            return 0.0
        return min(max(elapsed / expected, 0.0), 0.99)

    def _compute_step_eta_unlocked(self, now: float) -> Optional[float]:
        stage = self._current_stage
        if stage is None:
            return None
        done = self._current_stage_units_done
        total = self._current_stage_units_total
        if done is not None and total is not None and total > 0:
            avg = self._stage_unit_avg_seconds.get(stage)
            if stage == "activation_regions" and self._activation_tile_seconds is not None:
                avg = self._activation_tile_seconds
            if avg is None and done > 0 and self._current_stage_started_at is not None:
                elapsed = max(now - self._current_stage_started_at, 0.0)
                avg = max(elapsed / done, 0.01)
            if avg is not None:
                remaining_units = max(total - done, 0.0)
                return self._smooth_step_eta_unlocked(remaining_units * max(avg, 0.01))
        if self._current_stage_started_at is None:
            return None
        elapsed = max(now - self._current_stage_started_at, 0.0)
        expected = self._estimate_stage_total_seconds_unlocked(stage, now)
        return self._smooth_step_eta_unlocked(max(expected - elapsed, 0.0))

    def _compute_total_eta_unlocked(self, step_eta: Optional[float], now: float) -> Optional[float]:
        stage = self._current_stage
        if stage is None:
            return None
        stage_idx = STAGE_ORDER.index(stage)
        remaining = step_eta if step_eta is not None else 0.0
        for future_stage in STAGE_ORDER[stage_idx + 1 :]:
            remaining += self._estimate_stage_total_seconds_unlocked(future_stage, now)
        if remaining <= 0:
            self._smoothed_total_eta_seconds = 0.0
            return 0.0
        if self._smoothed_total_eta_seconds is None:
            self._smoothed_total_eta_seconds = remaining
        else:
            self._smoothed_total_eta_seconds = (
                (1.0 - self._eta_alpha) * self._smoothed_total_eta_seconds
                + self._eta_alpha * remaining
            )
        return self._smoothed_total_eta_seconds

    def _estimate_stage_total_seconds_unlocked(self, stage: str, now: Optional[float] = None) -> float:
        observed = self._stage_observed_seconds.get(stage)
        if observed is not None and observed > 0:
            return observed
        if stage == self._current_stage:
            ts = now if now is not None else time.monotonic()
            done = self._current_stage_units_done
            total = self._current_stage_units_total
            if done is not None and total is not None and total > 0:
                avg = self._stage_unit_avg_seconds.get(stage)
                if stage == "activation_regions" and self._activation_tile_seconds is not None:
                    avg = self._activation_tile_seconds
                if avg is not None:
                    return max(avg * total, 0.01)
                if done > 0 and self._current_stage_started_at is not None:
                    elapsed = max(ts - self._current_stage_started_at, 0.0)
                    return max((elapsed / done) * total, 0.01)
            if self._current_stage_started_at is not None:
                elapsed = max(ts - self._current_stage_started_at, 0.0)
                default_estimate = DEFAULT_STAGE_SECONDS.get(stage, 0.1)
                return max(default_estimate, elapsed)
        return DEFAULT_STAGE_SECONDS.get(stage, 0.1)

    def _smooth_step_eta_unlocked(self, raw_value: float) -> float:
        raw_value = max(raw_value, 0.0)
        if raw_value <= 0:
            self._smoothed_step_eta_seconds = 0.0
            return 0.0
        if self._smoothed_step_eta_seconds is None:
            self._smoothed_step_eta_seconds = raw_value
        else:
            self._smoothed_step_eta_seconds = (
                (1.0 - self._eta_alpha) * self._smoothed_step_eta_seconds
                + self._eta_alpha * raw_value
            )
        return self._smoothed_step_eta_seconds
