import json
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

PROFILE_VERSION = 2
DEFAULT_PROFILE_PATH = Path.home() / ".decoding-explorer" / "bead_progress_profile.json"
DEFAULT_ALPHA = 0.35


class EtaProfileStore:
    def __init__(self, path: Optional[str] = None, alpha: float = DEFAULT_ALPHA):
        self.path = Path(path) if path else DEFAULT_PROFILE_PATH
        self.alpha = float(alpha)
        self._lock = Lock()
        self._data = self._read()

    def _read(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self._empty()
        if not isinstance(raw, dict):
            return self._empty()
        if raw.get("version") != PROFILE_VERSION:
            return self._empty()
        modes = raw.get("modes")
        if not isinstance(modes, dict):
            raw["modes"] = {}
        return raw

    @staticmethod
    def _empty() -> dict:
        return {"version": PROFILE_VERSION, "modes": {}}

    def load(self, mode: str) -> Dict[str, float]:
        with self._lock:
            mode_data = self._data.get("modes", {}).get(mode, {})
            per_pix = mode_data.get("per_pixel_sq_seconds", {})
            if not isinstance(per_pix, dict):
                return {}
            return {
                str(stage): float(value)
                for stage, value in per_pix.items()
                if isinstance(value, (int, float)) and value > 0
            }

    def runs(self, mode: str) -> int:
        with self._lock:
            value = self._data.get("modes", {}).get(mode, {}).get("runs", 0)
            return int(value) if isinstance(value, int) and value >= 0 else 0

    def record_run(self, mode: str, observed_per_pixel_sq: Dict[str, float]):
        with self._lock:
            modes = self._data.setdefault("modes", {})
            mode_data = modes.setdefault(mode, {"per_pixel_sq_seconds": {}, "runs": 0})
            ema = mode_data.setdefault("per_pixel_sq_seconds", {})
            for stage, observed in observed_per_pixel_sq.items():
                if not isinstance(observed, (int, float)) or observed <= 0:
                    continue
                previous = ema.get(stage)
                if not isinstance(previous, (int, float)) or previous <= 0:
                    ema[stage] = float(observed)
                else:
                    ema[stage] = self.alpha * float(observed) + (
                        1.0 - self.alpha
                    ) * float(previous)
            mode_data["runs"] = int(mode_data.get("runs", 0)) + 1

    def save(self):
        with self._lock:
            self._data["version"] = PROFILE_VERSION
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
