#!/usr/bin/env python3
import json
import pathlib
import shutil
import sys

import numpy as np

# ── Setup Paths ──────────────────────────────────────────────────────────────

# Get the path from command line argument
if len(sys.argv) < 2:
    print("Usage: python openvino_export.py <path_to_model_folder>")
    sys.exit(1)

# Resolving absolute paths prevents "SameFileError" logic issues
input_path = pathlib.Path(sys.argv[1]).resolve()
MODEL_NAME = input_path.name
BASE_DIR = input_path.parent

# Set output directory (e.g., assets/model_name)
# If you want it in a specific 'assets' folder, adjust here:
OUTPUT_DIR = input_path

# ── Dependency Checks ────────────────────────────────────────────────────────


def _require(pkg):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"ERROR: '{pkg}' not found.")
        sys.exit(1)


tf = _require("tensorflow")
ov = _require("openvino")
_ = _require("stardist")

from openvino import convert_model, save_model
from stardist.models import StarDist2D

# ── Load Model ───────────────────────────────────────────────────────────────

print(f"Loading custom model '{MODEL_NAME}' from {BASE_DIR}...")
model = StarDist2D(None, name=MODEL_NAME, basedir=str(BASE_DIR))

print(
    f"  Axes: {model.config.axes} | Rays: {model.config.n_rays} | Grid: {model.config.grid}"
)

# ── Convert to OpenVINO ──────────────────────────────────────────────────────

print("\nConverting to OpenVINO IR ...")
ov_model = convert_model(model.keras_model)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
xml_path = OUTPUT_DIR / f"{MODEL_NAME}.xml"
save_model(ov_model, str(xml_path))
print(f"  Saved IR: {xml_path}")

# ── Handle Metadata ──────────────────────────────────────────────────────────

# Avoid SameFileError: Only copy if source and dest are different
src_config = input_path / "config.json"
dst_config = OUTPUT_DIR / "config.json"

if src_config.exists() and src_config.resolve() != dst_config.resolve():
    shutil.copy(src_config, dst_config)
    print("  Copied config.json to output directory.")
else:
    print("  Note: config.json already exists in target directory, skipping copy.")

# Save thresholds (even if default) so the app doesn't have to guess
thresholds = {
    "prob": float(model.thresholds.prob) if model.thresholds else 0.5,
    "nms": float(model.thresholds.nms) if model.thresholds else 0.4,
}
with open(OUTPUT_DIR / "thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)
    print("  Saved thresholds.json")

# ── Verification ─────────────────────────────────────────────────────────────

probe = np.random.rand(1, 256, 256, 1).astype(np.float32)
tf_prob, tf_dist = model.keras_model.predict(probe, verbose=0)

core = ov.Core()
compiled = core.compile_model(xml_path, "CPU")
ov_result = compiled(probe)
ov_prob, ov_dist = np.asarray(ov_result[0]), np.asarray(ov_result[1])

diff = np.abs(tf_prob - ov_prob).max()
print(f"\nVerification: {'OK' if diff < 1e-4 else 'WARNING'}")
print(f"Max difference: {diff:.2e}")
