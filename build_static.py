"""
Run this locally to bundle data into data.json for static deployment.
- previous: parsed from all_data_and_stats.xlsx (V1 sheet)
- latest:   read from KC_out_aggressive stats.json files
Usage: python build_static.py
"""
import glob
import json

import pandas as pd

LATEST_DIR = "H:\\unprocessed\\aligned_corrected\\KC_out_aggressive"
EXCEL_PATH = "all_data_and_stats.xlsx"


def parse_excel_previous(xlsx_path):
    df = pd.read_excel(xlsx_path, sheet_name="V1", header=None)
    runs = []
    current = None

    for _, row in df.iterrows():
        col0, col1, col2, col3 = row[0], row[1], row[2], row[3]

        if pd.isna(col0):
            continue

        # Header row: col2 is a float < 1 (filtered fraction), col1 is a large bead count
        if isinstance(col2, float) and col2 < 1 and isinstance(col1, (int, float)) and col1 > 1000:
            if current:
                runs.append(current)
            filtered_pct = round(float(col2) * 100, 2)
            invalid_pct = round(float(col3) * 100, 4)
            current = {
                "name": str(col0),
                "total_beads": int(col1),
                "valid": None,          # raw count not in Excel
                "valid_pct": round(100 - filtered_pct, 1),
                "invalid": None,        # raw count not in Excel
                "invalid_pct": invalid_pct,
                "filtered": None,
                "filtered_pct": filtered_pct,
                "protein_counts": {},
            }
        elif current is not None and isinstance(col0, str) and not pd.isna(col1):
            # Protein pair row: (protein, count, protein, count)
            current["protein_counts"][str(col0)] = int(col1)
            if not pd.isna(col2) and not pd.isna(col3):
                current["protein_counts"][str(col2)] = int(col3)

    if current:
        runs.append(current)

    return runs


def load_latest(output_dir):
    files = sorted(glob.glob(f"{output_dir}/*/stats.json"))
    stats = []
    for path in files:
        with open(path) as f:
            stats.append(json.load(f))
    return stats


previous = parse_excel_previous(EXCEL_PATH)
latest = load_latest(LATEST_DIR)

output = {"previous": previous, "latest": latest}

with open("data.json", "w") as f:
    json.dump(output, f)

print(f"previous: {len(previous)} runs from Excel (V1 sheet)")
print(f"latest:   {len(latest)} runs from {LATEST_DIR}")
print("Written to data.json")
