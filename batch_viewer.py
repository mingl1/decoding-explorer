import glob
import json
import sys

import pandas as pd
import streamlit as st

# Parse --output-dir from argv (streamlit passes args after --)
output_dir = "H:\\unprocessed\\aligned_corrected\\KC_out_aggressive2"
for i, arg in enumerate(sys.argv):
    if arg == "--output-dir" and i + 1 < len(sys.argv):
        output_dir = sys.argv[i + 1]
        break

st.set_page_config(page_title="Batch Ensemble Viewer", layout="wide")
st.title("Batch Ensemble Viewer")

# Load all stats
stats_files = sorted(glob.glob(f"{output_dir}/*/stats.json"))
all_stats = []
for path in stats_files:
    with open(path) as f:
        all_stats.append(json.load(f))

# Sidebar
with st.sidebar:
    st.caption(f"Output dir: `{output_dir}`")
    st.markdown("---")
    if not all_stats:
        st.warning("No runs found.")
        selected = None
    else:
        run_names = [s["name"] for s in all_stats]
        selected = st.radio("Select run", ["(overview)"] + run_names)

if not all_stats:
    st.info(f"No stats.json files found under `{output_dir}`.")
    st.stop()


def calc_invalid_pct(s):
    num_combos = len(s.get("protein_counts", {})) / 2
    if not num_combos or not s["valid"]:
        return 0.0
    return round(s["invalid"] / (s["valid"] / num_combos) * 100, 2)

# Overview tab
if selected == "(overview)":
    st.subheader("Overview")
    rows = [
        {
            "Run": s["name"],
            "Total beads": s["total_beads"],
            "Valid %": s["valid_pct"],
            "Invalid %": calc_invalid_pct(s),
            "Filtered %": s["filtered_pct"],
        }
        for s in all_stats
    ]
    st.dataframe(pd.DataFrame(rows).set_index("Run"), use_container_width=True)

# Run detail
else:
    stats = next(s for s in all_stats if s["name"] == selected)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total beads", stats["total_beads"])
    c2.metric("Valid %", f"{stats['valid_pct']}%")
    c3.metric("Invalid %", f"{calc_invalid_pct(stats)}%")
    c4.metric("Filtered %", f"{stats['filtered_pct']}%")

    st.markdown("---")

    if "margin_sweep" in stats:
        sweep_data = stats["margin_sweep"]
        selected = sweep_data["selected_margin_ratio"]
        st.subheader(f"Margin ratio sweep  *(selected: {selected}x)*")
        rows = [
            {
                "Ratio": f"{v['min_margin_ratio']}x" + (" ✓" if v["min_margin_ratio"] == selected else ""),
                "Valid %": v["valid_pct"],
                "Invalid %": v["invalid_pct"],
                "Filtered %": v["filtered_pct"],
                "Score (V-I)": v.get("score", round(v["valid_pct"] - v["invalid_pct"], 2)),
            }
            for v in sweep_data["sweep"].values()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Ratio"), use_container_width=True)
        st.markdown("---")

    if stats["protein_counts"]:
        st.subheader("Protein counts")
        protein_df = pd.DataFrame(
            list(stats["protein_counts"].items()), columns=["Protein", "Count"]
        ).set_index("Protein").sort_values("Count", ascending=False)
        st.bar_chart(protein_df)
        st.dataframe(protein_df, use_container_width=True)
    else:
        st.info("No protein counts available.")
