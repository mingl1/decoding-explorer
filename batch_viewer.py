import glob
import json
import sys

import pandas as pd
import streamlit as st

# Parse --output-dir from argv (streamlit passes args after --)
output_dir = "./batch_results"
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

# Overview tab
if selected == "(overview)":
    st.subheader("Overview")
    rows = [
        {
            "Run": s["name"],
            "Total beads": s["total_beads"],
            "Valid %": s["valid_pct"],
            "Invalid %": s["invalid_pct"],
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
    c3.metric("Invalid %", f"{stats['invalid_pct']}%")
    c4.metric("Filtered %", f"{stats['filtered_pct']}%")

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
