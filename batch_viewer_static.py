import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Batch Ensemble Viewer", layout="wide")
st.title("Batch Ensemble Viewer")

data = json.loads(Path("data.json").read_text())


def calc_invalid_pct(s):
    if s.get("invalid_pct") is not None:
        return s.get("invalid_pct", 0.0)
    invalid = s.get("invalid")
    valid = s.get("valid")
    unique_cycle_combos = s.get("unique_cycle_combos")
    if (
        invalid is not None
        and valid not in (None, 0)
        and unique_cycle_combos not in (None, 0)
    ):
        return round((invalid * unique_cycle_combos / valid) * 100, 2)
    num_combos = len(s.get("protein_counts", {})) / 2
    if invalid is None or not num_combos or not valid:
        return 0.0
    return round(invalid / (valid / num_combos) * 100, 2)


# Sidebar
with st.sidebar:
    dataset_label = st.radio("Dataset", ["Previous", "Latest"])
    dataset_key = "latest" if dataset_label == "Latest" else "previous"
    all_stats = data[dataset_key]

    st.markdown("---")
    if not all_stats:
        st.warning("No runs found.")
        selected = None
    else:
        run_names = [s["name"] for s in all_stats]
        selected = st.radio("Select run", ["(overview)"] + run_names)

if not all_stats:
    st.info("No runs available for this dataset.")
    st.stop()

# Overview
if selected == "(overview)":
    st.subheader("Overview")

    if dataset_key == "latest":
        prev_lookup = {s["name"]: s for s in data.get("previous", [])}

        def fmt_valid(val, diff):
            if diff is None:
                return f"{val}%"
            arrow = "↑" if diff > 0 else "↓"
            sign = "+" if diff > 0 else ""
            return f"{val}%  {arrow} {sign}{diff}%"

        def fmt_invalid(val, diff):
            if diff is None:
                return f"{val}%"
            # for invalid, decrease is shown as improvement (↓ is good)
            arrow = "↓" if diff < 0 else "↑"
            sign = "+" if diff > 0 else ""
            return f"{val}%  {arrow} {sign}{diff}%"

        def color_valid(val):
            if "↑" in str(val):
                return "color: green"
            if "↓" in str(val):
                return "color: red"
            return ""

        def color_invalid(val):
            if "↓" in str(val):
                return "color: green"
            if "↑" in str(val):
                return "color: red"
            return ""

        def color_gain(val):
            if not isinstance(val, (int, float)):
                return ""
            return "color: green" if val > 0 else "color: red" if val < 0 else ""

        rows = []
        for s in all_stats:
            prev = prev_lookup.get(s["name"])
            valid_pct = s["valid_pct"]
            inv_pct = calc_invalid_pct(s)

            if prev:
                v_diff = round(valid_pct - prev["valid_pct"], 2)
                i_diff = round(inv_pct - calc_invalid_pct(prev), 2)
                total_gain = round(v_diff - i_diff, 2)
            else:
                v_diff = i_diff = total_gain = None

            rows.append(
                {
                    "Run": s["name"],
                    "Total beads": s["total_beads"],
                    "Valid %": fmt_valid(valid_pct, v_diff),
                    "Invalid %": fmt_invalid(inv_pct, i_diff),
                    "Filtered %": f"{s['filtered_pct']}%",
                    "Total Gain %": total_gain,
                }
            )

        df = pd.DataFrame(rows).set_index("Run")
        styled = (
            df.style.map(color_valid, subset=["Valid %"])
            .map(color_invalid, subset=["Invalid %"])
            .map(color_gain, subset=["Total Gain %"])
            .format(
                {
                    "Total Gain %": lambda v: (
                        f"+{v}%"
                        if isinstance(v, float) and v > 0
                        else (f"{v}%" if isinstance(v, float) else "—")
                    )
                }
            )
        )
        st.dataframe(styled, use_container_width=True)

    else:
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
        selected_ratio = sweep_data["selected_margin_ratio"]
        st.subheader(f"Margin ratio sweep  *(selected: {selected_ratio}x)*")
        rows = [
            {
                "Ratio": f"{v['min_margin_ratio']}x"
                + (" ✓" if v["min_margin_ratio"] == selected_ratio else ""),
                "Valid %": v["valid_pct"],
                "Invalid %": v["invalid_pct"],
                "Filtered %": v["filtered_pct"],
                "Score (V-I)": v.get(
                    "score", round(v["valid_pct"] - v["invalid_pct"], 2)
                ),
            }
            for v in sweep_data["sweep"].values()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Ratio"), use_container_width=True)
        st.markdown("---")

    if stats["protein_counts"]:
        st.subheader("Protein counts")
        protein_df = (
            pd.DataFrame(
                list(stats["protein_counts"].items()), columns=["Protein", "Count"]
            )
            .set_index("Protein")
            .sort_values("Count", ascending=False)
        )
        st.bar_chart(protein_df)
        st.dataframe(protein_df, use_container_width=True)
    else:
        st.info("No protein counts available.")
