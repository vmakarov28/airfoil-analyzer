"""
Page 3 — Analysis Lab
Multi-airfoil, multi-Re, multi-Ncrit batch analysis with Plotly dashboards.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from components.shared import (
    fetch_uiuc_airfoils, get_airfoil_coords, analyze_airfoil,
    compute_key_metrics, generate_csv, make_zip, format_re, init_session_state,
)
from components.plotting_engine import make_polar_figure, make_shape_figure

init_session_state()
uiuc_airfoils = fetch_uiuc_airfoils()
all_names = sorted(uiuc_airfoils.keys())

PLOT_OPTIONS = [
    "CL vs CD", "CL vs Alpha", "CL/CD vs Alpha",
    "CD vs Alpha", "Cm vs Alpha", "Confidence vs Alpha", "Cpmin vs Alpha",
]

st.title("📊 Analysis Lab")
st.markdown("Select airfoils, set conditions, and run batch NeuralFoil analysis — unlimited combinations.")

# Handle preload from Reference Library
if "lab_preload" in st.session_state:
    preload = st.session_state.pop("lab_preload")
    if preload["name"] not in st.session_state.my_airfoils:
        st.session_state.my_airfoils[preload["name"]] = preload["coords"]
    st.toast(f"'{preload['name']}' loaded into Lab", icon="📊")

# ── Airfoil selector ──────────────────────────────────────────────────────────
with st.expander("🛫 Select Airfoils", expanded=True):
    sel_uiuc = st.multiselect("UIUC Database", all_names, key="lab_uiuc")
    sel_fav  = st.multiselect("My Favorites",  list(st.session_state.my_airfoils.keys()), key="lab_fav")
    uploaded = st.file_uploader("Upload .dat files", type=["dat", "txt"],
                                accept_multiple_files=True, key="lab_upload")

    airfoil_map = {}
    for s in sel_uiuc:
        c = get_airfoil_coords(s, uiuc_airfoils[s])
        if c is not None:
            airfoil_map[s] = c
    for s in sel_fav:
        airfoil_map[s] = st.session_state.my_airfoils[s]
    for u in uploaded or []:
        c = get_airfoil_coords(file=u)
        if c is not None:
            airfoil_map[u.name] = c

    if airfoil_map:
        fig_sel = make_shape_figure(list(airfoil_map.values()), list(airfoil_map.keys()))
        st.plotly_chart(fig_sel, use_container_width=True)
        st.caption(f"{len(airfoil_map)} airfoil(s) selected.")
    else:
        st.info("No airfoils selected yet. Choose from UIUC, Favorites, or upload files.")

# ── Parameters ───────────────────────────────────────────────────────────────
with st.expander("⚙️ Analysis Parameters", expanded=True):
    p1, p2 = st.columns(2)
    with p1:
        re_input = st.text_input(
            "Reynolds Numbers (semicolon-separated)",
            value="500000; 1000000",
            help="e.g. 200000; 500000; 1000000",
            key="lab_re",
        )
        try:
            res = [float(p.strip().replace(",", "")) for p in re_input.split(";") if p.strip()]
            if not res:
                raise ValueError
        except:
            st.error("Invalid Re values")
            res = [1_000_000]

        ncrit_input = st.text_input(
            "N-crit Values (semicolon-separated)",
            value="9",
            help="e.g. 5; 9; 11",
            key="lab_ncrit",
        )
        try:
            ncrits = [float(p.strip()) for p in ncrit_input.split(";") if p.strip()]
            if not ncrits:
                raise ValueError
        except:
            st.error("Invalid Ncrit values")
            ncrits = [9.0]

    with p2:
        alpha_range = st.slider("Alpha Range (°)", -20.0, 30.0, (-15.0, 25.0), 0.5, key="lab_alpha")
        alpha_step  = st.select_slider("Alpha step (°)", options=[0.1, 0.25, 0.5, 1.0, 2.0], value=0.5, key="lab_step")
        model_size  = st.selectbox(
            "NeuralFoil model size",
            ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"],
            index=5, key="lab_model",
        )

    alphas = np.arange(alpha_range[0], alpha_range[1] + alpha_step / 2, alpha_step)

    plot_types = st.multiselect(
        "Plots to display",
        PLOT_OPTIONS,
        default=PLOT_OPTIONS[:4],
        key="lab_plots",
    )

# ── Run ───────────────────────────────────────────────────────────────────────
total_runs = len(airfoil_map) * len(res) * len(ncrits)
st.info(f"**{total_runs}** analysis run(s) will be computed.", icon="🔢")

if st.button("🚀 Run Analysis", type="primary", key="lab_run", disabled=not airfoil_map):
    aero_list, run_names = [], []
    progress = st.progress(0.0)
    status = st.empty()

    tasks = [
        (name, coords, re, ncrit)
        for name, coords in airfoil_map.items()
        for re in res
        for ncrit in ncrits
    ]

    for i, (name, coords, re, ncrit) in enumerate(tasks):
        label = f"{name} | Re={format_re(re)}, Ncrit={ncrit}"
        status.caption(f"⚙️ Computing: {label}…")
        aero = analyze_airfoil(coords, alphas, re, model_size, ncrit)
        if aero is not None:
            aero_list.append(aero)
            run_names.append(label)
        progress.progress((i + 1) / len(tasks))

    status.empty()
    progress.empty()

    if aero_list:
        st.session_state["lab_results"] = {"aeros": aero_list, "names": run_names}
        # Also store single for Wing Designer compatibility
        if len(aero_list) == 1:
            st.session_state["analysis_data"] = {"aero": aero_list[0], "name": run_names[0]}
        st.success(f"✅ {len(aero_list)} run(s) completed!")
    else:
        st.error("All analysis runs failed. Check coordinate validity.")

# ── Results ───────────────────────────────────────────────────────────────────
if "lab_results" in st.session_state and st.session_state["lab_results"]:
    res_data = st.session_state["lab_results"]
    aero_list = res_data["aeros"]
    run_names = res_data["names"]

    st.markdown("---")
    st.subheader("📈 Results")

    if plot_types:
        fig = make_polar_figure(aero_list, plot_types, run_names)
        st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    metrics = [compute_key_metrics(a) for a in aero_list]
    df = pd.DataFrame(metrics, index=run_names)
    st.subheader("📋 Key Metrics")
    st.dataframe(df.style.highlight_max(axis=0, color="#2a4030")
                         .highlight_min(axis=0, color="#4a1a1a"), use_container_width=True)

    # Downloads
    st.subheader("💾 Export")
    dl1, dl2 = st.columns(2)
    zip_buf = make_zip(run_names, aero_list)
    dl1.download_button("⬇ Download all CSVs (ZIP)", zip_buf, "analysis_results.zip",
                        mime="application/zip", key="lab_dl_zip")
    metrics_csv = df.to_csv().encode()
    dl2.download_button("⬇ Download metrics CSV", metrics_csv, "metrics.csv",
                        mime="text/csv", key="lab_dl_metrics")
