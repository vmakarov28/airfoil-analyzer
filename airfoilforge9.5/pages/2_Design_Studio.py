"""
Page 2 — Design Studio
NACA generator, modifier, and merge tool in a unified workspace.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np

from components.shared import (
    fetch_uiuc_airfoils, get_airfoil_coords,
    generate_naca_4digit, generate_naca_5digit, generate_naca_6series,
    modify_airfoil, merge_airfoils,
    generate_dat, export_to_svg,
    init_session_state,
)
from components.plotting_engine import make_shape_figure

init_session_state()
uiuc_airfoils = fetch_uiuc_airfoils()
all_names = sorted(uiuc_airfoils.keys())

st.title("⚗️ Design Studio")
st.markdown("Generate NACA airfoils, modify existing ones, or blend two designs together.")

tab_gen, tab_mod, tab_merge = st.tabs(["✨ NACA Generator", "✏️ Modifier", "🔀 Merge"])

# ─────────────────────────────────────────────────────────────────────────────
with tab_gen:
    sub_4, sub_5, sub_6 = st.tabs(["4-Digit", "5-Digit", "6-Series"])

    with sub_4:
        st.markdown("**NACA 4-Digit Series**")
        c1, c2, c3, c4 = st.columns(4)
        m = c1.slider("Max Camber (%)", 0, 9, 0, key="m4") / 100
        p = c2.slider("Camber Position (tenths)", 0, 9, 4, key="p4") / 10
        t = c3.slider("Thickness (%)", 1, 40, 12, key="t4") / 100
        n_pts = c4.slider("Points", 50, 300, 100, key="n4")
        name4 = f"NACA {int(m*100):01d}{int(p*10):01d}{int(t*100):02d}"
        st.caption(f"Designation: **{name4}**")
        coords4 = generate_naca_4digit(m, p, t, n_pts)
        fig4 = make_shape_figure([coords4], [name4])
        st.plotly_chart(fig4, use_container_width=True)
        _c1, _c2, _c3 = st.columns(3)
        _c1.download_button("⬇ .dat", generate_dat(coords4, name4), f"{name4}.dat", key="dl4")
        _c2.download_button("⬇ SVG", export_to_svg(coords4), f"{name4}.svg",
                            mime="image/svg+xml", key="svg4")
        if _c3.button("⭐ Save to Favorites", key="fav4"):
            st.session_state.my_airfoils[name4] = coords4
            st.toast(f"Saved '{name4}'", icon="⭐")

    with sub_5:
        st.markdown("**NACA 5-Digit Series**")
        c1, c2, c3, c4, c5 = st.columns(5)
        l   = c1.slider("Design CL Factor (1-5)", 1, 5, 2, key="l5")
        pd5 = c2.slider("Max Camber Pos (1-5)", 1, 5, 3, key="p5")
        q   = c3.radio("Type", [0, 1], format_func=lambda x: "Standard" if x == 0 else "Reflex", key="q5")
        t5  = c4.slider("Thickness (%)", 1, 40, 12, key="t5")
        n5  = c5.slider("Points", 50, 300, 100, key="n5")
        name5 = f"NACA {l}{pd5}{1 if q == 1 else 0}{t5:02d}"
        st.caption(f"Designation: **{name5}**")
        coords5 = generate_naca_5digit(l, pd5, q, t5, n5)
        fig5 = make_shape_figure([coords5], [name5])
        st.plotly_chart(fig5, use_container_width=True)
        _c1, _c2, _c3 = st.columns(3)
        _c1.download_button("⬇ .dat", generate_dat(coords5, name5), f"{name5}.dat", key="dl5")
        _c2.download_button("⬇ SVG", export_to_svg(coords5), f"{name5}.svg",
                            mime="image/svg+xml", key="svg5")
        if _c3.button("⭐ Save to Favorites", key="fav5"):
            st.session_state.my_airfoils[name5] = coords5
            st.toast(f"Saved '{name5}'", icon="⭐")

    with sub_6:
        st.markdown("**NACA 6-Series (Approximate)**")
        st.info("The 6-series camber line uses an approximate formulation. Verify critical designs with XFOIL.", icon="ℹ️")
        c1, c2, c3, c4, c5 = st.columns(5)
        ser = c1.selectbox("Series", [63, 64, 65], key="ser6")
        a6  = c2.slider("Mean Line a", 0.0, 1.0, 0.8, 0.1, key="a6")
        cl6 = c3.slider("Design CL", 0.0, 1.0, 0.0, 0.05, key="cl6")
        t6  = c4.slider("Thickness (%)", 1, 40, 12, key="t6") / 100
        n6  = c5.slider("Points", 50, 300, 100, key="n6")
        name6 = f"NACA {ser}-{int(cl6*10)}{int(t6*100):02d}"
        st.caption(f"Designation: **{name6}**")
        coords6 = generate_naca_6series(ser, a6, cl6, t6, n6)
        fig6 = make_shape_figure([coords6], [name6])
        st.plotly_chart(fig6, use_container_width=True)
        _c1, _c2, _c3 = st.columns(3)
        _c1.download_button("⬇ .dat", generate_dat(coords6, name6), f"{name6}.dat", key="dl6")
        _c2.download_button("⬇ SVG", export_to_svg(coords6), f"{name6}.svg",
                            mime="image/svg+xml", key="svg6")
        if _c3.button("⭐ Save to Favorites", key="fav6"):
            st.session_state.my_airfoils[name6] = coords6
            st.toast(f"Saved '{name6}'", icon="⭐")


# ─────────────────────────────────────────────────────────────────────────────
with tab_mod:
    st.markdown("Adjust the thickness and camber of any airfoil.")

    src_col, ctrl_col = st.columns([2, 1])
    with src_col:
        src = st.radio("Source", ["UIUC Database", "My Favorites", "Upload .dat"], key="mod_src", horizontal=True)
        if src == "UIUC Database":
            sel_mod = st.selectbox("Select airfoil", [""] + all_names, key="mod_uiuc")
            coords_mod = get_airfoil_coords(sel_mod, uiuc_airfoils.get(sel_mod)) if sel_mod else None
        elif src == "My Favorites":
            favs = list(st.session_state.my_airfoils.keys())
            sel_mod = st.selectbox("Select favorite", [""] + favs, key="mod_fav")
            coords_mod = st.session_state.my_airfoils.get(sel_mod) if sel_mod else None
        else:
            up = st.file_uploader("Upload .dat", type=["dat", "txt"], key="mod_upload")
            coords_mod = get_airfoil_coords(file=up) if up else None
            sel_mod = up.name if up else ""

    with ctrl_col:
        added_t = st.slider("Added Thickness", -0.1, 0.1, 0.0, 0.005, key="mod_t",
                            help="Fraction of chord added to thickness")
        added_c = st.slider("Added Camber", -0.1, 0.1, 0.0, 0.005, key="mod_c",
                            help="Fraction of chord added to camber")
        pitch_m = st.slider("Preview Pitch (°)", -30.0, 30.0, 0.0, 1.0, key="mod_pitch")

    if coords_mod is not None:
        coords_modified = modify_airfoil(coords_mod, added_t or None, added_c or None)
        name_mod = f"Modified {sel_mod}" if sel_mod else "Modified Airfoil"
        fig_mod = make_shape_figure([coords_mod, coords_modified],
                                    [sel_mod or "Original", name_mod], pitch=pitch_m)
        st.plotly_chart(fig_mod, use_container_width=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ .dat", generate_dat(coords_modified, name_mod), "modified.dat", key="dl_mod")
        d2.download_button("⬇ SVG", export_to_svg(coords_modified), "modified.svg",
                           mime="image/svg+xml", key="svg_mod")
        if d3.button("⭐ Save to Favorites", key="fav_mod"):
            st.session_state.my_airfoils[name_mod] = coords_modified
            st.toast(f"Saved '{name_mod}'", icon="⭐")
    else:
        st.info("Select or upload an airfoil to start modifying.")


# ─────────────────────────────────────────────────────────────────────────────
with tab_merge:
    st.markdown("Blend two airfoils together using a weighted average of their upper and lower surfaces.")

    def pick_airfoil(label, key_prefix):
        src = st.radio(f"{label} source", ["UIUC", "Favorites", "Upload"], key=f"{key_prefix}_src", horizontal=True)
        if src == "UIUC":
            sel = st.selectbox(f"{label}", [""] + all_names, key=f"{key_prefix}_uiuc")
            return get_airfoil_coords(sel, uiuc_airfoils.get(sel)) if sel else None, sel
        elif src == "Favorites":
            favs = list(st.session_state.my_airfoils.keys())
            sel = st.selectbox(f"{label}", [""] + favs, key=f"{key_prefix}_fav")
            return st.session_state.my_airfoils.get(sel) if sel else None, sel
        else:
            up = st.file_uploader(f"Upload {label}", type=["dat", "txt"], key=f"{key_prefix}_up")
            return (get_airfoil_coords(file=up) if up else None), (up.name if up else "")

    col_a, col_b = st.columns(2)
    with col_a:
        coords_a, name_a = pick_airfoil("First Airfoil", "merge_a")
        if coords_a is not None:
            st.plotly_chart(make_shape_figure([coords_a], [name_a or "A"]), use_container_width=True)
    with col_b:
        coords_b, name_b = pick_airfoil("Second Airfoil", "merge_b")
        if coords_b is not None:
            st.plotly_chart(make_shape_figure([coords_b], [name_b or "B"]), use_container_width=True)

    if coords_a is not None and coords_b is not None:
        ratio = st.slider("Blend Ratio (0 = A, 1 = B)", 0.0, 1.0, 0.5, 0.01, key="merge_ratio")
        coords_merged = merge_airfoils(coords_a, coords_b, ratio)
        name_merged = f"Merge {int((1-ratio)*100)}% {name_a or 'A'} + {int(ratio*100)}% {name_b or 'B'}"
        st.plotly_chart(
            make_shape_figure([coords_a, coords_b, coords_merged],
                              [name_a or "A", name_b or "B", "Merged"]),
            use_container_width=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.download_button("⬇ .dat", generate_dat(coords_merged, name_merged), "merged.dat", key="dl_merge")
        m2.download_button("⬇ SVG", export_to_svg(coords_merged), "merged.svg",
                           mime="image/svg+xml", key="svg_merge")
        if m3.button("⭐ Save to Favorites", key="fav_merge"):
            st.session_state.my_airfoils[name_merged] = coords_merged
            st.toast(f"Saved '{name_merged}'", icon="⭐")
    else:
        st.info("Select two airfoils to enable blending.")
