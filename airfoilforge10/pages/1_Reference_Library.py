"""
Page 1 — Reference Library
Browse, search, filter, and download UIUC airfoils.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import io
import zipfile

from components.shared import fetch_uiuc_airfoils, get_airfoil_coords, generate_dat, export_to_svg, init_session_state
from components.plotting_engine import make_shape_figure

init_session_state()

st.title("🔍 Reference Library")
st.markdown("Browse the full UIUC airfoil database. Search, filter, preview shapes, and download `.dat` files.")

with st.spinner("Loading UIUC database…"):
    uiuc_airfoils = fetch_uiuc_airfoils()
    all_names = sorted(uiuc_airfoils.keys())

st.success(f"✅ {len(all_names):,} airfoils loaded from UIUC database.")

# ── Search & Filter ──────────────────────────────────────────────────────────
col_search, col_filter = st.columns([3, 1])
with col_search:
    query = st.text_input(
        "🔎 Filter list",
        placeholder="Type to narrow — e.g. NACA 0012, Clark, Eppler…",
        key="lib_query",
    )
with col_filter:
    sort_order = st.selectbox("Sort", ["A → Z", "Z → A"], key="lib_sort")

filtered = [n for n in all_names if query.lower() in n.lower()] if query else all_names
if sort_order == "Z → A":
    filtered = list(reversed(filtered))

st.caption(f"Showing **{len(filtered):,}** of {len(all_names):,} airfoils — pick from dropdown below.")

# ── Searchable full-list selectbox ───────────────────────────────────────────
# Streamlit's selectbox natively supports typing to search within the list.
selected = st.selectbox(
    "Select airfoil",
    options=[""] + filtered,
    format_func=lambda x: "— choose an airfoil —" if x == "" else x,
    key="lib_selected",
)

# ── Detail view ───────────────────────────────────────────────────────────────
if selected:
    st.markdown("---")
    col_shape, col_actions = st.columns([3, 1])

    url = uiuc_airfoils[selected]
    with st.spinner(f"Loading {selected}…"):
        coords = get_airfoil_coords(selected, url)

    with col_shape:
        if coords is not None:
            pitch = st.slider("Pitch / rotation (°)", -30.0, 30.0, 0.0, 1.0, key="lib_pitch")
            halo  = st.checkbox("Show halo offset", key="lib_halo")
            fig   = make_shape_figure([coords], [selected], halo=halo, pitch=pitch)
            st.plotly_chart(fig, use_container_width=True)

    with col_actions:
        st.markdown(f"### {selected}")
        if coords is not None:
            t_approx = np.max(coords[:, 1]) - np.min(coords[:, 1])
            st.markdown(f"**Points:** {len(coords)}")
            st.markdown(f"**Approx. thickness:** {t_approx:.3f}c")

            dat_data = generate_dat(coords, selected)
            st.download_button("⬇ Download .dat", dat_data, f"{selected}.dat", key="dl_dat_lib")

            svg_data = export_to_svg(coords, halo=halo, pitch=pitch)
            st.download_button("⬇ Download SVG", svg_data, f"{selected}.svg",
                               mime="image/svg+xml", key="dl_svg_lib")

            if st.button("⭐ Add to Favorites", key="lib_fav"):
                st.session_state.my_airfoils[selected] = coords
                st.toast(f"'{selected}' added to favorites!", icon="⭐")

            if st.button("📊 Analyze in Lab →", type="primary", key="lib_to_lab"):
                st.session_state["lab_preload"] = {"name": selected, "coords": coords}
                st.switch_page("pages/3_Analysis_Lab.py")
        else:
            st.error("Could not load airfoil coordinates.")

# ── Bulk download ─────────────────────────────────────────────────────────────
with st.expander("📦 Bulk Download"):
    bulk_sel = st.multiselect("Select multiple airfoils for ZIP download", all_names, key="bulk_sel")
    if bulk_sel and st.button("Download ZIP", key="bulk_zip"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name in bulk_sel:
                u = uiuc_airfoils.get(name)
                if u:
                    c = get_airfoil_coords(name, u)
                    if c is not None:
                        zf.writestr(f"{name}.dat", generate_dat(c, name))
        buf.seek(0)
        st.download_button("⬇ Download selected .dat files", buf, "airfoils.zip",
                           mime="application/zip", key="bulk_zip_dl")

# ── My Favorites ──────────────────────────────────────────────────────────────
if st.session_state.my_airfoils:
    with st.expander(f"⭐ My Favorites ({len(st.session_state.my_airfoils)})"):
        fav_names = list(st.session_state.my_airfoils.keys())
        fav_sel = st.selectbox("View favorite", fav_names, key="fav_sel_lib")
        if fav_sel:
            fc = st.session_state.my_airfoils[fav_sel]
            fig2 = make_shape_figure([fc], [fav_sel])
            st.plotly_chart(fig2, use_container_width=True)
            dat2 = generate_dat(fc, fav_sel)
            st.download_button("⬇ Download .dat", dat2, f"{fav_sel}.dat", key="fav_dl_lib")
            if st.button("🗑 Remove from Favorites", key="fav_rm_lib"):
                del st.session_state.my_airfoils[fav_sel]
                st.rerun()
