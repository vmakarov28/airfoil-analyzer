"""
Page 5 — Wing Designer
Lifting-line wing performance estimates using airfoil polar from Analysis Lab.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d

from components.shared import (
    fetch_uiuc_airfoils, get_airfoil_coords, analyze_airfoil, init_session_state, format_re,
)
from components.plotting_engine import DARK_LAYOUT

init_session_state()

st.title("🛫 Wing Designer")
st.markdown("Estimate full-wing performance using lifting-line theory from your airfoil's polar.")

uiuc_airfoils = fetch_uiuc_airfoils()
all_names = sorted(uiuc_airfoils.keys())

# ── Polar source ──────────────────────────────────────────────────────────────
polar_source = st.radio(
    "Polar source",
    ["Use last Analysis Lab result", "Compute polar now"],
    key="wing_src", horizontal=True,
)

aero = None

if polar_source == "Use last Analysis Lab result":
    if "analysis_data" in st.session_state and "aero" in st.session_state.analysis_data:
        aero = st.session_state.analysis_data["aero"]
        st.success(f"Using polar: **{st.session_state.analysis_data.get('name', 'Analysis Result')}**")
    else:
        st.warning("No Analysis Lab result found. Run a single-airfoil analysis first, or switch to 'Compute polar now'.")
else:
    sel = st.selectbox("Select airfoil", [""] + all_names, key="wing_airfoil")
    re_w = st.number_input("Reynolds Number", 10_000, 10_000_000, 1_000_000, 10_000, key="wing_re")
    ncrit_w = st.number_input("N-crit", 1.0, 15.0, 9.0, 0.5, key="wing_ncrit")
    if sel and st.button("Compute Polar", key="wing_compute_polar"):
        with st.spinner("Running NeuralFoil…"):
            coords = get_airfoil_coords(sel, uiuc_airfoils[sel])
            if coords is not None:
                alphas = np.arange(-10, 20, 0.5)
                aero = analyze_airfoil(coords, alphas, re_w, "xlarge", ncrit_w)
                if aero:
                    st.session_state["analysis_data"] = {"aero": aero, "name": f"{sel} Re={format_re(re_w)}"}
                    st.success("Polar computed!")

    if "analysis_data" in st.session_state and "aero" in st.session_state.analysis_data:
        aero = st.session_state.analysis_data["aero"]

# ── Wing geometry ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Wing Geometry")
gc1, gc2, gc3 = st.columns(3)
AR   = gc1.number_input("Aspect Ratio (b²/S)", 1.0, 30.0, 8.0, 0.5, key="wing_AR")
span = gc2.number_input("Span (m)", 0.1, 60.0, 2.0, 0.1, key="wing_span")
e    = gc3.number_input("Oswald efficiency factor", 0.5, 1.0, 0.85, 0.01, key="wing_e")

gc4, gc5 = st.columns(2)
speed   = gc4.number_input("Flight Speed (m/s)", 1.0, 200.0, 20.0, 0.5, key="wing_speed")
density = gc5.number_input("Air Density (kg/m³)", 0.1, 1.4, 1.225, 0.001, key="wing_density",
                            help="Sea-level std: 1.225 kg/m³")

chord = span / AR
S = span * chord
st.caption(f"Derived mean chord: **{chord:.3f} m** | Wing area: **{S:.3f} m²**")

alpha_w = st.slider("Angle of Attack (°)", -10.0, 20.0, 5.0, 0.5, key="wing_alpha")

# ── Compute ───────────────────────────────────────────────────────────────────
if aero is not None:
    try:
        interp_cl = interp1d(aero["alpha"], aero["CL"], fill_value="extrapolate")
        interp_cd = interp1d(aero["alpha"], aero["CD"], fill_value="extrapolate")
        CL_2d = float(interp_cl(alpha_w))
        CD0   = float(interp_cd(alpha_w))
        CDi   = CL_2d**2 / (np.pi * AR * e)
        CD    = CD0 + CDi
        L     = 0.5 * density * speed**2 * S * CL_2d
        D     = 0.5 * density * speed**2 * S * CD
        LD    = L / D if D > 0 else float("inf")

        st.markdown("---")
        st.subheader("Performance Estimates")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("CL (2D airfoil)", f"{CL_2d:.4f}")
        m2.metric("CDi (induced)", f"{CDi:.5f}")
        m3.metric("Lift", f"{L:.1f} N", f"{L/9.81:.2f} kg")
        m4.metric("Drag", f"{D:.2f} N")
        m5.metric("L/D", f"{LD:.2f}")

        # Polar sweep plot: show L/D vs alpha across full range
        alphas_full = np.array(aero["alpha"])
        CL_arr = np.array(aero["CL"])
        CD0_arr = np.array(aero["CD"])
        CDi_arr = CL_arr**2 / (np.pi * AR * e)
        CD_arr  = CD0_arr + CDi_arr
        LD_arr  = np.where(CD_arr > 0, CL_arr / CD_arr, 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=alphas_full, y=LD_arr, mode="lines",
                                 name="Wing L/D", line=dict(color="#e74c3c", width=2.5)))
        fig.add_trace(go.Scatter(x=alphas_full, y=np.array(aero["CL/CD"]), mode="lines",
                                 name="Airfoil L/D (2D)", line=dict(color="#3498db", width=2, dash="dash")))
        fig.add_vline(x=alpha_w, line_dash="dot", line_color="#f39c12",
                      annotation_text=f"α={alpha_w}°", annotation_position="top")
        fig.update_layout(**DARK_LAYOUT, title="Wing vs Airfoil L/D",
                          xaxis_title="Alpha (°)", yaxis_title="L/D",
                          xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"), height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Induced drag distribution (elliptical assumption)
        y = np.linspace(-span/2, span/2, 200)
        Gamma_0 = (2 * L) / (density * speed * np.pi * span / 2)
        Gamma = Gamma_0 * np.sqrt(1 - (2*y/span)**2)
        di_dist = (density * speed * Gamma)**0 * (L / (np.pi * AR * e)) * np.ones_like(y)  # placeholder
        lift_dist = 0.5 * density * speed**2 * chord * CL_2d * np.sqrt(1 - (2*y/span)**2) * (4/np.pi)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y, y=lift_dist, mode="lines", fill="tozeroy",
                                  fillcolor="rgba(231,76,60,0.15)",
                                  name="Lift distribution", line=dict(color="#e74c3c", width=2)))
        fig2.update_layout(**DARK_LAYOUT, title="Spanwise Lift Distribution (Elliptic Approx.)",
                           xaxis_title="Spanwise position y (m)", yaxis_title="Lift per unit span (N/m)",
                           xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"), height=280)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as ex:
        st.error(f"Computation error: {ex}")
else:
    st.info("Select a polar source above and compute / load a polar to see wing performance.")
