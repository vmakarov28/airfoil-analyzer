"""
Page 4 — Optimizer
Multi-objective airfoil shape optimizer using AeroSandbox + SciPy fallback.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np

from components.shared import (
    generate_dat, export_to_svg, analyze_airfoil, compute_key_metrics, init_session_state,
)
from components.plotting_engine import make_shape_figure, make_polar_figure

init_session_state()

PLOT_OPTIONS = ["CL vs CD", "CL vs Alpha", "CL/CD vs Alpha", "CD vs Alpha"]

st.title("🎯 Optimizer")
st.markdown("Optimize airfoil thickness distribution for maximum L/D, subject to geometric constraints.")


def generate_naca_symmetric(yt_params, n_points=100):
    x = np.linspace(0, 1, n_points)
    yt = (yt_params[0] * np.sqrt(x)
          + yt_params[1] * x
          + yt_params[2] * x**2
          + yt_params[3] * x**3
          + yt_params[4] * x**4)
    yt = np.maximum(yt, 0)
    upper = np.column_stack((x[::-1], yt[::-1]))
    lower = np.column_stack((x, -yt))
    coords = np.vstack((upper, lower))
    coords[0, 1] = 0
    coords[-1, 1] = 0
    return coords


# ── Parameters ───────────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)
with col_l:
    Re = st.slider("Reynolds Number", 50_000, 5_000_000, 1_000_000, 50_000, key="opt_re",
                   format="%d")
    target_alpha = st.slider("Target Angle of Attack (°)", 0.0, 20.0, 5.0, 0.5, key="opt_alpha")
    n_points = st.slider("Coordinate Points", 50, 200, 100, key="opt_npts")

with col_r:
    st.markdown("**Constraints**")
    min_thickness = st.slider("Min thickness/chord (%)", 1, 20, 6, key="opt_mint") / 100
    max_thickness = st.slider("Max thickness/chord (%)", 5, 40, 24, key="opt_maxt") / 100
    n_crit = st.slider("N-crit", 1.0, 15.0, 9.0, 0.5, key="opt_ncrit")

method = st.selectbox(
    "Optimization method",
    ["Nelder-Mead (fast, gradient-free)", "AeroSandbox Opti (gradient-based)"],
    key="opt_method",
)

if st.button("🚀 Optimize", type="primary", key="opt_run"):
    with st.spinner("Optimizing airfoil shape… this may take 20–60 seconds."):
        alphas = np.linspace(-5, target_alpha + 5, 41)
        x0 = np.array([0.2969, -0.1260, -0.3516, 0.2843, -0.1015])

        def objective(params):
            try:
                coords = generate_naca_symmetric(params, n_points)
                t_max = np.max(coords[:n_points // 2, 1]) * 2
                if t_max < min_thickness or t_max > max_thickness:
                    return 100.0
                aero = analyze_airfoil(coords, alphas, Re, "large", n_crit)
                if aero is None:
                    return 100.0
                ld = np.array(aero["CL/CD"])
                target_idx = np.argmin(np.abs(np.array(aero["alpha"]) - target_alpha))
                return -float(ld[target_idx])
            except:
                return 100.0

        best_coords = generate_naca_symmetric(x0, n_points)  # fallback
        best_score = objective(x0)

        try:
            if "AeroSandbox" in method:
                import aerosandbox as asb
                import aerosandbox.numpy as anp
                opti = asb.Opti()
                yt_params = opti.variable(init_guess=x0)
                x = anp.linspace(0, 1, n_points)
                yt = (yt_params[0] * anp.sqrt(x)
                      + yt_params[1] * x
                      + yt_params[2] * x**2
                      + yt_params[3] * x**3
                      + yt_params[4] * x**4)
                t_approx = anp.max(yt)
                cl_approx = 2 * anp.pi * anp.deg2rad(target_alpha) * (1 + 0.8 * t_approx)
                cd_approx = 0.005 / anp.sqrt(Re) + 0.01 * t_approx**2
                opti.maximize(cl_approx / cd_approx)
                opti.subject_to([t_approx >= min_thickness, t_approx <= max_thickness])
                sol = opti.solve(verbose=False)
                yt_opt = sol(yt)
                coords_opt = np.vstack([
                    np.column_stack((x[::-1], yt_opt[::-1])),
                    np.column_stack((x, -yt_opt)),
                ])
                coords_opt[0, 1] = 0
                coords_opt[-1, 1] = 0
                best_coords = coords_opt
            else:
                from scipy.optimize import minimize
                result = minimize(
                    objective, x0,
                    method="Nelder-Mead",
                    options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4},
                )
                if result.fun < best_score:
                    best_coords = generate_naca_symmetric(result.x, n_points)
        except Exception as e:
            st.warning(f"Optimizer encountered an issue ({e}); showing best found so far.")

        st.session_state["opt_result"] = best_coords

    st.success("✅ Optimization complete!")

if "opt_result" in st.session_state:
    coords_opt = st.session_state["opt_result"]
    st.subheader("Optimized Shape")

    t_max = np.max(coords_opt[:n_points // 2, 1]) * 2
    st.metric("Max Thickness/Chord", f"{t_max:.3f} ({t_max*100:.1f}%)")

    fig = make_shape_figure([coords_opt], ["Optimized Airfoil"])
    st.plotly_chart(fig, use_container_width=True)

    # Quick analysis of result
    with st.spinner("Running analysis on optimized airfoil…"):
        alphas_dense = np.arange(-10, 20, 0.5)
        aero_opt = analyze_airfoil(coords_opt, alphas_dense, Re, "large", n_crit)

    if aero_opt is not None:
        metrics = compute_key_metrics(aero_opt)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max CL", f"{metrics.get('Max CL', 0):.4f}")
        m2.metric("Stall Angle", f"{metrics.get('Stall Angle (°)', 0):.1f}°")
        m3.metric("Min CD", f"{metrics.get('Min CD', 0):.5f}")
        m4.metric("Max L/D", f"{metrics.get('Max L/D', 0):.2f}")

        fig_polar = make_polar_figure([aero_opt], ["CL vs Alpha", "CL/CD vs Alpha"],
                                     ["Optimized Airfoil"])
        st.plotly_chart(fig_polar, use_container_width=True)

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.download_button("⬇ .dat", generate_dat(coords_opt, "Optimized Airfoil"),
                           "optimized.dat", key="opt_dl_dat")
    col_d2.download_button("⬇ SVG", export_to_svg(coords_opt), "optimized.svg",
                           mime="image/svg+xml", key="opt_dl_svg")
    if col_d3.button("⭐ Save to Favorites", key="opt_fav"):
        st.session_state.my_airfoils["Optimized Airfoil"] = coords_opt
        st.toast("Saved 'Optimized Airfoil'", icon="⭐")
