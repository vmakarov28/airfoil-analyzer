# airfoil_env\Scripts\activate
# cd C:\Users\aipla\Documents\Personal\Projects\app
# streamlit run app.py

# pip install gradio neuralfoil matplotlib pandas numpy
# app.py - Gradio app for Airfoil Database Analyzer

# Airfoil Analyzer Pro Streamlit App - Final Reworked Version
# Requirements: streamlit neuralfoil aerosandbox matplotlib numpy pandas requests beautifulsoup4 pillow svgwrite scipy streamlit-extras

import streamlit as st
import neuralfoil as nf
import aerosandbox as asb
import aerosandbox.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests
from bs4 import BeautifulSoup
import pandas as pd
import zipfile
import svgwrite
from scipy.interpolate import interp1d
from streamlit_extras.stylable_container import stylable_container
import re
from st_copy import copy_button 
from streamlit_option_menu import option_menu



# Global CSS for refinement
st.markdown("""
<style>
.row-widget.stHorizontal {
    gap: 0px !important;  /* Custom gap in pixels; adjust as needed (e.g., 4px for even closer) */
}
@media (max-width: 768px) {
    .row-widget.stHorizontal {
        gap: 0px !important;  /* Smaller on mobile */
    }
}
.card {
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease, rotateY 0.3s ease, rotateX 0.3s ease;
    text-align: center;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    cursor: pointer;
    margin: 10px;
}
.card:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.card:nth-child(odd):hover {
    transform: scale(1.05) rotateY(3deg);
}
.card:nth-child(even):hover {
    transform: scale(1.05) rotateX(3deg);
}
.stButton > button {
    width: auto;
    border: none;
    background-color: #444444 !important;  /* Default gray background for visibility */
    color: #ffffff !important;
}
.stButton > button[title="Compute the Reynolds number"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.stButton > button[title="Select this kinematic viscosity"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.section-spacer {
    margin-top: 20px;
}
.stRadio > div {
    background-color: inherit;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 10px;
}
.stRadio label {
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Airfoil Analyzer Pro", layout="wide", initial_sidebar_state="expanded")

# Define slugs and names
page_slugs = [
    "home", "airfoil-tools", "airfoil-generator",
    "airfoil-modifier", "merge-airfoils", "optimize-shape",
    "wing-estimator", "tutorials", "about"
]
page_names = [
    "Home", "Airfoil Tools", "Airfoil Generator",
    "Airfoil Modifier", "Merge Airfoils", "Optimize Shape",
    "Wing Estimator", "Tutorials", "About"
]
page_map = dict(zip(page_slugs, page_names))
reverse_map = dict(zip(page_names, page_slugs))

def format_re(re):
    if re < 1e3:
        return f"{int(re)}"
    elif re < 1e6:
        return f"{int(re / 1e3)}k"
    elif re < 1e9:
        return f"{int(re / 1e6)}M"
    else:
        return f"{re:.0e}"  # Fallback for extremes

@st.cache_data(ttl=3600)
def fetch_uiuc_airfoils():
    url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        airfoils = {}
        for link in links:
            if link['href'].endswith('.dat'):
                name = link.text.strip()
                dat_url = f"https://m-selig.ae.illinois.edu/ads/{link['href']}"
                airfoils[name] = dat_url
        return airfoils
    except Exception as e:
        st.error(f"Failed to fetch UIUC database: {str(e)}. Using fallback mode.")
        return {}

def parse_dat_file(content):
    try:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty file")
        # Assume header is first line
        name = lines[0]
        lines = lines[1:]
        # Detect format
        parts = lines[0].split()
        if len(parts) == 2:
            try:
                nu, nl = map(float, parts)
                if nu.is_integer() and nl.is_integer() and int(nu) >= 10 and int(nl) >= 10:
                    nu, nl = int(nu), int(nl)
                    # Parse upper, skipping non-numeric
                    upper = []
                    line_idx = 1
                    while len(upper) < nu and line_idx < len(lines):
                        try:
                            x, y = map(float, lines[line_idx].split()[:2])
                            upper.append([x, y])
                        except:
                            pass
                        line_idx += 1
                    # Parse lower
                    lower = []
                    while len(lower) < nl and line_idx < len(lines):
                        try:
                            x, y = map(float, lines[line_idx].split()[:2])
                            lower.append([x, y])
                        except:
                            pass
                        line_idx += 1
                    if len(upper) != nu or len(lower) != nl:
                        raise ValueError("Insufficient valid points in file")
                    upper = np.array(upper)
                    lower = np.array(lower)
                    coords = np.vstack((upper[::-1], lower[1:]))
                else:
                    # Fall back to SELIG
                    raise ValueError("Not a valid LEDNICER; treating as SELIG")
            except:
                # SELIG format
                coords = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = map(float, parts[:2])
                            coords.append([x, y])
                        except:
                            pass
                coords = np.array(coords)
        else:
            # SELIG format
            coords = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x, y = map(float, parts[:2])
                        coords.append([x, y])
                    except:
                        pass
            coords = np.array(coords)
        # Ensure closed
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack((coords, coords[0]))
        # Remove internal duplicates (preserve start/end)
        internal_coords = coords[1:-1]
        unique_internal_idx = np.unique(internal_coords, axis=0, return_index=True)[1]
        internal_coords = internal_coords[np.sort(unique_internal_idx)]
        coords = np.vstack((coords[0:1], internal_coords, coords[-1:]))
        # Normalize x to [0,1] if out of bounds
        if np.any(coords[:,0] < 0) or np.any(coords[:,0] > 1):
            st.warning("Coordinates x not in [0,1]; normalizing.")
            min_x = np.min(coords[:,0])
            max_x = np.max(coords[:,0])
            if max_x > min_x:
                coords[:,0] = (coords[:,0] - min_x) / (max_x - min_x)
        return coords
    except Exception as e:
        st.error(f"Invalid .dat file: {str(e)}")
        return None

def get_airfoil_coords(name=None, url=None, file=None):
    if file:
        try:
            content = file.getvalue().decode('utf-8', errors='ignore')
            return parse_dat_file(content)
        except:
            st.error("File decoding failed")
            return None
    elif name and url:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return parse_dat_file(response.text)
        except Exception as e:
            st.error(f"Failed to fetch {name}: {str(e)}")
            return None
    return None

def analyze_airfoil(coords, alphas, Re, model_size='xlarge', n_crit=9):
    try:
        if coords.shape[0] < 20 or coords.shape[1] != 2 or not np.all((0 <= coords[:,0]) & (coords[:,0] <= 1)):
            raise ValueError("Invalid coordinates: must be at least 20 points, Nx2, x in [0,1]")
        if coords[1,1] < 0:
            coords = coords[::-1]
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alphas,
            Re=Re,
            n_crit=n_crit,
            model_size=model_size
        )
        aero['alpha'] = np.array(alphas)
        aero['CL/CD'] = np.array(aero['CL']) / np.maximum(np.array(aero['CD']), 1e-6)
        return aero
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def create_plots(aero_data, plot_types, is_compare=False, names=None):
    if not aero_data:
        return None
    num_plots = len(plot_types)
    rows = (num_plots + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axs = axs.flatten()
    plot_map = {
        'CL vs CD': (('CD', 'CL'), 'Polar'),
        'CL vs Alpha': (('alpha', 'CL'), 'Lift Curve'),
        'CL/CD vs Alpha': (('alpha', 'CL/CD'), 'Efficiency'),
        'CD vs Alpha': (('alpha', 'CD'), 'Drag Curve'),
        'Cm vs Alpha': (('alpha', 'CM'), 'Moment Curve'),
        'Confidence vs Alpha': (('alpha', 'analysis_confidence'), 'Confidence'),
        'Cpmin vs Alpha': (('alpha', 'Cpmin'), 'Min Pressure')
    }
    colors = plt.cm.tab10(np.linspace(0, 1, len(aero_data) if is_compare else 1))
    i = 0
    for plot_type in plot_types:
        plot_entry = plot_map.get(plot_type, None)
        if plot_entry is None:
            continue
        (x_key, y_key), title = plot_entry  # Proper unpacking
        ax = axs[i]
        if is_compare:
            for j, (name, aero) in enumerate(zip(names, aero_data)):
                if x_key in aero and y_key in aero:
                    ax.plot(aero[x_key], aero[y_key], label=name, color=colors[j])
            ax.legend()
        else:
            aero = aero_data[0]
            if x_key in aero and y_key in aero:
                ax.plot(aero[x_key], aero[y_key], color='blue')
        ax.set_xlabel(x_key.capitalize())
        ax.set_ylabel(y_key)
        ax.set_title(title)
        ax.grid(True)
        i += 1
    for j in range(i, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def plot_airfoil_shape(coords_list, names=None, halo=False, pitch=0.0):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords_list)))
    for i, coords in enumerate(coords_list):
        theta = np.deg2rad(pitch)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords_rot = coords @ rot.T
        label = names[i] if names else "Airfoil"
        ax.plot(coords_rot[:, 0], coords_rot[:, 1], label=label, color=colors[i])
        if halo:
            offset = 0.01 * (np.max(coords_rot[:,1]) - np.min(coords_rot[:,1]))
            dirs = np.roll(coords_rot, -1, axis=0) - coords_rot
            norms_len = np.linalg.norm(dirs, axis=1)
            norms_len[norms_len == 0] = 1
            norms = np.column_stack((-dirs[:,1], dirs[:,0])) / norms_len[:, np.newaxis]
            halo_coords = coords_rot + offset * norms
            ax.plot(halo_coords[:, 0], halo_coords[:, 1], '--', color=colors[i])
    ax.set_aspect('equal')
    ax.set_title("Airfoil Shape")
    if names and len(names) > 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)  # Increased y for higher position
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Tight bbox to include external legend
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

def export_to_svg(coords, halo=False, pitch=0.0, scale=1000):
    y_range = np.max(coords[:,1]) - np.min(coords[:,1])
    height = max(200, scale * y_range * 1.2)
    dwg = svgwrite.Drawing(size=(scale, height))
    theta = np.deg2rad(pitch)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ rot.T
    min_y = np.min(coords_rot[:,1])
    coords_rot[:,1] -= min_y
    coords_rot *= scale
    points = [(float(x), float(y)) for x, y in coords_rot]
    dwg.add(dwg.polygon(points, stroke='black', fill='none'))
    if halo:
        offset = 0.01 * scale
        dirs = np.roll(coords_rot, -1, axis=0) - coords_rot
        norms_len = np.linalg.norm(dirs, axis=1)
        norms_len[norms_len == 0] = 1
        norms = np.column_stack((-dirs[:,1], dirs[:,0])) / norms_len[:, np.newaxis]
        halo_coords = coords_rot + offset * norms
        hpoints = [(float(x), float(y)) for x, y in halo_coords]
        dwg.add(dwg.polygon(hpoints, stroke='black', fill='none', stroke_dasharray='5,5'))
    buf = io.StringIO()  # Changed to StringIO for text
    dwg.write(buf)
    buf.seek(0)
    return buf.getvalue().encode('utf-8')  # Encode to bytes for download

def compute_key_metrics(aero):
    if aero is None or 'CL' not in aero or len(aero['CL']) == 0:
        return {}
    aero = {k: np.nan_to_num(np.array(v)) for k, v in aero.items()}
    max_cl_idx = np.argmax(aero['CL'])
    min_cd = np.min(aero['CD'])
    if min_cd == 0:
        min_cd = 1e-6
    return {
        'Max CL': aero['CL'][max_cl_idx],
        'Stall Angle': aero['alpha'][max_cl_idx],
        'Min CD': min_cd,
        'Max L/D': np.max(aero['CL/CD']),
        'Avg Confidence': np.mean(aero['analysis_confidence'])
    }

def generate_csv(aero):
    if aero is None:
        return None
    aero = {k: np.nan_to_num(np.array(v)) for k, v in aero.items()}
    df = pd.DataFrame(aero)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

def generate_dat(coords, name, fmt='SELIG'):
    buf = io.StringIO()
    buf.write(f"{name}\n")
    if fmt == 'LEDNICER':
        le_idx = np.argmin(coords[:,0])
        upper = coords[:le_idx+1][::-1]
        lower = coords[le_idx:]
        buf.write(f"{len(upper)} {len(lower)}\n")
        np.savetxt(buf, upper, fmt='%.6f')
        buf.write("\n")
        np.savetxt(buf, lower, fmt='%.6f')
    else:
        np.savetxt(buf, coords, fmt='%.6f')
    buf.seek(0)
    return buf.getvalue()

def generate_naca_4digit(m, p, t, n_points=100):
    if p <= 0 or p >= 1:
        if m > 0:
            st.warning("Invalid: m>0 with p=0 or 1; forcing symmetric airfoil.")
        m = 0
        p = 0.4
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    if m == 0:
        pass
    else:
        idx = x < p
        yc[idx] = m / p**2 * (2 * p * x[idx] - x[idx]**2)
        yc[~idx] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[~idx] - x[~idx]**2)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    # Force sharp TE if approximate blunt
    coords[0,1] = 0
    coords[-1,1] = 0
    return coords

def generate_naca_5digit(l, p_digit, q, t, n_points=100):
    t = 0.01 * t
    cl_design = (3/2) * (l / 10.0)
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    if q == 0:  # Standard, vectorized
        xcm = 0.05 * p_digit
        r_old = 0.1
        diff, tol = 1, 1e-6
        while diff > tol:
            r = xcm + r_old * np.sqrt(r_old / 3)
            diff = abs(r - r_old)
            r_old = r
        qm = (3*r -7*r**2 +8*r**3 -4*r**4) / np.sqrt(r*(1-r)) - 1.5*(1-2*r)*(np.pi/2 - np.arcsin(1-2*r))
        k1 = 6 * cl_design / qm
        idx = x < r
        yc[idx] = (k1 / 6) * (x[idx]**3 - 3*r*x[idx]**2 + r**2*(3-r)*x[idx])
        dyc[idx] = (k1 / 6) * (3*x[idx]**2 - 6*r*x[idx] + r**2*(3-r))
        yc[~idx] = (k1 / 6) * r**3 * (1 - x[~idx])
        dyc[~idx] = - (k1 / 6) * r**3
    else:  # Reflex, hardcoded to 231
        st.warning("Reflex uses 231 series example; customize for others.")
        r = 0.2170
        k1 = 15.793
        k21 = -0.1010
        idx = x < r
        yc[idx] = (k1 / 6) * ( (x[idx] - r)**3 - k21*(1-r)**3*x[idx] - r**3*x[idx] + r**3 )
        dyc[idx] = (k1 / 6) * ( 3*(x[idx] - r)**2 - k21*(1-r)**3 - r**3 )
        yc[~idx] = (k1 / 6) * ( k21*(x[~idx] - r)**3 - k21*(1-r)**3*x[~idx] - r**3*x[~idx] + r**3 )
        dyc[~idx] = (k1 / 6) * ( 3*k21*(x[~idx] - r)**2 - k21*(1-r)**3 - r**3 )
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    coords[0,1] = 0
    coords[-1,1] = 0
    return coords

def generate_naca_6series(series, a, cl_design, t, n_points=100):
    if cl_design != 0:
        st.warning("Cambered 6-series approximate; for accurate, use advanced tools.")
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if cl_design != 0:
        m = cl_design / (2 * np.pi * (1 + a))
        idx = x <= a
        yc[idx] = m * (x[idx] / a**2) * (2 * a - x[idx])
        dyc_dx[idx] = m * (2 * a - 2 * x[idx]) / a**2
        yc[~idx] = m * ((1 - x[~idx]) / (1 - a)**2) * (1 + x[~idx] - 2 * a)
        dyc_dx[~idx] = m * ((1 - x[~idx]) * (1 - 2 * a) - (1 + x[~idx] - 2 * a)) / (1 - a)**2
    if series == 63:
        a4 = -0.1015
    elif series == 64 or series == 65:
        a4 = -0.1036
    else:
        a4 = -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    coords[0,1] = 0
    coords[-1,1] = 0
    return coords

def modify_airfoil(coords, added_t=None, added_c=None):
    le_idx = np.argmin(coords[:,0])
    upper = coords[:le_idx+1][::-1]
    lower = coords[le_idx:]
    # Resample to common x-grid to ensure same length
    x_common = np.linspace(0, 1, max(len(upper), len(lower)))
    interp_yu = interp1d(upper[:,0], upper[:,1], kind='linear', fill_value="extrapolate")
    interp_yl = interp1d(lower[:,0], lower[:,1], kind='linear', fill_value="extrapolate")
    y_u = interp_yu(x_common)
    y_l = interp_yl(x_common)
    if added_t is not None:
        curr_t = np.max(y_u - y_l)
        if curr_t > 0:
            new_t = curr_t + added_t
            if new_t <= 0:
                st.warning("Added thickness results in non-positive value; skipping thickness change.")
            else:
                scale = new_t / curr_t
                y_u *= scale
                y_l *= scale
        else:
            st.warning("Current thickness <=0; skipping thickness change.")
    if added_c is not None:
        yc_add = added_c * (1 - x_common)**2 * x_common
        y_u += yc_add
        y_l += yc_add
    upper = np.column_stack((x_common, y_u))
    lower = np.column_stack((x_common, y_l))
    coords = np.vstack((upper[::-1], lower[1:]))
    coords[:,0] = coords[:,0] / np.max(coords[:,0]) if np.max(coords[:,0]) > 0 else coords[:,0]
    return coords

def merge_airfoils(coords1, coords2, ratio=0.5):
    le1 = np.argmin(coords1[:,0])
    upper1 = coords1[:le1+1][::-1]
    lower1 = coords1[le1:]
    le2 = np.argmin(coords2[:,0])
    upper2 = coords2[:le2+1][::-1]
    lower2 = coords2[le2:]
    x_common = np.linspace(0, 1, max(len(upper1), len(upper2)))
    interp_u1 = interp1d(upper1[:,0], upper1[:,1], kind='linear', fill_value="extrapolate")
    interp_l1 = interp1d(lower1[:,0], lower1[:,1], kind='linear', fill_value="extrapolate")
    interp_u2 = interp1d(upper2[:,0], upper2[:,1], kind='linear', fill_value="extrapolate")
    interp_l2 = interp1d(lower2[:,0], lower2[:,1], kind='linear', fill_value="extrapolate")
    y_u = (1 - ratio) * interp_u1(x_common) + ratio * interp_u2(x_common)
    y_l = (1 - ratio) * interp_l1(x_common) + ratio * interp_l2(x_common)
    upper = np.column_stack((x_common, y_u))
    lower = np.column_stack((x_common, y_l))
    coords = np.vstack((upper[::-1], lower))  # Fixed: full lower, LE duplicate ok for sharp
    return coords

def optimize_shape(Re, target_alpha, n_points=100):
    opti = asb.Opti()
    yt_params = opti.variable(init_guess=np.array([0.2969, -0.1260, -0.3516, 0.2843, -0.1015]))
    x = anp.linspace(0, 1, n_points)
    yt = yt_params[0]*anp.sqrt(x) + yt_params[1]*x + yt_params[2]*x**2 + yt_params[3]*x**3 + yt_params[4]*x**4
    t = anp.max(yt)
    cl = 2 * anp.pi * anp.deg2rad(target_alpha) * (1 + 0.8 * t)
    cd = 0.005 / anp.sqrt(Re) + 0.01 * t**2
    opti.maximize(cl / cd)
    try:
        sol = opti.solve(verbose=False)
        yt_opt = sol(yt)
    except:
        st.error("Optimization failed; using NACA 0012.")
        return generate_naca_4digit(0, 0.4, 0.12, n_points)
    coords = np.vstack([np.column_stack((x[::-1], yt_opt[::-1])), np.column_stack((x, -yt_opt))])
    return coords


def reynolds_calculator(page_key=""):
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Reynolds number calculator")
        velocity = st.number_input("Velocity (m/s)", min_value=0.0, value=10.0, help="m/s", key=f"velocity_{page_key}")  # 32.369 mph equivalent
        chord = st.number_input("Chord width (meters)", min_value=0.0, value=0.2, help="m", key=f"chord_{page_key}")  # 0.65617 ft equivalent
        
        # Consolidated viscosity input with two-way sync
        default_visc = 1.2462e-5
        if 'kin_visc' not in st.session_state:
            st.session_state['kin_visc'] = default_visc
        
        # Sync shared to widget state before rendering (forces update on button press)
        widget_key = f"kin_visc_{page_key}"
        st.session_state[widget_key] = st.session_state['kin_visc']
        
        kin_visc = st.number_input("Kinematic Viscosity (m²/s)", min_value=0.0, help="m²/s", key=widget_key, format="%e")
        
        # Sync widget back to shared (updates if user manually edits)
        st.session_state['kin_visc'] = kin_visc
        
        # Auto-compute and display Reynolds number in minimalist boxed field
        if kin_visc > 0:
            Re = velocity * chord / kin_visc
            re_text = f"{Re:.0f}"
            with stylable_container(
                key=f"re_box_container_{page_key}",
                css_styles="""
                    div {
                        padding: 3px 5px;  /* Reduced vertical padding for less height */
                        border-radius: 10px;
                        background-color: #2c2c2c;  /* Subtler dark gray */
                        text-align: center;
                        color: #ffffff;
                        display: flex;  /* Flex for inline alignment */
                        align-items: center;
                        justify-content: center;
                    }
                    /* Ensure button aligns inline */
                    button {
                        margin-left: 10px;
                        font-size: 1.2em;
                        background-color: #333333;  /* Default button bg */
                        color: #ffffff;
                        border: none;
                        border-radius: 4px;
                        padding: 2px 6px;
                        cursor: pointer;
                        transition: background-color 0.3s ease, transform 0.3s ease;
                    }
                    button:hover {
                        background-color: #444444;
                        transform: scale(1.05);
                    }
                    /* Style for Copied! state as green bubble */
                    button[kind="primary"] {  /* Target after copy (st-copy sets kind) */
                        background-color: #4CAF50 !important;  /* Modern green */
                        border-radius: 20px !important;  /* High radius for bubble/pill shape */
                        padding: 4px 12px !important;  /* Increased padding for bubble effect */
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);  /* Subtle shadow for depth */
                        color: #ffffff !important;
                        font-size: 0.9em;  /* Slightly smaller for clean fit */
                    }
                """
            ):
                col_re_text, col_re_copy = st.columns([4, 1])
                with col_re_text:
                    st.markdown(f"<strong>Reynolds Number: {re_text}</strong>", unsafe_allow_html=True)
                with col_re_copy:
                    copy_button(
                        re_text,
                        tooltip="Copy to clipboard",
                        copied_label="Copied!",
                        icon="📋",  # Clipboard emoji for icon
                        key=f"copy_re_{page_key}"
                    )
        else:
            st.error("Invalid inputs—viscosity must be positive.")
        
        st.markdown(r"""
        
        """)
        st.markdown(r"""
        
                    
        The Reynolds number is a dimensionless value that measures the ratio of inertial forces to viscous forces and describes the degree of laminar or turbulent flow.
        Systems that operate at the same Reynolds number will have the same flow characteristics even if the fluid, speed and characteristic lengths vary.
        
        It is calculated with:
        Re = ρvl / μ = vl / ν
        
                    
        Where:    
        v = Velocity of the fluid  
        l = The characteristic length, the chord width of an airfoil  
        ρ = The density of the fluid  
        μ = The dynamic viscosity of the fluid  
        ν = The kinematic viscosity of the fluid
                    
        """)
    with right_col:
        st.subheader("Kinematic Viscosity")
        st.markdown("Example kinematic viscosity values for air and water at 1 atm and various temperatures")
        # Air table
        st.markdown("Air----------------------------------------------------------°C-----------------------------------°F")
        air_values = [
            {"Kinematic m²/s": "1.2462E-5", "°C": "-10", "°F": "14"},
            {"Kinematic m²/s": "1.3324E-5", "°C": "0", "°F": "32"},
            {"Kinematic m²/s": "1.4207E-5", "°C": "10", "°F": "50"},
            {"Kinematic m²/s": "1.5111E-5", "°C": "20", "°F": "68"}
        ]
        for i, val in enumerate(air_values):
            col_a1, col_a2, col_a3, col_a4 = st.columns([2, 1, 1, 1])
            col_a1.write(val["Kinematic m²/s"])
            col_a2.write(val["°C"])
            col_a3.write(val["°F"])
            with stylable_container(
                key=f"air_use_container_{page_key}_{i}",
                css_styles="""
                    button {
                        background-color: #27ae60 !important;  /* Nephritis Green */
                        color: #ffffff !important;
                        box-shadow: 0 2px 4px rgba(39,174,96,0.3);
                        transition: background-color 0.3s ease, transform 0.3s ease;
                    }
                    button:hover {
                        background-color: #219954 !important;
                        transform: scale(1.05);
                    }
                """
            ):
                if col_a4.button("Use", key=f"air_use_{page_key}_{i}", help="Select this kinematic viscosity"):
                    st.session_state.kin_visc = float(val["Kinematic m²/s"])  # Update shared
                    st.rerun()
        # Water table
        st.markdown("Water---------------------------------------------------°C-----------------------------------°F")
        water_values = [
            {"Kinematic m²/s": "1.6438E-6", "°C": "1", "°F": "33.8"},
            {"Kinematic m²/s": "1.2676E-6", "°C": "10", "°F": "50"},
            {"Kinematic m²/s": "9.7937E-7", "°C": "20", "°F": "68"}
        ]
        for i, val in enumerate(water_values):
            col_w1, col_w2, col_w3, col_w4 = st.columns([2, 1, 1, 1])
            col_w1.write(val["Kinematic m²/s"])
            col_w2.write(val["°C"])
            col_w3.write(val["°F"])
            with stylable_container(
                key=f"water_use_container_{page_key}_{i}",
                css_styles="""
                    button {
                        background-color: #27ae60 !important;  /* Nephritis Green */
                        color: #ffffff !important;
                        box-shadow: 0 2px 4px rgba(39,174,96,0.3);
                        transition: background-color 0.3s ease, transform 0.3s ease;
                    }
                    button:hover {
                        background-color: #219954 !important;
                        transform: scale(1.05);
                    }
                """
            ):
                if col_w4.button("Use", key=f"water_use_{page_key}_{i}", help="Select this kinematic viscosity"):
                    st.session_state.kin_visc = float(val["Kinematic m²/s"])
                    st.rerun()

# Session State
if 'my_airfoils' not in st.session_state:
    st.session_state.my_airfoils = {}
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}
if 'compare_data' not in st.session_state:
    st.session_state.compare_data = {}
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = {}
if 'page' not in st.session_state:
    st.session_state.page = "home"    
if 'theme' not in st.session_state:
    st.session_state.theme = "Dark"  # Enforce dark mode always

# Sidebar Navigation - Modern Icon-Based Menu
with st.sidebar:
    st.markdown("### ✈️ Airfoil Analyzer Pro")
    st.markdown("---")
    
    # Icon mapping for each page
    page_icons = {
        "home": "house-fill",
        "airfoil-tools": "tools",
        "airfoil-generator": "lightning-fill",
        "airfoil-modifier": "pencil-square",
        "merge-airfoils": "union",
        "optimize-shape": "graph-up-arrow",
        "wing-estimator": "speedometer2",
        "tutorials": "book-fill",
        "about": "info-circle-fill"
    }
    
    selected_page = option_menu(
        menu_title=None,
        options=page_names,
        icons=[page_icons[slug] for slug in page_slugs],
        menu_icon="cast",
        default_index=page_slugs.index(st.session_state.page),
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#1e1e1e"
            },
            "icon": {
                "color": "#ffffff",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px",
                "padding": "10px 15px",
                "border-radius": "8px",
                "color": "#ffffff",
                "background-color": "#2b2b2b",
                "--hover-color": "#3d3d3d",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #e74c3c 0%, #c0392b 100%)",
                "color": "#ffffff",
                "font-weight": "600",
                "box-shadow": "0 4px 12px rgba(231, 76, 60, 0.4)",
                "transform": "translateX(5px)"
            },
        },
        key="main_navigation"
    )
    
    # Update page state if changed
    selected_slug = reverse_map[selected_page]
    if selected_slug != st.session_state.page:
        st.session_state.page = selected_slug
        st.rerun()
    
    st.markdown("---")

# Enhanced sidebar styling
st.markdown("""
<style>
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 2px solid #333333;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #1a1a1a;
    }
    
    /* Make sidebar title more prominent */
    [data-testid="stSidebar"] h3 {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin-bottom: 0 !important;
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Smooth scrolling for sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Hover effect enhancement */
    .nav-link:hover {
        transform: translateX(3px) !important;
    }
</style>
""", unsafe_allow_html=True)

# Apply Dark Mode CSS always (light mode removed)
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .card {
        border-color: #555555;
        background-color: #333333;
    }
    input, select, textarea {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    .stButton > button {
        padding: 0.1rem 1rem;
        border-radius: 4px;
        border: none;
        color: #ffffff;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
        width: auto;
        height: auto;
        margin: 0.5rem 0;
        font-size: 1rem;
        background-color: #444444 !important;  /* Default gray for visibility */
    }
    .stButton > button:hover {
        transform: scale(1.05);
    }
    /* Primary (e.g., Start Analyzing, Compare, Optimize) */
    .stButton > button[kind="primary"] {
        background-color: #e74c3c !important;  /* Alizarin Red */
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #c0392b !important;
    }
    /* Use buttons (viscosity) - Green for positive/select actions */
    .stButton > button[title="Select this kinematic viscosity"] {
        background-color: #27ae60 !important;  /* Nephritis Green */
        box-shadow: 0 2px 4px rgba(39,174,96,0.3);
    }
    .stButton > button[title="Select this kinematic viscosity"]:hover {
        background-color: #219954 !important;
    }
    /* Calculate button - Blue for computational actions */
    .stButton > button[title="Compute the Reynolds number"] {
        background-color: #3498db !important;  /* Belize Blue */
        box-shadow: 0 2px 4px rgba(52,152,219,0.3);
    }
    .stButton > button[title="Compute the Reynolds number"]:hover {
        background-color: #2c81ba !important;
    }
    /* Add to Favorites - Purple for creative/saving actions */
    .stButton > button[title="Add this airfoil to favorites"] {
        background-color: #8e44ad !important;  /* Wisteria Purple */
        box-shadow: 0 2px 4px rgba(142,68,173,0.3);
    }
    .stButton > button[title="Add this airfoil to favorites"]:hover {
        background-color: #783993 !important;
    }
    /* Generate/Compute Graphs - Turquoise for generation/processing */
    .stButton > button[title="Generate the airfoil"], .stButton > button[title="Compute the performance graphs"] {
        background-color: #16a085 !important;  /* Green Sea Turquoise */
        box-shadow: 0 2px 4px rgba(22,160,133,0.3);
    }
    .stButton > button[title="Generate the airfoil"]:hover, .stButton > button[title="Compute the performance graphs"]:hover {
        background-color: #13896f !important;
    }
    /* Download buttons - Orange for output/export actions */
    .stDownloadButton > button {
        background-color: #f39c12 !important;  /* Sunflower Orange */
        box-shadow: 0 2px 4px rgba(243,156,18,0.3);
    }
    .stDownloadButton > button:hover {
        background-color: #d68910 !important;
    }
    table {
        background-color: #333333;
        color: #ffffff;
    }
    .stExpander {
        background-color: #333333;
        color: #ffffff;
    }
    .stTab {
        background-color: #333333;
        color: #ffffff;
    }
    .stSlider > div > div > div > div {
        background-color: #444444;
    }
    .stSelectbox > div > div {
        background-color: #333333;
    }
</style>
""", unsafe_allow_html=True)

uiuc_airfoils = fetch_uiuc_airfoils()
all_names = sorted(uiuc_airfoils.keys())

plot_types_options = ['CL vs CD', 'CL vs Alpha', 'CL/CD vs Alpha', 'CD vs Alpha', 'Cm vs Alpha', 'Confidence vs Alpha', 'Cpmin vs Alpha']

if st.session_state.page == "home":
    # Hero Section
    st.title("Airfoil Analyzer Pro")
    st.subheader("The Ultimate NeuralFoil-Powered Successor to AirfoilTools.com")
    st.markdown("Fast, intuitive airfoil analysis with UIUC database integration—perfect for engineers and enthusiasts.")
    if st.button("Start Analyzing", type="primary"):
        st.session_state.page = "airfoil-tools"
        st.rerun()

    # Features Overview - Card Grid
    st.header("Key Features")
    features = [
        {"title": "Airfoil Tools", "desc": "Search, analyze, compare airfoils with UIUC integration.", "slug": "airfoil-tools"},
        {"title": "Airfoil Generator", "desc": "Create NACA 4/5/6-series airfoils with custom parameters.", "slug": "airfoil-generator"},
        {"title": "Airfoil Modifier", "desc": "Adjust thickness and camber of existing airfoils.", "slug": "airfoil-modifier"},
        {"title": "Merge Airfoils", "desc": "Blend two airfoils for hybrid designs.", "slug": "merge-airfoils"},
        {"title": "Optimize Shape", "desc": "Optimize airfoil for maximum L/D at target conditions.", "slug": "optimize-shape"},
        {"title": "Wing Estimator", "desc": "Estimate full wing performance from airfoil polars.", "slug": "wing-estimator"},
        {"title": "Tutorials", "desc": "Guides for using the app effectively.", "slug": "tutorials"},
        {"title": "About", "desc": "App details, credits, and limitations.", "slug": "about"}
    ]

    # In the "home" page section
    cols = st.columns(3, gap="small")  # Width divided equally; gap affects spacing (small/medium/large)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            with stylable_container(
                key=f"feature_container_{feat['slug']}",
                css_styles="""
                    button {
                        background-color: #333333;
                        color: #ffffff;
                        padding: 25px;
                        border: 1px solid #555555;
                        border-radius: 12px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
                        text-align: center;
                        height: 120px;  /* Fixed height for uniformity */
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                        cursor: pointer;
                        margin: 15px 0;
                        width: 400px !important;  # Main width control: adjust this pixel value for all widgets
                        min-width: 300px !important;  # Ensures minimum on small screens
                        max-width: 700px !important;  # Caps on large screens for uniformity
                        font-size: 14px;
                        overflow: hidden;
                        box-sizing: border-box;
                    }
                    button > div > p {  /* Target title and desc for non-primary */
                        margin: 0;
                    }
                    button > div > p:first-child {  /* Title styling */
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    button > div > p:last-child {  /* Desc styling */
                        font-size: 14px;
                        color: #cccccc;
                    }
                    button:hover {
                        background-color: #404040;
                        transform: scale(1.05);
                        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
                    }
                    button:nth-child(odd):hover {
                        transform: scale(1.05) rotateY(3deg);
                    }
                    button:nth-child(even):hover {
                        transform: scale(1.05) rotateX(3deg);
                    }
                """
            ):
                if st.button(f"{feat['title']}\n\n{feat['desc']}", key=f"home_btn_{feat['slug']}"):
                    st.session_state.page = feat['slug']
                    st.rerun()

    # Quick Tools
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.header("Quick Tools")
    reynolds_calculator("home")  # Existing, with help tooltips for inputs

    # Footer
    st.markdown("---")
    st.markdown("Footer: [UIUC Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) | [NeuralFoil GitHub](https://github.com/peterdsharpe/NeuralFoil)")

elif st.session_state.page == "airfoil-tools":
    st.title("Airfoil Tools")
    tab1, tab2, tab3 = st.tabs(["Search Database", "Compare Airfoils", "Reynolds Calculator"])

    with tab1:
        # Combined searchable selectbox (Streamlit handles search for long lists)
        selected = st.selectbox(
            "Search Airfoil Name",
            all_names,
            index=None,  # Start with no selection (optional blank)
            help="Type to search the UIUC database and select an airfoil (e.g., 'NACA 0012'). Details will appear after selection.",
            key="selected_at"
        )
        
        if selected:
            dat_url = uiuc_airfoils[selected]
            st.markdown(f"**{selected}**")
            coords = get_airfoil_coords(selected, dat_url)
            if coords is not None:
                st.image(plot_airfoil_shape([coords]), caption="Airfoil Shape")
                dat_content = requests.get(dat_url).content
                st.download_button("Download .dat", dat_content, file_name=f"{selected}.dat")
                if st.button("Add to Favorites", key="add_fav_at", help="Add this airfoil to favorites"):
                    if selected in st.session_state.my_airfoils:
                        st.warning("Overwriting existing favorite with same name.")
                    st.session_state.my_airfoils[selected] = coords
                    st.balloons()
                # Performance Graphs
                st.markdown("Enter semicolon-separated lists for batch (e.g., 1000000;5000000). Note: Large lists may take time—limit to 5-10 for best performance.")
                re_input = st.text_input("Reynolds Numbers (semicolon-separated, e.g., 1000000 or 10,000; 20,000)", value="1000000", help="Enter one for single analysis or multiple for batch (commas optional in numbers).", key="re_input_at")
                try:
                    parts = [p.strip() for p in re_input.split(';') if p.strip()]  # Split on ';'
                    res = [float(p.replace(',', '')) for p in parts]  # Remove internal ',' and convert
                    if not res:
                        raise ValueError("No valid Re values")
                except ValueError:
                    st.error("Invalid Re input; use numbers separated by semicolons (commas optional in numbers).")
                    res = []
                ncrit_input = st.text_input("N Crit Values (semicolon-separated, e.g., 9.0 or 5.0; 9.0)", value="9.0", help="Enter one for single or multiple transition criteria (commas optional in numbers).", key="ncrit_input_at")
                try:
                    parts = [p.strip() for p in ncrit_input.split(';') if p.strip()]  # Split on ';'
                    ncrits = [float(p.replace(',', '')) for p in parts]  # Remove internal ',' and convert
                    if not ncrits:
                        raise ValueError("No valid Ncrit values")
                except ValueError:
                    st.error("Invalid Ncrit input; use numbers separated by semicolons (commas optional in numbers).")
                    ncrits = []
                alpha_range = st.slider("Alpha Range (°)", -20.0, 30.0, (-15.0, 25.0), step=0.5, key="alpha_range_at")  # Single range slider with dual knobs
                alpha_min, alpha_max = alpha_range
                if alpha_min > alpha_max:
                    st.error("Alpha Min > Max; swapping values.")
                    alpha_min, alpha_max = alpha_max, alpha_min
                alpha_step = 0.1  # Fixed at 0.1°
                alphas = np.arange(alpha_min, alpha_max + alpha_step/2, alpha_step)
                if len(alphas) == 0:
                    st.error("Invalid alpha range; using default.")
                    alphas = np.linspace(-5, 15, 21)
                plot_types = st.multiselect("Select Plots", plot_types_options, default=plot_types_options[:3], key="plot_types_at")
                if st.button("Compute Graphs", key="compute_graphs_at", help="Compute the performance graphs"):
                    if not res or not ncrits:
                        st.error("Enter valid Re and Ncrit values")
                    else:
                        with st.spinner("Computing..."):
                            aero_list = []
                            names = []
                            progress = st.progress(0)
                            total = len(res) * len(ncrits)
                            count = 0
                            for re in res:
                                for ncrit in ncrits:
                                    aero = analyze_airfoil(coords, alphas, re, 'xxxlarge', ncrit)
                                    if aero:
                                        aero_list.append(aero)
                                        re_str = format_re(re)
                                        names.append(f"Re={re_str}, Ncrit={ncrit}")
                                    count += 1
                                    progress.progress(count / total)
                        if aero_list:
                            st.session_state.search_batch_data = {'aeros': aero_list, 'names': names}  # Temp storage for download
                            if len(aero_list) == 1:
                                st.session_state.analysis_data = {'aero': aero_list[0], 'name': names[0]}  # Set for Wing Estimator compatibility
                            st.success("Computation complete!")
                            st.image(create_plots(aero_list, plot_types, True, names))
                            metrics_list = [compute_key_metrics(a) for a in aero_list]
                            metrics_df = pd.DataFrame(metrics_list, index=names)
                            st.table(metrics_df)
                            zip_buf = io.BytesIO()
                            with zipfile.ZipFile(zip_buf, 'w') as zf:
                                for name, aero in zip(names, aero_list):
                                    csv_data = generate_csv(aero)
                                    if csv_data:
                                        zf.writestr(f"{name}.csv", csv_data)
                            zip_buf.seek(0)
                            st.download_button("Download ZIP", zip_buf, f"{selected}_batch.zip", mime="application/zip")
        else:
            st.markdown("Select an airfoil from the dropdown to view details.")

    with tab2:
        with st.expander("Select Airfoils"):
            selected_uiuc = st.multiselect("UIUC Airfoils", all_names, key="selected_uiuc_ca")
            selected_my = st.multiselect("My Favorites", list(st.session_state.my_airfoils.keys()), key="selected_my_ca")
            uploaded = st.file_uploader("Upload .dat Files", type=["dat", "txt"], accept_multiple_files=True, key="uploaded_ca")
            airfoils = []
            names = []
            coords_list = []
            for s in selected_uiuc:
                dat_url = uiuc_airfoils[s]
                coords = get_airfoil_coords(s, dat_url)
                if coords is not None:
                    airfoils.append(coords)
                    names.append(s)
                    coords_list.append(coords)
            for s in selected_my:
                coords = st.session_state.my_airfoils[s]
                airfoils.append(coords)
                names.append(s)
                coords_list.append(coords)
            for u in uploaded or []:
                coords = get_airfoil_coords(file=u)
                if coords is not None:
                    airfoils.append(coords)
                    names.append(u.name)
                    coords_list.append(coords)
            if coords_list:
                st.image(plot_airfoil_shape(coords_list, names), caption="Shapes")

        with st.expander("Parameters"):
            Re = st.slider("Re", 10000, 10000000, 1000000, 10000, key="re_ca")
            alpha_min = st.slider("Alpha Min (°)", -20.0, 0.0, -15.0, 0.5, key="alpha_min_ca")
            alpha_max = st.slider("Alpha Max (°)", 0.0, 30.0, 25.0, 0.5, key="alpha_max_ca")
            alpha_step = st.slider("Alpha Step (°)", 0.1, 5.0, 0.5, 0.1, key="alpha_step_ca")
            if alpha_min > alpha_max:
                st.error("Alpha Min > Max; swapping values.")
                alpha_min, alpha_max = alpha_max, alpha_min
            alphas = np.arange(alpha_min, alpha_max + alpha_step/2, alpha_step)
            if len(alphas) == 0:
                st.error("Invalid alpha range; using default.")
                alphas = np.linspace(-5, 15, 21)
            model_size = st.selectbox("Model Size", ['xxsmall', 'xsmall', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'xxxlarge'], index=5, key="model_size_ca")

        plot_types = st.multiselect("Plots", plot_types_options, default=plot_types_options[:3], key="plot_types_ca")

        if st.button("Compare", type="primary", key="compare_ca"):
            if not airfoils:
                st.error("Select airfoils")
            else:
                with st.spinner("Comparing..."):
                    aero_list = []
                    progress = st.progress(0)
                    step = 1.0 / len(airfoils)
                    for i, coords in enumerate(airfoils):
                        aero = analyze_airfoil(coords, alphas, Re, model_size)
                        if aero:
                            aero_list.append(aero)
                        progress.progress((i+1)*step)
                if aero_list:
                    st.session_state.compare_data = {'aeros': aero_list, 'names': names}
                    st.success("Comparison complete!")
                    st.image(create_plots(aero_list, plot_types, True, names))
                    metrics_list = [compute_key_metrics(a) for a in aero_list]
                    metrics_df = pd.DataFrame(metrics_list, index=names)
                    st.table(metrics_df)
                    confs = [m['Avg Confidence'] for m in metrics_list]
                    st.markdown("Average Confidences: " + ", ".join(f"{n}: {c:.2f}" for n, c in zip(names, confs)))

        if 'compare_data' in st.session_state and st.session_state.compare_data:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w') as zf:
                for name, aero in zip(st.session_state.compare_data['names'], st.session_state.compare_data['aeros']):
                    csv_data = generate_csv(aero)
                    if csv_data:
                        zf.writestr(f"{name}.csv", csv_data)
            zip_buf.seek(0)
            st.download_button("Download ZIP", zip_buf, "comparison.zip", mime="application/zip", key="download_zip_ca")

    with tab3:
        st.title("Reynolds Number Calculator")
        reynolds_calculator("airfoil-tools")  # Updated calculator

elif st.session_state.page == "airfoil-generator":
    st.title("Airfoil Generator")
    tab1, tab2, tab3 = st.tabs(["NACA 4-Digit", "NACA 5-Digit", "NACA 6-Series"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        m = col1.slider("Max Camber (%)", 0, 9, 0, key="m_4digit_ag") / 100
        p = col2.slider("Camber Position (tenths)", 0, 9, 4, key="p_4digit_ag") / 10
        t = col3.slider("Thickness (%)", 1, 40, 12, key="t_4digit_ag") / 100
        n_points = st.slider("Points", 50, 200, 100, key="n_points_4digit_ag")
        if st.button("Generate 4-Digit", key="gen_4digit_ag", help="Generate the airfoil"):
            coords = generate_naca_4digit(m, p, t, n_points)
            name = f"NACA {int(m*100):02d}{int(p*10)}{int(t*100):02d}"
            st.image(plot_airfoil_shape([coords]), caption=name)
            dat_data = generate_dat(coords, name)
            st.download_button("Download .dat", dat_data, f"{name}.dat", key="download_4digit_ag")
            if st.button("Add to Favorites", key="add_fav_4digit_ag", help="Add this airfoil to favorites"):
                if name in st.session_state.my_airfoils:
                    st.warning("Overwriting existing favorite with same name.")
                st.session_state.my_airfoils[name] = coords
                st.balloons()

    with tab2:
        col1, col2, col3 = st.columns(3)
        l = col1.slider("Design CL Factor (1-5)", 1, 5, 2, key="l_5digit_ag")
        p_digit = col2.slider("Max Camber Pos Digit (1-5)", 1, 5, 3, key="p_digit_5digit_ag")
        q = col3.radio("Type", [0, 1], format_func=lambda x: "Standard" if x==0 else "Reflex", key="q_5digit_ag")
        t = st.slider("Thickness (%)", 1, 40, 12, key="t_5digit_ag")
        n_points = st.slider("Points", 50, 200, 100, key="n_points_5digit_ag")
        if st.button("Generate 5-Digit", key="gen_5digit_ag", help="Generate the airfoil"):
            coords = generate_naca_5digit(l, p_digit, q, t, n_points)
            name = f"NACA {l}{p_digit}{1 if q==1 else 0} {t:02d}"
            st.image(plot_airfoil_shape([coords]), caption=name)
            dat_data = generate_dat(coords, name)
            st.download_button("Download .dat", dat_data, f"{name}.dat", key="download_5digit_ag")
            if st.button("Add to Favorites", key="add_fav_5digit_ag", help="Add this airfoil to favorites"):
                if name in st.session_state.my_airfoils:
                    st.warning("Overwriting existing favorite with same name.")
                st.session_state.my_airfoils[name] = coords
                st.balloons()

    with tab3:
        series = st.selectbox("Series", [63, 64, 65], key="series_6series_ag")
        a = st.slider("Mean Line a", 0.0, 1.0, 0.8, 0.1, key="a_6series_ag")
        cl_design = st.slider("Design CL", 0.0, 1.0, 0.0, 0.05, key="cl_design_6series_ag")
        t = st.slider("Thickness (%)", 1, 40, 12, key="t_6series_ag") / 100
        n_points = st.slider("Points", 50, 200, 100, key="n_points_6series_ag")
        if st.button("Generate 6-Series", key="gen_6series_ag", help="Generate the airfoil"):
            coords = generate_naca_6series(series, a, cl_design, t, n_points)
            name = f"NACA {series}- {int(cl_design*10)} {int(t*100):02d}"
            st.image(plot_airfoil_shape([coords]), caption=name)
            dat_data = generate_dat(coords, name)
            st.download_button("Download .dat", dat_data, f"{name}.dat", key="download_6series_ag")
            if st.button("Add to Favorites", key="add_fav_6series_ag", help="Add this airfoil to favorites"):
                if name in st.session_state.my_airfoils:
                    st.warning("Overwriting existing favorite with same name.")
                st.session_state.my_airfoils[name] = coords
                st.balloons()

elif st.session_state.page == "airfoil-modifier":
    st.title("Modify Airfoil")
    uiuc_sel = st.selectbox("UIUC Airfoil", [""] + all_names, key="uiuc_sel_am")
    dat_url = uiuc_airfoils.get(uiuc_sel)
    my_sel = st.selectbox("My Favorites", [""] + list(st.session_state.my_airfoils.keys()), key="my_sel_am")
    uploaded = st.file_uploader("Upload .dat", type=["dat", "txt"], key="uploaded_am")
    coords = get_airfoil_coords(uiuc_sel, dat_url, uploaded)
    if coords is None:
        coords = st.session_state.my_airfoils.get(my_sel)
    if coords is not None:
        added_t = st.slider("Added Thickness", -0.1, 0.1, None, 0.01, key="added_t_am")
        added_c = st.slider("Added Camber", -0.1, 0.1, None, 0.01, key="added_c_am")
        if added_t is not None or added_c is not None:
            coords = modify_airfoil(coords, added_t, added_c)
        st.image(plot_airfoil_shape([coords]), caption="Modified")
        name = "Modified Airfoil"
        dat_data = generate_dat(coords, name)
        st.download_button("Download .dat", dat_data, "modified.dat", key="download_am")
        if st.button("Add to Favorites", key="add_fav_am", help="Add this airfoil to favorites"):
            if name in st.session_state.my_airfoils:
                st.warning("Overwriting existing favorite with same name.")
            st.session_state.my_airfoils[name] = coords
            st.balloons()

elif st.session_state.page == "merge-airfoils":
    st.title("Merge Airfoils")
    col1, col2 = st.columns(2)
    with col1:
        uiuc1 = st.selectbox("First Airfoil UIUC", [""] + all_names, key="uiuc1_ma")
        dat_url1 = uiuc_airfoils.get(uiuc1)
        my1 = st.selectbox("First from Favorites", [""] + list(st.session_state.my_airfoils.keys()), key="my1_ma")
        upload1 = st.file_uploader("Upload First .dat", type=["dat", "txt"], key="upload1_ma")
        coords1 = get_airfoil_coords(uiuc1, dat_url1, upload1)
        if coords1 is None:
            coords1 = st.session_state.my_airfoils.get(my1)
        if coords1 is not None:
            st.image(plot_airfoil_shape([coords1]), caption=uiuc1 or my1 or (upload1.name if upload1 else "First Airfoil"))
    with col2:
        uiuc2 = st.selectbox("Second Airfoil UIUC", [""] + all_names, key="uiuc2_ma")
        dat_url2 = uiuc_airfoils.get(uiuc2)
        my2 = st.selectbox("Second from Favorites", [""] + list(st.session_state.my_airfoils.keys()), key="my2_ma")
        upload2 = st.file_uploader("Upload Second .dat", type=["dat", "txt"], key="upload2_ma")
        coords2 = get_airfoil_coords(uiuc2, dat_url2, upload2)
        if coords2 is None:
            coords2 = st.session_state.my_airfoils.get(my2)
        if coords2 is not None:
            st.image(plot_airfoil_shape([coords2]), caption=uiuc2 or my2 or (upload2.name if upload2 else "Second Airfoil"))
    if coords1 is not None and coords2 is not None:
        ratio = st.slider("Blend Ratio (0=First, 1=Second)", 0.0, 1.0, 0.5, 0.05, key="ratio_ma")
        coords = merge_airfoils(coords1, coords2, ratio)
        st.image(plot_airfoil_shape([coords]), caption="Merged Airfoil")
        name = "Merged Airfoil"
        dat_data = generate_dat(coords, name)
        st.download_button("Download .dat", dat_data, "merged.dat", key="download_ma")
        if st.button("Add to Favorites", key="add_fav_ma", help="Add this airfoil to favorites"):
            if name in st.session_state.my_airfoils:
                st.warning("Overwriting existing favorite with same name.")
            st.session_state.my_airfoils[name] = coords
            st.balloons()
    else:
        st.error("Select two airfoils")

elif st.session_state.page == "optimize-shape":
    st.title("Optimize Airfoil Shape")
    Re = st.slider("Re", 10000, 10000000, 1000000, 10000, key="re_os")
    target_alpha = st.slider("Target Alpha (°)", 0.0, 20.0, 5.0, 0.5, key="target_alpha_os")
    n_points = st.slider("Points", 50, 200, 100, key="n_points_os")
    if st.button("Optimize", type="primary", key="optimize_os"):
        with st.spinner("Optimizing..."):
            coords = optimize_shape(Re, target_alpha, n_points)
        st.image(plot_airfoil_shape([coords]), caption="Optimized")
        name = "Optimized Airfoil"
        dat_data = generate_dat(coords, name)
        st.download_button("Download .dat", dat_data, "optimized.dat", key="download_os")
        if st.button("Add to Favorites", key="add_fav_os", help="Add this airfoil to favorites"):
            if name in st.session_state.my_airfoils:
                st.warning("Overwriting existing favorite with same name.")
            st.session_state.my_airfoils[name] = coords
            st.balloons()

elif st.session_state.page == "wing-estimator":
    st.title("Wing Performance Estimator")
    use_analysis = st.checkbox("Use Last Analysis Polar", key="use_analysis_we")
    if use_analysis and 'analysis_data' in st.session_state and 'aero' in st.session_state.analysis_data:
        aero = st.session_state.analysis_data['aero']
    else:
        aero = None
        st.warning("Run a single airfoil analysis first to generate a polar, or uncheck the box.")
    AR = st.number_input("Aspect Ratio", min_value=1.0, value=6.0, key="ar_we")
    span = st.number_input("Span (m)", min_value=0.1, value=10.0, key="span_we")
    chord = st.number_input("Average Chord (m)", min_value=0.1, value=1.0, key="chord_we")
    speed = st.number_input("Speed (m/s)", min_value=1.0, value=30.0, key="speed_we")
    density = st.number_input("Air Density (kg/m³)", min_value=0.1, value=1.225, help="Sea level ~1.225", key="density_we")
    alpha = st.slider("Alpha (°)", -10.0, 20.0, 5.0, key="alpha_we")
    if aero:
        interp_cl = interp1d(aero['alpha'], aero['CL'], fill_value="extrapolate")
        interp_cd = interp1d(aero['alpha'], aero['CD'], fill_value="extrapolate")
        CL = interp_cl(alpha)
        CD0 = interp_cd(alpha)
        e = 0.9
        induced_cd = CL**2 / (np.pi * AR * e)
        CD = CD0 + induced_cd
        L = 0.5 * density * speed**2 * (span * chord) * CL
        D = 0.5 * density * speed**2 * (span * chord) * CD
        st.markdown(f"Estimated Lift: {L:.2f} N or {L/9.81:.2f} kg")
        st.markdown(f"Estimated Drag: {D:.2f} N")
        st.markdown(f"L/D: {L/D:.2f}" if D > 0 else "Infinite L/D")
    else:
        st.error("No polar available")

elif st.session_state.page == "tutorials":
    st.title("Tutorials")
    st.markdown(r"""
    ### Getting Started
    - Search: Filter and download from UIUC.
    - Analyze: Upload or select, set params, run. Ensure sharp TE for best results; blunt may affect accuracy.
    - Compare: Multi-select, shared params.
    - Generate: Use sliders for NACA params.
    ### Advanced
    - Modifier: Adjust thickness/camber.
    - Merge: Blend two designs.
    - Optimize: Simple shape opt for max L/D.
    Note: For questions, check About.
    """)

elif st.session_state.page == "about":
    st.title("About")
    st.markdown(r"""
    Credits: UIUC Airfoil Database, NeuralFoil by Peter Sharpe, AeroSandbox.
    This app is open-source inspired; deploy on Streamlit Cloud via GitHub.
    For API: https://x.ai/api
    Limitations: Reflex 5-digit and 6-series approximate camber; verify with external tools for precision. Ensure sharp TE inputs for optimal NeuralFoil results.
    Note: If using an older NeuralFoil version, Mach support may be unavailable—update with `pip install neuralfoil --upgrade`.
    Run locally: python -m venv env; env\Scripts\activate; pip install streamlit neuralfoil aerosandbox matplotlib numpy pandas requests beautifulsoup4 pillow svgwrite scipy; streamlit run app.py
    """)
