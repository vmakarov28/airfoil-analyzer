"""
Shared utilities, helpers, data functions, and global styles used across all pages.
"""

import streamlit as st
import neuralfoil as nf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.interpolate import interp1d
import io
import zipfile
import svgwrite


# ── Global styles (injected on every page via init_session_state) ─────────────

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stApp"] {
    background-color: #070b12 !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #0a0f1a 100%) !important;
    border-right: 1px solid rgba(231,76,60,0.25) !important;
    box-shadow: 4px 0 30px rgba(231,76,60,0.08) !important;
}

[data-testid="stSidebarNav"] a {
    font-size: 1.05rem !important; font-weight: 500 !important;
    padding: 10px 16px !important; border-radius: 8px !important;
    margin: 2px 6px !important; display: block !important;
    transition: all 0.2s ease !important; color: #b0b8c8 !important;
    letter-spacing: 0.01em !important;
    border-left: 3px solid transparent !important;
    text-decoration: none !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(231,76,60,0.10) !important;
    color: #ff7060 !important;
    border-left-color: rgba(231,76,60,0.5) !important;
}
[data-testid="stSidebarNav"] a[aria-selected="true"] {
    background: rgba(231,76,60,0.14) !important;
    color: #ff6b5b !important;
    border-left-color: #e74c3c !important;
    box-shadow: inset 0 0 20px rgba(231,76,60,0.08),
                0 0 16px rgba(231,76,60,0.12) !important;
}
[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNav"] a p,
[data-testid="stSidebarNav"] li { font-size: 1.05rem !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #e74c3c, #c0392b) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 4px 14px rgba(231,76,60,0.4) !important;
    height: auto !important; min-height: unset !important;
    padding: 10px 24px !important; border-radius: 8px !important;
    font-size: 0.95rem !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(231,76,60,0.6) !important;
    transform: translateY(-2px) !important;
}
.stDownloadButton > button {
    background: linear-gradient(90deg, #f39c12, #d68910) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(243,156,18,0.35) !important;
}
input, select, textarea {
    background-color: #1a2035 !important; color: #e0e0e0 !important;
    border: 1px solid #2d3a55 !important; border-radius: 6px !important;
}
[data-testid="stSelectbox"] > div > div  { background-color: #1a2035 !important; }
[data-testid="stMultiSelect"] > div > div { background-color: #1a2035 !important; }
[data-testid="stTabs"] button { color: #9aa0ad !important; border-bottom: 2px solid transparent; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e74c3c !important; border-bottom: 2px solid #e74c3c !important; font-weight: 600;
}
.stDataFrame, .stTable { background-color: #0d1525 !important; color: #e0e0e0 !important; }
table { border-collapse: collapse; width: 100%; }
th { background-color: #1a2035 !important; color: #e74c3c !important; padding: 8px 12px; }
td { background-color: #0d1525 !important; color: #e0e0e0 !important;
     padding: 6px 12px; border-bottom: 1px solid #1a2035; }
.stExpander { background-color: #0d1525 !important;
              border: 1px solid #1a2035 !important; border-radius: 8px !important; }
.stAlert { border-radius: 8px !important; }
hr { border-color: #1a2035 !important; }
iframe { border: none !important; }
</style>
"""

_SIDEBAR_BRAND_HTML = """
<div style="text-align:center;padding:20px 0 10px;">
  <div style="font-size:2.4rem;margin-bottom:5px;line-height:1;">✈️</div>
  <div style="font-size:1.4rem;font-weight:700;letter-spacing:0.04em;
               background:linear-gradient(135deg,#e74c3c 0%,#f39c12 60%,#e74c3c 100%);
               background-size:200% auto;
               animation:afShimmer 3s linear infinite;
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">AirfoilForge Pro</div>
  <div style="font-size:0.63rem;color:#3a4a62;letter-spacing:0.1em;
               margin-top:4px;text-transform:uppercase;">
    NeuralFoil · AeroSandbox · UIUC
  </div>
  <div style="margin:12px auto 0;width:55%;height:1px;
               background:linear-gradient(90deg,transparent,rgba(231,76,60,0.5),transparent);">
  </div>
  <div style="margin-top:10px;padding:0 14px;text-align:left;">
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                 color:#2a3a52;font-weight:700;">Navigation</div>
  </div>
</div>
<style>
@keyframes afShimmer{0%{background-position:0% center}100%{background-position:200% center}}
</style>
"""


def inject_global_styles():
    """Inject dark-theme CSS + sidebar branding on the current page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(_SIDEBAR_BRAND_HTML, unsafe_allow_html=True)


# ── Formatting ───────────────────────────────────────────────────────────────

def format_re(re):
    if re < 1e3:
        return f"{int(re)}"
    elif re < 1e6:
        return f"{int(re / 1e3)}k"
    elif re < 1e9:
        return f"{int(re / 1e6)}M"
    return f"{re:.0e}"


# ── UIUC Database ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_uiuc_airfoils():
    url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        airfoils = {}
        for link in soup.find_all("a", href=True):
            if link["href"].endswith(".dat"):
                name = link.text.strip()
                dat_url = f"https://m-selig.ae.illinois.edu/ads/{link['href']}"
                airfoils[name] = dat_url
        return airfoils
    except Exception as e:
        st.error(f"Failed to fetch UIUC database: {e}. Using fallback mode.")
        return {}


# ── .dat parsing ─────────────────────────────────────────────────────────────

def parse_dat_file(content):
    try:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty file")
        name = lines[0]
        lines = lines[1:]
        parts = lines[0].split()
        coords = None
        if len(parts) == 2:
            try:
                nu, nl = map(float, parts)
                if nu.is_integer() and nl.is_integer() and int(nu) >= 10 and int(nl) >= 10:
                    nu, nl = int(nu), int(nl)
                    upper, lower = [], []
                    line_idx = 1
                    while len(upper) < nu and line_idx < len(lines):
                        try:
                            x, y = map(float, lines[line_idx].split()[:2])
                            upper.append([x, y])
                        except:
                            pass
                        line_idx += 1
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
                    raise ValueError("Not LEDNICER")
            except:
                pass
        if coords is None:
            raw = []
            for line in lines:
                p = line.split()
                if len(p) >= 2:
                    try:
                        x, y = map(float, p[:2])
                        raw.append([x, y])
                    except:
                        pass
            coords = np.array(raw)
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack((coords, coords[0]))
        internal = coords[1:-1]
        idx = np.unique(internal, axis=0, return_index=True)[1]
        internal = internal[np.sort(idx)]
        coords = np.vstack((coords[0:1], internal, coords[-1:]))
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] > 1):
            mn, mx = np.min(coords[:, 0]), np.max(coords[:, 0])
            if mx > mn:
                coords[:, 0] = (coords[:, 0] - mn) / (mx - mn)
        return coords
    except Exception as e:
        st.error(f"Invalid .dat file: {e}")
        return None


def get_airfoil_coords(name=None, url=None, file=None):
    if file:
        try:
            content = file.getvalue().decode("utf-8", errors="ignore")
            return parse_dat_file(content)
        except:
            st.error("File decoding failed")
            return None
    elif name and url:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return parse_dat_file(r.text)
        except Exception as e:
            st.error(f"Failed to fetch {name}: {e}")
            return None
    return None


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_airfoil(coords, alphas, Re, model_size="xlarge", n_crit=9):
    try:
        if coords.shape[0] < 20 or coords.shape[1] != 2 or not np.all(
            (0 <= coords[:, 0]) & (coords[:, 0] <= 1)
        ):
            raise ValueError("Invalid coordinates")
        if coords[1, 1] < 0:
            coords = coords[::-1]
        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=alphas,
            Re=Re,
            n_crit=n_crit,
            model_size=model_size,
        )
        aero["alpha"] = np.array(alphas)
        aero["CL/CD"] = np.array(aero["CL"]) / np.maximum(np.array(aero["CD"]), 1e-6)
        return aero
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None


def compute_key_metrics(aero):
    if aero is None or "CL" not in aero or len(aero["CL"]) == 0:
        return {}
    aero = {k: np.nan_to_num(np.array(v)) for k, v in aero.items()}
    max_cl_idx = np.argmax(aero["CL"])
    min_cd = max(np.min(aero["CD"]), 1e-6)
    return {
        "Max CL": round(float(aero["CL"][max_cl_idx]), 4),
        "Stall Angle (°)": round(float(aero["alpha"][max_cl_idx]), 2),
        "Min CD": round(float(min_cd), 6),
        "Max L/D": round(float(np.max(aero["CL/CD"])), 2),
        "Avg Confidence": round(float(np.mean(aero["analysis_confidence"])), 4),
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


# ── NACA Generators ──────────────────────────────────────────────────────────

def generate_naca_4digit(m, p, t, n_points=100):
    if p <= 0 or p >= 1:
        m, p = 0, 0.4
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    if m != 0:
        idx = x < p
        yc[idx] = m / p ** 2 * (2 * p * x[idx] - x[idx] ** 2)
        yc[~idx] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~idx] - x[~idx] ** 2)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    theta = np.arctan(np.gradient(yc, x))
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    coords[0, 1] = 0
    coords[-1, 1] = 0
    return coords


def generate_naca_5digit(l, p_digit, q, t, n_points=100):
    t = 0.01 * t
    cl_design = (3 / 2) * (l / 10.0)
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    if q == 0:
        xcm = 0.05 * p_digit
        r = 0.1
        for _ in range(1000):
            r_new = xcm + r * np.sqrt(r / 3)
            if abs(r_new - r) < 1e-6:
                break
            r = r_new
        qm = (3 * r - 7 * r**2 + 8 * r**3 - 4 * r**4) / np.sqrt(r * (1 - r)) - 1.5 * (1 - 2 * r) * (np.pi / 2 - np.arcsin(1 - 2 * r))
        k1 = 6 * cl_design / qm
        idx = x < r
        yc[idx] = (k1 / 6) * (x[idx] ** 3 - 3 * r * x[idx] ** 2 + r**2 * (3 - r) * x[idx])
        dyc[idx] = (k1 / 6) * (3 * x[idx] ** 2 - 6 * r * x[idx] + r**2 * (3 - r))
        yc[~idx] = (k1 / 6) * r**3 * (1 - x[~idx])
        dyc[~idx] = -(k1 / 6) * r**3
    else:
        r, k1, k21 = 0.2170, 15.793, -0.1010
        idx = x < r
        yc[idx] = (k1 / 6) * ((x[idx] - r) ** 3 - k21 * (1 - r) ** 3 * x[idx] - r**3 * x[idx] + r**3)
        dyc[idx] = (k1 / 6) * (3 * (x[idx] - r) ** 2 - k21 * (1 - r) ** 3 - r**3)
        yc[~idx] = (k1 / 6) * (k21 * (x[~idx] - r) ** 3 - k21 * (1 - r) ** 3 * x[~idx] - r**3 * x[~idx] + r**3)
        dyc[~idx] = (k1 / 6) * (3 * k21 * (x[~idx] - r) ** 2 - k21 * (1 - r) ** 3 - r**3)
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    coords[0, 1] = 0; coords[-1, 1] = 0
    return coords


def generate_naca_6series(series, a, cl_design, t, n_points=100):
    x = np.linspace(0, 1, n_points)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if cl_design != 0:
        m = cl_design / (2 * np.pi * (1 + a))
        idx = x <= a
        yc[idx] = m * (x[idx] / a**2) * (2 * a - x[idx])
        dyc_dx[idx] = m * (2 * a - 2 * x[idx]) / a**2
        yc[~idx] = m * ((1 - x[~idx]) / (1 - a) ** 2) * (1 + x[~idx] - 2 * a)
        dyc_dx[~idx] = m * ((1 - x[~idx]) * (1 - 2 * a) - (1 + x[~idx] - 2 * a)) / (1 - a) ** 2
    a4 = -0.1036 if series in (64, 65) else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + a4 * x**4)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = np.vstack([np.column_stack((xu[::-1], yu[::-1])), np.column_stack((xl, yl))])
    coords[0, 1] = 0; coords[-1, 1] = 0
    return coords


# ── Airfoil operations ───────────────────────────────────────────────────────

def modify_airfoil(coords, added_t=None, added_c=None):
    le_idx = np.argmin(coords[:, 0])
    upper = coords[: le_idx + 1][::-1]
    lower = coords[le_idx:]
    x_common = np.linspace(0, 1, max(len(upper), len(lower)))
    y_u = interp1d(upper[:, 0], upper[:, 1], kind="linear", fill_value="extrapolate")(x_common)
    y_l = interp1d(lower[:, 0], lower[:, 1], kind="linear", fill_value="extrapolate")(x_common)
    if added_t is not None:
        curr_t = np.max(y_u - y_l)
        if curr_t > 0:
            new_t = curr_t + added_t
            if new_t > 0:
                y_u *= new_t / curr_t
                y_l *= new_t / curr_t
    if added_c is not None:
        yc_add = added_c * (1 - x_common) ** 2 * x_common
        y_u += yc_add
        y_l += yc_add
    coords = np.vstack((np.column_stack((x_common, y_u))[::-1], np.column_stack((x_common, y_l))[1:]))
    if np.max(coords[:, 0]) > 0:
        coords[:, 0] /= np.max(coords[:, 0])
    return coords


def merge_airfoils(coords1, coords2, ratio=0.5):
    def split(c):
        le = np.argmin(c[:, 0])
        return c[: le + 1][::-1], c[le:]
    u1, l1 = split(coords1)
    u2, l2 = split(coords2)
    x = np.linspace(0, 1, max(len(u1), len(u2)))
    y_u = (1 - ratio) * interp1d(u1[:, 0], u1[:, 1], fill_value="extrapolate")(x) + ratio * interp1d(u2[:, 0], u2[:, 1], fill_value="extrapolate")(x)
    y_l = (1 - ratio) * interp1d(l1[:, 0], l1[:, 1], fill_value="extrapolate")(x) + ratio * interp1d(l2[:, 0], l2[:, 1], fill_value="extrapolate")(x)
    return np.vstack((np.column_stack((x, y_u))[::-1], np.column_stack((x, y_l))))


# ── Export helpers ───────────────────────────────────────────────────────────

def generate_dat(coords, name, fmt="SELIG"):
    buf = io.StringIO()
    buf.write(f"{name}\n")
    if fmt == "LEDNICER":
        le_idx = np.argmin(coords[:, 0])
        upper = coords[: le_idx + 1][::-1]
        lower = coords[le_idx:]
        buf.write(f"{len(upper)} {len(lower)}\n")
        np.savetxt(buf, upper, fmt="%.6f")
        buf.write("\n")
        np.savetxt(buf, lower, fmt="%.6f")
    else:
        np.savetxt(buf, coords, fmt="%.6f")
    buf.seek(0)
    return buf.getvalue()


def export_to_svg(coords, halo=False, pitch=0.0, scale=1000):
    y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
    height = max(200, scale * y_range * 1.2)
    dwg = svgwrite.Drawing(size=(scale, height))
    theta = np.deg2rad(pitch)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ rot.T
    coords_rot[:, 1] -= np.min(coords_rot[:, 1])
    coords_rot *= scale
    points = [(float(x), float(y)) for x, y in coords_rot]
    dwg.add(dwg.polygon(points, stroke="black", fill="none"))
    if halo:
        offset = 0.01 * scale
        dirs = np.roll(coords_rot, -1, axis=0) - coords_rot
        nlen = np.linalg.norm(dirs, axis=1)
        nlen[nlen == 0] = 1
        norms = np.column_stack((-dirs[:, 1], dirs[:, 0])) / nlen[:, np.newaxis]
        halo_coords = coords_rot + offset * norms
        hpoints = [(float(x), float(y)) for x, y in halo_coords]
        dwg.add(dwg.polygon(hpoints, stroke="black", fill="none", stroke_dasharray="5,5"))
    buf = io.StringIO()
    dwg.write(buf)
    buf.seek(0)
    return buf.getvalue().encode("utf-8")


def make_zip(names, aero_list):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, aero in zip(names, aero_list):
            csv_data = generate_csv(aero)
            if csv_data:
                safe = name.replace("/", "_").replace(" ", "_")
                zf.writestr(f"{safe}.csv", csv_data)
    buf.seek(0)
    return buf


# ── Session state init ───────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "my_airfoils": {},
        "analysis_data": {},
        "compare_data": {},
        "batch_data": {},
        "kin_visc": 1.2462e-5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Automatically inject global styles + sidebar branding on every page
    inject_global_styles()
