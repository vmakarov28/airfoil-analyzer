"""
Page 8 — About
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from components.shared import init_session_state

init_session_state()

st.title("ℹ️ About AirfoilForge Pro")

st.markdown("""
## What is AirfoilForge Pro?

AirfoilForge Pro is a completely free, open-source airfoil engineering platform built for the RC, drone, UAV, 
and student aerospace community. It is designed as the definitive free successor to AirfoilTools.com, combining:

- **NeuralFoil** — a lightning-fast, differentiable surrogate model for 2D aerodynamics
- **AeroSandbox** — a gradient-based MDO framework for optimization
- **UIUC Airfoil Database** — 1,600+ real-world airfoil coordinate files
- **Plotly** — fully interactive, exportable charts

## Credits

| Library | Author | License |
|---------|--------|---------|
| [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) | Peter Sharpe (MIT) | MIT |
| [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) | Peter Sharpe (MIT) | MIT |
| [UIUC Coordinate Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) | Prof. M. Selig (UIUC) | Public domain |
| [Streamlit](https://streamlit.io) | Snowflake Inc. | Apache 2.0 |
| [Plotly](https://plotly.com) | Plotly Technologies Inc. | MIT |

## Limitations

- **NeuralFoil accuracy**: surrogate model trained on XFOIL data. Typical error: CL ≈ 1–3%, CD ≈ 5–10%.
- **Stall modeling**: NeuralFoil is less accurate near and beyond stall (α > 15°).
- **NACA 5-digit reflex**: the reflex (q=1) series uses hardcoded 231 coefficients.
- **NACA 6-series**: cambered 6-series uses an approximate camber line — verify with XFOIL for precision.
- **Sharp trailing edge**: NeuralFoil performs best with sharp TE. Blunt TE may reduce accuracy.
- **No viscous interaction**: the wing estimator uses simple lifting-line with no viscous corrections.

## Deploy your own instance

```bash
git clone https://github.com/your-fork/airfoilforge
cd airfoilforge
pip install -r requirements.txt
streamlit run app.py
```

Or click "Deploy" in Streamlit Community Cloud (free).

## requirements.txt

```
streamlit
neuralfoil
aerosandbox
numpy
pandas
requests
beautifulsoup4
pillow
svgwrite
scipy
plotly
streamlit-option-menu
```

## Roadmap

- [ ] PDF report export (reportlab)
- [ ] Reference polar overlay from AirfoilTools CSV cache
- [ ] Hybrid NeuralFoil + XFOIL reference comparison
- [ ] DXF / STEP export (via CadQuery)
- [ ] 3D wing visualization (Plotly 3D)
- [ ] Shareable URL deep-linking (query params)
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.8rem;'>AirfoilForge Pro — 100% free, no login, no limits.</div>",
    unsafe_allow_html=True,
)
