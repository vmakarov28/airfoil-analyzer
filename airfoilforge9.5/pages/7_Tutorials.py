"""
Page 7 — Tutorials
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from components.shared import init_session_state

init_session_state()

st.title("📚 Tutorials")
st.markdown("Step-by-step guides to get the most out of AirfoilForge Pro.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Getting Started",
    "Analysis Lab",
    "Design Studio",
    "Tips & Tricks",
])

with tab1:
    st.markdown("""
## Getting Started

### 1. Find an Airfoil
Go to **Reference Library** → type in the search box (e.g. "NACA 0012", "Clark Y", "Eppler").
Click an airfoil name to see its shape, then download the `.dat` file or send it straight to the Analysis Lab.

### 2. Understand Reynolds Number
Use the **Reynolds Calculator** to find the Re for your flight conditions.  
Re = velocity × chord / kinematic viscosity.  
- RC models: Re ≈ 50,000–500,000  
- Drones / UAV: Re ≈ 100,000–1,000,000  
- General aviation: Re ≈ 1,000,000–10,000,000

### 3. Run Your First Analysis
Go to **Analysis Lab**:
1. Select an airfoil from the UIUC dropdown.
2. Enter your Re value(s).
3. Click **Run Analysis**.
4. Examine CL–α, CL/CD–α curves, and the key metrics table.

### 4. Compare Multiple Airfoils
In **Analysis Lab**, select several airfoils from the multiselect.  
All curves appear on the same interactive Plotly charts — hover for exact values, click legend items to hide/show.
""")

with tab2:
    st.markdown("""
## Analysis Lab Deep Dive

### Alpha Range
Start with −15° to 25° and reduce if NeuralFoil confidence drops below 0.5 near stall.

### Multiple Re / Ncrit
Enter semicolon-separated values: `200000; 500000; 1000000`  
This creates a separate run for every (Re × Ncrit) combination.

### Model Size
| Size | Speed | Accuracy |
|------|-------|----------|
| xxsmall | ⚡⚡⚡ | ★★☆ |
| xlarge (default) | ⚡⚡ | ★★★★ |
| xxxlarge | ⚡ | ★★★★★ |

Use **xlarge** or **xxlarge** for final analysis; smaller sizes for rapid exploration.

### N-crit
- N=9: typical low-turbulence wind tunnel / clean air  
- N=5: higher turbulence / rough conditions  
- N=11: very clean laminar conditions

### Confidence Score
NeuralFoil outputs an `analysis_confidence` between 0 and 1.  
Values > 0.85 are generally reliable. Near stall or at very low Re, confidence may drop.

### Export
All results export as CSVs in a ZIP. Use the metrics CSV for direct comparison in Excel or Python.
""")

with tab3:
    st.markdown("""
## Design Studio Guide

### NACA Generator
| Series | Best for |
|--------|----------|
| 4-digit | Classic, simple, well-understood (e.g. 0012 = symmetric, 2412 = cambered) |
| 5-digit | Higher-lift cambered designs (e.g. 23012) |
| 6-series | Low-drag, laminar-flow designs (approximate in this tool) |

**Tip:** Use n_points = 150–200 for smoother curves in analysis.

### Modifier
- Increasing **thickness** delays stall and increases structural depth, but raises drag.
- Adding **camber** increases zero-lift CL and L/D at low-moderate α.
- Large modifications (> 0.05c) may produce unrealistic geometries — always verify the shape visually.

### Merge
The blend slider linearly interpolates upper and lower surface y-coordinates.  
0.5 = 50% of each. Try 0.3 / 0.7 for biased blends.  
Save interesting designs to Favorites before they're lost!
""")

with tab4:
    st.markdown("""
## Tips & Tricks

### Keyboard shortcuts
- **Tab** — move between input fields
- **Enter** — submit a text input

### Workflow recommendation
1. Reynolds Calculator → find your Re  
2. Reference Library → shortlist 3–5 candidates  
3. Analysis Lab → compare them at your Re  
4. Design Studio → tweak the winner  
5. Optimizer → final squeeze  
6. Wing Designer → full-vehicle estimate

### Accuracy notes
- NeuralFoil is a surrogate model trained on XFOIL data — expect ≈1–3% error in CL, ≈5–10% in CD.
- For sharp trailing edges, accuracy is best. Blunt TE may degrade results.
- Very thin airfoils (< 4% thickness) at high α may show unreliable predictions.
- Always validate critical designs with XFOIL, RANS CFD, or wind tunnel testing.

### Favourites persist during your session
Favorites are stored in browser session state — they reset when you close the tab.  
Export your `.dat` files to keep designs permanently.

### Open source
AirfoilForge Pro is built on:
- [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) by Peter Sharpe (MIT license)
- [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) by Peter Sharpe
- [UIUC Airfoil Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) by Prof. Michael Selig
""")
