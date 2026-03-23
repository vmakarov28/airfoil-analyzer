"""
Reynolds number calculator component — reusable across pages.
"""

import streamlit as st
import streamlit.components.v1 as components
from components.plotting_engine import make_reynolds_gauge


def reynolds_calculator(page_key=""):
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Reynolds Number Calculator")

        velocity = st.number_input(
            "Velocity (m/s)", min_value=0.0, value=10.0, key=f"velocity_{page_key}"
        )
        chord = st.number_input(
            "Chord width (m)", min_value=0.0, value=0.2, key=f"chord_{page_key}"
        )

        if "kin_visc" not in st.session_state:
            st.session_state["kin_visc"] = 1.2462e-5
        widget_key = f"kin_visc_{page_key}"
        st.session_state[widget_key] = st.session_state["kin_visc"]
        kin_visc = st.number_input(
            "Kinematic Viscosity (m²/s)",
            min_value=0.0,
            key=widget_key,
            format="%e",
            help="Air at 20 °C ≈ 1.51×10⁻⁵ m²/s",
        )
        st.session_state["kin_visc"] = kin_visc

        if kin_visc > 0:
            Re = velocity * chord / kin_visc
            re_formatted = f"{Re:,.0f}"

            # Reynolds number display + copy button in one row
            st.markdown(
                f"""
                <div style="background:#2c2c2c;border-radius:10px;padding:12px 20px;
                            text-align:center;font-size:1.3rem;font-weight:700;color:#e74c3c;
                            letter-spacing:1px;margin:10px 0 4px 0;">
                  Re = {re_formatted}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Copy-to-clipboard button rendered as an inline HTML component
            # Uses navigator.clipboard API which works in modern browsers over HTTPS/localhost
            copy_html = f"""
            <div style="display:flex;justify-content:center;margin-bottom:10px;">
              <button id="copy-re-btn-{page_key}"
                onclick="(function(){{
                  var val = '{re_formatted}';
                  if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(val).then(function() {{
                      var btn = document.getElementById('copy-re-btn-{page_key}');
                      btn.innerHTML = '✅&nbsp;&nbsp;Copied!';
                      btn.style.borderColor = '#27ae60';
                      btn.style.color = '#27ae60';
                      setTimeout(function() {{
                        btn.innerHTML = '📋&nbsp;&nbsp;Copy Re to Clipboard';
                        btn.style.borderColor = '#3d4451';
                        btn.style.color = '#9aa0ad';
                      }}, 2000);
                    }});
                  }} else {{
                    var ta = document.createElement('textarea');
                    ta.value = val;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                    var btn = document.getElementById('copy-re-btn-{page_key}');
                    btn.innerHTML = '✅&nbsp;&nbsp;Copied!';
                    btn.style.borderColor = '#27ae60';
                    btn.style.color = '#27ae60';
                    setTimeout(function() {{
                      btn.innerHTML = '📋&nbsp;&nbsp;Copy Re to Clipboard';
                      btn.style.borderColor = '#3d4451';
                      btn.style.color = '#9aa0ad';
                    }}, 2000);
                  }}
                }})();"
                style="
                  background: transparent;
                  border: 1px solid #3d4451;
                  border-radius: 8px;
                  color: #9aa0ad;
                  cursor: pointer;
                  font-size: 0.85rem;
                  font-family: 'Inter', 'Segoe UI', sans-serif;
                  padding: 7px 18px;
                  transition: all 0.2s ease;
                  letter-spacing: 0.02em;
                "
                onmouseover="this.style.borderColor='#e74c3c';this.style.color='#e74c3c';"
                onmouseout="if(!this.innerHTML.includes('Copied')){{this.style.borderColor='#3d4451';this.style.color='#9aa0ad';}}"
              >
                📋&nbsp;&nbsp;Copy Re to Clipboard
              </button>
            </div>
            """
            components.html(copy_html, height=52)

            st.plotly_chart(
                make_reynolds_gauge(Re), use_container_width=True, config={"displayModeBar": False}
            )
        else:
            st.error("Viscosity must be positive.")

        st.markdown(
            r"""
            **Formula:** Re = ρvl / μ = vl / ν

            | Symbol | Meaning |
            |--------|---------|
            | v | Fluid velocity |
            | l | Characteristic length (chord) |
            | ρ | Fluid density |
            | μ | Dynamic viscosity |
            | ν | Kinematic viscosity |
            """
        )

    with right_col:
        st.subheader("Kinematic Viscosity Reference")
        st.markdown("Select a value to auto-fill the calculator.")

        air_values = [
            {"label": "Air at -10 °C (14 °F)", "value": 1.2462e-5},
            {"label": "Air at 0 °C (32 °F)",   "value": 1.3324e-5},
            {"label": "Air at 10 °C (50 °F)",  "value": 1.4207e-5},
            {"label": "Air at 20 °C (68 °F)",  "value": 1.5111e-5},
            {"label": "Air at 30 °C (86 °F)",  "value": 1.6035e-5},
        ]
        water_values = [
            {"label": "Water at 1 °C (33.8 °F)", "value": 1.6438e-6},
            {"label": "Water at 10 °C (50 °F)",  "value": 1.2676e-6},
            {"label": "Water at 20 °C (68 °F)",  "value": 9.7937e-7},
        ]

        st.markdown("**Air (1 atm)**")
        for i, v in enumerate(air_values):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"`{v['value']:.4e}` — {v['label']}")
            if c2.button("Use", key=f"air_{page_key}_{i}"):
                st.session_state["kin_visc"] = v["value"]
                st.rerun()

        st.markdown("---")
        st.markdown("**Water (1 atm)**")
        for i, v in enumerate(water_values):
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"`{v['value']:.4e}` — {v['label']}")
            if c2.button("Use", key=f"water_{page_key}_{i}"):
                st.session_state["kin_visc"] = v["value"]
                st.rerun()
