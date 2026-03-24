"""
Page 6 — Reynolds Number Calculator (standalone page)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from components.shared import init_session_state
from components.reynolds_calculator import reynolds_calculator

init_session_state()

st.title("📐 Reynolds Number Calculator")
st.markdown("Compute your Reynolds number and identify the flow regime at a glance.")

reynolds_calculator("standalone")
