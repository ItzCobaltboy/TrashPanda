import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import os

# Load the route map if available
if not os.path.exists("truck_routes.html"):
    st.error("Route visualization not found. Run the optimization script first.")
    st.stop()

# Display header
st.title("🚛 Smart Garbage Collection Dashboard")

# Sidebar for simulation controls
st.sidebar.header("Simulation Controls")
n_bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=50, value=10)
n_trucks = st.sidebar.slider("Number of Trucks", min_value=1, max_value=5, value=2)
capacity = st.sidebar.slider("Truck Capacity", min_value=100, max_value=500, value=250)

st.markdown("### Bin Fill Levels and Priorities")

# Simulate some data (In production, load actual data)
bins_df = pd.DataFrame({
    'bin_id': np.arange(n_bins),
    'lat': np.random.uniform(17.385, 17.45, n_bins),
    'lon': np.random.uniform(78.45, 78.49, n_bins),
    'fill_level': np.random.randint(40, 100, n_bins),
})

bins_df['priority'] = bins_df['fill_level'] + np.random.randn(n_bins) * 5

st.dataframe(bins_df)

st.markdown("### Truck Routes Visualization")

# Show map
with open("truck_routes.html", "r", encoding="utf-8") as f:
    html_data = f.read()

st.components.v1.html(html_data, height=600, scrolling=True)

st.markdown("""
---
Made with ❤️ using Streamlit, Q-learning, and KNN.
Contact for real-time bin data integration and scheduling optimization.
""")
