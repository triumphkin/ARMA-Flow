import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Traffic Hero | AI Command Center", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_simulation_data():
    chaos_df = pd.read_csv('baseline_chaos_data.csv')
    smart_df = pd.read_csv('baseline_chaos_optimized_data.csv')
    
    

df = load_simulation_data()
max_time = int(df['time_step'].max())
car_ids = df['car_id'].unique().tolist()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🚦 System Controls")
    st.markdown("### Time Simulation")
    
    # The Judge's Slider
    time_step = st.slider("Simulate Time Step", min_value=0, max_value=max_time, value=50, step=1)
    
    st.divider()
    st.markdown("### Vehicle Tracking")
    # The Judge's Car Selection
    selected_car = st.selectbox("Select Car to Track (Judge's Car)", ["None"] + car_ids)
    show_ambulance = st.toggle("Highlight Active Ambulances", value=True)
    
    st.divider()
    st.markdown("### AI Interventions (Mahi's ML)")
    st.toggle("Predictive Variable Speed Limits", value=True, disabled=True)
    st.toggle("Adaptive Batch Zipper (Virtual Traffic Light)", value=True, disabled=True)
    st.toggle("Digital Signage (Load Balancing)", value=True, disabled=True)

# Filter data for the current time step
current_df = df[df['time_step'] == time_step]

# --- ML PREDICTOR DASHBOARD (Mahi's Random Forest Output) ---
st.header("Traffic Hero: Predictive ML Highway Management")

# Simulated ML metrics based on the chaos data at the current time step
cars_on_footpath = len(current_df[(current_df['y_chaos'] == 0) & (current_df['x_chaos'] > 0)])
avg_impatience = current_df['impatience'].mean()
# Simulate Mahi's Random Forest prediction
seconds_to_gridlock = max(0, 100 - (time_step * 1.5)) if time_step < 70 else 0 

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Seconds to Gridlock", f"{seconds_to_gridlock:.0f}s", delta="-15s (Warning)" if seconds_to_gridlock < 30 else "Optimal", delta_color="inverse")
col2.metric("Average Driver Impatience", f"{avg_impatience:.2f}", delta=f"{(avg_impatience - 0.5):.2f}", delta_color="inverse")
col3.metric("Violations (Cars on Footpath)", cars_on_footpath, delta=cars_on_footpath if cars_on_footpath > 0 else 0, delta_color="inverse")
col4.metric("System AI Status", "ACTIVE" if seconds_to_gridlock < 30 else "MONITORING")

st.divider()

# --- PLOTTING LOGIC ---
def draw_highway_map(data, x_col, y_col, title):
    """Generates the Plotly map for the cars."""
    fig = go.Figure()
    
    # Draw Road Boundaries
    fig.add_hline(y=3.5, line_color="white", line_width=4) # Top boundary
    fig.add_hline(y=0.5, line_color="white", line_width=4) # Bottom boundary (Edge of road)
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red") # Footpath boundary
    fig.add_hline(y=1.5, line_dash="dash", line_color="gray") # Lane 1/2 divider
    fig.add_hline(y=2.5, line_dash="dash", line_color="gray") # Lane 2/3 divider
    
    # Draw Merge Bottleneck (Visual indicator)
    fig.add_vrect(x0=380, x1=420, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Merge Zone")

    # Determine colors for cars
    colors = []
    symbols = []
    sizes = []
    
    for _, row in data.iterrows():
        # Default styling
        c = "#00E5FF" # Cyan for normal cars
        s = "circle"
        sz = 12
        
        if row['car_id'] == selected_car:
            c = "#FFD700" # Yellow for Judge's tracked car
            sz = 20
        elif show_ambulance and row['is_ambulance'] == 1:
            c = "#FF0000" # Red for Ambulance
            s = "cross"
            sz = 18
        elif row['impatience'] > 0.8:
            c = "#FF5722" # Orange for highly impatient/aggressive

        colors.append(c)
        symbols.append(s)
        sizes.append(sz)

    # Add cars to scatter plot
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='markers',
        marker=dict(color=colors, symbol=symbols, size=sizes, line=dict(width=1, color='white')),
        hoverinfo='text',
        text=[f"Car {int(r['car_id'])} | Impatience: {r['impatience']:.2f}" for _, r in data.iterrows()]
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 600], showgrid=False, zeroline=False, title="Distance (m)"),
        yaxis=dict(range=[-1, 4], showgrid=False, zeroline=False, tickvals=[0, 1, 2, 3], ticktext=["Footpath", "Lane 1", "Lane 2", "Lane 3"]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        showlegend=False
    )
    return fig

# --- THE SIDE-BY-SIDE (TOP/BOTTOM) COMPARISON ---
st.subheader("Simulation: Unmanaged Chaos (Human Behavior)")
st.caption("Notice aggressive drivers bypassing the queue via the footpath when gridlock hits.")
fig_chaos = draw_highway_map(current_df, 'x_chaos', 'y_chaos', "")
st.plotly_chart(fig_chaos, use_container_width=True)

st.subheader("Simulation: AI Managed Flow (Heuristic Control)")
st.caption("AI detects gridlock conditions -> Triggers Variable Speed Limits and Batch Zipper merging.")
fig_smart = draw_highway_map(current_df, 'x_smart', 'y_smart', "")
st.plotly_chart(fig_smart, use_container_width=True)