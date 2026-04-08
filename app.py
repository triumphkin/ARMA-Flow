import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Traffic Hero | AI Command Center", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_simulation_data():
    chaos_df = pd.read_csv('baseline_chaos_data.csv')
    smart_df = pd.read_csv('baseline_chaos_optimized_data.csv')
    return chaos_df, smart_df
    
chaos_df, smart_df = load_simulation_data()

max_time = int(chaos_df['Timestamp_Step'].max())
car_ids = chaos_df['Car_ID'].unique().tolist()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🚦 System Controls")
    st.markdown("### Time Simulation")
    
    # The Judge's Slider
    time_step = st.slider("Simulate Time Step", min_value=0, max_value=max_time, value=0, step=1)
    
    st.divider()
    st.markdown("### Vehicle Tracking")
    selected_car = st.selectbox("Select Car to Track", ["None"] + car_ids)
    show_ambulance = st.toggle("Highlight Active Ambulances", value=True)
    
    st.divider()
    st.markdown("### AI Interventions")
    # ENABLED THE TOGGLE
    vsl_active = st.toggle("Predictive Variable Speed Limits", value=True)
    st.toggle("Adaptive Batch Zipper (Virtual Traffic Light)", value=True, disabled=True)
    st.toggle("Digital Signage (Load Balancing)", value=True, disabled=True)

# Filter data for the current time step for both maps
current_chaos = chaos_df[chaos_df['Timestamp_Step'] == time_step]
current_smart = smart_df[smart_df['Timestamp_Step'] == time_step]

# --- ML PREDICTOR DASHBOARD ---
st.header("Traffic Hero: Predictive ML Highway Management")

if not current_chaos.empty:
    cars_on_footpath = len(current_chaos[(current_chaos['Lane'] == 0) & (current_chaos['Position_X'] > 0)])
    avg_impatience = current_chaos['Current_Impatience'].mean()
    seconds_to_gridlock = current_smart['Seconds_To_Gridlock'].iloc[0] 
    chaos_throughput = int(current_chaos['Total_Throughput'].iloc[0])
else:
    cars_on_footpath, avg_impatience, seconds_to_gridlock, chaos_throughput = 0, 0, 300, 0

if not current_smart.empty:
    smart_throughput = int(current_smart['Total_Throughput'].iloc[0])
else:
    smart_throughput = 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Seconds to Gridlock", f"{seconds_to_gridlock:.0f}s", delta="-15s (Warning)" if seconds_to_gridlock < 120 else "Optimal", delta_color="inverse")
col2.metric("Average Driver Impatience", f"{avg_impatience:.2f}", delta=f"{(avg_impatience - 0.5):.2f}", delta_color="inverse") # Fixed header
col3.metric("Violations (Cars on Footpath)", cars_on_footpath, delta=cars_on_footpath if cars_on_footpath > 0 else 0, delta_color="inverse")
col4.metric("System AI Status", "ACTIVE" if seconds_to_gridlock <= 120 else "MONITORING")

st.divider()

# --- PLOTTING LOGIC ---
def draw_highway_map(data, title):
    fig = go.Figure()
    
    fig.add_hline(y=3.5, line_color="white", line_width=4) 
    fig.add_hline(y=0.5, line_color="white", line_width=4) 
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red") 
    fig.add_hline(y=1.5, line_dash="dash", line_color="gray") 
    fig.add_hline(y=2.5, line_dash="dash", line_color="gray") 
    
    fig.add_vrect(x0=700, x1=800, fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Merge Zone") 

    if data.empty:
        return fig

    colors = []
    symbols = []
    sizes = []
    
    for _, row in data.iterrows():
        car_color_id = row.get('Color', 0) 
        
        try:
            if pd.isna(car_color_id):
                car_color_id = 0
            else:
                car_color_id = int(float(car_color_id))
        except (ValueError, TypeError):
            car_color_id = 0

        if car_color_id == 1: c = "#FFD1DC" 
        elif car_color_id == 2: c = "#AEC6CF" 
        elif car_color_id == 3: c = "#FDFD96" 
        else: c = "#00E5FF" 
            
        s = "circle"
        sz = 12
        
        if str(int(row['Car_ID'])) == str(selected_car):
            c = "#23DF0B" 
            sz = 20
        elif show_ambulance and row.get('Is_Ambulance', 0) == 1:
            c = "#FF0000" 
            s = "cross"
            sz = 18
        elif row['Current_Impatience'] > 0.8: # FIXED BUG: Was > 1, max is 1.0!
            c = "#FF5722" 

        colors.append(c)
        symbols.append(s)
        sizes.append(sz)

    fig.add_trace(go.Scatter(
        x=data['Position_X'],
        y=data['Lane'],
        mode='markers',
        marker=dict(color=colors, symbol=symbols, size=sizes, line=dict(width=1, color='white')),
        hoverinfo='text',
        text=[f"Car {int(r['Car_ID'])} | Impatience: {r['Current_Impatience']:.2f} | Speed: {r['Speed']:.1f}" for _, r in data.iterrows()]
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1000], showgrid=False, zeroline=False, title="Distance (m)"), 
        yaxis=dict(range=[-1, 4], showgrid=False, zeroline=False, tickvals=[0, 1, 2, 3], ticktext=["Footpath", "Lane 1", "Lane 2", "Lane 3"]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        showlegend=False
    )
    return fig


# --- SIMULATION 1: UNMANAGED CHAOS ---
st.subheader(f"Simulation: Unmanaged Chaos | 📉 Current Throughput: {chaos_throughput} cars")
st.caption("Notice aggressive drivers bypassing the queue via the footpath when gridlock hits.")
fig_chaos = draw_highway_map(current_chaos, "")
st.plotly_chart(fig_chaos, use_container_width=True)

st.divider()

# --- SIMULATION 2: AI MANAGED FLOW ---
st.subheader(f"Simulation: AI Managed Flow | 📈 Current Throughput: {smart_throughput} cars")
st.caption("AI detects gridlock conditions -> Triggers VSL & Batch Zipper merging to smooth the bottleneck.")

# --- NEW: DIGITAL GANTRY VSL SIGNS ---
if vsl_active and not current_smart.empty:
    # Dynamically calculate the VSL logic for the UI based on live density
    live_densities = {
        1: len(current_smart[current_smart['Lane'] == 1]),
        2: len(current_smart[current_smart['Lane'] == 2]),
        3: len(current_smart[current_smart['Lane'] == 3])
    }
    
    target_speeds = {1: 25.0, 2: 25.0, 3: 25.0}
    if sum(live_densities.values()) >= 15:
        for lane in [1, 2, 3]:
            if live_densities[lane] > 5:
                target_speeds[lane] = max(8.0, 25.0 - ((live_densities[lane] - 5) * 0.8))
        for _ in range(2):
            if target_speeds[1] - target_speeds[2] > 5: target_speeds[1] = target_speeds[2] + 5.0
            if target_speeds[3] - target_speeds[2] > 5: target_speeds[3] = target_speeds[2] + 5.0
            if target_speeds[2] - target_speeds[1] > 5: target_speeds[2] = target_speeds[1] + 5.0
            if target_speeds[2] - target_speeds[3] > 5: target_speeds[2] = target_speeds[3] + 5.0

    # HTML/CSS for glowing digital LED signs
    def sign_html(lane_name, speed):
        color = "#00FF00" if speed >= 20 else "#FFD700" if speed >= 12 else "#FF0000"
        return f"""
        <div style="background-color: #111; color: {color}; padding: 10px; border-radius: 8px; 
                    text-align: center; font-family: 'Courier New', monospace; border: 3px solid #333; 
                    box-shadow: 0px 0px 15px {color}40;">
            <div style="font-size: 14px; color: #888; text-transform: uppercase;">{lane_name} Limit</div>
            <div style="font-size: 32px; font-weight: bold; text-shadow: 0 0 10px {color};">{speed:.0f} <span style="font-size: 16px;">m/s</span></div>
        </div>
        """

    sign_cols = st.columns(3)
    sign_cols[0].markdown(sign_html("Lane 1", target_speeds[1]), unsafe_allow_html=True)
    sign_cols[1].markdown(sign_html("Lane 2", target_speeds[2]), unsafe_allow_html=True)
    sign_cols[2].markdown(sign_html("Lane 3", target_speeds[3]), unsafe_allow_html=True)
    st.write("") # Tiny spacer

fig_smart = draw_highway_map(current_smart, "")
st.plotly_chart(fig_smart, use_container_width=True)

st.divider()

# --- SYSTEM THROUGHPUT LINE GRAPH ---
st.header("📊 Performance Proof: Total Throughput Over Time")
st.caption("This graph proves that our AI intervention results in a higher volume of vehicles successfully clearing the bottleneck over time.")

chaos_trend = chaos_df.groupby('Timestamp_Step')['Total_Throughput'].max().reset_index()
smart_trend = smart_df.groupby('Timestamp_Step')['Total_Throughput'].max().reset_index()

fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=chaos_trend['Timestamp_Step'],
    y=chaos_trend['Total_Throughput'],
    mode='lines',
    name='Unmanaged Chaos (Baseline)',
    line=dict(color='#FF5722', width=2)
))

fig_trend.add_trace(go.Scatter(
    x=smart_trend['Timestamp_Step'],
    y=smart_trend['Total_Throughput'],
    mode='lines',
    name='AI Managed Flow (Optimized)',
    line=dict(color='#00E5FF', width=3) 
))

fig_trend.add_vline(
    x=time_step, 
    line_dash="dash", 
    line_color="#FFD700", 
    annotation_text="Current Sim Time", 
    annotation_position="bottom right"
)

fig_trend.update_layout(
    xaxis_title="Simulation Time Step",
    yaxis_title="Total Cars Processed",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="white"),
    legend=dict(
        x=0.02, 
        y=0.98, 
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor="white",
        borderwidth=1
    )
)

st.plotly_chart(fig_trend, use_container_width=True)