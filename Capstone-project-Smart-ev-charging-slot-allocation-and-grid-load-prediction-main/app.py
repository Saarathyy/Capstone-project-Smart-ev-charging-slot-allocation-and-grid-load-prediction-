import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(page_title="EV Grid Intelligence", layout="wide")
st_autorefresh(interval=5000, key="refresh")

st.title("⚡ Smart EV Charging & Grid Intelligence Dashboard")

# ---------------- LOAD RESOURCES ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("energy_demand_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please upload 'energy_demand_model.pkl'.")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clean_hourly_data.csv")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload 'clean_hourly_data.csv'.")
        return pd.DataFrame()

model = load_model()
hourly = load_data()

if model is None or hourly.empty:
    st.stop()

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("⚙ Forecast Control Panel")

selected_date = st.sidebar.date_input("Select Forecast Date", datetime.date.today())
now = datetime.datetime.now()
selected_hour = st.sidebar.slider("Select Hour of Day", 0, 23, now.hour)
selected_station = st.sidebar.selectbox("Select Charging Station", ["Station A", "Station B", "Station C"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Simulating: {selected_date} @ {selected_hour}:00")

# ---------------- DATA PREP ----------------
# Mock multi-station data
if "Station A" not in hourly.columns:
    hourly["Station A"] = hourly["energy_demand"]
    hourly["Station B"] = hourly["energy_demand"] * 1.12
    hourly["Station C"] = hourly["energy_demand"] * 0.88

station_series = hourly[selected_station]

# Get feature columns
if hasattr(model, "feature_names_in_"):
    feature_columns = model.feature_names_in_
else:
    feature_columns = [
        "hour", "day_of_week", "month", "is_weekend", "prev_hour_demand",
        "rolling_mean_3", "rolling_mean_6", "day_of_year", "prev_2_hour_demand",
        "prev_3_hour_demand", "rolling_std_3", "week_of_year"
    ]

# ---------------- PREDICTION LOGIC ----------------
def get_features_for_datetime(target_date, target_hour, historical_df):
    dt = datetime.datetime.combine(target_date, datetime.time(target_hour))
    day_of_week = dt.weekday()
    last_known = historical_df.iloc[-1]
    
    input_data = {
        "hour": target_hour,
        "day_of_week": day_of_week,
        "month": dt.month,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "day_of_year": dt.timetuple().tm_yday,
        "week_of_year": dt.isocalendar()[1],
        "prev_hour_demand": last_known.get("energy_demand", 0),
        "prev_2_hour_demand": historical_df.iloc[-2]["energy_demand"] if len(historical_df) > 1 else 0,
        "prev_3_hour_demand": historical_df.iloc[-3]["energy_demand"] if len(historical_df) > 2 else 0,
        "rolling_mean_3": last_known.get("rolling_mean_3", 0),
        "rolling_mean_6": last_known.get("rolling_mean_6", 0),
        "rolling_std_3": last_known.get("rolling_std_3", 0),
    }
    df = pd.DataFrame([input_data])
    valid_cols = [c for c in feature_columns if c in df.columns]
    return df[valid_cols]

# KPI Calc
input_df = get_features_for_datetime(selected_date, selected_hour, hourly)
try:
    predicted_demand = model.predict(input_df)[0]
except:
    predicted_demand = 0

STATION_CAPACITY = 100.0
SLOT_POWER = 7.0
available_power = STATION_CAPACITY - predicted_demand
slots_open = int(max(0, available_power) // SLOT_POWER)
utilization_pct = (predicted_demand / STATION_CAPACITY) * 100

# ---------------- DASHBOARD UI ----------------
c1, c2, c3 = st.columns(3)
c1.metric(f"Predicted Load ({selected_hour}:00)", f"{predicted_demand:.2f} kW")
c2.metric("Available Smart Slots", f"{slots_open} / 14")
c3.metric("Grid Utilization", f"{utilization_pct:.1f}%", delta_color="inverse")

st.markdown("---")

# Charts
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🔮 24-Hour Forecast")
    forecast_vals = []
    for h in range(24):
        f_df = get_features_for_datetime(selected_date, h, hourly)
        try:
            pred = model.predict(f_df)[0]
        except:
            pred = 0
        forecast_vals.append(pred)
        
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=list(range(24)), y=forecast_vals, mode='lines+markers', name='Predicted Load', line=dict(color='#AB63FA', width=3), fill='tozeroy'))
    fig_forecast.add_trace(go.Scatter(x=[selected_hour], y=[predicted_demand], mode='markers', marker=dict(color='red', size=12), name='Selected Time'))
    fig_forecast.update_layout(title=f"Load Profile: {selected_date}", xaxis_title="Hour", yaxis_title="kW", template="plotly_dark", height=400)
    st.plotly_chart(fig_forecast, use_container_width=True)

with col_right:
    st.subheader("⚡ Grid Status")
    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=utilization_pct, title={'text': "Load %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': [{'range': [0, 60], 'color': "#00CC96"}, {'range': [60, 85], 'color': "#FFA15A"}, {'range': [85, 100], 'color': "#EF553B"}]}))
    fig_gauge.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ---------------- FIXED HEATMAP SECTION ----------------
c_hist, c_heat = st.columns(2)

with c_hist:
    st.subheader(f"📉 Recent History ({selected_station})")
    fig_hist = px.line(station_series.tail(48), title="Last 48 Hours Observed", markers=True)
    fig_hist.update_layout(template="plotly_dark", showlegend=False, xaxis_title="Index", yaxis_title="kW")
    st.plotly_chart(fig_hist, use_container_width=True)

with c_heat:
    st.subheader("🔥 Average Load Heatmap")
    
    if 'day_of_week' in hourly.columns and 'hour' in hourly.columns:
        # 1. Map integers to readable Day Names
        day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        hourly['day_name'] = hourly['day_of_week'].map(day_map)
        
        # 2. Create Pivot Table (Hour x Day)
        pivot = hourly.pivot_table(
            index='hour', 
            columns='day_name', 
            values='energy_demand', 
            aggfunc='mean'
        )
        
        # 3. Enforce Correct Day Order (Mon -> Sun)
        ordered_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Only include days that actually exist in the data
        existing_days = [d for d in ordered_days if d in pivot.columns]
        pivot = pivot[existing_days]
        
        # 4. Plot Heatmap
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="Day of Week", y="Hour of Day", color="Load (kW)"),
            x=existing_days,
            y=pivot.index,
            aspect="auto",
            color_continuous_scale="RdYlBu_r", # Red = High Load (Hot)
            title="Weekly Load Patterns"
        )
        fig_heat.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Insufficient data to generate Heatmap.")

st.caption(f"System Time: {now.strftime('%H:%M:%S')} | Model: GradientBoostingRegressor")