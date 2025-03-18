import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("dataset_updated.csv")
    return data

df = load_data()

# Sidebar filters
st.sidebar.header("Filter Data")
selected_route = st.sidebar.selectbox("Select Route", df["Route"].unique())
filtered_df = df[df["Route"] == selected_route]

# App Title
st.title("GSRTC Data Dashboard")

# Key Metrics
st.subheader("Key Insights")
st.metric(label="Average Occupancy Rate", value=f"{filtered_df['Occupancy_Rate'].mean():.2f}%")
st.metric(label="Average Fuel Consumption", value=f"{filtered_df['Fuel_Consumption'].mean():.2f} Liters")
st.metric(label="Average Delay", value=f"{filtered_df['Delay_Minutes'].mean():.2f} min")

# Visualization
st.subheader("Route-wise Occupancy Distribution")
fig_occupancy = px.histogram(filtered_df, x="Occupancy_Rate", nbins=20, title="Occupancy Rate Distribution")
st.plotly_chart(fig_occupancy)

st.subheader("Fuel Consumption Analysis")
fig_fuel = px.line(filtered_df, x="Trip_Date", y="Fuel_Consumption", title="Fuel Consumption Over Time")
st.plotly_chart(fig_fuel)

st.subheader("Delay Trends")
fig_delay = px.box(filtered_df, y="Delay_Minutes", title="Delay Distribution per Route")
st.plotly_chart(fig_delay)

st.subheader("Revenue Insights")
fig_revenue = px.bar(filtered_df, x="Route", y="Revenue", title="Revenue per Route")
st.plotly_chart(fig_revenue)

# Demand Forecasting Placeholder
st.subheader("Demand Forecasting")
st.write("Coming soon: ML-based predictions for passenger demand!")
