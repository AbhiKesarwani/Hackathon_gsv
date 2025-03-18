import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset with caching to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv("dataset_updated.csv")  # Ensure the correct file is used
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names
    return df

df = load_data()

# Debugging: Print column names to ensure "Route" exists
st.write("Columns in Dataset:", df.columns)

# Ensure "Route" column exists
if "Route" in df.columns:
    selected_route = st.sidebar.selectbox("Select Route", df["Route"].unique())
else:
    st.sidebar.error("Column 'Route' not found in dataset")

# Filter data based on selected route
filtered_df = df[df["Route"] == selected_route]

# KPI Cards
st.subheader(f"📊 Key Metrics for {selected_route}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Average Fuel Consumption", value=f"{filtered_df['Fuel_Consumption'].mean():.2f} Liters")

with col2:
    st.metric(label="Average Delay", value=f"{filtered_df['Delay_Minutes'].mean():.2f} min")

with col3:
    st.metric(label="Revenue per Trip", value=f"₹{(filtered_df['Ticket_Price'] * filtered_df['Seats_Booked']).mean():.2f}")

# Visualization: Fuel Consumption Trend
st.subheader("⛽ Fuel Consumption Over Time")
if "Date" in df.columns:
    fig_fuel = px.line(filtered_df, x="Date", y="Fuel_Consumption", title="Fuel Consumption Trend")
    st.plotly_chart(fig_fuel)
else:
    st.warning("Date column not found in dataset. Cannot plot fuel trend.")

# Visualization: Ticket Price Distribution
st.subheader("💰 Ticket Price Distribution")
fig_price = px.histogram(filtered_df, x="Ticket_Price", title="Distribution of Ticket Prices", nbins=20)
st.plotly_chart(fig_price)

# Visualization: Delay Analysis
st.subheader("⏳ Average Delay by Route")
fig_delay = px.bar(df, x="Route", y="Delay_Minutes", title="Average Delay per Route", color="Delay_Minutes")
st.plotly_chart(fig_delay)

# Utilization Rate Calculation
if "Seats_Booked" in df.columns and "Total_Seats_Available" in df.columns:
    df["Utilization_Rate"] = (df["Seats_Booked"] / df["Total_Seats_Available"]) * 100
    df["Occupancy_Category"] = df["Utilization_Rate"].apply(lambda x: "Low" if x < 40 else "Moderate" if x < 70 else "High")
    
    # Donut Chart: Bus Occupancy
    st.subheader("🚍 Bus Utilization Category")
    fig_occupancy = px.pie(df, names="Occupancy_Category", title="Occupancy Levels", hole=0.4)
    st.plotly_chart(fig_occupancy)
else:
    st.warning("Seats_Booked or Total_Seats_Available columns missing.")

# Footer
st.markdown("📌 **GSRTC Data Analysis Dashboard** - Powered by Streamlit & Plotly")
