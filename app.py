import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset efficiently
DATA_PATH = "updated_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    if df.empty:
        st.error("❌ Dataset is empty. Please upload valid data.")
        st.stop()
else:
    st.error("❌ Dataset not found. Please upload a valid file.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Demand Forecasting", "Upload Data"])

# Home Page
if page == "Home":
    st.title("🚌 GSRTC Data-Driven Insights Dashboard")
    st.write("""
    ## 🚀 Problem Statement
    GSRTC currently operates with fragmented data across multiple departments, leading to inefficiencies in decision-making, inaccurate demand forecasting, and suboptimal resource utilization.

    ## ⚠ Challenges
    - Lack of a centralized data platform
    - Inefficient scheduling and resource allocation
    - Difficulty in predicting passenger demand
    - High operational costs due to fuel inefficiencies

    ## 🎯 Expected Outcomes
    - A unified data platform integrating multiple data sources
    - Improved decision-making with real-time insights
    - Accurate demand forecasting to optimize scheduling
    - Enhanced customer satisfaction with better service planning
    """)

# Dataset Page
elif page == "Dataset":
    st.title("📂 Dataset Overview")
    st.write("""
    ### 🔍 About the Dataset
    - *Synthetic Data* designed to simulate real-world GSRTC operations.
    - Includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - Helps in *resource allocation, demand forecasting, and operational efficiency*.
    
    ### 📊 Sample Data:
    """)

    # Remove Unnamed Columns
    df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    st.dataframe(df_cleaned, height=400, width=1000)  # Enables scrolling in both directions
    st.download_button("⬇ Download Dataset", df_cleaned.to_csv(index=False), "dataset.csv", "text/csv")


# EDA Portal
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    st.write("Below are key insights from our Power BI analysis.")

    st.image("gsrtc_dashboard.png", caption="📊 GSRTC Power BI Dashboard")
    st.image("avg_fuel_consumption.png", caption="⛽ Average Fuel Consumption")
    st.image("avg_profit_per_trip_by_route.png", caption="💰 Avg Profit Per Trip by Route")
    st.image("seats_booked_by_destination.png", caption="🛋 Seats Booked by Destination")
    st.image("seats_booked_per_month.png", caption="📆 Seats Booked Per Month")
    st.image("total_trips_by_delay_status.png", caption="⏳ Total Trips by Delay Status")
    st.image("total_trips_by_occupancy.png", caption="🚌 Total Trips by Occupancy")

# Demand Forecasting Portal
elif page == "Demand Forecasting":
    st.title("📈 Passenger Demand Forecasting")
    st.write("Using **Exponential Smoothing** for fast and efficient demand prediction.")
    
    st.image("Screenshot 2025-03-09 131032.png", caption="🚌 Deamnd Forecasting")   

# Data Upload Portal
elif page == "Upload Data":
    st.title("📤 Upload New Data")

    uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read uploaded file
        new_data = pd.read_csv(uploaded_file, index_col=0).loc[:, ~df.columns.str.contains('^Unnamed')]

        # Remove any accidental unnamed columns in the uploaded file
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

        # Ensure column consistency
        expected_columns = set(df.columns)
        uploaded_columns = set(new_data.columns)

        if expected_columns != uploaded_columns:
            st.error("❌ Column mismatch! Ensure the uploaded file has the same structure as the dataset.")
            st.write("### ✅ Expected Columns:", list(expected_columns))
            st.write("### 🔄 Uploaded Columns:", list(uploaded_columns))
        else:
            # Append new data
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            st.success("✅ Data successfully uploaded!")

            # Reload dataset properly
            df = pd.read_csv(DATA_PATH, index_col=0).loc[:, ~df.columns.str.contains('^Unnamed')]

            st.write("### 🔍 Updated Dataset Preview")
            st.dataframe(df.tail(10))

