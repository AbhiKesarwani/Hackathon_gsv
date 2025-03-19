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
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # ✅ Remove "Unnamed" columns
    if df.empty:
        st.error("❌ Dataset is empty. Please upload valid data.")
        st.stop()
else:
    st.error("❌ Dataset not found. Please upload a valid file.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Demand Forecasting", "Upload Data"])

# Add a red background note
warning_html = """
    <div style="background-color:#ffcccc; padding:10px; border-radius:5px; text-align:center;">
        <h4 style="color:red;">🚧 This section is under process. Some functionalities may not work as expected. 🚧</h4>
    </div>
"""

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
    st.markdown(warning_html, unsafe_allow_html=True)  # Display the red warning box
    st.write("This section will provide demand forecasting using predictive models.")
    st.title("📈 Passenger Demand Forecasting")
    st.write("Using **SARIMA** for fast and efficient demand prediction.")
    
    st.image("Demand Forecast.png", caption="🚌 Demand Forecasting")   

# Data Upload Portal
elif page == "Upload Data":
    st.title("📤 Upload New Data")

    uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read uploaded file and clean it
        new_data = pd.read_csv(uploaded_file)
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]  # ✅ Remove "Unnamed" columns

        expected_columns = list(df.columns)
        uploaded_columns = list(new_data.columns)

        # **Check for extra columns in uploaded file**
        extra_columns = set(uploaded_columns) - set(expected_columns)
        if extra_columns:
            st.error("❌ Column mismatch! The uploaded file contains **unexpected columns**.")
            st.write("### ❌ Unexpected Columns in Upload:", sorted(list(extra_columns)))
            st.stop()

        # **Check for missing columns in uploaded file**
        missing_columns = set(expected_columns) - set(uploaded_columns)
        if missing_columns:
            st.warning("⚠️ Some columns are missing in the uploaded file. They will be filled with NaN.")
            for col in missing_columns:
                new_data[col] = pd.NA  # Fill missing columns with null

        # **Ensure the column order is exactly the same**
        new_data = new_data[expected_columns]

        # **Append new data**
        new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
        st.success("✅ Data successfully uploaded!")

        # Reload the full dataset
        df = pd.read_csv(DATA_PATH)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # ✅ Ensure "Unnamed" columns are removed

        # **Download Button for Full Data**
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Dataset", data=csv, file_name="full_dataset.csv", mime="text/csv")

        st.write("### 🔍 Updated Dataset Preview")
        st.dataframe(df)

 
