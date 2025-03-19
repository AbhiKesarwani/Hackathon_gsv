import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load dataset efficiently
DATA_PATH = "updated_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True)
    
    # Remove Unnamed Columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if df.empty:
        st.error("❌ Dataset is empty. Please upload valid data.")
        st.stop()
else:
    st.error("❌ Dataset not found. Please upload a valid file.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("bus_icon.png", width=100)
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📂 Dataset", "📊 EDA", "📈 Demand Forecasting", "📤 Upload Data"])

# Home Page
if page == "🏠 Home":
    st.title("🚌 GSRTC Data-Driven Insights Dashboard")
    st.image("dashboard_banner.png", use_column_width=True)
    st.write("""
    ## 🚀 Problem Statement
    GSRTC currently operates with fragmented data across multiple departments, leading to inefficiencies in decision-making, inaccurate demand forecasting, and suboptimal resource utilization.

    ## ⚠️ Challenges
    - ❌ Lack of a centralized data platform
    - 📉 Inefficient scheduling and resource allocation
    - 🔮 Difficulty in predicting passenger demand
    - ⛽ High operational costs due to fuel inefficiencies

    ## 🎯 Expected Outcomes
    ✅ A unified data platform integrating multiple data sources  
    ✅ Improved decision-making with real-time insights  
    ✅ Accurate demand forecasting to optimize scheduling  
    ✅ Enhanced customer satisfaction with better service planning  
    """)

# Dataset Page
elif page == "📂 Dataset":
    st.title("📊 Dataset Overview")
    st.write("""
    ### 🔍 About the Dataset
    - **Comprehensive Data** covering all aspects of GSRTC bus operations.
    - **Essential for Demand Forecasting, Revenue Analysis, and Resource Planning.**
    """)

    # Display dataset
    st.subheader("🗂 Sample Data:")
    st.dataframe(df, height=400, width=1000)
    
    # Allow users to download cleaned dataset
    st.download_button("⬇️ Download Dataset", df.to_csv(index=False), "dataset.csv", "text/csv")

    # Display dataset columns for reference
    st.subheader("📝 Column Descriptions")
    st.write("""
    - **Date:** Date of the trip.  
    - **Day_of_Week:** Day corresponding to the trip's date.  
    - **Time_Slot:** Time period of the journey (Morning, Afternoon, Night).  
    - **Route_ID:** Unique identifier for the bus route.  
    - **Route_Name:** Name of the route (start to end location).  
    - **Bus_ID:** Unique identifier for the bus.  
    - **Driver_ID:** Unique identifier for the driver.  
    - **Bus_Type:** Type of bus (AC, Non-AC, Sleeper, etc.).  
    - **Total_Seats:** Number of seats available in the bus.  
    - **Seats_Booked:** Number of seats booked for the trip.  
    - **Occupancy_Rate (%):** Percentage of seats occupied.  
    - **Revenue_Per_Trip:** Total revenue generated from ticket sales.  
    - **Fuel_Consumption_Liters:** Fuel consumed during the trip in liters.  
    - **Fuel_Efficiency_KMPL:** Fuel efficiency of the bus in kilometers per liter.  
    - **Weather:** Weather conditions during the journey.  
    - **Festival_Season:** Indicates if the trip falls during a festival season.  
    - **Holiday:** Indicates if the trip falls on a public holiday.  
    """)

# EDA Portal
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.write("📊 Below are key insights from our Power BI analysis.")
    
    st.image("gsrtc_dashboard.png", caption="📊 GSRTC Power BI Dashboard", use_column_width=True)
    st.image("avg_fuel_consumption.png", caption="⛽ Average Fuel Consumption", use_column_width=True)
    st.image("avg_profit_per_trip_by_route.png", caption="💰 Avg Profit Per Trip by Route", use_column_width=True)

# Demand Forecasting Portal
elif page == "📈 Demand Forecasting":
    st.title("📈 Passenger Demand Forecasting")
    st.write("Using **Exponential Smoothing** for fast and efficient demand prediction.")

    df_daily = df.groupby('Date').agg({'Seats_Booked': 'sum'}).reset_index().sort_values(by='Date')

    df_daily['Seats_Booked'] = pd.to_numeric(df_daily['Seats_Booked'], errors='coerce')
    df_daily = df_daily.dropna(subset=['Seats_Booked'])

    if len(df_daily) < 10:
        st.error("❌ Not enough data points for forecasting. Please upload more historical data.")
    else:
        train_size = int(len(df_daily) * 0.8)
        train, test = df_daily[:train_size], df_daily[train_size:]

        try:
            model = ExponentialSmoothing(train['Seats_Booked'], trend="add", seasonal="add", seasonal_periods=7)
            model_fit = model.fit()
        except ValueError as e:
            st.error(f"❌ Model Training Error: {e}")
            st.stop()

        test_forecast = model_fit.forecast(len(test))
        rmse = np.sqrt(mean_squared_error(test['Seats_Booked'], test_forecast))

        future_steps = st.slider("📅 Select Forecast Duration (Days)", min_value=7, max_value=90, value=30)
        future_forecast = model_fit.forecast(future_steps)
        future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_daily['Date'], df_daily['Seats_Booked'], label="Actual Data", color="blue")
        ax.plot(df_daily['Date'][len(train):], test_forecast, label="Test Forecast", linestyle="dashed", color="green")
        ax.plot(future_dates, future_forecast, label=f"Next {future_steps} Days Forecast", linestyle="dashed", color="red")
        ax.legend()
        st.pyplot(fig)

# Data Upload Portal
elif page == "📤 Upload Data":
    st.title("📤 Upload New Data")
    
    uploaded_file = st.file_uploader("📁 Choose a CSV file", type="csv")

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

        if set(new_data.columns) != set(df.columns):
            st.error("❌ Column mismatch! Ensure the uploaded file has the same structure as the dataset.")
            st.write("### ✅ Expected Columns:", list(df.columns))
            st.write("### 🔄 Uploaded Columns:", list(new_data.columns))
        else:
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            st.success("✅ Data successfully uploaded and added to the existing dataset!")

            df = pd.read_csv(DATA_PATH).loc[:, ~df.columns.str.contains('^Unnamed')]

            st.write("### 🔍 Updated Dataset Preview")
            st.dataframe(df.tail(10))
