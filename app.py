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
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], dayfirst=True, usecols=[
        'Date', 'Route_ID', 'Seats_Booked', 'Fuel_Consumption_Liters', 'Total_Seats', 'Ticket_Price'
    ])
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

    ## ⚠️ Challenges
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
    - **Synthetic Data** designed to simulate real-world GSRTC operations.
    - Includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - Helps in **resource allocation, demand forecasting, and operational efficiency**.

    ### 📊 Sample Data:
    """)

    st.dataframe(df, height=400, width=1000)  # Enables scrolling in both directions
    st.download_button("⬇️ Download Dataset", df.to_csv(index=False), "dataset.csv", "text/csv")

# EDA Portal
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")
    st.write("Below are key insights from our Power BI analysis.")

    st.image("gsrtc_dashboard.png", caption="📊 GSRTC Power BI Dashboard")
    st.image("avg_fuel_consumption.png", caption="⛽ Average Fuel Consumption")
    st.image("avg_profit_per_trip_by_route.png", caption="💰 Avg Profit Per Trip by Route")
    st.image("seats_booked_by_destination.png", caption="🛋️ Seats Booked by Destination")
    st.image("seats_booked_per_month.png", caption="📆 Seats Booked Per Month")
    st.image("total_trips_by_delay_status.png", caption="⏳ Total Trips by Delay Status")
    st.image("total_trips_by_occupancy.png", caption="🚌 Total Trips by Occupancy")

# Demand Forecasting Portal
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Demand Forecasting Portal
elif page == "Demand Forecasting":
    st.title("📊 Passenger Demand Forecasting (Optimized)")
    st.write("Using **Exponential Smoothing** for fast and efficient demand prediction.")

    # Preprocessing: Aggregate by Date
    df_daily = df.groupby('Date').agg({'Seats_Booked': 'sum'}).reset_index().sort_values(by='Date')

    # **Fix Missing or Invalid Values**
    df_daily['Seats_Booked'] = pd.to_numeric(df_daily['Seats_Booked'], errors='coerce')  # Ensure numeric
    df_daily = df_daily.dropna(subset=['Seats_Booked'])  # Drop rows with NaN

    # **Ensure Sufficient Data**
    if len(df_daily) < 10:  # Check if dataset has enough points
        st.error("Not enough data points for forecasting. Please upload more historical data.")
    else:
        # Train-Test Split
        train_size = int(len(df_daily) * 0.8)
        train, test = df_daily[:train_size], df_daily[train_size:]

        # **Fit Exponential Smoothing Model**
        try:
            model = ExponentialSmoothing(train['Seats_Booked'], trend="add", seasonal="add", seasonal_periods=7)
            model_fit = model.fit()
        except ValueError as e:
            st.error(f"Model Training Error: {e}")
            st.stop()

        # **Forecast Test Data**
        test_forecast = model_fit.forecast(len(test))

        # **Evaluate Model Performance**
        rmse = np.sqrt(mean_squared_error(test['Seats_Booked'], test_forecast))
        st.write(f"📊 **Model RMSE:** {rmse:.2f} (Lower is better)")

        # **User Input for Future Forecasting**
        future_steps = st.slider("📅 Select Forecast Duration (Days)", min_value=7, max_value=90, value=30)

        # **Future Demand Forecast**
        future_forecast = model_fit.forecast(future_steps)

        # **Create Future Dates**
        future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

        # **Visualization**
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_daily['Date'], df_daily['Seats_Booked'], label="Actual Data", color="blue")
        ax.plot(df_daily['Date'][len(train):], test_forecast, label="Test Forecast", linestyle="dashed", color="green")
        ax.plot(future_dates, future_forecast, label=f"Next {future_steps} Days Forecast", linestyle="dashed", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Seats Booked")
        ax.set_title("📈 Exponential Smoothing - Demand Forecasting")
        ax.legend()
        ax.grid()

        st.pyplot(fig)

        # **Display Insights**
        st.subheader("🔎 Key Insights")
        peak_demand = future_forecast.max()
        low_demand = future_forecast.min()

        st.write(f"✔️ **Highest Predicted Demand:** {peak_demand:.0f} seats")
        st.write(f"⚠️ **Lowest Predicted Demand:** {low_demand:.0f} seats")
        st.write("🚀 **Business Impact:** Helps optimize fleet allocation, fuel efficiency, and revenue planning.")

# Data Upload Portal
elif page == "Upload Data":
    st.title("📤 Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        if set(new_data.columns) == set(df.columns):
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            st.success("✅ Data successfully uploaded and merged!")
        else:
            st.error("❌ Column mismatch! Ensure the uploaded file has the same structure as the dataset.")
