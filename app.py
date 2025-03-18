import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
DATA_PATH = "updated_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Dataset not found!")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Demand Forecasting", "Upload Data"])

# Home Page
if page == "Home":
    st.title("GSRTC Data-Driven Insights Dashboard")
    st.write("""
    ## Problem Statement
    GSRTC currently operates with fragmented data across multiple departments, leading to inefficiencies in decision-making, inaccurate demand forecasting, and suboptimal resource utilization.
    
    ## Challenges
    - Lack of a centralized data platform
    - Inefficient scheduling and resource allocation
    - Difficulty in predicting passenger demand
    - High operational costs due to fuel inefficiencies
    
    ## Expected Outcomes
    - A unified data platform integrating multiple data sources
    - Improved decision-making with real-time insights
    - Accurate demand forecasting to optimize scheduling
    - Enhanced customer satisfaction with better service planning
    """)

# Dataset Page
# Dataset Page
elif page == "Dataset":
    st.title("Dataset Overview")
    st.write("""
    ### About the Dataset
    - This dataset is **synthetically generated** to simulate real-world GSRTC operations.
    - It includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - The dataset is structured to support insights into **resource allocation, demand forecasting, and operational efficiency**.
    
    ### Columns in the Dataset:
    - **Date**: The date of the trip  
    - **Route**: The route name (e.g., "Ahmedabad-Vadodara")  
    - **Initial_Point** & **Final_Point**: Departure and arrival locations  
    - **Bus_Type**: Type of bus (AC/Non-AC)  
    - **Seats_Booked**: Number of seats booked for the trip  
    - **Total_Seats_Available**: Maximum seating capacity  
    - **Fuel_Consumption**: Fuel used per trip (in liters)  
    - **Delay_Minutes**: Delay in minutes for the trip  
    - **Ticket_Price**: Cost of a single ticket  
    - **Distance**: Distance covered by the bus (in km)  

    **You can scroll through the dataset below:**
    """)

    st.dataframe(df, height=400, width=1000)  # Enables scrolling in both directions
    st.download_button("Download Dataset", df.to_csv(index=False), "dataset.csv", "text/csv")

# EDA Portal
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.subheader("Fuel Consumption Analysis")
    fig, ax = plt.subplots()
    sns.lineplot(x=df["Date"], y=df["Fuel_Consumption"], ax=ax)
    st.pyplot(fig)
    
    st.subheader("Occupancy Rate Distribution")
    df['Utilization_Rate'] = (df['Seats_Booked'] / df['Total_Seats_Available']) * 100
    fig, ax = plt.subplots()
    sns.histplot(df["Utilization_Rate"], bins=20, ax=ax)
    st.pyplot(fig)

# Demand Forecasting Portal
if page == "Demand Forecasting":
    st.title("📊 Passenger Demand Forecasting")

    # Data Preprocessing
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')
    df_daily = df.groupby('Date').agg({'Seats_Booked': 'sum'}).reset_index()
    df_daily = df_daily.sort_values(by='Date')

    # ADF Test for Stationarity
    def adf_test(series):
        result = adfuller(series)
        return result[1]  # Return p-value

    # Check Stationarity
    p_value = adf_test(df_daily["Seats_Booked"])
    st.write(f"🔍 **ADF Test p-value:** {p_value:.5f}")
    
    # If Not Stationary, Apply Differencing
    if p_value > 0.05:
        df_daily["Seats_Booked_Diff"] = df_daily["Seats_Booked"].diff().dropna()
        st.write("❌ Data is NOT stationary. Applied differencing.")
    else:
        st.write("✅ Data is stationary. No differencing applied.")

    # Train-Test Split
    train_size = int(len(df_daily) * 0.8)
    train, test = df_daily[:train_size], df_daily[train_size:]

    # Train SARIMA Model
    sarima_model = SARIMAX(train['Seats_Booked'], 
                            order=(1, 1, 1),  
                            seasonal_order=(1, 1, 1, 60),  
                            enforce_stationarity=False, 
                            enforce_invertibility=False)

    sarima_result = sarima_model.fit()

    # Forecast for Test Data
    test_forecast = sarima_result.get_forecast(steps=len(test))
    test_forecast_mean = test_forecast.predicted_mean

    # Model Evaluation
    rmse = np.sqrt(mean_squared_error(test['Seats_Booked'], test_forecast_mean))
    st.write(f"📊 **SARIMA Model RMSE:** {rmse:.2f}")

    # Forecast Future Demand for Next 30 Days
    future_steps = 30
    sarima_forecast_next_month = sarima_result.forecast(steps=future_steps)

    # Create Future Dates
    future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

    # Plot Demand Forecasting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_daily['Date'], df_daily['Seats_Booked'], label="Actual Data", color="blue")
    ax.plot(df_daily['Date'][:len(train)], sarima_result.fittedvalues, label="Fitted Values", linestyle="dotted", color="orange")
    ax.plot(df_daily['Date'][len(train):], test_forecast_mean, label="Test Forecast", linestyle="dashed", color="green")
    ax.plot(future_dates, sarima_forecast_next_month, label="Next 30 Days Forecast", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seats Booked")
    ax.set_title("SARIMA Model - Demand Forecasting")
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)

# Data Upload Portal
elif page == "Upload Data":
    st.title("Upload New Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        if set(new_data.columns) == set(df.columns):
            new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
            st.success("Data successfully uploaded and merged!")
        else:
            st.error("Column mismatch! Ensure the uploaded file has the same structure as the dataset.")
