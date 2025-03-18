import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Load dataset
DATA_PATH = "dataset_updated.csv"
if os.path.exists("dataset_updated.csv"):
    df = pd.read_csv("dataset_updated.csv")
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
elif page == "Demand Forecasting":
    st.title("Passenger Demand Forecasting")
    st.write("Forecasting future passenger demand using SARIMA model.")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Train SARIMA Model
    model = SARIMAX(df['Seats_Booked'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.forecast(steps=30)
    
    st.line_chart(forecast)

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
