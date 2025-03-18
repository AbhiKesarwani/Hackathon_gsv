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
elif page == "Dataset":
    st.title("Dataset Overview")
    st.write("""
    ### About the Dataset
    - This dataset is **synthetically generated** to simulate real-world GSRTC operations.
    - It includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - The dataset is structured to support insights into **resource allocation, demand forecasting, and operational efficiency**.
    
    ### Columns in the Dataset:
    Date - Date of the trip.
Day_of_Week - Day corresponding to the trip's date.
Time_Slot - Time period of the journey (Morning, Afternoon, Night).
Route_ID - Unique identifier for the bus route.
Route_Name - Name of the route (start to end location).
Bus_ID - Unique identifier for the bus.
Driver_ID - Unique identifier for the driver.
Bus_Type - Type of bus (AC, Non-AC, Sleeper, etc.).
Total_Seats - Number of seats available in the bus.
Seats_Booked - Number of seats booked for the trip.
Occupancy_Rate (%) - Percentage of seats occupied.
Ticket_Price - Price per ticket for the journey.
Revenue_Per_Trip - Total revenue generated from ticket sales.
Fuel_Consumption_Liters - Fuel consumed during the trip in liters.
Fuel_Efficiency_KMPL - Fuel efficiency of the bus in kilometers per liter.
Maintenance_Status - Indicates the current maintenance status.
Last_Service_Date - Date when the bus was last serviced.
Breakdown_Incidents - Number of breakdown incidents recorded.
Disruption_Likelihood - Probability of trip disruption.
Risk_Score - Risk assessment score for the trip.
Delay_Probability - Probability of delay during the trip.
Expense_Per_Trip - Total operational expenses per trip.
Profit_Per_Trip - Profit earned per trip after expenses.
Weather - Weather conditions during the journey.
Festival_Season - Indicates if the trip falls during a festival season.
Holiday - Indicates if the trip falls on a public holiday.
Special_Event - Mentions any special event affecting the route.
Avg_Travel_Time (mins) - Average time taken for the trip in minutes.
Delay_Mins - Actual delay in minutes.
Start_Desitination - Incorrect duplicate of Start_Destination.
End_Destination - Destination point of the route.
Distance_KM - Distance covered during the trip in kilometers.
Start_Destination - Starting point of the route.
Avg_Travel_Time_Mins - Duplicate of Avg_Travel_Time (mins).
Weather_Impact - Impact of weather on the journey.
Event_Type - Type of event affecting the trip (Festival, Holiday, etc.). 

    **You can scroll through the dataset below:**
    """)

    st.dataframe(df, height=400, width=1000)  # Enables scrolling in both directions
    st.download_button("Download Dataset", df.to_csv(index=False), "dataset.csv", "text/csv")

# EDA Portal
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    st.write("Below are key insights from our Power BI analysis.")

    st.image("gsrtc_dashboard.png", caption="gsrtc_dashboard.png")
    st.image("avg_fuel_consumption.png", caption="avg_fuel_consumption.png")
    st.image("avg_profit_per_trip_by_route.png", caption="avg_profit_per_trip_by_route.png")
    st.image("seats_booked_by_destination.png", caption="seats_booked_by_destination.png")
    st.image("seats_booked_per_month.png", caption="seats_booked_per_month.png")
    st.image("total_trips_by_delay_status.png", caption="total_trips_by_delay_status.png")
    st.image("total_trips_by_delay_occupancy.png", caption="total_trips_by_delay_occupancy.png")
    


# Demand Forecasting Portal


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
