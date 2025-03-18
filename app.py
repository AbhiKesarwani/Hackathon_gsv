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
    
    st.image("fuel_efficiency_chart.png", caption="Fuel Consumption Analysis")
    st.image("occupancy_trends.png", caption="Passenger Occupancy Trends")


# Demand Forecasting Portal
if page == "Demand Forecasting":
    st.title("📊 Passenger Demand Forecasting")
    st.write("Using SARIMA model to predict future passenger demand.")

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

    # User Input for Future Forecasting
    future_steps = st.slider("Select Future Forecast Duration (Days)", min_value=7, max_value=90, value=30)
    
    # Forecast Future Demand
    sarima_forecast_next = sarima_result.forecast(steps=future_steps)

    # Create Future Dates
    future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_daily['Date'], df_daily['Seats_Booked'], label="Actual Data", color="blue")
    ax.plot(df_daily['Date'][:len(train)], sarima_result.fittedvalues, label="Fitted Values", linestyle="dotted", color="orange")
    ax.plot(df_daily['Date'][len(train):], test_forecast_mean, label="Test Forecast", linestyle="dashed", color="green")
    ax.plot(future_dates, sarima_forecast_next, label=f"Next {future_steps} Days Forecast", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seats Booked")
    ax.set_title("📈 SARIMA Model - Demand Forecasting")
    ax.legend()
    ax.grid()
    
    st.pyplot(fig)

    # Display Insights
    st.subheader("🔎 Key Insights")
    peak_demand = sarima_forecast_next.max()
    low_demand = sarima_forecast_next.min()
    
    st.write(f"✔️ **Highest Predicted Demand:** {peak_demand:.0f} seats")
    st.write(f"⚠️ **Lowest Predicted Demand:** {low_demand:.0f} seats")
    st.write("🚀 **Business Impact:** This forecast helps in optimizing fleet allocation, fuel efficiency, and revenue planning.")

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
