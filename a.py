import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import numpy as np
from taipy.gui import Gui, State
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Load Dataset
DATA_PATH = "updated_data.csv"
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame()

# Define Pages
def home_page(state):
    return """
    <|container|
    ## GSRTC Data-Driven Insights Dashboard
    
    ### Problem Statement
    GSRTC currently operates with fragmented data across multiple departments, leading to inefficiencies in decision-making, inaccurate demand forecasting, and suboptimal resource utilization.
    
    ### Challenges
    - Lack of a centralized data platform
    - Inefficient scheduling and resource allocation
    - Difficulty in predicting passenger demand
    - High operational costs due to fuel inefficiencies
    
    ### Expected Outcomes
    - A unified data platform integrating multiple data sources
    - Improved decision-making with real-time insights
    - Accurate demand forecasting to optimize scheduling
    - Enhanced customer satisfaction with better service planning
    |>
    """

def dataset_page(state):
    if df.empty:
        return "<|text|Error: Dataset not loaded. Please upload the dataset.|>"
    return """
    <|container|
    ## Dataset Overview
    
    ### About the Dataset
    - This dataset is synthetically generated to simulate real-world GSRTC operations.
    - It includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - The dataset supports insights into resource allocation, demand forecasting, and operational efficiency.
    
    <|table|data=df|width=100%|height=400px|>
    <|button|label=Download Dataset|on_action=download_data|>
    |>
    """

def eda_page(state):
    return """
    <|container|
    ## Exploratory Data Analysis
    
    Below are key insights from our Power BI analysis:
    
    <|img src="gsrtc_dashboard.png" width="90%"|>
    <|img src="avg_fuel_consumption.png" width="90%"|>
    <|img src="avg_profit_per_trip_by_route.png" width="90%"|>
    <|img src="seats_booked_by_destination.png" width="90%"|>
    <|img src="seats_booked_per_month.png" width="90%"|>
    <|img src="total_trips_by_delay_status.png" width="90%"|>
    <|img src="total_trips_by_delay_occupancy.png" width="90%"|>
    |>
    """

def demand_forecasting_page(state):
    if df.empty:
        return "<|text|Error: Dataset not loaded. Please upload the dataset.|>"

    df_daily = df.groupby('Date').agg({'Seats_Booked': 'sum'}).reset_index()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily = df_daily.sort_values(by='Date')

    # Check for Stationarity
    p_value = adfuller(df_daily["Seats_Booked"])[1]
    if p_value > 0.05:
        df_daily["Seats_Booked_Diff"] = df_daily["Seats_Booked"].diff().dropna()
        message = "❌ Data is NOT stationary. Applied differencing."
    else:
        message = "✅ Data is stationary. No differencing applied."

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

    # Future Forecasting
    future_steps = state.future_steps if 'future_steps' in state else 30
    sarima_forecast_next = sarima_result.forecast(steps=future_steps)
    future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

    # Plot Forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_daily['Date'], df_daily['Seats_Booked'], label="Actual Data", color="blue")
    ax.plot(future_dates, sarima_forecast_next, label=f"Next {future_steps} Days Forecast", linestyle="dashed", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seats Booked")
    ax.set_title("📈 SARIMA Model - Demand Forecasting")
    ax.legend()
    ax.grid()
    
    fig.savefig("sarima_forecast.png")

    return f"""
    <|container|
    ## Passenger Demand Forecasting
    
    {message}
    📊 SARIMA Model RMSE: {rmse:.2f}
    
    <|slider|value={future_steps}|min=7|max=90|on_change=update_forecast|label=Future Forecast Duration (Days)|>
    <|img src="sarima_forecast.png" width="90%"|>
    
    🔎 *Key Insights*
    ✔ Highest Predicted Demand: {sarima_forecast_next.max():.0f} seats  
    ⚠ Lowest Predicted Demand: {sarima_forecast_next.min():.0f} seats  
    🚀 Business Impact: Optimizes fleet allocation, fuel efficiency, and revenue planning.
    |>
    """

def upload_page(state):
    return """
    <|container|
    ## Upload New Data
    
    <|file_selector|label=Choose a CSV file|on_action=upload_data|extensions=.csv|>
    |>
    """

# Download Dataset
def download_data(state):
    df.to_csv("dataset_download.csv", index=False)

# Upload Data
def upload_data(state, payload):
    uploaded_file = payload['args'][0]
    new_data = pd.read_csv(uploaded_file)
    if set(new_data.columns) == set(df.columns):
        new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
        state.new_data_uploaded = True
    else:
        state.new_data_uploaded = False

# Update Forecast
def update_forecast(state, payload):
    state.future_steps = payload['args'][0]

# Define Page Routing
pages = {
    "Home": home_page,
    "Dataset": dataset_page,
    "EDA": eda_page,
    "Demand Forecasting": demand_forecasting_page,
    "Upload Data": upload_page,
}

# Define GUI Layout
page_selector = "<|toggle|theme=dark|value=Home|label=Navigation|options={['Home', 'Dataset', 'EDA', 'Demand Forecasting', 'Upload Data']}|on_change=switch_page|>"

def switch_page(state, payload):
    state.current_page = payload['args'][0]

# Initialize GUI
state = State()
state.current_page = "Home"

gui = Gui(page_selector + "<|part|render={pages[state.current_page](state)}|>")
gui.run(title="GSRTC Data Insights Dashboard", port=8501)
