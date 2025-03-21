import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gdown
import pickle
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from twilio.rest import Client


def set_background(image_url):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: url({image_url}) no-repeat center center fixed;
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
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




# Define paths
MODEL_PATH = "xgb_model.pkl"
SCALER_PATH = "scaler.pkl"


# **Load Pretrained Model Efficiently**
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at {MODEL_PATH}. Please upload a valid model file.")
        st.stop()
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"⚠ Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ Scaler file not found at {SCALER_PATH}. Please upload a valid scaler file.")
        st.stop()
    
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"⚠ Error loading scaler: {e}")
        st.stop()

# Load the model
xgb_model = load_model()

# Load the scaler
scaler = load_scaler()

# Sidebar Navigation
# Sidebar with Logo and Navigation
st.sidebar.image("logo.png", use_container_width=True)  # Add your logo here
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "EDA", "Predictive Maintenance", "Demand Forecasting", "Upload Data", "Query"])

# Add a red background note
warning_html = """
    <div style="background-color:#ffcccc; padding:10px; border-radius:5px; text-align:center;">
        <h4 style="color:red;">🚧 This section is under process. Some functionalities may not work as expected. 🚧</h4>
    </div>
"""



# Home Page
if page == "Home":
    set_background("bg_1.jpg")
    st.title("🚌 GSRTC Data-Driven Insights Dashboard")

    st.markdown("What They Expected from Us")

    st.markdown("""
    1. **Unified Data Platform** - A centralized system integrating all key operational data for seamless access. ✅  

    2. **Exploratory Data Analysis (EDA)** - In-depth data analysis to uncover insights on fuel consumption, delays, and occupancy. ✅

    3. **Predictive Maintenance** - – Machine learning models to predict vehicle breakdowns before they happen. ✅

    4. **Demand Forecasting** – Advanced time-series models to forecast passenger demand accurately. ✅
    
    5. **Enhanced Decision Making** -  – Actionable insights to optimize routes, fuel efficiency, and revenue. ✅

    📌 *With these milestones achieved, we are now moving towards more advanced analytics, real-time anomaly detection, and geospatial route analysis!*
    """)

        # Add project progress section
    # st.markdown("## 📌 *Project Achievements So Far*")
    
    # st.markdown("""
    # ✅ *Data Cleaning & Preprocessing*
    # - Removed duplicate and unnamed columns.
    # - Ensured uniform data format and column consistency.
    # - Filled missing values in uploaded data while maintaining integrity.

    # ✅ *Exploratory Data Analysis (EDA)*
    # - *Power BI Dashboards* for visual insights 📊.
    # - Examined trends in fuel consumption, delays, and route profitability.
    # - Analyzed seasonal variations in demand and travel patterns.

    # ✅ *Demand Forecasting*
    # - Implemented *Exponential Smoothing* for accurate passenger predictions.
    # - Developed a forecasting model to optimize bus scheduling.
    # - Integrated *RMSE Evaluation* to ensure model accuracy.

    # ✅ *Dynamic Data Upload & Management*
    # - Allowed *new data uploads* while ensuring column consistency.
    # - *Handled missing columns* by filling them with null values.
    # - Prevented incorrect column mapping to avoid data corruption.

    # ✅ *Downloadable Reports*
    # - Users can download *full datasets* post-processing.
    # - Ensured no data loss during downloads.

    # 🚧 *Upcoming Enhancements*:
    # - *Predictive Maintenance (Under Process) 🔧*
    #     - Machine learning models to *predict breakdowns*.
    #     - Reduce unexpected failures and improve efficiency.
    # - *Advanced Time-Series Forecasting*
    #     - Implementing *SARIMA, LSTM, Prophet* for better accuracy.
    # - *Geospatial Route Analysis*
    #     - Visualizing route performance and bottlenecks using maps.
    # - *Real-time Anomaly Detection*
    #     - Identifying unusual delays, fuel inefficiencies, and disruptions.""")
    
    st.write("""
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
    set_background("bg_1.jpg")
    st.write("""
    ### 🔍 About the Dataset
    - Synthetic Data designed to simulate real-world GSRTC operations.
    - Includes details about bus trips, fuel consumption, occupancy rates, delays, ticket prices, and more.
    - Helps in resource allocation, demand forecasting, and operational efficiency.
    
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


# **Predictive Maintenance Portal**
elif page == "Predictive Maintenance":
    st.title("🔧 Predictive Maintenance Analysis")

    uploaded_file = st.file_uploader("📁 Upload Maintenance Data (CSV)", type="csv")

    if uploaded_file is not None:
    # Load uploaded data
        new_data = pd.read_csv(uploaded_file)

        # Debug: Print column names
        st.write("📝 Expected feature names:", scaler.feature_names_in_)
        st.write("📂 Uploaded dataset columns:", new_data.columns.tolist())

        # Define expected features
        expected_features = scaler.feature_names_in_
    
        # Check for missing columns
        missing_cols = set(expected_features) - set(new_data.columns)
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        # Ensure correct order & type
        new_data = new_data[expected_features]
        new_data = new_data.apply(pd.to_numeric, errors="coerce")

        # Standardize data
        new_X_scaled = scaler.transform(new_data)  # ✅ No more feature mismatch error

        # Predict Maintenance Status
        predictions = xgb_model.predict(new_X_scaled)
        new_data["Predicted_Maintenance_Status"] = predictions

        # Display Results
        st.write("### ✅ Maintenance Predictions:")
        st.dataframe(new_data)

        # Download Predictions
        st.download_button("⬇ Download Predictions", new_data.to_csv(index=False), "maintenance_predictions.csv", "text/csv")

        # Count plot of maintenance predictions
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=new_data["Predicted_Maintenance_Status"], palette=["green", "red"], ax=ax)
        ax.set_title("🔧 Predicted Maintenance Status Distribution")
        ax.set_xlabel("Maintenance Status (0 = Good, 1 = Needs Repair)")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Good Condition", "Needs Repair"])

        # Display in Streamlit
        st.pyplot(fig)

# Demand Forecasting Portal
elif page == "Demand Forecasting":
    st.title("📈 Passenger Demand Forecasting")
    st.write("Using *SARIMA* for fast and efficient demand prediction.")
    
    st.image("Demand Forecast.png", caption="🚌 Seats Booked") 

    # --- Load Seat Booking Forecast ---
    DATA_PATH_2 = "forecast_seat_book.csv"
    if os.path.exists(DATA_PATH_2):  # ✅ Fixed variable name
        df_forecast_2 = pd.read_csv(DATA_PATH_2)
        if df_forecast_2.empty:
            st.error("❌ Dataset is empty. Please upload valid data.")
            st.stop()
    else:
        st.error("❌ Dataset not found. Please upload a valid file.")
        st.stop()

    st.dataframe(df_forecast_2, height=400, width=1000)
    st.download_button("⬇ Download Seat Forecast", df_forecast_2.to_csv(index=False), "forecast_seat.csv", "text/csv")

    st.image("delay_mins.jpg", caption="🚌 Delay Minutes") 

    DATA_PATH_1 = "forecast_delay_mins.csv"
    if os.path.exists(DATA_PATH_1):
        df_forecast_1 = pd.read_csv(DATA_PATH_1)  # ✅ First, read the CSV
        if df_forecast_1.empty:
            st.error("❌ Dataset is empty. Please upload valid data.")
            st.stop()
    else:
        st.error("❌ Dataset not found. Please upload a valid file.")
        st.stop()


    st.dataframe(df_forecast_1, height=400, width=1000)
    st.download_button("⬇ Download Delay Forecast", df_forecast_1.to_csv(index=False), "forecast_delay.csv", "text/csv")

    

    # --- Load Fuel Consumption Forecast ---
    st.image("fuel_cons.jpg", caption="🚌 Fuel Consumption")
    DATA_PATH_3 = "forecast_consumption_fuel.csv"
    if os.path.exists(DATA_PATH_3):
        df_forecast_3 = pd.read_csv(DATA_PATH_3)
        if df_forecast_3.empty:
            st.error("❌ Dataset is empty. Please upload valid data.")
            st.stop()
    else:
        st.error("❌ Dataset not found. Please upload a valid file.")
        st.stop()

    st.dataframe(df_forecast_3, height=400, width=1000)
    st.download_button("⬇ Download Fuel Forecast", df_forecast_3.to_csv(index=False), "forecast_fuel.csv", "text/csv")


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

        # *Check for extra columns in uploaded file*
        extra_columns = set(uploaded_columns) - set(expected_columns)
        if extra_columns:
            st.error("❌ Column mismatch! The uploaded file contains *unexpected columns*.")
            st.write("### ❌ Unexpected Columns in Upload:", sorted(list(extra_columns)))
            st.stop()

        # *Check for missing columns in uploaded file*
        missing_columns = set(expected_columns) - set(uploaded_columns)
        if missing_columns:
            st.warning("⚠ Some columns are missing in the uploaded file. They will be filled with NaN.")
            for col in missing_columns:
                new_data[col] = pd.NA  # Fill missing columns with null

        # *Ensure the column order is exactly the same*
        new_data = new_data[expected_columns]

        # *Append new data*
        new_data.to_csv(DATA_PATH, mode='a', header=False, index=False)
        st.success("✅ Data successfully uploaded!")

        # Reload the full dataset
        df = pd.read_csv(DATA_PATH)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # ✅ Ensure "Unnamed" columns are removed

        # *Download Button for Full Data*
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download Full Dataset", data=csv, file_name="full_dataset.csv", mime="text/csv")

        st.write("### 🔍 Updated Dataset Preview")
        st.dataframe(df)

elif page=="Query":

    # Twilio credentials 
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_PHONE_NUMBER = "+13309021484"
    YOUR_PHONE_NUMBER = "+918700442643"

    # Function to send SMS
    def send_sms(message):
        try:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=YOUR_PHONE_NUMBER
             )
            return "Message Sent Successfully!"
        except Exception as e:
            return f"Error: {str(e)}"

    # Streamlit UI
    st.title("User Request Portal")

    st.write("Enter your request below, and it will be sent as an SMS.")

    # User input
    user_request = st.text_area("Your Request:", "")

    if st.button("Send Request via SMS"):
        if user_request.strip():
            response = send_sms(user_request)
            st.success(response)
        else:
            st.error("Please enter a request before sending.")

