# GSRTC Data-Driven Insights Platform

## 🚀 Overview
The **GSRTC Data-Driven Insights Platform** is a comprehensive tool designed to optimize operations, enhance decision-making, and improve service delivery for GSRTC. This platform integrates **predictive maintenance, demand forecasting, and data visualization** to provide actionable insights based on real-time and historical data.

## 📌 Features & Achievements

- **Unified Data Platform:** A centralized repository, easily accessible across all devices.
- **Exploratory Data Analysis (EDA):** Power BI dashboards for data-driven decision-making.
- **Predictive Maintenance:** Machine learning model to predict vehicle breakdowns and minimize maintenance costs.
- **Demand Forecasting:** Accurate predictions of passenger demand to optimize scheduling and resource allocation.
- **Dynamic Data Upload & Management:** Ensures seamless integration of new data with real-time updates.
- **Standardized Data Processing:** Automated feature engineering, data cleaning, and anomaly detection.
- **Scalability & Accessibility:** Hosted on **Streamlit Cloud** for easy access and deployment.

## 🛠️ Installation & Setup

### 🔹 Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd gsrtc-insights
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

### 🔹 Streamlit Cloud Deployment
The platform is hosted on **Streamlit Cloud**, and you can access it here:  
🔗 **[GSRTC Data-Driven Insights Platform](https://gsvhackathon.streamlit.app/)**

## 📂 Project Structure
```
/project_root
│-- app.py                  # Main Streamlit App
│-- updated_data.csv        # Dataset used for analysis
│-- xgb_model.pkl           # Trained Machine Learning Model
│-- scaler.pkl              # Scaler for feature standardization
│-- gsrtc_dashboard.png     # Power BI Dashboard Image
│-- requirements.txt        # List of dependencies
│-- README.md               # Project Documentation
```

## 📊 Dataset & Model Details
- **Dataset:** Includes parameters such as fuel consumption, breakdown incidents, disruption likelihood, risk scores, and profitability per trip.
- **Machine Learning Model:** Uses **XGBoost** for predictive maintenance.
- **Data Preprocessing:** Includes feature engineering, standardization, and handling missing values.
- **SMOTE:** Used for handling class imbalance in predictive maintenance.

## 🎯 Expected Outcomes
- **Improved Decision-Making:** Data-driven insights for GSRTC operations.
- **Enhanced Efficiency:** Optimized scheduling, cost reduction, and improved service delivery.
- **Predictive Maintenance:** Reduced vehicle breakdowns and unplanned downtimes.
- **Accurate Forecasting:** Better resource planning based on demand trends.
- **Increased Customer Satisfaction:** Improved passenger experience through enhanced scheduling and reliability.

## 👥 Contributors
- [@anantj09](https://github.com/anantj09)
- [@ajiteshchanna](https://github.com/ajiteshchanna)
- [@amVICKY](https://github.com/amVICKY)

## 📜 License
This project is licensed under the **MIT License**.

---
🔹 *For any issues or suggestions, feel free to open an issue on GitHub!*

