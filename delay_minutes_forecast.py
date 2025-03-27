import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Step 1: Load Dataset
file_path = "D:/coding/python/dataset_updated.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)

# Step 2: Preprocess Data
# Drop unnecessary columns (if any)
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Aggregate 'Seats_Booked' by Date for daily forecasting
df_daily = df.groupby('Date').agg({'Delay_Mins': 'sum'}).reset_index()

# Ensure data is sorted chronologically
df_daily = df_daily.sort_values(by='Date')

# Step 3: Check for Stationarity using Augmented Dickey-Fuller (ADF) Test
def check_stationarity(data):
    result = adfuller(data)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    
    if result[1] < 0.05:
        print("✅ Data is stationary (No differencing needed).")
        return False  # No differencing needed
    else:
        print("❌ Data is NOT stationary (Differencing required).")
        return True  # Differencing needed

needs_differencing = check_stationarity(df_daily['Delay_Mins'])

# Step 4: Plot ACF & PACF to determine (p, q)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ACF Plot
sm.graphics.tsa.plot_acf(df_daily['Delay_Mins'].diff().dropna() if needs_differencing else df_daily['Delay_Mins'], 
                          lags=40, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

# PACF Plot
sm.graphics.tsa.plot_pacf(df_daily['Delay_Mins'].diff().dropna() if needs_differencing else df_daily['Delay_Mins'], 
                           lags=40, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()

# Step 5: Train-Test Split (80-20 split)
train_size = int(len(df_daily) * 0.8)
train, test = df_daily[:train_size], df_daily[train_size:]

# Step 6: Train Optimized SARIMA Model
p, d, q = (1, 1, 1)  # Use values from PACF/ACF analysis
P, D, Q, s = (1, 1, 1, 45)  # Weekly seasonality

sarima_model = SARIMAX(train['Delay_Mins'], 
                        order=(p, d, q), 
                        seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False, 
                        enforce_invertibility=False)

sarima_result = sarima_model.fit()

# Step 7: Forecast for Test Period
test_forecast = sarima_result.get_forecast(steps=len(test))
test_forecast_mean = test_forecast.predicted_mean

# Evaluate Model Performance
rmse = np.sqrt(mean_squared_error(test['Delay_Mins'], test_forecast_mean))
print(f"SARIMA Test RMSE: {rmse:.2f}")

# Step 8: Forecast for the Next 30 Days
future_steps = 30
sarima_forecast_next_month = sarima_result.forecast(steps=future_steps)

# Create future dates
future_dates = pd.date_range(start=df_daily['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps)

# Step 9: Visualize the Results
plt.figure(figsize=(12, 6))
plt.plot(df_daily['Date'], df_daily['Delay_Mins'], label="Actual Data", color="blue")
plt.plot(test['Date'], test_forecast_mean, label="Test Forecast", linestyle="dashed", color="green")
plt.plot(future_dates, sarima_forecast_next_month, label="Next 30 Days Forecast", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Delay_Mins")
plt.title("SARIMA Demand Forecast with Stationarity Check & ACF/PACF")
plt.legend()
plt.grid()
plt.show()
