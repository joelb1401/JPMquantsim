import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load and prepare the data
df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
df['Prices'] = df['Prices'].astype(float)

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(df['Dates'], df['Prices'])
plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Create an interpolation function for historical data
interp_func = interp1d(df['Dates'].astype(int), df['Prices'], kind='cubic', fill_value='extrapolate')

# Fit ARIMA model for forecasting
model = ARIMA(df['Prices'], order=(1, 1, 1))
results = model.fit()

# Function to estimate price for a given date
def estimate_price(date):
    date = pd.to_datetime(date)
    if date < df['Dates'].min() or date > df['Dates'].max() + timedelta(days=365):
        return None
    
    if date <= df['Dates'].max():
        return float(interp_func(date.astype(int)))
    else:
        days_ahead = (date - df['Dates'].max()).days
        forecast = results.forecast(steps=days_ahead)
        return float(forecast.iloc[-1])

# Example usage
print(f"Estimated price on 2023-06-15: {estimate_price('2023-06-15'):.2f}")
print(f"Estimated price on 2024-12-31: {estimate_price('2024-12-31'):.2f}")
print(f"Estimated price on 2025-09-30: {estimate_price('2025-09-30'):.2f}")

# Visualize historical data and forecast
future_dates = pd.date_range(start=df['Dates'].max() + timedelta(days=1), periods=365)
future_prices = [estimate_price(date) for date in future_dates]

plt.figure(figsize=(12, 6))
plt.plot(df['Dates'], df['Prices'], label='Historical Data')
plt.plot(future_dates, future_prices, label='Forecast', linestyle='--')
plt.title('Natural Gas Prices: Historical Data and Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Analyze seasonal patterns
df['Month'] = df['Dates'].dt.month
monthly_avg = df.groupby('Month')['Prices'].mean().reindex(range(1, 13))

plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='bar')
plt.title('Average Natural Gas Prices by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()