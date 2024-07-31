import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load and prepare the data
df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
df['Prices'] = df['Prices'].astype(float)

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

def price_storage_contract(injection_dates, withdrawal_dates, volumes, max_storage,
                           storage_cost_per_day, injection_cost, withdrawal_cost):
    
    if sum(volumes) != 0:
        return "Error: Total injection volume must equal total withdrawal volume"
    
    total_value = 0
    current_storage = 0
    cash_flows = []

    # Combine and sort all dates
    all_dates = [(date, 'inject', vol) for date, vol in zip(injection_dates, volumes) if vol > 0] + \
                [(date, 'withdraw', vol) for date, vol in zip(withdrawal_dates, volumes) if vol < 0]
    all_dates.sort(key=lambda x: x[0])

    for date, action, volume in all_dates:
        # Check if storage limit is exceeded
        if current_storage + volume > max_storage or current_storage + volume < 0:
            return "Error: Storage limit exceeded or insufficient gas for withdrawal"

        price = estimate_price(date)
        if price is None:
            return f"Error: Unable to estimate price for date {date}"
        
        if action == 'inject':
            cost = volume * price + abs(volume) * injection_cost
            total_value -= cost
            cash_flows.append((-cost, date, f"Inject {volume:.2f} units at {price:.2f}"))
        else:  # withdraw
            revenue = abs(volume) * price - abs(volume) * withdrawal_cost
            total_value += revenue
            cash_flows.append((revenue, date, f"Withdraw {abs(volume):.2f} units at {price:.2f}"))

        current_storage += volume

    # Calculate storage costs
    start_date = min(injection_dates + withdrawal_dates)
    end_date = max(injection_dates + withdrawal_dates)
    storage_days = (end_date - start_date).days + 1
    total_storage_cost = storage_cost_per_day * storage_days
    total_value -= total_storage_cost
    cash_flows.append((-total_storage_cost, end_date, f"Total storage cost for {storage_days} days"))

    return total_value, cash_flows

# Example usage
injection_dates = [datetime(2024, 6, 1), datetime(2024, 7, 1)]
withdrawal_dates = [datetime(2024, 12, 1), datetime(2025, 1, 1)]
volumes = [500000, 500000, -500000, -500000]  # Positive for injection, negative for withdrawal
max_storage = 1500000
storage_cost_per_day = 1000
injection_cost = 0.05
withdrawal_cost = 0.05

contract_value, cash_flows = price_storage_contract(
    injection_dates, withdrawal_dates, volumes, max_storage,
    storage_cost_per_day, injection_cost, withdrawal_cost
)

print(f"Contract Value: ${contract_value:,.2f}")
print("\nCash Flows:")
for flow, date, description in cash_flows:
    print(f"{date.strftime('%Y-%m-%d')}: ${flow:,.2f} - {description}")

# Visualize historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Dates'], df['Prices'], label='Historical Data')
future_dates = pd.date_range(start=df['Dates'].max() + timedelta(days=1), periods=365)
future_prices = [estimate_price(date) for date in future_dates]
plt.plot(future_dates, future_prices, label='Forecast', linestyle='--')
plt.scatter(injection_dates + withdrawal_dates, 
            [estimate_price(date) for date in injection_dates + withdrawal_dates],
            color='red', zorder=5, label='Contract Dates')
plt.title('Natural Gas Prices: Historical Data, Forecast, and Contract Dates')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()