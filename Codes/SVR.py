!pip install yfinance
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Load data
end_date = datetime.now()
start_date = '2023-01-01'  # Start date
data = yf.download('BTC-USD', start=start_date, end=end_date)

# Prepare data
data['Date'] = data.index
data['Days'] = (data['Date'] - data['Date'].min()).dt.days
X = data[['Days']]
y = data['Close']

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Train Weighted SVR model
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr.fit(X_scaled, y_scaled)

# Forecast prices for the next week
future_days = np.array([X['Days'].max() + i for i in range(1, 8)]).reshape(-1, 1)
future_days_scaled = scaler_X.transform(future_days)
predicted_scaled = svr.predict(future_days_scaled)
predicted_prices = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))

# Plotting the actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Actual Prices', color='blue')
future_dates = [data['Date'].max() + timedelta(days=i) for i in range(1, 8)]
plt.plot(future_dates, predicted_prices, label='Predicted Prices', color='orange')

# Highlight the entire dataset including the test period
plt.plot(data['Date'], data['Close'], label='Test Data', color='lightblue', alpha=0.5)

plt.title('BTC-USD Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Model evaluation
y_pred = svr.predict(X_scaled)
y_pred_inverse = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y, y_pred_inverse))
mae = mean_absolute_error(y, y_pred_inverse)
r_squared = r2_score(y, y_pred_inverse)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r_squared}')
