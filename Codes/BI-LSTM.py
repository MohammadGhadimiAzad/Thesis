!pip install yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler

# دانلود داده‌ها
data = yf.download('BTC-USD', start='2023-01-01', end='2023-11-01')
data = data[['Close']]

# مقیاس داده‌ها
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# تقسیم داده‌ها به مجموعه آموزش و تست
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# ایجاد مجموعه‌های داده برای آموزش و پیش‌بینی
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 7  # به عنوان مثال، 7 روز برای پیش‌بینی
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# تغییر شکل داده‌ها برای Bi-LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ساخت مدل Bi-LSTM
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# آموزش مدل
model.fit(X_train, y_train, epochs=50, batch_size=32)

# پیش‌بینی
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ارزیابی مدل
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_prices))
mae = mean_absolute_error(y_test_actual, predicted_prices)
r2 = r2_score(y_test_actual, predicted_prices)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")

# ترسیم نمودار
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size + time_step + 1:], y_test_actual, label='Actual Prices', color='blue')
plt.plot(data.index[train_size + time_step + 1:], predicted_prices, label='Predicted Prices', color='red')
plt.title('BTC-USD Price Prediction with Bi-LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
