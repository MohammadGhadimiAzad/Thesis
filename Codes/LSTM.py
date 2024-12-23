!pip install yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# بارگذاری داده‌های BTC-USD
ticker = "BTC-USD"
data = yf.download(ticker, start="2023-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# استفاده از قیمت پایانی برای پیش‌بینی
data = data[['Close']]

# نرمال‌سازی داده‌ها
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# تقسیم داده‌ها به داده‌های آموزش و تست
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# تابع برای ایجاد داده‌های ورودی و خروجی
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# ایجاد مجموعه داده‌های آموزشی و تست
time_step = 5
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# تغییر شکل ورودی به [نمونه‌ها، زمان، ویژگی‌ها]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ساخت مدل LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# کامپایل مدل
model.compile(optimizer='adam', loss='mean_squared_error')

# آموزش مدل
model.fit(X_train, y_train, epochs=100, batch_size=32)

# پیش‌بینی بر روی داده‌های تست
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # بازگرداندن به مقیاس اصلی

# محاسبه پارامترهای ارزیابی
y_test_actual = scaler.inverse_transform(test_data[time_step + 1:])  # قیمت‌های واقعی
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
r_squared = r2_score(y_test_actual, predictions)

# چاپ نتایج
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r_squared}")

# ترسیم نمودار
plt.figure(figsize=(14, 5))
plt.plot(data.index[-len(y_test_actual):], y_test_actual, label='Actual Price', color='blue')
plt.plot(data.index[-len(predictions):], predictions, label='Predicted Price', color='red')
plt.title('BTC-USD Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
