!pip install yfinance  
import yfinance as yf  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

# دانلود داده‌های بیت‌کوین  
ticker = 'BTC-USD'  
# data = yf.download(ticker, period='1y', interval='1d')  
data = yf.download(ticker, start="2020-11-01", end="2021-04-01", interval = "1d")

# محاسبه Bollinger Bands  
def calculate_bollinger_bands(df, window=20, num_std=2):  
    df['SMA'] = df['Close'].rolling(window=window).mean()  
    df['STD'] = df['Close'].rolling(window=window).std()  
    df['Upper Band'] = df['SMA'] + (df['STD'] * num_std)  
    df['Lower Band'] = df['SMA'] - (df['STD'] * num_std)  
    return df  

# محاسبه RSI  
def calculate_rsi(df, window=14):  
    delta = df['Close'].diff()  
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  
    rs = gain / loss  
    df['RSI'] = 100 - (100 / (1 + rs))  
    return df  

# محاسبه Ichimoku  
def calculate_ichimoku(df):  
    df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2  
    df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2  
    df['Senkou_span_a'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)  
    df['Senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)  
    df['Chikou_span'] = df['Close'].shift(-26)  
    return df  

# محاسبه میانگین متحرک نمایی (EMA)
def calculate_ema_20(data, window=20):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_ema(df):
    df['EMA_20'] = calculate_ema_20(df)
    return df

# محاسبه شیب خط  
def calculate_slope(prices):  
    x = np.arange(len(prices))  
    slope, _ = np.polyfit(x, prices, 1)  # درجه 1 برای خطوط مستقیم  
    return slope  

# محاسبه سیگنال‌ها  
def generate_signal(df):  
    df = df.dropna()  
    
    if len(df) == 0:  
        return "No valid data available for analysis."  

    close_last = df['Close'].iloc[-1]  
    close_last_value = close_last.item()  
    upper_band_last = df['Upper Band'].iloc[-1]  
    lower_band_last = df['Lower Band'].iloc[-1]  
    rsi_last = df['RSI'].iloc[-1]  
    senkou_a_last = df['Senkou_span_a'].iloc[-1]  
    senkou_b_last = df['Senkou_span_b'].iloc[-1]  

    #if isinstance(senkou_a_last, pd.Series) or isinstance(senkou_b_last, pd.Series):  
    senkou_a_last = senkou_a_last.item()  
    senkou_b_last = senkou_b_last.item()  

    signal = []  
    
    #if isinstance(upper_band_last, (float, int)) and isinstance(close_last_value, (float, int)):  
    if close_last_value > upper_band_last:  
        signal.append("Overbought (Bollinger)")  
    elif close_last_value < lower_band_last:  
        signal.append("Oversold (Bollinger)")  
    
    #if isinstance(rsi_last, (float, int)):  
    if rsi_last > 70:  
        signal.append("Overbought (RSI)")  
    elif rsi_last < 30:  
        signal.append("Oversold (RSI)")  

    # if isinstance(senkou_a_last, (float, int)) and isinstance(senkou_b_last, (float, int)):  
    if close_last_value > senkou_a_last and close_last_value > senkou_b_last:  
        signal.append("Bullish (Ichimoku)")  
    elif close_last_value < senkou_a_last and close_last_value < senkou_b_last:  
        signal.append("Bearish (Ichimoku)")  

    #EWA, threshold = 0.05
    threshold = 0.05
    ewa_last = df['EMA_20'].iloc[-1]
    if close_last_value > ewa_last * (1 + threshold):
        signal.append("Overbought (EWA)")
    if close_last_value > ewa_last * (1 + threshold):
        signal.append("Oversold (EWA)")
    
    slope = calculate_slope(df['Close'].values)  
    
    result = ""
    signal = signal[0]
    if "Overbought" in signal:  
        result = "زمان خوبی برای سرمایه گذاری نیست (خرید بیش از حد)"  
    elif "Oversold" in signal:  
        result = "زمان مناسب برای سرمایه گذاری (افزایش فروش)"  
    elif "Bullish" in signal:  
        result = "زمان مناسب برای سرمایه گذاری (صعودی)"  
    elif "Bearish" in signal:  
        result = "زمان مناسبی برای سرمایه گذاری نیست (نزولی)"  
    else:  
        result = "بازار خنثی"  

    return result, slope  

# اعمال محاسبات بر روی داده‌ها  
data = calculate_bollinger_bands(data)  
data = calculate_rsi(data)  
data = calculate_ichimoku(data)  
data = calculate_ema(data)  

# ارزیابی وضعیت  
investment_signal, slope = generate_signal(data)  
print(f"\n\nنتیجه ارزیابی: {investment_signal}")
print(f"شیب خط قیمت‌های BTC: {slope.item()}")  

# نمایش نمودار (اختیاری)  
plt.figure(figsize=(15, 10))  
plt.plot(data['Close'], label='BTC-USD Price')  
plt.plot(data['Upper Band'], label='Upper Band', linestyle='--')  
plt.plot(data['Lower Band'], label='Lower Band', linestyle='--')  
plt.title("Bollinger Bands with BTC-USD Price")  
plt.legend()  
plt.show()
