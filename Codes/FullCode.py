!pip install yfinance tensorflow --quiet

# integrated_cpoa_with_indicators_and_gru.py
# اجرا در Colab: !pip install yfinance tensorflow --quiet
# سپس: python integrated_cpoa_with_indicators_and_gru.py
import os
import warnings
warnings.filterwarnings("ignore")

# --- نصب وابستگی (در محیط تعاملی مثل Colab uncomment کنید) ---
# !pip install yfinance tensorflow --quiet

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# برای مدل Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

np.random.seed(42)

# ---------------------------
# تنظیمات کاربری
# ---------------------------
SYMBOLS = ["BTC-USD", "ETH-USD", "LTC-USD"]  # مثال: چند دارایی انتخابی
START = "2022-01-01"
END = datetime.now().strftime("%Y-%m-%d")
LOOKBACK_DAYS = 30
MIN_DAYS_FOR_FEATURES = 120

USE_DL = True and TF_AVAILABLE     # اگر TF نصب نشده باشد به صورت خودکار خاموش می‌شود
EPOCHS = 10        # برای آزمایش کوتاه; در کار نهایی افزایش دهید
TIME_STEP = 14     # برای ورودی GRU (مثلاً 14 روز)
OUTPUT_CSV = "pareto_front_results.csv"

# ---------------------------
# توابع اندیکاتورها
# ---------------------------
def calculate_bollinger_bands(df, window=20, num_std=2):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['STD'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['SMA'] + (df['STD'] * num_std)
    df['BB_Lower'] = df['SMA'] - (df['STD'] * num_std)
    return df

def calculate_rsi(df, window=14):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_ichimoku(df):
    df = df.copy()
    high = df['High']; low = df['Low']; close = df['Close']
    df['Tenkan_sen'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Kijun_sen'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df['Senkou_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
    df['Senkou_B'] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df['Chikou'] = close.shift(-26)
    return df

def calculate_ema(df, span=20):
    df = df.copy()
    df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    return df

def calculate_basic_features(df):
    df = df.copy()
    df['momentum_7d'] = df['Close'].pct_change(7)
    df['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
    df['return_1d'] = df['Close'].pct_change(1)
    return df

# ---------------------------
# مدل پیش‌بینی GRU (یکبار آموزش و سپس پیش‌بینی بخش تست)
# ---------------------------
def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(32, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_predict_gru(close_series, time_step=TIME_STEP, epochs=EPOCHS):
    """
    ورودی: close_series: pd.Series indexed by date
    خروجی: pd.Series از 'predicted_return' با ایندکس بخشی از تاریخ‌ها (معمولاً بخش تست)
    روش: داده رو 80/20 تقسیم می‌کنیم، مدل روی بخش آموزش آموزش می‌یابد و روی بخش تست پیش‌بینی می‌کند.
    """
    arr = close_series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(arr)
    
    train_size = int(len(arr_scaled) * 0.8)
    if train_size <= time_step + 1:
        # داده کم است؛ پیش‌بینی نکن
        return pd.Series(dtype=float, index=close_series.index)
    
    X_train, y_train = [], []
    for i in range(time_step, train_size):
        X_train.append(arr_scaled[i-time_step:i, 0])
        y_train.append(arr_scaled[i, 0])
    X_train = np.array(X_train).reshape((-1, time_step, 1))
    y_train = np.array(y_train)
    
    X_test, test_idx = [], []
    for i in range(train_size, len(arr_scaled)):
        if i - time_step < 0:
            continue
        X_test.append(arr_scaled[i-time_step:i, 0])
        test_idx.append(close_series.index[i])
    if len(X_test) == 0:
        return pd.Series(dtype=float, index=close_series.index)
    X_test = np.array(X_test).reshape((-1, time_step, 1))
    
    if USE_DL and TF_AVAILABLE:
        model = create_gru_model((time_step, 1))
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        preds_scaled = model.predict(X_test).reshape(-1, 1)
        preds = scaler.inverse_transform(preds_scaled).reshape(-1)
    else:
        # fallback ساده: از میانگین تغییرات استفاده کن (baseline)
        last_values = [arr.flatten()[i-time_step:i] for i in range(train_size, len(arr_scaled))]
        preds = [v[-1] for v in last_values]
    
    # تبدیل به بازده روزانه پیش‌بینی‌شده: (pred - prev)/prev
    prevs = []
    for i in range(train_size, len(arr_scaled)):
        prevs.append(close_series.values[i-1])
    preds_returns = (np.array(preds).astype(float) - np.array(prevs).astype(float)) / (np.array(prevs).astype(float) + 1e-9)
    
    result_series = pd.Series(preds_returns, index=test_idx)
    result_series.name = "predicted_return"
    return result_series

# ---------------------------
# دانلود داده‌ها و محاسبه فیچرها
# ---------------------------
def download_and_compute(symbols, start=START, end=END):
    historical = {}
    for s in symbols:
        try:
            df = yf.download(s, start=start, end=end, progress=False, auto_adjust=False)
            if df.shape[0] < MIN_DAYS_FOR_FEATURES:
                print(f"  ⚠️ {s}: داده ناکافی ({len(df)} روز) — نادیده گرفته شد")
                continue
            # محاسبه اندیکاتورها
            df = calculate_basic_features(df)
            df = calculate_bollinger_bands(df)
            df = calculate_rsi(df)
            df = calculate_ichimoku(df)
            df = calculate_ema(df, span=20)
            historical[s] = df
            print(f"  ✅ {s}: بارگذاری و فیچرها آماده ({len(df)} روز)")
        except Exception as e:
            print(f"  ❌ {s}: خطا - {e}")
    return historical

# ---------------------------
# تبدیل فیچرها به فرمت مورد نیاز CPOA و اضافه کردن پیش‌بینی‌ها
# ---------------------------
def add_predicted_feature_to_hist(historical):
    predictions = {}
    for s, df in historical.items():
        predicted = train_predict_gru(df['Close'], time_step=TIME_STEP, epochs=EPOCHS)
        # قرار دادن مقادیر پیش‌بینی در همان ایندکس (ریبا شده)
        historical[s] = historical[s].copy()
        historical[s]['predicted_return'] = np.nan
        # only fill where predictions exist
        for idx, val in predicted.items():
            if idx in historical[s].index:
                historical[s].loc[idx, 'predicted_return'] = val
        predictions[s] = predicted
        print(f"  🔮 {s}: predicted feature length = {len(predicted)}")
    return historical, predictions

# ---------------------------
# کلاس MultiObjectiveCPOA (تعدیل شده برای فیچرهای بیشتر)
# ---------------------------
class MultiObjectiveCPOA:
    def __init__(self, population_size=20, max_generations=6, transaction_cost=0.001, lookback_days=LOOKBACK_DAYS):
        self.population_size = population_size
        self.max_generations = max_generations
        self.transaction_cost = transaction_cost
        self.lookback_days = lookback_days
        self.symbols = None
        self.num_assets = 0

    def initialize_population(self):
        pop = []
        for _ in range(self.population_size):
            w = np.random.dirichlet(np.ones(self.num_assets))
            strat = {
                'weights': w,
                'volatility_threshold': float(np.random.uniform(0.02, 0.1)),
                'risk_appetite': float(np.random.uniform(0.2, 1.2)),
                'rebalance_frequency': int(np.random.choice([7, 14, 30]))
            }
            pop.append(strat)
        return pop

    def evaluate_strategy(self, strategy, features_list, price_data):
        dates = price_data.index.to_list()
        if len(dates) < self.lookback_days + 2:
            return -1000.0, -1000.0

        rets = []
        prev_weights = np.zeros(self.num_assets)

        for t in range(self.lookback_days, len(dates)):
            cur_date = dates[t]
            prev_date = dates[t - 1]
            daily_ret = 0.0

            for i, sym in enumerate(self.symbols):
                feats = features_list[i]
                if cur_date not in feats.index:
                    continue
                # خواندن فیچرهای اصلی
                try:
                    mom = feats.loc[cur_date, 'momentum_7d']
                    vol = feats.loc[cur_date, 'volatility_30d']
                    rsi = feats.loc[cur_date, 'RSI']
                    pred = feats.loc[cur_date, 'predicted_return'] if 'predicted_return' in feats.columns else np.nan
                    bb_u = feats.loc[cur_date, 'BB_Upper']
                    bb_l = feats.loc[cur_date, 'BB_Lower']
                    ema20 = feats.loc[cur_date, 'EMA_20']
                except Exception:
                    continue

                if pd.isna(mom) or pd.isna(vol):
                    continue
                w = strategy['weights'][i]

                # تصمیم‌گیری ترکیبی: امتیازدهی ساده مبتنی بر فیچرها
                score = 0.0
                # RSI
                if not pd.isna(rsi):
                    if rsi < 30:
                        score += 0.6
                    elif rsi > 70:
                        score -= 0.6
                # Bollinger
                price_now = price_data.loc[cur_date, sym]
                price_prev = price_data.loc[prev_date, sym]
                if not pd.isna(price_now) and not pd.isna(price_prev) and price_prev != 0:
                    # oversold if price < lower band
                    if not pd.isna(bb_l) and price_now < bb_l:
                        score += 0.5
                    if not pd.isna(bb_u) and price_now > bb_u:
                        score -= 0.5
                # predicted_return
                if not pd.isna(pred):
                    score += float(pred) * 2.0  # وزن دادن به پیش‌بینی مدل

                # volatility-based penalization
                if not pd.isna(vol) and vol > strategy['volatility_threshold']:
                    score -= 0.3

                asset_ret = 0.0
                try:
                    asset_ret = (price_now / price_prev) - 1.0
                except Exception:
                    asset_ret = 0.0

                # final contribution: وزن * risk_appetite * (asset_ret + score factor)
                daily_ret += w * strategy['risk_appetite'] * (asset_ret + 0.5 * score)

            # هزینه تراکنش در rebalance
            if (t - self.lookback_days) % int(strategy['rebalance_frequency']) == 0:
                turnover = np.sum(np.abs(strategy['weights'] - prev_weights))
                cost = self.transaction_cost * turnover
                daily_ret -= cost
                prev_weights = strategy['weights'].copy()

            rets.append(daily_ret)

        if len(rets) == 0:
            return -1000.0, -1000.0

        rets = pd.Series(rets)
        eps = 1e-9
        sharpe = (rets.mean() / (rets.std() + eps)) * np.sqrt(252)  # annualized
        total = (1 + rets).prod() - 1
        return float(sharpe), float(total)

    def dominates(self, a, b):
        return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])

    def pareto_front(self, scores):
        front = []
        for i, s in enumerate(scores):
            dominated = False
            for j, t in enumerate(scores):
                if i != j and self.dominates(t, s):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    def mutate(self, strategy):
        new_w = strategy['weights'] + np.random.normal(0, 0.05, size=self.num_assets)
        new_w = np.clip(new_w, 0, None)
        if new_w.sum() <= 0:
            new_w = np.ones_like(new_w) / len(new_w)
        else:
            new_w = new_w / new_w.sum()
        new_s = strategy.copy()
        new_s['weights'] = new_w
        # تغییر کوچک در پارامترها
        new_s['volatility_threshold'] = float(np.clip(new_s['volatility_threshold'] + np.random.normal(0, 0.005), 0.01, 0.5))
        new_s['risk_appetite'] = float(np.clip(new_s['risk_appetite'] + np.random.normal(0, 0.05), 0.05, 2.0))
        return new_s

    def optimize(self, historical_data, min_days=MIN_DAYS_FOR_FEATURES):
        print("🔍 آماده‌سازی فیچرها و هم‌ترازسازی ایندکس‌ها...")
        features_dict = {}
        price_series_dict = {}
        valid_symbols = []

        for sym, df in historical_data.items():
            if 'Close' not in df.columns:
                continue
            f = df[['momentum_7d','volatility_30d','RSI','BB_Upper','BB_Lower','EMA_20','predicted_return','return_1d']].copy()
            f_clean = f.dropna(how='any')
            if len(f_clean) < min_days:
                print(f"  ⚠️ {sym}: داده فیچر ناکافی ({len(f_clean)} روز) — چشم‌پوشی شد")
                continue
            features_dict[sym] = f_clean
            price_series_dict[sym] = df['Close'].reindex(f_clean.index)
            valid_symbols.append(sym)

        if len(valid_symbols) < 2:
            raise RuntimeError("دادهٔ کافی برای حداقل 2 دارایی موجود نیست. لیست نمادها را بررسی کنید.")

        # محاسبه ایندکس مشترک
        common_index = None
        for sym in valid_symbols:
            idx = features_dict[sym].index
            if common_index is None:
                common_index = idx
            else:
                common_index = common_index.intersection(idx)
        if common_index is None or len(common_index) < self.lookback_days + 2:
            raise RuntimeError("ایندکس مشترک خیلی کوتاه است؛ بازهٔ زمانی یا لیست نمادها را تغییر دهید.")

        # ساخت price_data
        price_data_dict = {}
        for sym in valid_symbols:
            price_series = price_series_dict[sym].reindex(common_index)
            price_data_dict[sym] = price_series
        price_data = pd.DataFrame(price_data_dict).dropna(how='any')
        if len(price_data) < self.lookback_days + 2:
            raise RuntimeError("پس از هم‌ترازی، طول سری زمانی برای محاسبات کافی نیست.")

        # هم‌ترازسازی features_list
        features_list = []
        for sym in valid_symbols:
            feats = features_dict[sym].reindex(price_data.index)
            features_list.append(feats)

        self.symbols = valid_symbols
        self.num_assets = len(features_list)
        print(f"✅ دارایی‌های قابل استفاده: {self.num_assets}, طول سری زمانی نهایی: {len(price_data)}")

        # الگوریتم تکاملی
        population = self.initialize_population()
        for gen in range(self.max_generations):
            scores = []
            for s in population:
                try:
                    score = self.evaluate_strategy(s, features_list, price_data)
                    scores.append(score)
                except Exception as e:
                    print(f"خطا در ارزیابی استراتژی: {e}")
                    scores.append((-1000.0, -1000.0))

            front_idx = self.pareto_front(scores)
            front_strats = [population[i] for i in front_idx]
            new_pop = front_strats.copy()
            # تولید فرزندان تا کامل شدن جمعیت
            while len(new_pop) < self.population_size:
                parent = population[np.random.randint(0, len(population))]
                new_pop.append(self.mutate(parent))
            population = new_pop
            print(f"نسل {gen+1}/{self.max_generations}: جمعیت={len(population)}, عدد در مرز پارتو={len(front_idx)}")

        # محاسبه نتایج نهایی و مرز پارتو
        final_scores = []
        for s in population:
            try:
                score = self.evaluate_strategy(s, features_list, price_data)
                final_scores.append(score)
            except Exception:
                final_scores.append((-1000.0, -1000.0))
        final_front_idx = self.pareto_front(final_scores)
        pareto_set = [(population[i], final_scores[i]) for i in final_front_idx]

        return pareto_set, valid_symbols

# ---------------------------
# اجرای اصلی
# ---------------------------
if __name__ == "__main__":
    print("📥 دانلود داده‌ها و آماده‌سازی فیچرها...")
    hist = download_and_compute(SYMBOLS, start=START, end=END)
    if len(hist) < 2:
        raise SystemExit("دیتای کافی دریافت نشد. لطفاً نمادها و بازه زمانی را بررسی کنید.")
    print("\n🔮 ساخت فیچر پیش‌بینی (مدل GRU / fallback) ...")
    hist, preds = add_predicted_feature_to_hist(hist)

    # اجرای CPOA
    print("\n⚙️ شروع بهینه‌سازی چندهدفه (CPOA)...")
    cpoa = MultiObjectiveCPOA(population_size=20, max_generations=5, transaction_cost=0.001, lookback_days=LOOKBACK_DAYS)
    pareto_set, used_symbols = cpoa.optimize(hist, min_days=MIN_DAYS_FOR_FEATURES)

    # ذخیره نتایج
    rows = []
    for strat, (sharpe, tot) in pareto_set:
        rows.append({
            "Sharpe": sharpe,
            "TotalReturn": tot,
            "VolatilityThreshold": strat['volatility_threshold'],
            "RiskAppetite": strat['risk_appetite'],
            "RebalanceFreq": strat['rebalance_frequency'],
            "Weights": ",".join([f"{w:.6f}" for w in strat['weights']])
        })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ نتایج مرز پارتو در: {OUTPUT_CSV}")
        # نمودار
        plt.figure(figsize=(8,6))
        plt.scatter(df_out["Sharpe"], df_out["TotalReturn"], alpha=0.7)
        plt.xlabel("Sharpe Ratio"); plt.ylabel("Total Return"); plt.title("Pareto Front (Sharpe vs Total Return)")
        plt.grid(True, alpha=0.3)
        plt.show()
        best_idx = df_out["Sharpe"].idxmax()
        best = df_out.loc[best_idx]
        print(f"\n🏆 بهترین استراتژی (براساس Sharpe):")
        print(best.to_string())
    else:
        print("هیچ استراتژی روی مرز پارتو تولید نشد.")
