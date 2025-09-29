!pip install yfinance

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedCPOA:
    def __init__(self, num_assets, population_size=30, max_iterations=50):
        self.num_assets = num_assets
        self.population_size = population_size
        self.max_iterations = max_iterations
        
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            weights = np.random.dirichlet(np.ones(self.num_assets))
            strategy = {
                'weights': weights,
                'momentum_window': np.random.choice([7, 14, 21]),
                'volatility_threshold': np.random.uniform(0.03, 0.08),
                'risk_appetite': np.random.uniform(0.3, 0.8),
                'rebalance_frequency': np.random.choice([7, 14, 30])
            }
            population.append(strategy)
        return population
    
    def extract_features(self, prices, volume):
        features = {}
        
        # محاسبه بازده‌ها
        features['returns_1d'] = prices.pct_change(1)
        features['returns_7d'] = prices.pct_change(7)
        features['returns_30d'] = prices.pct_change(30)
        
        # نوسانات
        features['volatility_7d'] = prices.pct_change().rolling(7).std()
        features['volatility_30d'] = prices.pct_change().rolling(30).std()
        
        # مومنتوم
        features['momentum_7d'] = prices.pct_change(7)
        features['momentum_30d'] = prices.pct_change(30)
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # حجم
        features['volume_ma'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma']
        
        #return pd.DataFrame(features)
        # ترکیب تمام features در یک DataFrame
        df_list = []
        for feature_name, feature_df in features.items():
            temp_df = feature_df.reset_index()
            temp_df.columns = ['Date', feature_name]  # تغییر نام ستون‌ها
            df_list.append(temp_df)

        # ادغام تمام DataFrame‌ها بر اساس تاریخ
        from functools import reduce
        final_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), df_list)

        return final_df
    
    def calculate_strategy_returns(self, strategy, features_dict, price_data):
        portfolio_returns = []
        
        for crypto_idx in range(self.num_assets):
            crypto_features = features_dict[crypto_idx]
            weight = strategy['weights'][crypto_idx]
            
            if weight > 0.01 and len(crypto_features) > 30:
                # استفاده از مومنتوم برای پیش‌بینی
                momentum_signal = crypto_features['momentum_7d'].iloc[-1]
                volatility = crypto_features['volatility_30d'].iloc[-1]
                
                if abs(momentum_signal) > 0.02 and volatility < strategy['volatility_threshold']:
                    adjusted_return = momentum_signal * weight * strategy['risk_appetite']
                    portfolio_returns.append(adjusted_return)
        
        if portfolio_returns:
            return np.mean(portfolio_returns)
        return 0
    
    def evaluate_strategy(self, strategy, features_dict, price_data, lookback_days=30):
        returns = []
        
        for day in range(lookback_days, len(price_data)):
            daily_return = self.calculate_strategy_returns(strategy, features_dict, price_data.iloc[:day])
            returns.append(daily_return)
        
        if len(returns) == 0:
            return -1000
        
        returns = pd.Series(returns)
        
        # محاسبه معیارهای ارزیابی
        total_return = np.prod(1 + returns) - 1
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365)
        
        negative_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / (np.std(negative_returns) + 1e-8) if len(negative_returns) > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean()
        
        # امتیاز نهایی
        score = (0.3 * sharpe_ratio + 
                 0.25 * (total_return * 100) + 
                 0.2 * sortino_ratio + 
                 0.15 * (1 / (abs(max_drawdown) + 0.01)) + 
                 0.1 * win_rate)
        
        return score
    
    def mutate_parameter(self, current, best, possible_values=None):
        if possible_values:
            current_idx = possible_values.index(current) if current in possible_values else len(possible_values)//2
            new_idx = max(0, min(len(possible_values)-1, 
                               current_idx + np.random.randint(-1, 2)))
            return possible_values[new_idx]
        else:
            mutation = np.random.normal(0, 0.1)
            return max(0.01, min(0.99, current + mutation))
    
    def evolve_strategy(self, strategy, global_best, iteration):
        new_weights = np.zeros_like(strategy['weights'])
        
        for i in range(self.num_assets):
            if np.random.random() < 0.3:  # 30% احتمال اکتشاف
                mutation = np.random.normal(0, 0.05)
                new_weights[i] = max(0, strategy['weights'][i] + mutation)
            else:  # 70% احتمال بهره‌برداری
                learning_rate = 0.3
                new_weights[i] = (1 - learning_rate) * strategy['weights'][i] + learning_rate * global_best['weights'][i]
        
        # نرمال‌سازی
        new_weights = new_weights / np.sum(new_weights)
        
        new_strategy = {
            'weights': new_weights,
            'momentum_window': self.mutate_parameter(
                strategy['momentum_window'], 
                global_best['momentum_window'],
                [7, 14, 21]
            ),
            'volatility_threshold': self.mutate_parameter(
                strategy['volatility_threshold'],
                global_best['volatility_threshold']
            ),
            'risk_appetite': self.mutate_parameter(
                strategy['risk_appetite'],
                global_best['risk_appetite']
            ),
            'rebalance_frequency': strategy['rebalance_frequency']
        }
        
        return new_strategy
    
    def optimize(self, historical_data):
        print("🔍 شروع استخراج ویژگی‌ها...")
        features_dict = {}
        price_data = pd.DataFrame()
        
        for i, (crypto, data) in enumerate(historical_data.items()):
            if len(data) > 30:  # فقط رمزارزهای با داده کافی
                features_dict[i] = self.extract_features(data['Close'], data['Volume'])
                price_data[crypto] = data['Close']
        
        if len(features_dict) == 0:
            print("❌ داده کافی برای آنالیز وجود ندارد")
            return None, -1000
        
        self.num_assets = len(features_dict)
        population = self.initialize_population()
        global_best = None
        best_score = -np.inf
        
        print("🚀 شروع فرآیند بهینه‌سازی...")
        for iteration in range(self.max_iterations):
            scores = []
            
            for strategy in population:
                score = self.evaluate_strategy(strategy, features_dict, price_data)
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    global_best = strategy.copy()
                    print(f"✅ تکرار {iteration+1}: بهترین امتیاز = {best_score:.4f}")
            
            # ایجاد نسل جدید
            new_population = []
            elite_count = max(2, self.population_size // 5)
            
            # انتخاب بهترین‌ها (Elitism)
            elite_indices = np.argsort(scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # تولید استراتژی‌های جدید
            while len(new_population) < self.population_size:
                parent = population[np.random.randint(0, self.population_size)]
                child = self.evolve_strategy(parent, global_best, iteration)
                new_population.append(child)
            
            population = new_population
        
        return global_best, best_score

def get_crypto_data(symbols, start_date, end_date):
    print(f"📥 دریافت داده‌های از {start_date} تا {end_date}...")
    data = {}
    successful_symbols = []
    
    for symbol in symbols:
        try:
            crypto_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(crypto_data) > 30:  # حداقل 30 روز داده
                data[symbol] = crypto_data
                successful_symbols.append(symbol)
                print(f"✅ {symbol}: {len(crypto_data)} روز داده")
            else:
                print(f"❌ {symbol}: داده ناکافی")
        except Exception as e:
            print(f"❌ {symbol}: خطا - {e}")
    
    print(f"🎯 تعداد رمزارزهای موفق: {len(successful_symbols)} از {len(symbols)}")
    return data, successful_symbols

def predict_future_performance(best_strategy, historical_data, prediction_days=30):
    print("\n🔮 پیش‌بینی عملکرد یک ماه آینده...")
    
    # محاسبه متوسط بازده روزانه تاریخی
    total_historical_return = 0
    count = 0
    
    for crypto, data in historical_data.items():
        if len(data) > 30:
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                avg_return = returns.mean()
                total_historical_return += avg_return
                count += 1
    
    if count == 0:
        return "❌ امکان پیش‌بینی وجود ندارد"
    
    avg_daily_return = total_historical_return / count
    predicted_monthly_return = (1 + avg_daily_return) ** prediction_days - 1
    
    # تعدیل بر اساس استراتژی
    strategy_multiplier = best_strategy['risk_appetite'] * 1.5
    adjusted_return = predicted_monthly_return * strategy_multiplier
    
    return adjusted_return

# لیست ۵۰ رمزارز محبوب
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "BUSD-USD", "XRP-USD",
                  "ADA-USD", "DOGE-USD", "DOT-USD", "SHIB-USD", "BCH-USD", "AVAX-USD", "FUN-USD", 
                  "LTC-USD", "LINK-USD", "TRX-USD", "CRO-USD", "NEAR-USD", "SOL-USD", "ATOM-USD", 
                  "ETC-USD", "ALGO-USD", "ICP-USD", "XLM-USD", "XMR-USD", "FIL-USD", "VET-USD", 
                  "FLOW-USD", "QNT-USD", "EGLD-USD", "MANA-USD", "XTZ-USD", "SAND-USD", "AXS-USD", 
                  "HBAR-USD", "EOS-USD", "CRV-USD", "KCS-USD", "BAT-USD", "DAI-USD", "RUNE-USD", 
                  "HNT-USD", "MKR-USD", "LRC-USD", "ZEN-USD", "ONE-USD", "ZIL-USD", "AAVE-USD", "GALA-USD"]

def main():
    print("🎯 الگوریتم پیشرفته بهینه‌سازی پرتفوی رمزارز (CPOA)")
    print("=" * 60)
    
    # تاریخ‌ها
    end_date = datetime.now()
    start_date_train = end_date - timedelta(days=180)  # 6 ماه داده آموزشی
    start_date_test = end_date - timedelta(days=30)   # 1 ماه داده تست
    
    # دریافت داده‌های آموزشی
    historical_data, successful_symbols = get_crypto_data(
        CRYPTO_SYMBOLS, start_date_train, end_date
    )
    
    if len(historical_data) < 5:
        print("❌ داده کافی برای تحلیل وجود ندارد")
        return
    
    print(f"\n📊 تحلیل داده‌های {len(historical_data)} رمزارز...")
    
    # اجرای الگوریتم بهینه‌سازی
    cpoa = AdvancedCPOA(num_assets=len(historical_data), population_size=20, max_iterations=30)
    best_strategy, best_score = cpoa.optimize(historical_data)
    
    if best_strategy is None:
        print("❌ الگوریتم نتوانست راه‌حل مناسبی پیدا کند")
        return
    
    print("\n" + "=" * 60)
    print("🎉 نتایج نهایی بهینه‌سازی")
    print("=" * 60)
    
    # نمایش بهترین استراتژی
    print(f"🏆 بهترین امتیاز بهینه‌سازی: {best_score:.4f}")
    print(f"📈 پنجره مومنتوم بهینه: {best_strategy['momentum_window']} روز")
    print(f"⚡ آستانه نوسان بهینه: {best_strategy['volatility_threshold']:.3f}")
    print(f"🎯 اشتهای ریسک بهینه: {best_strategy['risk_appetite']:.3f}")
    print(f"🔄 فرکانس بازتعادل: هر {best_strategy['rebalance_frequency']} روز")
    
    # نمایش ۱۰ رمزارز برتر
    print(f"\n🏅 ۱۰ رمزارز برتر پیشنهادی:")
    crypto_weights = list(zip(successful_symbols, best_strategy['weights']))
    crypto_weights.sort(key=lambda x: x[1], reverse=True)
    
    for i, (crypto, weight) in enumerate(crypto_weights[:10]):
        if weight > 0.01:
            print(f"{i+1:2d}. {crypto:8s}: {weight:.2%}")
    
    # پیش‌بینی عملکرد
    predicted_return = predict_future_performance(best_strategy, historical_data)
    
    if isinstance(predicted_return, float):
        print(f"\n🔮 پیش‌بینی بازده یک ماه آینده: {predicted_return:.2%}")
        
        if predicted_return > 0.1:
            print("💚 پیش‌بینی: بازار صعودی - مناسب برای سرمایه‌گذاری")
        elif predicted_return > 0:
            print("💛 پیش‌بینی: بازار خنثی - سرمایه‌گذاری با احتیاط")
        else:
            print("🔴 پیش‌بینی: بازار نزولی - بهتر است منتظر بمانید")
    else:
        print(predicted_return)
    
    print(f"\n💡 توصیه‌ها:")
    print("- پرتفوی پیشنهادی را به صورت پلکانی اجرا کنید")
    print("- همواره از مدیریت ریسک استفاده کنید")
    print("- عملکرد را به طور منظم مانیتور کنید")
    print("- در صورت تغییر شرایط بازار، استراتژی را بازبینی کنید")

if __name__ == "__main__":
    main()
