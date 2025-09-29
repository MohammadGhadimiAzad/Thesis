!pip install yfinance --quiet

# اجرا در سل دوم (پس از نصب yfinance)
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

np.random.seed(42)
OUTPUT_CSV = "/content/sample_data/pareto_front_results.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

class MultiObjectiveCPOA:
    def __init__(self, population_size=20, max_generations=8, transaction_cost=0.001, lookback_days=30):
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
                'volatility_threshold': float(np.random.uniform(0.03, 0.08)),
                'risk_appetite': float(np.random.uniform(0.3, 0.8)),
                'rebalance_frequency': int(np.random.choice([7, 14, 30]))
            }
            pop.append(strat)
        return pop

    def extract_features(self, prices: pd.Series):
        prices = prices.sort_index()
        
        # محاسبه فیچرها و اطمینان از 1 بعدی بودن
        momentum_7d = prices.pct_change(7).squeeze()
        volatility_30d = prices.pct_change().rolling(30).std().squeeze()
        
        # ایجاد DataFrame با اطمینان از 1 بعدی بودن داده‌ها
        feats = pd.DataFrame({
            'momentum_7d': momentum_7d.values if hasattr(momentum_7d, 'values') else momentum_7d,
            'volatility_30d': volatility_30d.values if hasattr(volatility_30d, 'values') else volatility_30d
        }, index=prices.index)
        
        return feats

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
                try:
                    mom = feats.loc[cur_date, 'momentum_7d']
                    vol = feats.loc[cur_date, 'volatility_30d']
                except Exception:
                    continue
                if pd.isna(mom) or pd.isna(vol):
                    continue
                w = strategy['weights'][i]
                if abs(mom) > 0.02 and vol < strategy['volatility_threshold']:
                    try:
                        p_now = price_data.loc[cur_date, sym]
                        p_prev = price_data.loc[prev_date, sym]
                    except KeyError:
                        continue
                    if pd.isna(p_now) or pd.isna(p_prev) or p_prev == 0:
                        continue
                    asset_ret = (p_now / p_prev) - 1.0
                    daily_ret += w * strategy['risk_appetite'] * asset_ret

            if (t - self.lookback_days) % strategy['rebalance_frequency'] == 0:
                turnover = np.sum(np.abs(strategy['weights'] - prev_weights))
                cost = self.transaction_cost * turnover
                daily_ret -= cost
                prev_weights = strategy['weights'].copy()

            rets.append(daily_ret)

        if len(rets) == 0:
            return -1000.0, -1000.0
            
        rets = pd.Series(rets)
        eps = 1e-9
        sharpe = (rets.mean() / (rets.std() + eps)) * np.sqrt(365)
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
        return new_s

    def optimize(self, historical_data, min_days=200):
        print("🔍 آماده‌سازی فیچرها و هم‌ترازسازی ایندکس‌ها...")

        # 1) تولید فیچرها و پاکسازی
        features_dict = {}
        price_series_dict = {}
        valid_symbols = []

        for sym, df in historical_data.items():
            if 'Close' not in df.columns:
                continue
            
            # اطمینان از سری 1 بعدی
            close_prices = df['Close']
            if close_prices.ndim > 1:
                close_prices = close_prices.squeeze()
                
            f = self.extract_features(close_prices)
            f_clean = f.dropna(subset=['momentum_7d', 'volatility_30d'])
            if len(f_clean) < min_days:
                print(f"  ⚠️ {sym}: داده فیچر ناکافی ({len(f_clean)} روز) — چشم‌پوشی شد")
                continue
            features_dict[sym] = f_clean
            price_series_dict[sym] = close_prices.reindex(f_clean.index)
            valid_symbols.append(sym)

        if len(valid_symbols) < 2:
            raise RuntimeError("دادهٔ کافی برای حداقل 2 دارایی موجود نیست. لیست نمادها را بررسی کنید.")

        # 2) محاسبه ایندکس مشترک
        common_index = None
        for sym in valid_symbols:
            idx = features_dict[sym].index
            if common_index is None:
                common_index = idx
            else:
                common_index = common_index.intersection(idx)

        if common_index is None or len(common_index) < self.lookback_days + 2:
            raise RuntimeError("ایندکس مشترک خیلی کوتاه است؛ بازهٔ زمانی یا لیست نمادها را تغییر دهید.")

        # 3) ساخت price_data با اطمینان از 1 بعدی بودن ستون‌ها
        price_data_dict = {}
        for sym in valid_symbols:
            price_series = price_series_dict[sym].reindex(common_index)
            # اطمینان از 1 بعدی بودن داده‌ها
            if price_series.ndim > 1:
                price_series = price_series.squeeze()
            price_data_dict[sym] = price_series
            
        price_data = pd.DataFrame(price_data_dict)
        price_data = price_data.dropna(how='any')
        
        if len(price_data) < self.lookback_days + 2:
            raise RuntimeError("پس از هم‌ترازی، طول سری زمانی برای محاسبات کافی نیست.")

        # 4) هم‌ترازسازی features_list
        features_list = []
        for sym in valid_symbols:
            feats = features_dict[sym].reindex(price_data.index)
            features_list.append(feats)

        # ذخیره نمادها و تعداد assets
        self.symbols = valid_symbols
        self.num_assets = len(features_list)
        print(f"✅ دارایی‌های قابل استفاده: {self.num_assets}, طول سری زمانی نهایی: {len(price_data)}")

        # 5) الگوریتم تکاملی
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
            
            while len(new_pop) < self.population_size:
                parent = population[np.random.randint(0, len(population))]
                new_pop.append(self.mutate(parent))
                
            population = new_pop
            print(f"نسل {gen+1}/{self.max_generations}: جمعیت={len(population)}, تعداد در مرز پارتو={len(front_idx)}")

        # محاسبه نتایج نهایی
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

# ---------------------- اجرای اصلی ----------------------
if __name__ == "__main__":
    symbols = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "BUSD-USD", "XRP-USD",
                  "ADA-USD", "DOGE-USD", "DOT-USD", "SHIB-USD", "BCH-USD", "AVAX-USD", "FUN-USD", 
                  "LTC-USD", "LINK-USD", "TRX-USD", "CRO-USD", "NEAR-USD", "SOL-USD", "ATOM-USD", 
                  "ETC-USD", "ALGO-USD", "ICP-USD", "XLM-USD", "XMR-USD", "FIL-USD", "VET-USD", 
                  "FLOW-USD", "QNT-USD", "EGLD-USD", "MANA-USD", "XTZ-USD", "SAND-USD", "AXS-USD", 
                  "HBAR-USD", "EOS-USD", "CRV-USD", "KCS-USD", "BAT-USD", "DAI-USD", "RUNE-USD", 
                  "HNT-USD", "MKR-USD", "LRC-USD", "ZEN-USD", "ONE-USD", "ZIL-USD", "AAVE-USD", "GALA-USD"]
    
    end = datetime.now()
    start = end - timedelta(days=365)

    # دانلود داده‌ها
    historical = {}
    print("📥 دانلود داده‌ها (yfinance)...")
    for s in symbols:
        try:
            df = yf.download(s, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) >= 200:
                historical[s] = df
                print(f"  ✅ {s}: {len(df)} روز")
            else:
                print(f"  ⚠️ {s}: نادیده گرفته شد ({len(df)} روز)")
        except Exception as e:
            print(f"  ❌ {s}: خطا - {e}")

    if len(historical) < 2:
        print("نمادهای دریافتی:", list(historical.keys()))
        raise SystemExit("دیتای کافی دریافت نشد. لطفاً اتصال اینترنت و نمادها را چک کن.")

    cpoa = MultiObjectiveCPOA(population_size=20, max_generations=5, transaction_cost=0.001, lookback_days=30)
    
    try:
        pareto_set, used_symbols = cpoa.optimize(historical, min_days=180)
        print(f"✅ بهینه‌سازی با موفقیت انجام شد. {len(pareto_set)} استراتژی در مرز پارتو")

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
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ نتایج مرز پارتو در: {OUTPUT_CSV}")

        # نمایش نمودار
        if not df_out.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_out["Sharpe"], df_out["TotalReturn"], c="blue", alpha=0.7)
            plt.xlabel("Sharpe Ratio")
            plt.ylabel("Total Return")
            plt.title("Pareto Front (Sharpe vs Total Return)")
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # نمایش بهترین استراتژی
            best_idx = df_out["Sharpe"].idxmax()
            best = df_out.loc[best_idx]
            print(f"\n🏆 بهترین استراتژی:")
            print(f"   Sharpe Ratio: {best['Sharpe']:.4f}")
            print(f"   Total Return: {best['TotalReturn']:.4f}")
            print(f"   Volatility Threshold: {best['VolatilityThreshold']:.4f}")
            print(f"   Risk Appetite: {best['RiskAppetite']:.4f}")
            print(f"   Rebalance Frequency: {best['RebalanceFreq']} روز")
            
        else:
            print("هیچ استراتژی روی مرز پارتو تولید نشد.")
            
    except Exception as e:
        print(f"❌ خطا در بهینه‌سازی: {e}")
        import traceback
        traceback.print_exc()
