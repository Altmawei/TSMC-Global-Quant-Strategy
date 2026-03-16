import numpy as np
import pandas as pd
import yfinance as yf
import os
import requests
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TAIFEX_Contest_V50:

    def __init__(self, initial_capital=2000000):
        self.initial_capital = initial_capital
        self.file = "strategy_report.csv"
        self.target_vol = 0.008 
        self.tg_token = os.getenv("TG_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        self.balance, self.peak = self.load_equity()

    def load_equity(self):
        if os.path.isfile(self.file):
            try:
                df = pd.read_csv(self.file)
                if not df.empty:
                    return float(df.iloc[-1]["equity"]), float(df["equity"].max())
            except: pass
        return self.initial_capital, self.initial_capital

    def save_equity(self):
        df = pd.DataFrame([{"date": datetime.now().strftime("%Y-%m-%d"), "equity": self.balance}])
        try:
            df.to_csv(self.file, mode="a", index=False, header=not os.path.isfile(self.file))
        except: pass

    def get_data(self):
        """針對台指 VIX 缺失，改用台股歷史波動率逆推，拒絕美股替代"""
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="250d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(1)
        df = raw.ffill()

        try:
            tw_vix = yf.download("^TWVIX", period="250d", progress=False)['Close']
            if tw_vix.dropna().empty: raise ValueError("Empty")
            df["^TWVIX"] = tw_vix
        except:
            # 使用台股 HV 逆推 VIX (模擬台灣權證市場定價特性)
            print("警告：抓不到台指 VIX，啟動 HV 擬合演算法...")
            hv = df["^TWII"].pct_change().rolling(20).std() * np.sqrt(252) * 100
            df["^TWVIX"] = (hv * 1.15 + 1.5).ffill() # 針對台股波動補償

        return df.dropna(subset=["^TWII"])

    def z_score(self, series, window=20):
        if len(series) < window: return 0
        std = series.tail(window).std()
        if std < 1e-8 or np.isnan(std): return 0
        return (series.iloc[-1] - series.tail(window).mean()) / std

    def factor_model(self, df):
        adr = (df["TSM"] / 5 * df["TWD=X"]) / df["2330.TW"]
        f1 = self.z_score(adr); f2 = self.z_score(df["SOXX"].pct_change())
        f3 = -self.z_score(df["^TWVIX"])
        f4 = self.z_score(df["^TWOII"] / df["^TWII"])
        f5 = -self.z_score(df["TWD=X"])
        f6 = df["^TWII"].pct_change().tail(10).mean() * 100
        basis = (df["^TWII"] * 1.002) - df["^TWII"]
        f7 = self.z_score(basis)
        f8 = -df["^TWVIX"].pct_change().tail(5).mean()
        return np.nan_to_num([f1, f2, f3, f4, f5, f6, f7, f8])

    def get_market_regime(self, df):
        ret = df["^TWII"].pct_change().fillna(0)
        vol = ret.rolling(20).std().fillna(0)
        X = np.column_stack([ret.tail(150), vol.tail(150)])
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
        means = [X[kmeans.labels_ == i, 0].mean() for i in range(4)]
        rank = np.where(np.argsort(means) == kmeans.labels_[-1])[0][0]
        return ["panic", "bear", "range", "bull"][rank]

    def calculate_sizing(self, df, score):
        returns = df["^TWII"].pct_change().dropna()
        recent_vol = returns.tail(20).std()
        if recent_vol < 1e-8: return 0
        # 實戰限制：槓桿降至 2.5，採半凱利原則 (0.5 倍)
        lev = min(self.target_vol / recent_vol, 2.5)
        dd = (self.peak - self.balance) / self.peak
        if dd > 0.05: lev *= 0.5
        if dd > 0.10: lev = 0
        edge = (min(abs(score) / 3, 0.5)) * 0.5
        return self.balance * lev * edge

    def generate_orders(self):
        df = self.get_data()
        if df.empty: return "error", "N/A", 0, 0
        factors = self.factor_model(df); score = np.mean(factors)
        regime = self.get_market_regime(df)
        idx_price = df["^TWII"].iloc[-1]
        trade_cap = self.calculate_sizing(df, score)
        if abs(score) < 0.15: regime = "neutral"

        # 單位計算
        tx = max(1, int(trade_cap / 2000000))
        mtx = max(1, int(trade_cap / 400000))
        tsmc_f = max(0, int(trade_cap / (df["2330.TW"].iloc[-1] * 200)))
        
        # 選擇權履約價計算 (四捨五入至百位數)
        call_strike = int(round(idx_price + 100, -2))
        put_strike = int(round(idx_price - 100, -2))

        if regime == "bull":
            orders = f"TX多 {tx}口 | MTX多 {mtx}口 | 台積期多 {tsmc_f}口\nOPT: 買入週台指 Call @ {call_strike}"
        elif regime == "bear":
            orders = f"TX空 {tx}口 | MTX空 {mtx}口 | 台積期空 {tsmc_f}口\nOPT: 買入週台指 Put @ {put_strike}"
        elif regime == "panic":
            orders = "全面避險 | 僅保留黃金期貨多單 | 買入深度價外Put (MDD防護)"
        else:
            orders = "低波動模式 | 僅微台 1 口維持參賽資格"

        return regime, orders, score, df["2330.TW"].iloc[-1]

    def report(self):
        regime, orders, score, price = self.generate_orders()
        msg = f"=== TAIFEX V50 ===\n狀態: {regime.upper()} | 分數: {score:.3f}\n台積電: {price:.1f}\n--- 執行指令 ---\n{orders}\n================"
        print(msg); self.send_telegram(msg); self.save_equity()

    def send_telegram(self, msg):
        if self.tg_token and self.tg_chat_id:
            try: requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg}, timeout=5)
            except: pass

if __name__ == "__main__":
    bot = TAIFEX_Contest_V50(); bot.report()
