import numpy as np
import pandas as pd
import yfinance as yf
import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header
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

    def _to_scalar(self, val):
        """極強效轉型工具：處理 Series, DataFrame 或 MultiIndex 資料"""
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                temp = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
                return float(temp.iloc[-1])
            return float(val)
        except:
            return 0.0

    def get_data(self):
        """抓取台美股數據，包含 VIX 擬合邏輯"""
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="250d", progress=False)
        
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(-1)
        
        df = raw.ffill()

        try:
            tw_vix = yf.download("^TWVIX", period="250d", progress=False)['Close']
            if isinstance(tw_vix, pd.DataFrame): tw_vix = tw_vix.iloc[:, 0]
            if tw_vix.dropna().empty: raise ValueError("Empty")
            df["^TWVIX"] = tw_vix
        except:
            print("警告：抓不到台指 VIX，啟動 HV 擬合演算法...")
            tw_price = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
            tw_returns = tw_price.pct_change()
            hv = tw_returns.rolling(20).std() * np.sqrt(252) * 100
            df["^TWVIX"] = (hv * 1.15 + 1.5).ffill() 

        return df.dropna(subset=["^TWII"])

    def z_score(self, series, window=20):
        if len(series) < window: return 0.0
        s = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
        tail = s.tail(window)
        std = tail.std()
        if std < 1e-8 or np.isnan(std) or np.isinf(std): return 0.0
        return float((s.iloc[-1] - tail.mean()) / std)

    def factor_model(self, df):
        def get_s(name):
            col = df[name]
            return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

        tsm = get_s("TSM"); twd = get_s("TWD=X"); tsmc = get_s("2330.TW")
        soxx = get_s("SOXX"); twii = get_s("^TWII"); twoii = get_s("^TWOII")
        vix = get_s("^TWVIX")

        # 核心 V50 因子計算
        f1 = self.z_score((tsm / 5 * twd) / tsmc)
        f2 = self.z_score(soxx.pct_change())
        f3 = -self.z_score(vix)
        f4 = self.z_score(twoii / twii)
        f5 = -self.z_score(twd)
        f6 = self.z_score(twii.pct_change(), window=10) * 10
        
        idx_p = twii; basis = (idx_p * 1.002) - idx_p
        f7 = self.z_score(basis)
        f8 = -self.z_score(vix.pct_change(), window=5)
        
        factors = [self._to_scalar(f) for f in [f1, f2, f3, f4, f5, f6, f7, f8]]
        return np.nan_to_num(factors)

    def get_market_regime(self, df):
        twii = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
        ret = twii.pct_change().fillna(0)
        vol = ret.rolling(20).std().fillna(0)
        X = np.column_stack([ret.tail(150), vol.tail(150)])
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
        means = [X[kmeans.labels_ == i, 0].mean() for i in range(4)]
        rank = np.where(np.argsort(means) == kmeans.labels_[-1])[0][0]
        return ["panic", "bear", "range", "bull"][rank]

    def calculate_sizing(self, df, score):
        twii = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
        returns = twii.pct_change().dropna()
        recent_vol = self._to_scalar(returns.tail(20).std())
        if recent_vol < 1e-8: return 0.0
        lev = min(self.target_vol / recent_vol, 2.5)
        dd = (self.peak - self.balance) / self.peak
        if dd > 0.05: lev *= 0.5
        if dd > 0.10: lev = 0.0
        edge = (min(abs(self._to_scalar(score)) / 3, 0.5)) * 0.5
        return float(self.balance * lev * edge)

    def generate_orders(self):
        df = self.get_data()
        if df.empty: return "error", "N/A", 0.0, 0.0
        factors = self.factor_model(df); score = float(np.mean(factors))
        regime = self.get_market_regime(df)
        
        twii_last = self._to_scalar(df["^TWII"])
        tsmc_last = self._to_scalar(df["2330.TW"])
        trade_cap = self.calculate_sizing(df, score)
        if abs(score) < 0.15: regime = "neutral"

        # 台指期單位計算
        tx = max(1, int(trade_cap / 2000000))
        mtx = max(1, int(trade_cap / 400000))
        tsmc_f = max(0, int(trade_cap / (tsmc_last * 200 + 1)))
        
        call_strike = int(round(twii_last + 100, -2))
        put_strike = int(round(twii_last - 100, -2))

        if regime == "bull":
            orders = f"TX多 {tx}口 | MTX多 {mtx}口 | 台積期多 {tsmc_f}口\nOPT: 買入週台指 Call @ {call_strike}"
        elif regime == "bear":
            orders = f"TX空 {tx}口 | MTX空 {mtx}口 | 台積期空 {tsmc_f}口\nOPT: 買入週台指 Put @ {put_strike}"
        elif regime == "panic":
            orders = "全面避險 | 僅保留黃金期貨多單 | 買入深度價外Put (MDD防護)"
        else:
            orders = "低波動模式 | 僅微台 1 口維持參賽資格"

        return regime, orders, score, tsmc_last

    def send_email(self, content):
        """參考金銀比模型加入的郵件功能"""
        sender = 'jeffreylin1201@gmail.com'
        receivers = ['jeffreylin1201@gmail.com'] 
        password = 'udcgkrdfdfoqznsn' 

        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header("V50 Strategy Bot", 'utf-8')
        message['To'] = Header("Strategy Admin", 'utf-8')
        message['Subject'] = Header(f"V50 策略報告 - {datetime.now().strftime('%m/%d %H:%M')}", 'utf-8')

        try:
            smtp_obj = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp_obj.login(sender, password)
            smtp_obj.sendmail(sender, receivers, message.as_string())
            print("郵件發送成功")
        except Exception as e:
            print(f"郵件發送失敗: {e}")

    def report(self):
        regime, orders, score, price = self.generate_orders()
        msg = f"=== TAIFEX V50 ===\n狀態: {regime.upper()} | 分數: {score:.3f}\n台積電: {price:.1f}\n--- 執行指令 ---\n{orders}\n================"
        print(msg)
        self.send_telegram(msg)
        self.send_email(msg) # 呼叫郵件發送
        self.save_equity()

    def send_telegram(self, msg):
        if self.tg_token and self.tg_chat_id:
            try: requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg}, timeout=10)
            except: pass

if __name__ == "__main__":
    bot = TAIFEX_Contest_V50(); bot.report()
