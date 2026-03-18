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
from sklearn.linear_model import Ridge  # 用於每日自動計算權重
from statsmodels.tsa.arima.model import ARIMA

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
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                temp = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
                return float(temp.iloc[-1])
            return float(val)
        except: return 0.0

    def get_data(self):
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="300d", progress=False) # 增加長度以供回歸
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(-1)
        
        # 去噪處理
        df = raw.ffill().ewm(span=5, adjust=False).mean()

        try:
            tw_vix = yf.download("^TWVIX", period="300d", progress=False)['Close']
            if isinstance(tw_vix, pd.DataFrame): tw_vix = tw_vix.iloc[:, 0]
            df["^TWVIX"] = tw_vix.ffill()
        except:
            tw_price = df["^TWII"]
            hv = tw_price.pct_change().rolling(20).std() * np.sqrt(252) * 100
            df["^TWVIX"] = (hv * 1.15 + 1.5).ffill() 

        return df.dropna(subset=["^TWII"])

    def z_score(self, s, w=20):
        if len(s) < w: return 0.0
        tail = s.tail(w)
        std = tail.std()
        return float((s.iloc[-1] - tail.mean()) / (std + 1e-8))

    def get_optimized_weights(self, df):
        """實作：每日自動回歸找權重"""
        try:
            # 1. 建立歷史因子矩陣 (簡化版計算過去 60 天的因子表現)
            # 為了效能與穩定度，我們拿最近 60 天做回歸
            lookback = 60
            y = df["^TWII"].pct_change().shift(-1).dropna().tail(lookback)
            
            # 建立特徵 X (這裡我們用簡化的 ADR 和 SOXX 當代表，實際運算會消耗較多資源)
            f1_hist = (df["TSM"]/5 * df["TWD=X"] / df["2330.TW"]).pct_change().tail(lookback)
            f2_hist = df["SOXX"].pct_change().tail(lookback)
            
            X = np.column_stack([f1_hist, f2_hist])
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            
            # 將係數標準化作為權重 (確保總和約為 1)
            raw_weights = np.abs(model.coef_)
            return raw_weights / (np.sum(raw_weights) + 1e-8)
        except:
            return np.array([0.15, 0.15]) # 失敗時的預設權重

    def factor_model(self, df):
        # 8 大因子
        f1 = self.z_score((df["TSM"] / 5 * df["TWD=X"]) / df["2330.TW"])
        f2 = self.z_score(df["SOXX"].pct_change())
        f3 = -self.z_score(df["^TWVIX"])
        f4 = self.z_score(df["^TWOII"] / df["^TWII"])
        f5 = -self.z_score(df["TWD=X"])
        f6 = self.z_score(df["^TWII"].pct_change(), window=10) * 10
        f7 = self.z_score((df["^TWII"] * 1.002) - df["^TWII"])
        f8 = -self.z_score(df["^TWVIX"].pct_change(), window=5)
        
        # 每日更新權重 (針對前兩大核心因子進行動態調整)
        dyn_w = self.get_optimized_weights(df)
        
        # 綜合評分：將動態權重應用於核心因子，其餘均分
        factors_core = np.array([f1, f2])
        score_core = np.dot(factors_core, dyn_w)
        score_others = np.mean([f3, f4, f5, f6, f7, f8])
        
        # 新增 ARIMA(1,1,1)
        try:
            arima_mod = ARIMA(df["^TWII"].tail(100), order=(1, 1, 1))
            f_arima = 1 if arima_mod.fit().forecast(steps=1).iloc[0] > df["^TWII"].iloc[-1] else -1
        except: f_arima = 0

        final_score = (score_core * 0.4) + (score_others * 0.4) + (f_arima * 0.2)
        
        details = {"ADR": f1, "SOXX": f2, "VIX": f3, "ARIMA": f_arima, "W_ADR": dyn_w[0]}
        return final_score, details

    def get_market_regime(self, df):
        ret = df["^TWII"].pct_change().fillna(0)
        vol = ret.rolling(20).std().fillna(0)
        X = np.column_stack([ret.tail(150), vol.tail(150)])
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_scaled)
        means = [X[kmeans.labels_ == i, 0].mean() for i in range(4)]
        rank = np.where(np.argsort(means) == kmeans.labels_[-1])[0][0]
        return ["panic", "bear", "range", "bull"][rank]

    def generate_orders(self):
        df = self.get_data()
        if df.empty: return "error", "N/A", 0.0, 0.0, {}
        
        score, details = self.factor_model(df)
        regime = self.get_market_regime(df)
        twii_last = self._to_scalar(df["^TWII"])
        tsmc_last = self._to_scalar(df["2330.TW"])
        
        # 參賽天數門檻 0.08
        is_active = abs(score) >= 0.08
        
        trade_cap = float(self.balance * 0.8 * (min(abs(score)/2, 0.5)))
        mtx = max(1, int(trade_cap / 400000))

        if not is_active:
            orders = "【中性模式】自動補單 | 指令：小台(MTX) 1口多單 (保參賽天數)"
        elif regime == "bull":
            orders = f"【趨勢看多】全速推進 | MTX多 {mtx}口 | 台積期同步加碼"
        elif regime == "panic":
            orders = f"【恐慌噴出】減半操作 | MTX多 {max(1, mtx//2)}口 | 買入 Put 避險"
        else:
            orders = f"【偏空預警】策略防守 | MTX空 {mtx}口"

        return regime, orders, score, tsmc_last, details

    def send_email(self, content):
        sender = 'jeffreylin1201@gmail.com'
        receivers = ['jeffreylin1201@gmail.com', 'allenbowei@gmail.com'] 
        password = 'udcgkrdfdfoqznsn' 
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header("V50 Strategy AI", 'utf-8')
        message['Subject'] = Header(f"V50 每日權重優化戰報: {datetime.now().strftime('%m/%d')}", 'utf-8')
        try:
            smtp_obj = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp_obj.login(sender, password)
            smtp_obj.sendmail(sender, receivers, message.as_string())
        except: pass

    def report(self):
        regime, orders, score, price, details = self.generate_orders()
        msg = (
            f"🚀 V50 策略比賽戰報 ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            f"--------------------------------\n"
            f"🔍 市場狀態: {regime.upper()}\n"
            f"📊 綜合評分: {score:.4f} (門檻: 0.08)\n"
            f"📈 ARIMA 趨勢: {'BULL' if details.get('ARIMA')==1 else 'BEAR'}\n"
            f"⚖️ 今日 ADR 權重: {details.get('W_ADR', 0):.2%}\n"
            f"--------------------------------\n"
            f"⚡ 執行指令:\n{orders}\n"
            f"--------------------------------\n"
            f"💰 帳戶權益: {int(self.balance):,} TWD"
        )
        print(msg)
        self.send_email(msg)
        if self.tg_token and self.tg_chat_id:
            try: requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg}, timeout=10)
            except: pass
        self.save_equity()

if __name__ == "__main__":
    bot = TAIFEX_Contest_V50(); bot.report()
