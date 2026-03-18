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
from sklearn.linear_model import Ridge
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
        """極強效轉型：解決 TypeError"""
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                v = val.iloc[-1]
                while isinstance(v, (pd.Series, pd.DataFrame)):
                    v = v.iloc[0]
                return float(v)
            return float(val)
        except: return 0.0

    def get_data(self):
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="300d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(-1)
        
        # 指數平滑去噪
        df = raw.ffill().ewm(span=5, adjust=False).mean()

        try:
            vix_raw = yf.download("^TWVIX", period="300d", progress=False)
            if vix_raw.empty: raise ValueError("VIX Empty")
            v_close = vix_raw['Close']
            if isinstance(v_close, pd.DataFrame): v_close = v_close.iloc[:, 0]
            df["^TWVIX"] = v_close.ffill()
        except:
            # 修正 ValueError: 確保備用 VIX 是一維的
            tw_p = df["^TWII"]
            if isinstance(tw_p, pd.DataFrame): tw_p = tw_p.iloc[:, 0]
            hv = tw_p.pct_change().rolling(20).std() * np.sqrt(252) * 100
            df["^TWVIX"] = (hv * 1.15 + 1.5).ffill()

        return df.dropna(subset=["^TWII"])

    def z_score(self, s, window=20):
        """修正參數對齊問題"""
        if len(s) < window: return 0.0
        target = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        tail = target.tail(window)
        std = tail.std()
        if std < 1e-8 or np.isnan(std): return 0.0
        return float((self._to_scalar(target) - tail.mean()) / std)

    def get_optimized_weights(self, df):
        try:
            lookback = 60
            idx = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
            y = idx.pct_change().shift(-1).dropna().tail(lookback)
            f1_h = (df["TSM"]/5 * df["TWD=X"] / df["2330.TW"]).pct_change().tail(lookback)
            f2_h = df["SOXX"].pct_change().tail(lookback)
            X = np.column_stack([f1_h.values, f2_h.values])
            model = Ridge(alpha=1.0).fit(X, y.values)
            w = np.abs(model.coef_)
            return w / (np.sum(w) + 1e-8)
        except: return np.array([0.5, 0.5])

    def factor_model(self, df):
        f1 = self.z_score((df["TSM"] / 5 * df["TWD=X"]) / df["2330.TW"])
        f2 = self.z_score(df["SOXX"].pct_change())
        f3 = -self.z_score(df["^TWVIX"])
        f4 = self.z_score(df["^TWOII"] / df["^TWII"])
        f5 = -self.z_score(df["TWD=X"])
        f6 = self.z_score(df["^TWII"].pct_change(), window=10) * 10
        f7 = self.z_score((df["^TWII"] * 1.002) - df["^TWII"])
        f8 = -self.z_score(df["^TWVIX"].pct_change(), window=5)
        
        dyn_w = self.get_optimized_weights(df)
        score_core = f1 * dyn_w[0] + f2 * dyn_w[1]
        score_others = np.mean([f3, f4, f5, f6, f7, f8])
        
        try:
            t_series = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
            arima_fit = ARIMA(t_series.tail(100), order=(1, 1, 1)).fit()
            f_arima = 1 if arima_fit.forecast(steps=1).iloc[0] > self._to_scalar(t_series) else -1
        except: f_arima = 0

        final_score = (score_core * 0.4) + (score_others * 0.4) + (f_arima * 0.2)
        details = {"ADR": f1, "SOXX": f2, "VIX": f3, "ARIMA": f_arima, "W_ADR": dyn_w[0]}
        return final_score, details

    def get_market_regime(self, df):
        p = df["^TWII"].iloc[:, 0] if isinstance(df["^TWII"], pd.DataFrame) else df["^TWII"]
        ret = p.pct_change().fillna(0)
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
        tsmc_last = self._to_scalar(df["2330.TW"])
        is_active = abs(score) >= 0.08
        trade_cap = float(self.balance * 0.8 * (min(abs(score)/2, 0.5)))
        mtx = max(1, int(trade_cap / 400000))

        if not is_active:
            orders = "【中性模式】補單參賽 | MTX多 1口"
        elif regime == "bull":
            orders = f"【趨勢看多】MTX多 {mtx}口 | 加碼台積期"
        elif regime == "panic":
            orders = f"【恐慌噴出】減半 MTX多 {max(1, mtx//2)}口"
        else:
            orders = f"【偏空預警】MTX空 {mtx}口"

        return regime, orders, score, tsmc_last, details

    def send_email(self, content):
        sender = 'jeffreylin1201@gmail.com'
        receivers = ['jeffreylin1201@gmail.com', 'allenbowei@gmail.com'] 
        password = 'udcgkrdfdfoqznsn' 
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = Header(f"V50 每日權重戰報: {datetime.now().strftime('%m/%d')}", 'utf-8')
        try:
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login(sender, password); s.sendmail(sender, receivers, msg.as_string())
        except: pass

    def report(self):
        regime, orders, score, price, details = self.generate_orders()
        msg = (
            f"🚀 V50 策略戰報 ({datetime.now().strftime('%m/%d %H:%M')})\n"
            f"--------------------------------\n"
            f"🔍 狀態: {regime.upper()} | 分數: {score:.4f}\n"
            f"📈 ARIMA: {'BULL' if details.get('ARIMA')==1 else 'BEAR'}\n"
            f"⚖️ ADR權重: {details.get('W_ADR', 0):.2%}\n"
            f"⚡ 指令: {orders}\n"
            f"💰 餘額: {int(self.balance):,} TWD"
        )
        print(msg); self.send_email(msg)
        if self.tg_token and self.tg_chat_id:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg})
        self.save_equity()

if __name__ == "__main__":
    TAIFEX_Contest_V50().report()
