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
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                temp = val.iloc[:, 0] if isinstance(val, pd.DataFrame) else val
                return float(temp.iloc[-1])
            return float(val)
        except: return 0.0

    def get_data(self):
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

        # 計算 8 大因子
        f1 = self.z_score((tsm / 5 * twd) / tsmc) # ADR Premium
        f2 = self.z_score(soxx.pct_change())       # Semi Momentum
        f3 = -self.z_score(vix)                    # Risk Off (VIX)
        f4 = self.z_score(twoii / twii)            # OTC vs Main
        f5 = -self.z_score(twd)                    # FX (TWD strength)
        f6 = self.z_score(twii.pct_change(), window=10) * 10
        f7 = self.z_score((twii * 1.002) - twii)   # Basis
        f8 = -self.z_score(vix.pct_change(), window=5)
        
        factors = [self._to_scalar(f) for f in [f1, f2, f3, f4, f5, f6, f7, f8]]
        # 額外回傳因子明細供報告使用
        details = {"ADR": f1, "SOXX": f2, "VIX": f3, "FX": f5}
        return np.nan_to_num(factors), details

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

    def generate_orders(self):
        df = self.get_data()
        if df.empty: return "error", "N/A", 0.0, 0.0, {}
        
        factors, details = self.factor_model(df)
        score = float(np.mean(factors))
        regime = self.get_market_regime(df)
        
        twii_last = self._to_scalar(df["^TWII"])
        tsmc_last = self._to_scalar(df["2330.TW"])
        trade_cap = float(self.balance * 0.8 * (min(abs(score)/2, 0.5))) # 簡化 sizing

        # 28天門檻調降至 0.08
        strategy_mode = "ACTIVE" if abs(score) >= 0.08 else "NEUTRAL"

        tx = max(1, int(trade_cap / 2000000))
        mtx = max(1, int(trade_cap / 400000))
        tsmc_f = max(0, int(trade_cap / (tsmc_last * 200 + 1)))

        if strategy_mode == "NEUTRAL":
            orders = "【中性盤整】維持參賽天數 | 建議：微台(MTX) 1口"
        elif regime == "bull":
            orders = f"【趨勢看多】TX多 {tx}口 | MTX多 {mtx}口 | 台積期多 {tsmc_f}口"
        elif regime == "panic":
            orders = f"【恐慌噴出】縮減規模跟進 | MTX多 {max(1, mtx//2)}口 | 買入 Put 避險"
        else:
            orders = f"【偏空預警】TX空 {tx}口 | MTX空 {mtx}口"

        return regime, orders, score, tsmc_last, details

    def report(self):
        regime, orders, score, price, details = self.generate_orders()
        
        # 建立極詳細報告內容
        report_msg = (
            f"🚀 V50 策略比賽戰報 ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"
            f"--------------------------------\n"
            f"🔍 市場狀態: {regime.upper()}\n"
            f"📊 綜合評分: {score:.4f} (門檻: 0.08)\n"
            f"💎 台積電現價: {price:.1f}\n"
            f"--------------------------------\n"
            f"📈 因子分解 (Z-Score):\n"
            f"   - ADR 溢價: {details.get('ADR', 0):.2f}\n"
            f"   - 半導體動能: {details.get('SOXX', 0):.2f}\n"
            f"   - 市場風險(VIX): {details.get('VIX', 0):.2f}\n"
            f"   - 匯率強度: {details.get('FX', 0):.2f}\n"
            f"--------------------------------\n"
            f"⚡ 執行指令:\n{orders}\n"
            f"--------------------------------\n"
            f"💰 當前模擬權益: {int(self.balance):,} TWD"
        )
        
        print(report_msg)
        self.send_telegram(report_msg)
        self.send_email(report_msg)
        self.save_equity()

    def send_email(self, content):
        sender = 'jeffreylin1201@gmail.com'
        receivers = ['jeffreylin1201@gmail.com', 'allenbowei@gmail.com'] 
        password = 'udcgkrdfdfoqznsn' 
        message = MIMEText(content, 'plain', 'utf-8')
        message['From'] = Header("V50 Strategy Bot", 'utf-8')
        message['Subject'] = Header(f"V50 戰報: {datetime.now().strftime('%m/%d')}", 'utf-8')
        try:
            smtp_obj = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            smtp_obj.login(sender, password)
            smtp_obj.sendmail(sender, receivers, message.as_string())
        except: pass

    def send_telegram(self, msg):
        if self.tg_token and self.tg_chat_id:
            try: requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                               data={"chat_id": self.tg_chat_id, "text": msg}, timeout=10)
            except: pass

if __name__ == "__main__":
    bot = TAIFEX_Contest_V50(); bot.report()
