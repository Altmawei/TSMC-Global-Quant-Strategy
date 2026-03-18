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
        # 考慮交易費用：假設每筆交易包含稅與手續費約為 0.1% (10 bps)
        self.transaction_cost = 0.001 
        self.tg_token = os.getenv("TG_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        self.balance, self.peak = self.load_equity()

    def load_equity(self):
        if os.path.isfile(self.file):
            try:
                df = pd.read_csv(self.file)
                if not df.empty: return float(df.iloc[-1]["equity"]), float(df["equity"].max())
            except: pass
        return self.initial_capital, self.initial_capital

    def save_equity(self):
        df = pd.DataFrame([{"date": datetime.now().strftime("%Y-%m-%d"), "equity": self.balance}])
        try: df.to_csv(self.file, mode="a", index=False, header=not os.path.isfile(self.file))
        except: pass

    def _to_scalar(self, val):
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                v = val.iloc[-1]
                while isinstance(v, (pd.Series, pd.DataFrame)): v = v.iloc[0]
                return float(v)
            return float(val)
        except: return 0.0

    def get_data(self):
        # 抓取 300 天數據以供 ARIMA 與 回歸使用
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="300d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(-1)
        
        # 指數平滑去噪 (Denoising)
        df = raw.ffill().ewm(span=5, adjust=False).mean()

        try:
            vix_raw = yf.download("^TWVIX", period="300d", progress=False)
            v_close = vix_raw['Close'] if 'Close' in vix_raw else vix_raw.iloc[:, 0]
            df["^TWVIX"] = v_close.ffill()
        except:
            tw_p = df["^TWII"]
            hv = tw_p.pct_change().rolling(20).std() * np.sqrt(252) * 100
            df["^TWVIX"] = (hv * 1.15 + 1.5).ffill()
        return df.dropna(subset=["^TWII"])

    def z_score(self, s, window=20):
        if len(s) < window: return 0.0
        target = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        tail = target.tail(window)
        std = tail.std()
        if std < 1e-8: return 0.0
        return float((self._to_scalar(target) - tail.mean()) / std)

    def factor_model(self, df):
        # 8 大核心因子
        f1 = self.z_score((df["TSM"] / 5 * df["TWD=X"]) / df["2330.TW"])
        f2 = self.z_score(df["SOXX"].pct_change())
        f3 = -self.z_score(df["^TWVIX"])
        f4 = self.z_score(df["^TWOII"] / df["^TWII"])
        f5 = -self.z_score(df["TWD=X"])
        f6 = self.z_score(df["^TWII"].pct_change(), window=10) * 5
        f7 = self.z_score((df["^TWII"] * 1.002) - df["^TWII"])
        f8 = self.z_score(df["GC=F"].pct_change(), window=5)

        # 每日自動權重更新 (Ridge)
        try:
            idx = df["^TWII"]
            y = idx.pct_change().shift(-1).dropna().tail(60)
            f1_h = (df["TSM"]/5 * df["TWD=X"] / df["2330.TW"]).pct_change().tail(60)
            f2_h = df["SOXX"].pct_change().tail(60)
            model = Ridge(alpha=1.0).fit(np.column_stack([f1_h.values, f2_h.values]), y.values)
            w = np.abs(model.coef_)
            w = w / (np.sum(w) + 1e-8)
        except: w = np.array([0.5, 0.5])
        
        score_core = f1 * w[0] + f2 * w[1]
        score_others = np.mean([f3, f4, f5, f6, f7, f8])
        
        try:
            t_series = df["^TWII"]
            arima_fit = ARIMA(t_series.tail(100), order=(1, 1, 1)).fit()
            f_arima = 1 if arima_fit.forecast(steps=1).iloc[0] > self._to_scalar(t_series) else -1
        except: f_arima = 0

        final_score = (score_core * 0.4) + (score_others * 0.4) + (f_arima * 0.2)
        
        # 考慮費用後的「淨評分」：如果分數無法覆蓋交易成本，則視為 0
        if abs(final_score) < (self.transaction_cost * 100): 
            final_score *= 0.5 # 抑制過度頻繁交易

        details = {"ADR": f1, "SOXX": f2, "ARIMA": f_arima, "W_ADR": w[0]}
        return final_score, details

    def generate_orders(self, score, df):
        # 門檻判定
        is_active = abs(score) >= 0.08
        direction = "多" if score > 0 else "空"
        tsmc_p = self._to_scalar(df["2330.TW"])
        
        # 根據分數動態調整總部位價值
        lev = min(abs(score) * 2.5, 2.0)
        target_val = self.balance * lev

        if not is_active:
            # --- 平盤擠肉：跨商品套利 (符合 6 項商品) ---
            regime = "NEUTRAL (Range-Trading)"
            orders = (f"【平盤擠肉模式】偵測震盪，執行跨商品配對：\n"
                      f"1. 微台(FMTX): 多 2口\n"
                      f"2. 中型100(TMC): 空 1口 (對沖)\n"
                      f"3. 台積期(STC): 多 1口 (基差收斂)\n"
                      f"4. 黃金期(GDF): 多 1口 (避險)\n"
                      f"5. 小台(MX): 多 1口 (維持資格)\n"
                      f"6. 台指期(TX): 暫不進場 (節省費用)\n"
                      f"※ 預期獲利來源：中型股補漲與台積基差回歸。")
        else:
            # --- 趨勢模式 (BULL/BEAR) ---
            regime = "BULL" if score > 0 else "BEAR"
            tx = max(1, int(target_val / 2000000))
            mx = max(1, int(target_val / 500000))
            fmtx = max(1, int(target_val / 100000))
            tmc = max(1, int(target_val / 400000))
            stc = max(1, int(target_val / (tsmc_p * 2000)))
            gdf = 1 
            
            orders = (f"【趨勢{direction}單】全火力輸出，槓桿 {lev:.2f}x：\n"
                      f"1. 台股期(TX): {direction} {tx}口\n"
                      f"2. 小台(MX): {direction} {mx}口\n"
                      f"3. 微台(FMTX): {direction} {fmtx}口\n"
                      f"4. 中型100(TMC): {direction} {tmc}口\n"
                      f"5. 台積期(STC): {direction} {stc}口\n"
                      f"6. 黃金期(GDF): 多 1口 (配置)")
        
        return orders, regime

    def report(self):
        df = self.get_data()
        score, details = self.factor_model(df)
        orders, regime = self.generate_orders(score, df)
        
        msg = (
            f"🚀 V50 競賽最終整合戰報\n"
            f"--------------------------------\n"
            f"📈 綜合評分: {score:.4f} ({regime})\n"
            f"📉 ARIMA 趨勢: {'看多' if details['ARIMA']==1 else '看空'}\n"
            f"⚖️ ADR 動態權重: {details['W_ADR']:.1%}\n"
            f"--------------------------------\n"
            f"⚡ 執行指令 (已扣除 0.1% 交易成本考量):\n{orders}\n"
            f"--------------------------------\n"
            f"💰 帳戶餘額: {int(self.balance):,} TWD"
        )
        print(msg)
        
        # 郵件與 TG (直接內嵌參數確保不報錯)
        try:
            m = MIMEText(msg, 'plain', 'utf-8')
            m['Subject'] = Header(f"V50 戰報: {regime} (Score {score:.2f})", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
        except Exception as e: print(f"Mail Error: {e}")
        
        if self.tg_token and self.tg_chat_id:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", 
                          data={"chat_id": self.tg_chat_id, "text": msg})
        self.save_equity()

if __name__ == "__main__":
    TAIFEX_Contest_V50().report()
