import numpy as np
import pandas as pd
import yfinance as yf
import os, requests, smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from sklearn.linear_model import Ridge

class TAIFEX_V230_FailSafeFinal:

    def __init__(self, initial_capital=2000000):
        self.initial_capital = initial_capital
        self.file = "strategy_report.csv"
        self.tg_token = os.getenv("TG_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        self.cost_rate = 0.0015
        self.lambda_w = 0.95
        self.balance, self.peak, self.last_pos, self.last_nv, self.last_w = self.load_system_state()

    def load_system_state(self):
        if os.path.isfile(self.file):
            try:
                df = pd.read_csv(self.file)
                if not df.empty:
                    last = df.iloc[-1]
                    pos = last[[f"pos_{i}" for i in range(6)]].values
                    nv = last[[f"nv_{i}" for i in range(6)]].values
                    w = last[[f"w_{i}" for i in range(7)]].values if "w_6" in last else np.array([0.14]*7)
                    # 檢查載入的值是否合法
                    if np.isnan(float(last["equity"])): return self.initial_capital, self.initial_capital, np.zeros(6), np.zeros(6), np.array([0.14]*7)
                    return float(last["equity"]), float(df["equity"].max()), pos, nv, w
            except: pass
        return self.initial_capital, self.initial_capital, np.zeros(6), np.zeros(6), np.array([0.14]*7)

    def save_system_state(self, equity, pos, nv, w):
        # 熔斷機制：如果 equity 是 nan，絕對不儲存，防止損壞 CSV
        if np.isnan(equity) or equity < -1e15: return 
        state = {"date": datetime.now().strftime("%Y-%m-%d"), "equity": equity}
        for i in range(6): state[f"pos_{i}"] = pos[i]; state[f"nv_{i}"] = nv[i]
        for i in range(7): state[f"w_{i}"] = w[i]
        pd.DataFrame([state]).to_csv(self.file, mode="a", index=False, header=not os.path.isfile(self.file))

    def _to_scalar(self, val):
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)): return float(val.iloc[:,0].iloc[-1] if isinstance(val, pd.DataFrame) else val.iloc[-1])
            return float(val)
        except: return 0.0

    def get_data(self):
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        raw = yf.download(tickers, period="300d", progress=False)
        
        if raw.empty or len(raw) < 10: raise ValueError("Yahoo Finance 抓取失敗")
        
        df_close = raw['Close'].ffill().bfill()
        df_high = raw['High'].ffill().bfill() if 'High' in raw else df_close
        df_low = raw['Low'].ffill().bfill() if 'Low' in raw else df_close

        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close['^TWII']
        
        # 強化 Parkinson Vol 的穩定性
        log_hl = np.log(df_high['^TWII'] / (df_low['^TWII'] + 1e-8))
        df['park_vol'] = np.sqrt(1/(4*np.log(2)) * (log_hl**2)).rolling(10).mean().ffill().replace(0, 0.01)

        # Alpha 因子 (加上 Small epsilon 防止 0 除以 0)
        df['f1_adr'] = ((df_close["TSM"]/5 * df_close["TWD=X"]) / (df_close["2330.TW"] + 1e-8)).pct_change()
        df['f2_semi'] = df_close["SOXX"].pct_change()
        df['f3_basis'] = (df['Close'] - df['Close'].rolling(5).mean()).pct_change()
        df['f4_regime'] = (df_close["^TWOII"] / (df_close["^TWII"] + 1e-8)).pct_change()
        df['f5_retail'] = (df['Close'].pct_change() / (df['Close'].rolling(20).std() + 1e-8)).shift(1)
        df['f6_fx'] = -df_close['TWD=X'].pct_change()
        
        # VIX 修正
        try:
            v_raw = yf.download("^TWVIX", period="300d", progress=False)['Close']
            df["^TWVIX"] = v_raw.ffill().bfill()
        except:
            df["^TWVIX"] = (df['Close'].pct_change().rolling(20).std()*np.sqrt(252)*115+1.5)
        
        df['f7_vix'] = -df["^TWVIX"].pct_change()
        
        self.prices = df_close.iloc[-1]
        return df.dropna().ffill()

    def get_alpha(self, df):
        def zs(s): return (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
        feats = ['f1_adr','f2_semi','f3_basis','f4_regime','f5_retail','f6_fx','f7_vix']
        df_zs = df[feats].apply(zs)
        f_vec = df_zs.iloc[-1].values
        
        try:
            lookback = 60
            y = df["Close"].pct_change().shift(-1).dropna().tail(lookback)
            X = df_zs.tail(lookback+1).iloc[:-1]
            new_w = Ridge(alpha=2.0).fit(X, y).coef_
            smooth_w = self.lambda_w * self.last_w + (1 - self.lambda_w) * new_w
        except: smooth_w = self.last_w

        raw_score = np.dot(f_vec, smooth_w)
        # 熔斷：如果 raw_score 是 nan，直接返回 0
        if np.isnan(raw_score): return 0.0, smooth_w, df_zs
        
        vol_adj = 2.5 * (0.012 / (self._to_scalar(df['park_vol']) + 1e-8))
        score = np.tanh(raw_score * np.clip(vol_adj, 1.5, 4.0))
        return score, smooth_w, df_zs

    def execute_trading(self, alpha, df, w_vec, df_zs):
        p = self.prices
        # 合約名目價值 (加上防 nan 處理)
        current_nv = np.array([
            p.get("^TWII", 20000)*200, p.get("^TWII", 20000)*50, p.get("^TWII", 20000)*10,
            p.get("^TWOII", 200)*100, p.get("2330.TW", 1000)*2000, 
            p.get("GC=F", 2000)*100*p.get("TWD=X", 32)
        ])
        
        # PnL 計算 (確保沒有 NaN)
        gross_pnl = 0
        if np.all(self.last_nv > 0) and not np.any(np.isnan(current_nv)):
            item_rets = (current_nv / (self.last_nv + 1e-8)) - 1
            gross_pnl = np.sum(self.last_pos * self.last_nv * item_rets)
        
        if np.isnan(gross_pnl): gross_pnl = 0

        # 動態槓桿與部位
        lev = min((0.012 / (self._to_scalar(df['park_vol']) + 1e-8)) * abs(alpha) * 5, 2.0)
        if np.isnan(lev): lev = 0

        alloc = np.array([0.4, 0.2, 0.1, 0.1, 0.2, 0.05])
        alloc /= (np.sum(alloc) + 1e-8)
        
        target_val = self.balance * lev
        direction = np.sign(alpha) if abs(alpha) > 0.05 else 0
        
        if direction == 0:
            final_pos = np.zeros(6) # 如果訊號不明，部位歸零防止溢位
            t_cost = 0
        else:
            raw_pos = (target_val * alloc / (current_nv + 1e-8))
            # 修正：使用 np.nan_to_num 防止極大值
            final_pos = np.nan_to_num(raw_pos, nan=0, posinf=0, neginf=0).astype(int) * direction
            # 強制維持 1 口規則
            for i in range(6): 
                if final_pos[i] == 0: final_pos[i] = 1 * (direction if direction != 0 else 1)
            
            diff_nv = np.sum(np.abs(final_pos - self.last_pos) * current_nv)
            t_cost = diff_nv * self.cost_rate

        return final_pos, current_nv, (gross_pnl - t_cost), lev

    def report(self):
        try:
            df = self.get_data()
            alpha, w_vec, df_zs = self.get_alpha(df)
            pos, nv, pnl, lev = self.execute_trading(alpha, df, w_vec, df_zs)
            
            if np.isnan(pnl): pnl = 0
            self.balance += pnl
            self.peak = max(self.peak, self.balance)
            
            msg = (
                f"🚀 V230 防護穩定版戰報\n"
                f"--------------------------------\n"
                f"🧠 Alpha: {alpha:.4f}\n"
                f"⚡ 持倉口數: TX:{pos[0]} | MX:{pos[1]} | STC:{pos[4]}\n"
                f"💰 今日損益: {int(pnl):,} TWD\n"
                f"💰 累積權益: {int(self.balance):,} TWD"
            )
            print(msg)
            # 郵件發送 (代碼不變)
            m = MIMEText(msg, 'plain', 'utf-8')
            m['Subject'] = Header(f"V230 戰報", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
            self.save_system_state(self.balance, pos, nv, w_vec)
        except Exception as e:
            print(f"系統崩潰防護觸發: {e}")

if __name__ == "__main__":
    TAIFEX_V230_FailSafeFinal().report()
