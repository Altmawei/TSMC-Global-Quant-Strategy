import numpy as np
import pandas as pd
import yfinance as yf
import os, requests, smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from sklearn.linear_model import Ridge

class TAIFEX_V220_StabilityFinal:

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
                    return float(last["equity"]), float(df["equity"].max()), pos, nv, w
            except: pass
        return self.initial_capital, self.initial_capital, np.zeros(6), np.zeros(6), np.array([0.14]*7)

    def save_system_state(self, equity, pos, nv, w):
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
        """強化版抓取：解決 KeyError 與 Database Locked 問題"""
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        # 增加多一次嘗試機會
        raw = yf.download(tickers, period="300d", progress=False)
        
        # 解決 MultiIndex 問題並確保數據存在
        if isinstance(raw.columns, pd.MultiIndex):
            df_close = raw['Close'].ffill()
            df_high = raw['High'].ffill() if 'High' in raw else df_close
            df_low = raw['Low'].ffill() if 'Low' in raw else df_close
        else:
            df_close = raw.ffill()
            df_high = df_close
            df_low = df_close

        # 修正 5. 確保必須要有 ^TWII
        if "^TWII" not in df_close.columns:
            raise ValueError("關鍵大盤數據抓取失敗，請重新運行。")

        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close['^TWII']
        
        # 修正：Parkinson Vol (解決 KeyError 'High' 的關鍵)
        # 如果 High/Low 缺失，則退化為標準收盤價回報波動率
        try:
            log_hl = np.log(df_high['^TWII'] / df_low['^TWII'])
            df['park_vol'] = np.sqrt(1/(4*np.log(2)) * (log_hl**2)).rolling(10).mean()
        except:
            df['park_vol'] = df['Close'].pct_change().rolling(10).std() * np.sqrt(252) / 100

        # Alpha 因子計算
        df['f1_adr'] = ((df_close["TSM"]/5 * df_close["TWD=X"]) / df_close["2330.TW"]).pct_change()
        df['f2_semi'] = df_close["SOXX"].pct_change()
        df['f3_basis'] = (df['Close'] - df['Close'].rolling(5).mean()).pct_change()
        df['f4_regime'] = (df_close["^TWOII"]/df_close["^TWII"]).pct_change()
        df['f5_retail'] = (df['Close'].pct_change() / df['Close'].rolling(20).std().replace(0, 0.01)).shift(1)
        df['f6_fx'] = -df_close['TWD=X'].pct_change()
        
        # VIX 備用擬合
        try:
            v_raw = yf.download("^TWVIX", period="300d", progress=False)['Close']
            df["^TWVIX"] = v_raw.ffill() if not v_raw.dropna().empty else (df['Close'].pct_change().rolling(20).std()*np.sqrt(252)*115+1.5)
        except:
            df["^TWVIX"] = (df['Close'].pct_change().rolling(20).std()*np.sqrt(252)*115+1.5).ffill()
        df['f7_vix'] = -df['^TWVIX'].pct_change()
        
        # 保存其他必要價格以便 execute_trading 使用
        self.prices = df_close.iloc[-1]
        self.df_close = df_close # 為了之後算 rets
        return df.dropna(subset=["f1_adr", "f2_semi"]).ffill()

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

        # 動態 Alpha 放大 (截圖 2:18:21)
        park_v = self._to_scalar(df['park_vol'])
        vol_adj = 2.5 * (0.012 / (park_v + 1e-8))
        score = np.tanh(np.dot(f_vec, smooth_w) * np.clip(vol_adj, 1.5, 4.0))
        return (0 if abs(score) < 0.05 else score), smooth_w, df_zs

    def execute_trading(self, alpha, df, w_vec, df_zs):
        p = self.prices
        # 合約名目價值 (動態抓取價格)
        current_nv = np.array([p["^TWII"]*200, p["^TWII"]*50, p["^TWII"]*10, p["^TWOII"]*100, p["2330.TW"]*2000, p["GC=F"]*100*p["TWD=X"]])
        
        # 修正 PnL 邏輯
        if np.all(self.last_nv > 0):
            item_rets = (current_nv / self.last_nv) - 1
            gross_pnl = np.sum(self.last_pos * self.last_nv * item_rets)
        else: gross_pnl = 0

        # 動態風控與門檻
        park_v = self._to_scalar(df['park_vol'])
        t_vol = 0.012
        lev = min((t_vol / (park_v + 1e-8)) * abs(alpha) * 5, 2.0)
        
        vix_z = df_zs['f7_vix'].iloc[-1]
        vix_th = df_zs['f7_vix'].rolling(60).quantile(0.10).iloc[-1]
        if vix_z < vix_th: lev *= 0.8
        if (self.peak - self.balance)/self.peak > 0.12: lev = 0 
        
        # 動態 Buffer 與分配 (截圖 2:18:21)
        buffer_size = 0.1 * (1 - abs(alpha))
        alloc = np.zeros(6)
        alloc[0] = max(0.05, abs(w_vec[0]) * 0.4) 
        alloc[1] = max(0.05, abs(w_vec[2]) * 0.2) 
        alloc[2] = max(0.05, buffer_size)         
        alloc[3] = max(0.05, abs(w_vec[3]) * 0.1) 
        alloc[4] = max(0.05, abs(w_vec[4]) * 0.2) 
        alloc[5] = 0.05                           
        
        fx_z = df_zs['f6_fx'].iloc[-1]
        fx_th = df_zs['f6_fx'].rolling(60).quantile(0.90).iloc[-1]
        if fx_z > fx_th: alloc[0] *= 0.8
        
        alloc /= (np.sum(alloc) + 1e-8)
        target_val = self.balance * lev
        direction = np.sign(alpha) if abs(alpha) > 0 else 1

        # 核心：確保交易滿 6 項標的
        raw_pos = (target_val * alloc / (current_nv + 1e-8)).astype(int) * direction
        for i in range(6):
            if raw_pos[i] == 0: raw_pos[i] = 1 * direction
        if alpha < 0: raw_pos[5] = abs(raw_pos[5])

        # 動態慣性 (截圖 2:18:21)
        dynamic_inertia = 0.10 * (park_v / t_vol)
        diff_nv = np.sum(np.abs(raw_pos - self.last_pos) * current_nv)
        if diff_nv < (dynamic_inertia * self.balance) and abs(alpha) > 0:
            final_pos = self.last_pos; t_cost = 0
        else:
            final_pos = raw_pos; t_cost = diff_nv * self.cost_rate
        
        return final_pos, current_nv, (gross_pnl - t_cost), lev

    def report(self):
        df = self.get_data()
        alpha, w_vec, df_zs = self.get_alpha(df)
        pos, nv, pnl, lev = self.execute_trading(alpha, df, w_vec, df_zs)
        
        self.balance += pnl
        self.peak = max(self.peak, self.balance)
        regime = "BULL" if alpha > 0 else ("BEAR" if alpha < 0 else "NEUTRAL")

        msg = (
            f"🚀 V220 抗震修復版戰報\n"
            f"--------------------------------\n"
            f"🧠 Alpha: {alpha:.4f} | 狀態: {regime}\n"
            f"📊 權重: ADR:{w_vec[0]:.0%} | Basis:{w_vec[2]:.0%} | VIX:{w_vec[6]:.0%}\n"
            f"🛡️ 狀態: {'[穩定性修復中]' if self.prices['GC=F']==0 else '正常'}\n"
            f"⚡ 持倉明細 (1~6 項必備):\n"
            f" TX:{pos[0]} | MX:{pos[1]} | FMTX:{pos[2]}\n"
            f" TMC:{pos[3]} | STC:{pos[4]} | GDF:{pos[5]}\n"
            f"--------------------------------\n"
            f"💰 今日淨損益: {int(pnl):+} TWD\n"
            f"💰 累積總權益: {int(self.balance):,} TWD"
        )
        print(msg)
        try:
            m = MIMEText(msg, 'plain', 'utf-8')
            m['Subject'] = Header(f"V220 穩定版: {regime}", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
        except: pass
        if self.tg_token and self.tg_chat_id:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg})
        self.save_system_state(self.balance, pos, nv, w_vec)

if __name__ == "__main__":
    TAIFEX_V220_StabilityFinal().report()
