import numpy as np
import pandas as pd
import yfinance as yf
import os, requests, smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from sklearn.linear_model import Ridge

class TAIFEX_V210_ContestFinal:

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
        tickers = ["2330.TW", "TSM", "SOXX", "^TWII", "^TWOII", "TWD=X", "GC=F"]
        df = yf.download(tickers, period="300d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(-1)
        
        df['park_vol'] = np.sqrt(1/(4*np.log(2)) * ((np.log(df['High']/df['Low']))**2)).rolling(10).mean()
        df['f1_adr'] = ((df["TSM"]/5 * df["TWD=X"]) / df["2330.TW"]).pct_change()
        df['f2_semi'] = df["SOXX"].pct_change()
        df['f3_basis'] = (df['^TWII'] - df['^TWII'].rolling(5).mean()).pct_change()
        df['f4_regime'] = (df["^TWOII"]/df["^TWII"]).pct_change()
        df['f5_retail'] = (df['^TWII'].pct_change() / df['^TWII'].rolling(20).std()).shift(1)
        df['f6_fx'] = -df['TWD=X'].pct_change()

        try:
            v_raw = yf.download("^TWVIX", period="300d", progress=False)['Close']
            df["^TWVIX"] = v_raw.ffill() if not v_raw.dropna().empty else (df['^TWII'].pct_change().rolling(20).std()*np.sqrt(252)*115+1.5)
        except:
            df["^TWVIX"] = (df['^TWII'].pct_change().rolling(20).std()*np.sqrt(252)*115+1.5).ffill()
        df['f7_vix'] = -df['^TWVIX'].pct_change()
        return df.dropna(subset=["^TWII"])

    def get_alpha(self, df):
        def zs(s): return (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
        feats = ['f1_adr','f2_semi','f3_basis','f4_regime','f5_retail','f6_fx','f7_vix']
        df_zs = df[feats].apply(zs)
        f_vec = df_zs.iloc[-1].values
        
        try:
            lookback = 60
            y = df["^TWII"].pct_change().shift(-1).dropna().tail(lookback)
            X = df_zs.tail(lookback+1).iloc[:-1]
            new_w = Ridge(alpha=2.0).fit(X, y).coef_
            smooth_w = self.lambda_w * self.last_w + (1 - self.lambda_w) * new_w
        except: smooth_w = self.last_w

        vol_adj = 2.5 * (0.012 / (self._to_scalar(df['park_vol']) + 1e-8))
        score = np.tanh(np.dot(f_vec, smooth_w) * np.clip(vol_adj, 1.5, 4.0))
        return (0 if abs(score) < 0.05 else score), smooth_w, df_zs

    def execute_trading(self, alpha, df, w_vec, df_zs):
        twp, tsmcp, gp, twdp = self._to_scalar(df["^TWII"]), self._to_scalar(df["2330.TW"]), self._to_scalar(df["GC=F"]), self._to_scalar(df["TWD=X"])
        # 合約名目價值
        current_nv = np.array([twp*200, twp*50, twp*10, self._to_scalar(df["^TWOII"])*100, tsmcp*2000, gp*100*twdp])
        
        # PnL 校準
        gross_pnl = np.sum(self.last_pos * self.last_nv * ((current_nv/self.last_nv)-1)) if np.all(self.last_nv>0) else 0

        # 風控
        park_v = self._to_scalar(df['park_vol'])
        t_vol = 0.012
        lev = min((t_vol / (park_v + 1e-8)) * abs(alpha) * 5, 2.0)
        
        # 動態門檻
        vix_threshold = df_zs['f7_vix'].rolling(60).quantile(0.10).iloc[-1]
        if df_zs['f7_vix'].iloc[-1] < vix_threshold: lev *= 0.8
        if (self.peak - self.balance)/self.peak > 0.12: lev = 0 
        
        # 標的分散分配 (TX, MX, FMTX, TMC, STC, GDF)
        # 修正：確保這 6 個標的都有基礎分配權重，符合參賽資格
        alloc = np.zeros(6)
        alloc[0] = max(0.05, abs(w_vec[0]) * 0.4) # TX
        alloc[1] = max(0.05, abs(w_vec[2]) * 0.2) # MX
        alloc[2] = 0.1                            # FMTX (固定緩衝)
        alloc[3] = max(0.05, abs(w_vec[3]) * 0.1) # TMC
        alloc[4] = max(0.05, abs(w_vec[4]) * 0.2) # STC
        alloc[5] = 0.05                           # GDF (基礎避險)
        
        # 匯率動態調節
        fx_threshold = df_zs['f6_fx'].rolling(60).quantile(0.90).iloc[-1]
        if df_zs['f6_fx'].iloc[-1] > fx_threshold: alloc[0] *= 0.8
        
        alloc /= (np.sum(alloc) + 1e-8)

        target_val = self.balance * lev
        direction = np.sign(alpha) if abs(alpha) > 0 else 1 # 中性時假設微量看多以符合資格

        # 修正：部位計算邏輯。如果算出來是 0 口，強制補 1 口，確保符合「交易 6 項標的」規則
        raw_pos = (target_val * alloc / (current_nv + 1e-8)).astype(int) * direction
        for i in range(6):
            if raw_pos[i] == 0: raw_pos[i] = 1 * direction
            
        # 黃金避險特殊處理：若 alpha < 0 則做多避險
        if alpha < 0: raw_pos[5] = abs(raw_pos[5])

        # 交易慣性 (Inertia)
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
            f"🚀 V210 賽事決選版 (六項標的達標)\n"
            f"--------------------------------\n"
            f"🧠 Alpha: {alpha:.4f} | 狀態: {regime}\n"
            f"📊 權重分配: ADR:{w_vec[0]:.0%} | Basis:{w_vec[2]:.0%} | VIX:{w_vec[6]:.0%}\n"
            f"⚡ [必備] 六項持倉明細:\n"
            f" 1.大台(TX): {pos[0]} | 2.小台(MX): {pos[1]}\n"
            f" 3.微台(FMTX): {pos[2]} | 4.中型(TMC): {pos[3]}\n"
            f" 5.台積(STC): {pos[4]} | 6.黃金(GDF): {pos[5]}\n"
            f"--------------------------------\n"
            f"💰 今日損益: {int(pnl):+} TWD\n"
            f"💰 帳戶餘額: {int(self.balance):,} TWD"
        )
        print(msg)
        try:
            m = MIMEText(msg, 'plain', 'utf-8')
            m['Subject'] = Header(f"V210 決選版戰報: {regime}", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
        except: pass
        if self.tg_token and self.tg_chat_id:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg})
        self.save_system_state(self.balance, pos, nv, w_vec)

if __name__ == "__main__":
    TAIFEX_V210_ContestFinal().report()
