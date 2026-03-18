import numpy as np
import pandas as pd
import yfinance as yf
import os, requests, smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class TAIFEX_V160_FinalStandard:

    def __init__(self, initial_capital=2000000):
        self.initial_capital = initial_capital
        self.file = "strategy_report.csv"
        self.tg_token = os.getenv("TG_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        self.cost_rate = 0.0015
        self.lambda_w = 0.95
        self.inertia_threshold = 0.10
        # 載入狀態：包含昨日持倉、昨日合約價值、昨日權重向量
        self.balance, self.peak, self.last_pos, self.last_nv, self.last_w = self.load_system_state()

    def load_system_state(self):
        """修正：Position Persistence。嚴格讀取昨日狀態，防止 PnL 漂移。"""
        if os.path.isfile(self.file):
            try:
                df = pd.read_csv(self.file)
                if not df.empty:
                    last = df.iloc[-1]
                    pos = last[[f"pos_{i}" for i in range(6)]].values
                    nv = last[[f"nv_{i}" for i in range(6)]].values
                    w = last[[f"w_{i}" for i in range(5)]].values
                    return float(last["equity"]), float(df["equity"].max()), pos, nv, w
            except: pass
        return self.initial_capital, self.initial_capital, np.zeros(6), np.zeros(6), np.array([0.2]*5)

    def save_system_state(self, equity, pos, nv, w):
        state = {"date": datetime.now().strftime("%Y-%m-%d"), "equity": equity}
        for i in range(6): state[f"pos_{i}"] = pos[i]; state[f"nv_{i}"] = nv[i]
        for i in range(5): state[f"w_{i}"] = w[i]
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
        
        # 修正：Parkinson Volatility (用於 Risk Overlay)
        df['park_vol'] = np.sqrt(1/(4*np.log(2)) * ((np.log(df['High']/df['Low']))**2)).rolling(10).mean()
        
        # 修正：ML 特徵對齊 全部改用 pct_change，解決「水平 vs 變化」的統計偏誤。
        # ADR 比例修正 ：TSM/5 * FX / 2330
        df['f1_adr'] = ((df["TSM"]/5 * df["TWD=X"]) / df["2330.TW"]).pct_change()
        df['f2_semi'] = df["SOXX"].pct_change()
        df['f3_basis'] = (df['^TWII'] - df['^TWII'].rolling(5).mean()).pct_change()
        df['f4_regime'] = (df["^TWOII"]/df["^TWII"]).pct_change()
        df['f5_retail'] = (df['^TWII'].pct_change() / df['^TWII'].rolling(20).std()).shift(1)
        
        return df.dropna(subset=["^TWII"])

    def get_alpha(self, df):
        """修正：Ridge 訓練與實時訊號的穩定性 (1:49:44)。"""
        # 修正：實施 Z-Score 標準化，防止某因子量綱過大主導模型
        def zs_series(s): return (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-8)
        
        features = ['f1_adr','f2_semi','f3_basis','f4_regime','f5_retail']
        df_zs = df[features].apply(zs_series)
        f_vec = df_zs.iloc[-1].values
        
        try:
            # 修正：Lookback 增加至 60 天以提升 Ridge 穩定性 
            lookback = 60
            y = df["^TWII"].pct_change().shift(-1).dropna().tail(lookback)
            X = df_zs.tail(lookback+1).iloc[:-1] # 對齊昨日特徵算今日回報
            
            # 修正：增加 Regularization (alpha=2.0) 防止過度擬合 
            new_w = Ridge(alpha=2.0).fit(X, y).coef_
            smooth_w = self.lambda_w * self.last_w + (1 - self.lambda_w) * new_w
        except: smooth_w = self.last_w

        # Tanh 壓縮訊號至 [-1, 1]
        score = np.tanh(np.dot(f_vec, smooth_w) * 2.5)
        return (0 if abs(score) < 0.05 else score), smooth_w

    def execute_trading(self, alpha, df, w_vec):
        """修正：徹底對齊 PnL 時間線 (1:39:55) 與精細化對映 (1:46:59)。"""
        # 1. 建立今日名目價值 (NV)
        twp, tsmcp, gp, twdp = self._to_scalar(df["^TWII"]), self._to_scalar(df["2330.TW"]), self._to_scalar(df["GC=F"]), self._to_scalar(df["TWD=X"])
        # TX(200), MX(50), FMTX(10), TMC(100), STC(2000), GDF(100*FX)
        current_nv = np.array([twp*200, twp*50, twp*10, self._to_scalar(df["^TWOII"])*100, tsmcp*2000, gp*100*twdp])
        
        # 2. 修正：正確 PnL = 昨日持倉 * (今日價格 - 昨日價格)
        # 透過 (今日 NV / 昨日 NV) - 1 取得各商品的精確 Return
        if np.all(self.last_nv > 0):
            item_rets = (current_nv / self.last_nv) - 1
            gross_pnl = np.sum(self.last_pos * self.last_nv * item_rets)
        else: gross_pnl = 0

        # 3. 波動率風控
        vol_scaler = 0.012 / (self._to_scalar(df['park_vol']) + 1e-8)
        lev = min(vol_scaler * abs(alpha) * 5, 2.0)
        if (self.peak - self.balance)/self.peak > 0.12: lev = 0 
        
        # 4. 精細化因子對映 (1:46:59)
        alloc_weights = np.zeros(6)
        alloc_weights[0] = abs(w_vec[0]) * 0.4 # ADR -> TX
        alloc_weights[1] = abs(w_vec[2]) * 0.2 # Basis -> MX
        alloc_weights[2] = 0.1                 # FMTX (Buffer)
        alloc_weights[3] = abs(w_vec[3]) * 0.1 # Regime -> TMC
        alloc_weights[4] = abs(w_vec[4]) * 0.2 # Retail -> STC
        alloc_weights[5] = 0.1 if alpha < 0 else 0 # Gold Hedge
        alloc_weights /= (np.sum(alloc_weights) + 1e-8)

        target_val = self.balance * lev
        raw_target_pos = (target_val * alloc_weights / (current_nv + 1e-8)).astype(int) * np.sign(alpha)
        if alpha < 0: raw_target_pos[5] = abs(raw_target_pos[5]) # 避險永遠做多

        # 5. 修正：交易慣性 (Inertia)
        diff_nv = np.sum(np.abs(raw_target_pos - self.last_pos) * current_nv)
        if diff_nv < (self.inertia_threshold * self.balance) and abs(alpha) > 0:
            final_pos = self.last_pos; t_cost = 0
        else:
            final_pos = raw_target_pos; t_cost = diff_nv * self.cost_rate
        
        return final_pos, current_nv, (gross_pnl - t_cost), lev

    def report(self):
        df = self.get_data()
        alpha, w_vec = self.get_alpha(df)
        pos, nv, pnl, lev = self.execute_trading(alpha, df, w_vec)
        
        self.balance += pnl
        self.peak = max(self.peak, self.balance)
        regime = "BULL" if alpha > 0 else ("BEAR" if alpha < 0 else "NEUTRAL")

        msg = (
            f"🏆 V160 最終穩定版 (已修復 PnL 錯位與統計對齊)\n"
            f"--------------------------------\n"
            f"🧠 Alpha(Tanh): {alpha:.4f} | 槓桿: {lev:.2f}x\n"
            f"📊 權重分配: ADR:{w_vec[0]:.0%} | Basis:{w_vec[2]:.0%} | Retail:{w_vec[4]:.0%}\n"
            f"🛡️ 交易慣性: {'無變動' if np.array_equal(pos, self.last_pos) else '部位調整'}\n"
            f"⚡ 持倉向量: TX:{pos[0]} | MX:{pos[1]} | STC:{pos[4]} | GDF:{pos[5]}\n"
            f"--------------------------------\n"
            f"💰 今日淨損益: {int(pnl):+} TWD (基於昨日部位變動)\n"
            f"💰 累積帳戶權益: {int(self.balance):,} TWD"
        )
        print(msg)
        try:
            m = MIMEText(msg, 'plain', 'utf-8')
            m['Subject'] = Header(f"V160 終極戰報: {regime} ({datetime.now().strftime('%m/%d')})", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
        except: pass
        if self.tg_token and self.tg_chat_id:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/sendMessage", data={"chat_id": self.tg_chat_id, "text": msg})
        self.save_system_state(self.balance, pos, nv, w_vec)

if __name__ == "__main__":
    TAIFEX_V160_FinalStandard().report()
