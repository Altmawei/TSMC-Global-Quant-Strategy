import numpy as np
import pandas as pd
import yfinance as yf
import os, requests, smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from sklearn.linear_model import Ridge

class TAIFEX_V260_TheUltimate:

    def __init__(self, initial_capital=2000000):
        self.initial_capital = initial_capital
        self.file = "strategy_report.csv"
        self.tg_token = os.getenv("TG_TOKEN")
        self.tg_chat_id = os.getenv("TG_CHAT_ID")
        # 修正 4. 考慮雙邊成本 + 滑價
        self.cost_rate = 0.0020 # 提高至 20 bps 模擬真實磨耗
        self.lambda_w = 0.95
        self.balance, self.peak, self.last_pos, self.last_nv, self.last_w = self.load_system_state()

    def load_system_state(self):
        """核心修正：強制回溯最近一次有效數字，解決天文數字中毒問題"""
        if os.path.isfile(self.file):
            try:
                df = pd.read_csv(self.file)
                if not df.empty:
                    # 過濾掉小於 0 或極端大的異常值 (NaN 會自動過濾)
                    valid_df = df[(df['equity'] > 10000) & (df['equity'] < 1e10)]
                    if not valid_df.empty:
                        last = valid_df.iloc[-1]
                        bal = float(last["equity"])
                        pos = last[[f"pos_{i}" for i in range(6)]].values
                        nv = last[[f"nv_{i}" for i in range(6)]].values
                        w = last[[f"w_{i}" for i in range(7)]].values if "w_6" in last else np.array([0.14]*7)
                        return bal, float(valid_df["equity"].max()), pos, nv, w
            except: pass
        return self.initial_capital, self.initial_capital, np.zeros(6), np.zeros(6), np.array([0.14]*7)

    def save_system_state(self, equity, pos, nv, w):
        if np.isnan(equity) or equity <= 0 or equity > 1e10: return 
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
        raw = yf.download(tickers, period="300d", progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 20: raise ValueError("行情抓取失敗")
        
        df_close = raw['Close'].ffill().bfill()
        df_high = raw['High'].ffill().bfill() if 'High' in raw else df_close
        df_low = raw['Low'].ffill().bfill() if 'Low' in raw else df_close

        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close['^TWII']
        # Parkinson Vol
        log_hl_sq = (np.log(df_high['^TWII'] / (df_low['^TWII'] + 1e-9)))**2
        df['park_vol'] = np.sqrt(1/(4*np.log(2)) * log_hl_sq).rolling(10).mean().ffill().replace(0, 0.01)
        
        df['f1_adr'] = ((df_close["TSM"]/5 * df_close["TWD=X"]) / (df_close["2330.TW"] + 1e-9)).pct_change()
        df['f2_semi'] = df_close["SOXX"].pct_change()
        df['f3_basis'] = (df['Close'] - df['Close'].rolling(5).mean()).pct_change()
        df['f4_regime'] = (df_close["^TWOII"] / (df_close["^TWII"] + 1e-9)).pct_change()
        df['f5_retail'] = (df['Close'].pct_change() / (df['Close'].rolling(20).std() + 1e-9)).shift(1)
        df['f6_fx'] = -df_close['TWD=X'].pct_change()
        
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
        if np.isnan(raw_score): return 0.0, smooth_w, df_zs
        
        # 修正 1. 動態因子放大 (截圖 2:46:05)
        park_v = self._to_scalar(df['park_vol'])
        # 根據波動率動態調整 Tanh 斜率：低波動時放大信號，高波動時縮小
        score = np.tanh(raw_score * np.clip(2.5 * (0.012 / (park_v + 1e-9)), 1.5, 4.0))
        return score, smooth_w, df_zs

    def execute_trading(self, alpha, df, w_vec, df_zs):
        p = self.prices
        # 動態名目價值
        current_nv = np.array([p.get("^TWII")*200, p.get("^TWII")*50, p.get("^TWII")*10,
                               p.get("^TWOII")*100, p.get("2330.TW")*2000, p.get("GC=F")*100*p.get("TWD=X")])
        
        # PnL 計算 (對齊昨日部位)
        if np.all(self.last_nv > 0):
            item_rets = (current_nv / (self.last_nv + 1e-9)) - 1
            gross_pnl = np.sum(self.last_pos * self.last_nv * item_rets)
        else: gross_pnl = 0

        # 風控
        park_v = self._to_scalar(df['park_vol'])
        # 修正 5. 門檻隨波動調整 
        dynamic_threshold = 0.05 * (park_v / 0.012)
        direction = np.sign(alpha) if abs(alpha) > dynamic_threshold else 0
        
        lev = min((0.012 / (park_v + 1e-9)) * abs(alpha) * 5, 2.0)
        
        # 修正 2. VIX Percentile 降槓桿 
        vix_z = df_zs['f7_vix'].iloc[-1]
        vix_th = df_zs['f7_vix'].rolling(60).quantile(0.10).iloc[-1]
        if vix_z < vix_th: lev *= 0.8 # VIX 噴出時自動減碼

        target_val = self.balance * (lev if not np.isnan(lev) else 0)
        
        # 標的分配
        weights = np.array([0.4, 0.2, 0.1, 0.1, 0.2, 0.05]); weights /= np.sum(weights)
        raw_pos = (target_val * weights / (current_nv + 1e-9))
        final_pos = np.nan_to_num(raw_pos, nan=1).astype(int) * (direction if direction != 0 else 1)
        
        # 修正 3. 強制 6 標的最低 1 口 
        for i in range(6):
            if final_pos[i] == 0: final_pos[i] = 1 * (direction if direction != 0 else 1)
            if abs(final_pos[i]) > 500: final_pos[i] = 1 * (direction if direction != 0 else 1)
        
        if alpha < 0: final_pos[5] = abs(final_pos[5]) # 黃金避險

        diff_nv = np.sum(np.abs(final_pos - self.last_pos) * current_nv)
        return final_pos, current_nv, (gross_pnl - diff_nv * self.cost_rate), lev

    def report(self):
        try:
            df = self.get_data()
            alpha, w_vec, df_zs = self.get_alpha(df)
            pos, nv, pnl, lev = self.execute_trading(alpha, df, w_vec, df_zs)
            
            new_balance = self.balance + pnl
            if np.isnan(new_balance) or new_balance <= 0: raise ValueError("計算異常")
            self.balance = new_balance; self.peak = max(self.peak, self.balance)
            
            msg = (
                f"🏆 V260 賽事最終統整版\n"
                f"--------------------------------\n"
                f"🧠 Alpha: {alpha:.4f} | 槓桿: {lev:.2f}x\n"
                f"📊 因子權重: ADR:{w_vec[0]:.0%} | VIX:{w_vec[6]:.0%}\n"
                f"⚡ 持倉口數 (強制 6 標的): TX:{pos[0]} | MX:{pos[1]} | STC:{pos[4]} | GDF:{pos[5]}\n"
                f"💰 今日淨損益: {int(pnl):,} TWD\n"
                f"💰 累積總權益: {int(self.balance):,} TWD\n"
                f"📢 提示: 系統已根據市場波動自動調整 Alpha 放大倍數與 VIX 降槓桿門檻。"
            )
            print(msg)
            m = MIMEText(msg, 'plain', 'utf-8'); m['Subject'] = Header(f"V260 最終戰報", 'utf-8')
            s = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            s.login('jeffreylin1201@gmail.com', 'udcgkrdfdfoqznsn')
            s.sendmail('jeffreylin1201@gmail.com', ['jeffreylin1201@gmail.com'], m.as_string())
            self.save_system_state(self.balance, pos, nv, w_vec)
        except Exception as e: print(f"防護機制執行: {e}")

if __name__ == "__main__":
    TAIFEX_V260_TheUltimate().report()
