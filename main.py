import numpy as np
import yfinance as yf
from datetime import datetime

class TSMC_Grandmaster_Final:
    def __init__(self, adr_ratio=5):
        self.ratio = adr_ratio
        self.pos = 0             
        self.vol_filter = 0.0035  
        self.normal_gate = 60
        self.wed_gate = 75        
        self.fx_rate = self.update_fx_rate()

    def update_fx_rate(self):
        """自動抓取即時美金匯率"""
        try:
            ticker = yf.Ticker("TWD=X")
            current_fx = ticker.fast_info['last_price']
            print(f"--- [INFO] FX Rate Updated: {current_fx:.2f} ---")
            return current_fx
        except Exception as e:
            print(f"--- [WARN] FX Fetch Failed, using 32.5: {e} ---")
            return 32.5

    def get_market_context(self):
        now = datetime.now()
        is_wed = now.weekday() == 2
        time_str = now.strftime("%H:%M")
        
        if "08:45" <= time_str <= "09:30":
            status = "GOLDEN_OPENING"
        elif "09:31" <= time_str <= "13:25":
            status = "CORE_TRADING"
        else:
            status = "MARKET_CLOSED"
        return status, is_wed

    def calculate_alpha(self, d, status, is_wed):
        day_amp = (d['high'] - d['low']) / d['open']
        if day_amp < self.vol_filter:
            return 0, "Flat Market Filtered", 0

        score = 0
        premium = ((d['adr_usd'] / self.ratio) * self.fx_rate / d['tsmc_now']) - 1
        adr_w = 1.5 if status == "GOLDEN_OPENING" else 1.0 
        score += np.clip(premium * 2000 * adr_w, -35, 35)
        
        retail_w = -280 if is_wed else -160
        score += np.clip(d['retail_ratio'] * retail_w, -30, 30)
        
        score += np.clip((d['otc_pct'] - d['taiex_pct']) * 800, -15, 15)
        
        vix_gap = d['vix_now'] - d['vix_ma5']
        vix_penalty = -25 if vix_gap > 2.0 else (10 if vix_gap < -0.5 else 0)
        cb_score = 10 if d.get('cb_surge', False) else -5
        score += np.clip(vix_penalty + cb_score, -20, 20)

        gate = self.wed_gate if is_wed else self.normal_gate
        return score, gate, premium

    def trade_execution(self, score, gate, is_wed, vix_now):
        if abs(score) < gate:
            return f"【Wait】Score {score:.1f} < Gate {gate}"

        risk_mod = 0.5 if vix_now > 25 else 1.0
        order_type = "Market +2 Ticks" if abs(score) > 85 else "Limit Order"
        mode = "Settlement Mode" if is_wed else "Trend Mode"
        
        direction = "STRONG_LONG" if score >= gate else "STRONG_SHORT"
        items = ["TSMC Futures", "Micro TAIEX", "Mid 100", "Option Hedge"]
        
        return {
            "Direction": direction, "Mode": mode, "Order": order_type,
            "Items": items, "PositionSize": f"{1 * risk_mod} units", "FX": f"{self.fx_rate:.2f}"
        }

# --- GitHub 測試腳本 ---
if __name__ == "__main__":
    bot = TSMC_Grandmaster_Final()
    
    # 模擬今日數據測試
    test_market = {
        'adr_usd': 198.5, 'tsmc_now': 1055.0, 
        'retail_ratio': -0.15, 
        'otc_pct': 0.012, 'taiex_pct': 0.004,
        'open': 21000, 'high': 21300, 'low': 21000, 
        'vix_now': 14.5, 'vix_ma5': 15.2,
        'cb_surge': True
    }
    
    status, is_wed = bot.get_market_context()
    s, g, p = bot.calculate_alpha(test_market, status, is_wed)
    print(bot.trade_execution(s, g, is_wed, test_market['vix_now']))
