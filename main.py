import numpy as np
import yfinance as yf
import time
import pandas as pd
import os
from datetime import datetime

class TSMC_Global_Quant_Strategy_V7:
    """
    TSMC Global Quantitative Strategy - Version 7.4 (Auto-Report)
    功能：自動匯率重試 + 平盤過濾 + 結算日修正 + 每日 CSV 報表紀錄
    """
    
    def __init__(self, adr_ratio=5):
        self.ratio = adr_ratio
        self.vol_filter = 0.0035  
        self.normal_gate = 60     
        self.wed_gate = 75        
        self.fx_rate = self.update_fx_rate_with_retry()

    def update_fx_rate_with_retry(self):
        """自動抓取匯率，失敗會重複嘗試 5 次"""
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                ticker = yf.Ticker("TWD=X")
                current_fx = ticker.fast_info['last_price']
                if current_fx > 0:
                    print(f"--- [SUCCESS] 匯率抓取成功: {current_fx:.2f} ---")
                    return current_fx
            except Exception as e:
                retry_count += 1
                print(f"--- [RETRY] 匯率抓取失敗 (第 {retry_count} 次) ---")
                time.sleep(2) 
        return 32.5

    def get_market_context(self):
        """判斷交易時段與週三結算日"""
        now = datetime.now()
        is_wednesday = now.weekday() == 2 
        time_str = now.strftime("%H:%M")
        
        if "08:45" <= time_str <= "09:30":
            status = "GOLDEN_OPENING" 
        elif "09:31" <= time_str <= "13:25":
            status = "CORE_TRADING"
        else:
            status = "MARKET_CLOSED"
        return status, is_wednesday

    def calculate_alpha(self, d, status, is_wed):
        """多因子計分引擎"""
        # [Volatility Filter]
        day_amp = (d['high'] - d['low']) / d['open']
        if day_amp < self.vol_filter:
            return 0, "Flat_Filtered", 0

        score = 0
        # 1. ADR Premium
        premium = ((d['adr_usd'] / self.ratio) * self.fx_rate / d['tsmc_now']) - 1
        adr_weight = 1.5 if status == "GOLDEN_OPENING" else 1.0
        score += np.clip(premium * 2000 * adr_weight, -35, 35)

        # 2. Retail Sentiment
        retail_w = -280 if is_wed else -160
        score += np.clip(d['retail_ratio'] * retail_w, -30, 30)

        # 3. Momentum & VIX
        score += np.clip((d['otc_pct'] - d['taiex_pct']) * 800, -15, 15)
        vix_gap = d['vix_now'] - d['vix_ma5']
        vix_penalty = -25 if vix_gap > 2.0 else (10 if vix_gap < -0.5 else 0)
        cb_score = 10 if d.get('cb_surge') else -5
        score += np.clip(vix_penalty + cb_score, -20, 20)

        gate = self.wed_gate if is_wed else self.normal_gate
        return score, gate, premium

    def save_to_csv(self, report_data):
        """核心新增：自動將結果寫入 CSV 檔案"""
        filename = "strategy_report.csv"
        df_new = pd.DataFrame([report_data])
        
        # 如果檔案不存在則建立並加入標題，若存在則直接追加數據
        if not os.path.isfile(filename):
            df_new.to_csv(filename, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"--- [REPORT] 每日決策已同步至 {filename} ---")

# --- 自動執行與測試區 ---
if __name__ == "__main__":
    bot = TSMC_Global_Quant_Strategy_V7()
    
    # 模擬今日數據 (實戰時此處數據可由 API 帶入)
    test_market = {
        'adr_usd': 195.5, 'tsmc_now': 1055.0, 'retail_ratio': -0.15, 
        'otc_pct': 0.012, 'taiex_pct': 0.005,
        'open': 21000, 'high': 21300, 'low': 21000,
        'vix_now': 14.5, 'vix_ma5': 15.0, 'cb_surge': True
    }
    
    status, is_wed = bot.get_market_context()
    s, g, p = bot.calculate_alpha(test_market, status, is_wed)
    
    # 建立 CSV 紀錄格式
    report = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Score": round(s, 2),
        "Premium": f"{round(p*100, 2)}%",
        "FX_Rate": round(bot.fx_rate, 2),
        "Decision": "STRONG_LONG" if s >= g else ("STRONG_SHORT" if s <= -g else "WAIT"),
        "Market_Status": status,
        "Is_Wednesday": is_wed
    }
    
    # 輸出結果
    print("\n--- 實戰決策報告 ---")
    print(report)
    
    # 儲存至 CSV
    bot.save_to_csv(report)
