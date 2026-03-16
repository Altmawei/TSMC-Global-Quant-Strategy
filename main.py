import numpy as np
import yfinance as yf
import time
import pandas as pd
import os
from datetime import datetime

class TSMC_Combat_Commander_V7:
    def __init__(self, adr_ratio=5):
        self.ratio = adr_ratio
        self.fx_rate = self.update_fx_rate_with_retry()
        self.vol_filter = 0.0035

    def update_fx_rate_with_retry(self):
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                ticker = yf.Ticker("TWD=X")
                current_fx = ticker.fast_info['last_price']
                if current_fx > 0: return current_fx
            except:
                retry_count += 1
                time.sleep(2)
        return 32.5

    def get_signal(self, d):
        """核心邏輯：計算得分並轉化為實戰口數"""
        # [1] 平盤過濾：振幅不到 0.35%，直接回傳空手
        day_amp = (d['high'] - d['low']) / d['open']
        if day_amp < self.vol_filter:
            return "空手觀望", "市場波動不足，不建議進場避免磨損", 0

        # [2] Alpha 計算 (簡化呈現)
        premium = ((d['adr_usd'] / self.ratio) * self.fx_rate / d['tsmc_now']) - 1
        score = premium * 2000 # 基礎分
        
        # [3] 指令轉化邏輯
        risk_mod = 0.5 if d['vix_now'] > 25 else 1.0 # 恐慌時部位減半
        
        if score > 60:
            action = "全力做多"
            detail = f"買入：台積電股期 {int(2*risk_mod)} 口、微台 {int(5*risk_mod)} 口。避險：買入價外 Put 2 口。"
        elif score < -60:
            action = "全力放空"
            detail = f"賣出：台積電股期 {int(2*risk_mod)} 口、小台 {int(3*risk_mod)} 口。避險：買入價外 Call 2 口。"
        else:
            action = "小量試探"
            detail = "維持基本倉位，不建議重倉。"
            
        return action, detail, round(score, 2)

    def save_to_csv(self, report_data):
        filename = "strategy_report.csv"
        df_new = pd.DataFrame([report_data])
        if not os.path.isfile(filename):
            df_new.to_csv(filename, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    commander = TSMC_Combat_Commander_V7()
    
    # 模擬今日實戰數據
    mkt = {
        'adr_usd': 195.5, 'tsmc_now': 1055.0, 
        'open': 21000, 'high': 21300, 'low': 21000, 
        'vix_now': 14.5
    }
    
    action, detail, score = commander.get_signal(mkt)
    
    report = {
        "時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "作戰行動": action,
        "具體操作指令": detail,
        "策略得分": score,
        "參考匯率": round(commander.fx_rate, 2)
    }
    
    print(f"\n🚀 今日作戰指令：{action}")
    print(f"📋 執行細節：{detail}")
    
    commander.save_to_csv(report)
