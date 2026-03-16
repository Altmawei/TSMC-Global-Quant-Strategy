import numpy as np
import yfinance as yf
import time
from datetime import datetime

class TSMC_Global_Quant_Strategy_V7:
    def __init__(self, adr_ratio=5):
        # 1. 核心參數設定
        self.ratio = adr_ratio
        self.pos = 0             
        self.vol_filter = 0.0035  # 平盤過濾門檻 (0.35%)
        self.normal_gate = 60     # 平日進場門檻
        self.wed_gate = 75        # 結算日進場門檻 (更高要求)
        
        # 2. 初始化時即啟動【自動重試匯率抓取】
        self.fx_rate = self.update_fx_rate_with_retry()

    def update_fx_rate_with_retry(self):
        """
        核心改進：自動抓取美金匯率，若失敗會重複嘗試直到成功為止 (上限 5 次)
        """
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                ticker = yf.Ticker("TWD=X")
                # 抓取最新成交價
                current_fx = ticker.fast_info['last_price']
                if current_fx > 0:
                    print(f"--- [SUCCESS] 匯率抓取成功: {current_fx:.2f} ---")
                    return current_fx
                else:
                    raise ValueError("匯率數值異常")
            except Exception as e:
                retry_count += 1
                print(f"--- [RETRY] 匯率抓取失敗 (第 {retry_count} 次)，等待中... ---")
                time.sleep(2) # 停兩秒後重試
        
        print("--- [ERROR] 匯率重試次數達上限，使用保底值 32.5 ---")
        return 32.5

    def get_market_context(self):
        """判斷交易時段與是否為週三結算日"""
        now = datetime.now()
        is_wednesday = now.weekday() == 2 # 0是週一, 2是週三
        time_str = now.strftime("%H:%M")
        
        if "08:45" <= time_str <= "09:30":
            status = "GOLDEN_OPENING" # 開盤權重期
        elif "09:31" <= time_str <= "13:25":
            status = "CORE_TRADING"
        else:
            status = "MARKET_CLOSED"
        return status, is_wednesday

    def calculate_alpha(self, d, status, is_wed):
        """
        多因子計分引擎
        d: 包含行情數據的字典
        """
        # [0] 波動度過濾器 (平盤保護機制)
        day_amp = (d['high'] - d['low']) / d['open']
        if day_amp < self.vol_filter:
            return 0, "Flat Market Filtered (振幅不足，封盤)", 0

        score = 0
        
        # [1] ADR 領先指標 (權重 35)
        # 溢價公式: ((ADR/5) * 匯率 / 台股現貨) - 1
        premium = ((d['adr_usd'] / self.ratio) * self.fx_rate / d['tsmc_now']) - 1
        adr_weight = 1.5 if status == "GOLDEN_OPENING" else 1.0
        score += np.clip(premium * 2000 * adr_weight, -35, 35)

        # [2] 散戶籌碼指標 (權重 30)
        # 結算日加權 (retail_ratio > 0 代表散戶做多，得分會變負)
        retail_w = -280 if is_wed else -160
        score += np.clip(d['retail_ratio'] * retail_w, -30, 30)

        # [3] 內資動能 (15) & 恐慌情緒 (20)
        score += np.clip((d['otc_pct'] - d['taiex_pct']) * 800, -15, 15)
        
        vix_gap = d['vix_now'] - d['vix_ma5']
        vix_penalty = -25 if vix_gap > 2.0 else (10 if vix_gap < -0.5 else 0)
        score += np.clip(vix_penalty + (10 if d.get('cb_surge') else -5), -20, 20)

        gate = self.wed_gate if is_wed else self.normal_gate
        return score, gate, premium

    def trade_execution(self, score, gate, is_wed, vix_now):
        """輸出最終交易決策"""
        if abs(score) < gate:
            return f"【保持觀望】得分 ({score:.1f}) 未達標 ({gate})"

        # 風控：VIX 過高則減半部位
        risk_mod = 0.5 if vix_now > 25 else 1.0
        order_type = "市價衝刺 (Market +2)" if abs(score) > 85 else "限價掛單 (Limit)"
        
        direction = "STRONG_LONG" if score >= gate else "STRONG_SHORT"
        
        return {
            "方向": direction,
            "得分": f"{score:.2f}",
            "匯率": f"{self.fx_rate:.2f}",
            "執行建議": order_type,
            "單位數量": f"{1 * risk_mod} 單位",
            "組合": ["台積電股期", "微/小台", "中型100", "選擇權避險"]
        }

# --- GitHub 實戰執行測試區 ---
if __name__ == "__main__":
    # 建立機器人實例 (會自動抓匯率)
    bot = TSMC_Global_Quant_Strategy_V7()
    
    # 模擬輸入目前的市場數據 (測試用)
    test_market_data = {
        'adr_usd': 195.5,   # 昨晚 ADR 收盤價
        'tsmc_now': 1055.0, # 台股台積電現價
        'retail_ratio': -0.12, # 散戶看空 (反指標加分)
        'otc_pct': 0.012, 'taiex_pct': 0.005,
        'open': 21000, 'high': 21250, 'low': 21000, # 振幅足夠
        'vix_now': 14.5, 'vix_ma5': 15.0,
        'cb_surge': True
    }
    
    status, is_wed = bot.get_market_context()
    s, g, p = bot.calculate_alpha(test_market_data, status, is_wed)
    plan = bot.trade_execution(s, g, is_wed, test_market_data['vix_now'])
    
    print("\n========= 策略分析報告 =========")
    print(f"當前狀態: {status} | 結算日模式: {is_wed}")
    print(f"ADR 溢價率: {p*100:.2f}%")
    print(f"最終指令: {plan}")
