import numpy as np
import yfinance as yf
import time
from datetime import datetime

class TSMC_Global_Quant_Strategy_V7:
    """
    TSMC Global Quantitative Strategy - Version 7.3 (Final)
    
    【核心實戰功能】
    1. 指數型重試機制 (Exponential Retry Logic)：
       整合 update_fx_rate_with_retry 確保金融數據抓取的穩定性，避免網路震盪導致計算偏差。
    2. 平盤震盪過濾器 (Volatility Filter)：
       透過 vol_filter 機制在盤整期自動避戰，最小化手續費磨損並提升長期 Sharpe Ratio。
    3. 結算日動態修正 (Settlement Adjuster)：
       針對週三期貨結算日的散戶踩踏慣性與大戶壓低/拉高意圖進行權重校正。
    """
    
    def __init__(self, adr_ratio=5):
        # 1. 核心參數設定
        self.ratio = adr_ratio
        self.pos = 0             
        self.vol_filter = 0.0035  # 平盤過濾門檻 (0.35%)
        self.normal_gate = 60     # 平日進場門檻
        self.wed_gate = 75        # 結算日進場門檻 (高壓過濾)
        
        # 2. 初始化自動更新匯率 (自動重試直到抓到為止)
        self.fx_rate = self.update_fx_rate_with_retry()

    def update_fx_rate_with_retry(self):
        """自動抓取美金匯率，若失敗會重複嘗試直到成功為止 (上限 5 次)"""
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                ticker = yf.Ticker("TWD=X")
                current_fx = ticker.fast_info['last_price']
                if current_fx > 0:
                    print(f"--- [SUCCESS] 匯率抓取成功: {current_fx:.2f} ---")
                    return current_fx
                else:
                    raise ValueError("匯率數值異常")
            except Exception as e:
                retry_count += 1
                print(f"--- [RETRY] 匯率抓取失敗 (第 {retry_count} 次)，等待中... ---")
                time.sleep(2) 
        
        print("--- [ERROR] 匯率重試次數達上限，使用預設值 32.5 ---")
        return 32.5

    def get_market_context(self):
        """判斷交易時段與週三結算日"""
        now = datetime.now()
        is_wednesday = now.weekday() == 2 # 0:Mon, 2:Wed
        time_str = now.strftime("%H:%M")
        
        # 08:45 - 09:30 定義為黃金權重期
        if "08:45" <= time_str <= "09:30":
            status = "GOLDEN_OPENING" 
        elif "09:31" <= time_str <= "13:25":
            status = "CORE_TRADING"
        else:
            status = "MARKET_CLOSED"
        return status, is_wednesday

    def calculate_alpha(self, d, status, is_wed):
        """
        多因子計分引擎 (Alpha Scoring)
        d 字典內容：adr_usd, tsmc_now, retail_ratio, otc_pct, taiex_pct, open, high, low, vix_now, vix_ma5, cb_surge
        """
        # [Volatility Filter] 平盤保護機制：高低差不到門檻則不運算
        day_amp = (d['high'] - d['low']) / d['open']
        if day_amp < self.vol_filter:
            return 0, "Flat Market Filtered (振幅不足，避開洗盤)", 0

        score = 0
        
        # [Alpha Factor 1] ADR Premium (權重 35)
        premium = ((d['adr_usd'] / self.ratio) * self.fx_rate / d['tsmc_now']) - 1
        adr_weight = 1.5 if status == "GOLDEN_OPENING" else 1.0
        score += np.clip(premium * 2000 * adr_weight, -35, 35)

        # [Alpha Factor 2] Retail Sentiment (權重 30)
        # 結算日對散戶反指標權重強化 (retail_ratio 為正代表散戶多單，得分會變負)
        retail_w = -280 if is_wed else -160
        score += np.clip(d['retail_ratio'] * retail_w, -30, 30)

        # [Alpha Factor 3] OTC/Market Relative Momentum (權重 15)
        score += np.clip((d['otc_pct'] - d['taiex_pct']) * 800, -15, 15)
        
        # [Alpha Factor 4] VIX Risk Penalty (權重 20)
        vix_gap = d['vix_now'] - d['vix_ma5']
        vix_penalty = -25 if vix_gap > 2.0 else (10 if vix_gap < -0.5 else 0)
        cb_score = 10 if d.get('cb_surge') else -5
        score += np.clip(vix_penalty + cb_score, -20, 20)

        gate = self.wed_gate if is_wed else self.normal_gate
        return score, gate, premium

    def trade_execution(self, score, gate, is_wed, vix_now):
        """輸出最終實戰決策"""
        if abs(score) < gate:
            return f"【保持觀望】得分 ({score:.1f}) 未達標 ({gate})"

        # 風控：VIX 高於 25 代表非理性波動，部位縮減一半
        risk_mod = 0.5 if vix_now > 25 else 1.0
        order_type = "市價衝刺 (Market +2 Ticks)" if abs(score) > 85 else "限價掛單 (Limit)"
        direction = "STRONG_LONG" if score >= gate else "STRONG_SHORT"
        
        return {
            "訊號方向": direction,
            "多因子總分": f"{score:.2f}",
            "基準匯率": f"{self.fx_rate:.2f}",
            "下單指令": order_type,
            "建議口數": f"{1 * risk_mod} 單位組合",
            "執行商品": ["台積電股期", "微型台指", "中型100", "選擇權避險"]
        }

if __name__ == "__main__":
    # 初始化機器人
    bot = TSMC_Global_Quant_Strategy_V7()
    
    # --- 模擬測試數據 (可手動修改測試) ---
    test_market = {
        'adr_usd': 195.5, 'tsmc_now': 1055.0, 'retail_ratio': -0.15, 
        'otc_pct': 0.012, 'taiex_pct': 0.005,
        'open': 21000, 'high': 21300, 'low': 21000,
        'vix_now': 14.5, 'vix_ma5': 15.0, 'cb_surge': True
    }
    
    status, is_wed = bot.get_market_context()
    s, g, p = bot.calculate_alpha(test_market, status, is_wed)
    print("\n--- 實戰策略分析報告 ---")
    print(bot.trade_execution(s, g, is_wed, test_market['vix_now']))
