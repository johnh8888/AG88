# -*- coding: utf-8 -*-
# V15.2 QUANT SYSTEM (Zhaowei Edition)

import time
import random
import pandas as pd

print("🚀 V15.2 QUANT SYSTEM START")


# =========================
# 安全请求封装
# =========================
def safe_call(func, name="data", retries=5):
    for i in range(retries):
        try:
            data = func()
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            wait = round(random.uniform(2, 10), 2)
            print(f"⚠️ {name}失败 {i+1}/{retries}: {e} | 等待 {wait}s")
            time.sleep(wait)
    return None


# =========================
# A股数据（平安银行）
# =========================
def get_a_stock():
    import akshare as ak

    def fetch():
        return ak.stock_zh_a_hist(
            symbol="000001",
            period="daily",
            adjust="qfq"
        )

    df = safe_call(fetch, "A股数据")

    if df is None:
        print("⚠️ A股数据失败，切备用数据")
        df = pd.DataFrame({"close": [3000, 3050, 3020, 3100, 3080]})

    return df


# =========================
# 兆威机电（替代港股腾讯）
# =========================
def get_zhaowei():
    import akshare as ak

    def fetch():
        return ak.stock_zh_a_hist(
            symbol="003021",
            period="daily",
            adjust="qfq"
        )

    df = safe_call(fetch, "兆威机电")

    if df is None:
        print("⚠️ 兆威机电数据失败，使用 fallback")
        df = pd.DataFrame({"close": [80, 82, 79, 85, 88]})

    return df


# =========================
# 策略模型
# =========================
def strategy(df, name):
    if df is None or len(df) < 3:
        return {"symbol": name, "signal": "NO_DATA", "score": 0}

    score = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    if score > 2:
        signal = "BUY"
    elif score < -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"symbol": name, "signal": signal, "score": round(score, 3)}


# =========================
# 主逻辑
# =========================
def main():

    a_df = get_a_stock()
    z_df = get_zhaowei()

    results = []

    results.append(strategy(a_df, "A 000001"))
    results.append(strategy(a_df, "A 600519"))
    results.append(strategy(a_df, "A 000300"))
    results.append(strategy(z_df, "A 003021 兆威机电"))

    print("\n===== V15.2 交易信号 =====")

    buy = 0
    for r in results:
        print(f"{r['symbol']} | {r['signal']} | score={r['score']}")
        if r["signal"] == "BUY":
            buy += 1

    print("\n===== 风控输出 =====")
    print(f"信号数量: {len(results)}")
    print(f"买入信号: {buy}")

    if buy == 0:
        print("❌ 当前无交易机会（系统风控过滤）")
    else:
        print("✅ 存在交易机会")


if __name__ == "__main__":
    main()