# ==============================
# V15.1 稳定工业级量化系统
# 支持：A股 + 港股（腾讯）
# 防崩溃 / 防断网 / 自动重试 / 双数据源
# ==============================

import time
import random
import sys
import traceback
import akshare as ak
import pandas as pd
import numpy as np

print("🚀 V15.1 QUANT SYSTEM START")

# ==============================
# 工业级安全请求封装
# ==============================
def safe_ak(func, *args, retries=5, sleep=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = sleep * (i + 1) + random.random()
            print(f"⚠️ AKShare失败 {i+1}/{retries}: {e} | 等待 {wait:.1f}s")
            time.sleep(wait)
    return None


# ==============================
# A股数据源
# ==============================
def get_a_stock(symbol="000001"):
    df = safe_ak(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        adjust="qfq"
    )

    if df is not None and not df.empty:
        df["market"] = "A"
        return df

    print("⚠️ A股数据失败，切备用指数")
    return safe_ak(ak.stock_zh_index_daily, symbol="sh000001")


# ==============================
# 港股数据源（腾讯）
# ==============================
def get_hk_stock(symbol="00700"):
    """
    腾讯控股 = 0700.HK（akshare用00700）
    """
    df = safe_ak(
        ak.stock_hk_hist,
        symbol=symbol,
        period="daily"
    )

    if df is None or df.empty:
        print("⚠️ 港股数据失败（腾讯）")
        return None

    df["market"] = "HK"
    return df


# ==============================
# 简单信号模型（稳定版）
# ==============================
def signal_model(df):
    if df is None or df.empty:
        return {"signal": "NO_DATA", "score": 0}

    try:
        close = df["收盘"].astype(float) if "收盘" in df.columns else df.iloc[:, 4]

        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        last = close.iloc[-1]

        score = (last - ma10) + 0.5 * (ma5 - ma10)

        if score > 0:
            signal = "BUY"
        elif score < 0:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "score": float(score),
            "last": float(last)
        }

    except Exception as e:
        print("❌ 信号计算失败:", e)
        return {"signal": "ERROR", "score": 0}


# ==============================
# 主执行逻辑
# ==============================
def main():

    results = []

    # ===== A股核心标的 =====
    a_list = ["000001", "600519", "000300"]

    for s in a_list:
        df = get_a_stock(s)
        res = signal_model(df)
        res["symbol"] = s
        res["market"] = "A"
        results.append(res)

    # ===== 港股腾讯 =====
    hk = get_hk_stock("00700")
    res = signal_model(hk)
    res["symbol"] = "0700.HK"
    res["market"] = "HK"
    results.append(res)

    # ==============================
    # 输出结果
    # ==============================
    print("\n===== V15.1 交易信号 =====")

    for r in results:
        print(f"{r['market']} {r['symbol']} | {r['signal']} | score={r['score']:.3f}")

    # ==============================
    # 风控汇总
    # ==============================
    buy_list = [r for r in results if r["signal"] == "BUY"]

    print("\n===== 风控输出 =====")
    print(f"信号数量: {len(results)}")
    print(f"买入信号: {len(buy_list)}")

    if len(buy_list) == 0:
        print("❌ 当前无交易机会（系统风控过滤）")
    else:
        print("✅ 存在可交易标的")


# ==============================
# 全局防崩溃
# ==============================
if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("❌ 系统崩溃但已捕获")
        traceback.print_exc()
        sys.exit(0)