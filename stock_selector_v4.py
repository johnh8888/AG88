#!/usr/bin/env python3
# =========================
# A股因子模型 V5.1（数据稳定生产版）
# =========================

import os
import time
import random
import traceback
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

# =========================
# 配置
# =========================
class CFG:
    TOP_N = 10
    LOOKBACK_DAYS = 60
    MIN_AMOUNT = 2e8
    PRICE_LOW = 5

    BATCH_SIZE = 50          # 防止被封
    MAX_RETRY = 3            # 重试次数
    SLEEP_RANGE = (0.2, 1.2) # 随机延迟

    CACHE_FILE = "spot_cache.csv"
    ENABLE_CACHE = True


# =========================
# 工具：安全请求
# =========================
def safe_call(func, *args, retry=3, sleep_range=(0.2, 1.0), **kwargs):
    last_err = None
    for i in range(retry):
        try:
            time.sleep(random.uniform(*sleep_range))
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(2 ** i)
    return None


# =========================
# 股票列表（带缓存）
# =========================
def get_spot():
    if CFG.ENABLE_CACHE and os.path.exists(CFG.CACHE_FILE):
        try:
            df = pd.read_csv(CFG.CACHE_FILE)
            if len(df) > 1000:
                print("📦 使用缓存股票列表")
                return df
        except:
            pass

    print("🌐 拉取实时股票列表...")
    df = safe_call(ak.stock_zh_a_spot_em, retry=CFG.MAX_RETRY,
                   sleep_range=CFG.SLEEP_RANGE)

    if df is None or df.empty:
        raise RuntimeError("无法获取股票列表（所有数据源失败）")

    df.to_csv(CFG.CACHE_FILE, index=False)
    return df


# =========================
# 历史数据（稳定版）
# =========================
def fetch_one(code, start, end):
    try:
        df = safe_call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
            retry=CFG.MAX_RETRY,
            sleep_range=CFG.SLEEP_RANGE
        )
        if df is None or df.empty:
            return None

        df["code"] = code
        return df[["code", "日期", "开盘", "收盘", "最高", "最低", "成交额"]]

    except:
        return None


def get_hist_data(codes, start, end):
    frames = []

    for i, code in enumerate(codes):
        df = fetch_one(code, start, end)
        if df is not None:
            frames.append(df)

        # 分批休息（防封）
        if i % CFG.BATCH_SIZE == 0:
            time.sleep(random.uniform(0.5, 2))

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    hist.columns = ["code", "date", "open", "close", "high", "low", "amount"]
    hist["date"] = pd.to_datetime(hist["date"])
    return hist


# =========================
# 因子计算（简化稳定版）
# =========================
def calc_factors(df):
    if df.empty:
        return df

    df = df.sort_values(["code", "date"])

    def momentum(x):
        return x.pct_change(5)

    def vol(x):
        return x.pct_change().rolling(10).std()

    df["momentum"] = df.groupby("code")["close"].transform(momentum)
    df["volatility"] = df.groupby("code")["close"].transform(vol)
    df["money_flow"] = df["amount"].rolling(5).mean()

    df["tech"] = (
        df["close"] / df["close"].rolling(10).mean()
    )

    latest = df.groupby("code").tail(1)
    return latest.dropna()


# =========================
# 清洗
# =========================
def filter(df):
    return df[
        (df["amount"] > CFG.MIN_AMOUNT) &
        (df["close"] > CFG.PRICE_LOW)
    ]


# =========================
# 打分
# =========================
def score(df):
    df["final_score"] = (
        df["momentum"] * 0.35 +
        df["money_flow"] * 0.30 +
        df["volatility"] * 0.20 +
        df["tech"] * 0.15
    )
    return df


# =========================
# 主流程
# =========================
def main():
    try:
        print("=== V5.1 数据稳定引擎启动 ===")

        spot = get_spot()
        spot.to_csv("last_spot_snapshot.csv", index=False)

        codes = spot["代码"].tolist()

        end = datetime.now()
        start = end - timedelta(days=CFG.LOOKBACK_DAYS)

        print(f"📡 获取历史数据: {len(codes)}只股票")

        hist = get_hist_data(
            codes,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d")
        )

        if hist.empty:
            print("⚠️ 历史数据为空 → 输出兜底文件")
            pd.DataFrame(columns=[
                "code", "final_score", "momentum",
                "money_flow", "volatility", "tech"
            ]).to_csv("selected_stocks.csv", index=False)
            return

        print("🧠 计算因子")
        df = calc_factors(hist)

        if df.empty:
            print("⚠️ 因子为空 → 输出兜底文件")
            pd.DataFrame(columns=[
                "code", "final_score"
            ]).to_csv("selected_stocks.csv", index=False)
            return

        df = filter(df)
        df = score(df)

        result = df.sort_values("final_score", ascending=False).head(CFG.TOP_N)

        result.to_csv("selected_stocks.csv", index=False)

        print("\n🔥 TOP选股：")
        for _, r in result.iterrows():
            print(f"{r['code']}  score={r['final_score']:.3f}")

        print("\n✅ 完成")

    except Exception as e:
        print("❌ 系统崩溃，但已兜底输出")
        traceback.print_exc()

        pd.DataFrame({
            "error": [str(e)]
        }).to_csv("selected_stocks.csv", index=False)


if __name__ == "__main__":
    main()
