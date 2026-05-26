#!/usr/bin/env python3
# =========================
# A股因子模型 V5.2（终极数据稳定版）
# =========================

import os
import time
import traceback
import pandas as pd
import numpy as np
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

    CACHE_FILE = "spot_cache.csv"


# =========================
# 静态兜底股票池（关键）
# =========================
FALLBACK_CODES = [
    "000001", "000002", "000333", "600000", "600036",
    "600519", "600276", "601318", "601166", "601899"
]


# =========================
# Level 3：绝对兜底
# =========================
def get_fallback_spot():
    print("🧱 使用兜底股票池（防止系统崩溃）")
    return pd.DataFrame({
        "代码": FALLBACK_CODES,
        "名称": ["备用股"] * len(FALLBACK_CODES)
    })


# =========================
# Level 1 + Level 2
# =========================
def get_spot():
    # 1. 本地缓存
    if os.path.exists(CFG.CACHE_FILE):
        try:
            df = pd.read_csv(CFG.CACHE_FILE)
            if len(df) > 100:
                print("📦 使用本地缓存数据")
                return df
        except:
            pass

    # 2. AkShare
    try:
        print("🌐 尝试 AkShare 获取股票列表...")
        df = ak.stock_zh_a_spot_em()
        df.to_csv(CFG.CACHE_FILE, index=False)
        return df

    except Exception as e:
        print(f"⚠️ AkShare失败: {e}")

    # 3. fallback
    return get_fallback_spot()


# =========================
# 历史数据（安全版）
# =========================
def fetch_hist(code, start, end):
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )
        if df is None or df.empty:
            return None

        df["code"] = code
        return df

    except:
        return None


def get_hist(codes, start, end):
    frames = []

    for i, c in enumerate(codes):
        df = fetch_hist(c, start, end)
        if df is not None:
            frames.append(df)

        if i % 3 == 0:
            time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# =========================
# 因子（简化稳定版）
# =========================
def calc(df):
    if df.empty:
        return df

    df = df.sort_values(["code", "日期"])

    df["ret"] = df.groupby("code")["收盘"].pct_change(5)
    df["vol"] = df.groupby("code")["收盘"].pct_change().rolling(10).std()
    df["amt"] = df["成交额"].rolling(5).mean()

    latest = df.groupby("code").tail(1)
    return latest.dropna()


# =========================
# scoring
# =========================
def score(df):
    df["score"] = (
        df["ret"] * 0.4 +
        df["amt"] * 0.3 +
        (-df["vol"]) * 0.3
    )
    return df


# =========================
# 主程序
# =========================
def main():
    try:
        print("=== V5.2 数据稳定引擎 ===")

        spot = get_spot()
        spot.to_csv("last_spot_snapshot.csv", index=False)

        codes = spot["代码"].tolist()

        end = datetime.now()
        start = end - timedelta(days=CFG.LOOKBACK_DAYS)

        print(f"📡 拉取历史数据: {len(codes)}")

        hist = get_hist(
            codes,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d")
        )

        if hist.empty:
            print("⚠️ 历史数据为空 → 输出空文件")
            pd.DataFrame(columns=["code", "score"]).to_csv(
                "selected_stocks.csv", index=False
            )
            return

        print("🧠 计算因子")
        df = calc(hist)

        if df.empty:
            print("⚠️ 因子为空")
            pd.DataFrame(columns=["code", "score"]).to_csv(
                "selected_stocks.csv", index=False
            )
            return

        df = score(df)

        res = df.sort_values("score", ascending=False).head(CFG.TOP_N)

        res.to_csv("selected_stocks.csv", index=False)

        print("\n🔥 TOP:")
        print(res[["code", "score"]])

        print("✅ 完成")

    except Exception as e:
        print("❌ 系统级错误，但已兜底")
        traceback.print_exc()

        pd.DataFrame({"error": [str(e)]}).to_csv(
            "selected_stocks.csv", index=False
        )


if __name__ == "__main__":
    main()
