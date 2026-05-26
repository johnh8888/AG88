#!/usr/bin/env python3
# =========================
# A股因子模型 V4.0（GitHub Actions 容错版）
# =========================

import os, sys, traceback
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

# =========================
# 配置
# =========================
class CFG:
    TOP_N = 10
    MOMENTUM_PERIOD = 20
    VOL_PERIOD = 20
    MF_PERIOD = 5
    MIN_AMOUNT = 2e8
    LOOKBACK_DAYS = 60
    PRICE_LOW = 5
    TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
    TEST_SYMBOLS = 200

def get_hist_data(codes, start_date, end_date):
    frames = []
    for i, code in enumerate(codes):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")
            if df.empty:
                continue
            df["code"] = code
            frames.append(df[["code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]])
        except Exception:
            continue
    if not frames:
        # 不抛异常，而是返回空 DataFrame，让上游处理
        return pd.DataFrame()
    hist = pd.concat(frames, ignore_index=True)
    hist.columns = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    hist["date"] = pd.to_datetime(hist["date"])
    return hist

def calc_hist_factors(hist):
    if hist.empty:
        return pd.DataFrame()
    # ... 其余不变 ...

def process_factors(df, factor_cols):
    if df.empty:
        return df
    # ... 原有逻辑 ...

def get_final_score(df):
    if df.empty:
        return df
    factor_cols = ["momentum", "money_flow", "volatility", "tech"]
    df = process_factors(df, factor_cols)
    df["final_score"] = (
        df["momentum"] * 0.35 +
        df["money_flow"] * 0.30 +
        df["volatility"] * 0.20 +
        df["tech"] * 0.15
    )
    return df

def filter_stocks(df):
    if df.empty:
        return df
    df = df[df["amount"] > CFG.MIN_AMOUNT]
    df = df[df["close"] > CFG.PRICE_LOW]
    return df

def main():
    try:
        print(">>> 获取股票列表")
        spot = ak.stock_zh_a_spot_em()
        spot.to_csv("last_spot_snapshot.csv", index=False)

        if CFG.TEST_MODE:
            codes = spot["代码"].tolist()[:CFG.TEST_SYMBOLS]
        else:
            codes = spot["代码"].tolist()

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=CFG.LOOKBACK_DAYS)).strftime("%Y%m%d")

        print(f">>> 获取历史数据（{len(codes)} 只）")
        hist = get_hist_data(codes, start_date, end_date)

        if hist.empty:
            print("ERROR: 没有获取到任何历史数据，生成占位文件")
            pd.DataFrame(columns=["code","name","close","final_score","momentum","money_flow","volatility","tech"]).to_csv(
                "selected_stocks.csv", index=False, encoding="utf-8-sig"
            )
            return

        print(">>> 计算因子")
        latest = calc_hist_factors(hist)

        if latest.empty:
            print("ERROR: 因子计算后截面为空")
            pd.DataFrame(columns=["code","name","close","final_score","momentum","money_flow","volatility","tech"]).to_csv(
                "selected_stocks.csv", index=False, encoding="utf-8-sig"
            )
            return

        name_map = spot.set_index("代码")["名称"]
        latest["name"] = latest["code"].map(name_map)

        print(">>> 过滤与打分")
        latest = filter_stocks(latest)
        latest = get_final_score(latest)

        result = latest.sort_values("final_score", ascending=False).head(CFG.TOP_N)

        if result.empty:
            print("WARNING: 没有符合条件的股票")
            pd.DataFrame(columns=["code","name","close","final_score","momentum","money_flow","volatility","tech"]).to_csv(
                "selected_stocks.csv", index=False, encoding="utf-8-sig"
            )
        else:
            result[["code", "name", "close", "final_score",
                    "momentum", "money_flow", "volatility", "tech"]].to_csv(
                "selected_stocks.csv", index=False, encoding="utf-8-sig"
            )
            print("\n🔥 最终选股结果：")
            for _, row in result.iterrows():
                print(f"{row['code']} {row['name']}  总分={row['final_score']:.2f}")

        print("结果已保存至 selected_stocks.csv")

    except Exception as e:
        # 兜底：生成错误信息 CSV
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        pd.DataFrame({"error": [str(e)]}).to_csv("selected_stocks.csv", index=False)

if __name__ == "__main__":
    main()
