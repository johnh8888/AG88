# =========================
# A股因子模型 V3.0
# 多因子评分系统（专业版）
# =========================

import os
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
class CFG:
    TOP_N = 10
    LOOKBACK = 60
    MIN_AMOUNT = 1.5e8


# =========================
# DATA
# =========================
def get_data():
    raw = ak.stock_zh_a_spot_em()

    df = pd.DataFrame()
    df["code"] = raw["代码"]
    df["name"] = raw["名称"]
    df["price"] = raw["最新价"]
    df["pct"] = raw["涨跌幅"]
    df["amount"] = raw["成交额"]
    df["turnover"] = raw["换手率"]
    df["amplitude"] = raw["振幅"]

    return df


# =========================
# FACTOR 1: MOMENTUM
# =========================
def calc_momentum(df):
    df["momentum"] = (
        df["pct"] * 0.6
        + df["turnover"] * 0.2
        + (df["amount"] / 1e9) * 0.2
    )
    return df


# =========================
# FACTOR 2: MONEY FLOW
# =========================
def calc_money_flow(df):
    df["money_flow"] = (
        np.log(df["amount"].clip(lower=1)) * 0.5
        + df["turnover"] * 0.5
    )
    return df


# =========================
# FACTOR 3: VOLATILITY
# =========================
def calc_volatility(df):
    df["volatility"] = (
        -df["amplitude"] * 0.7
        - abs(df["pct"]) * 0.3
    )
    return df


# =========================
# FACTOR 4: TECH STRUCTURE
# =========================
def calc_tech(df):
    df["tech"] = (
        (df["pct"] > 0).astype(int) * 1.0
        + (df["price"] > 10).astype(int) * 0.5
    )
    return df


# =========================
# FACTOR MODEL
# =========================
def build_factors(df):
    df = calc_momentum(df)
    df = calc_money_flow(df)
    df = calc_volatility(df)
    df = calc_tech(df)

    return df


# =========================
# NORMALIZATION
# =========================
def normalize(df, cols):
    for c in cols:
        df[c] = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
    return df


# =========================
# FINAL SCORE
# =========================
def score(df):
    df = build_factors(df)

    df = normalize(df, ["momentum", "money_flow", "volatility", "tech"])

    df["final_score"] = (
        df["momentum"] * 0.35
        + df["money_flow"] * 0.30
        + df["volatility"] * 0.20
        + df["tech"] * 0.15
    )

    return df


# =========================
# FILTER
# =========================
def filter_stock(df):
    df = df.copy()

    df = df[
        (df["amount"] > CFG.MIN_AMOUNT)
        & (df["price"] > 3)
        & (~df["name"].str.contains("ST|退", na=False))
    ]

    return df


# =========================
# MAIN
# =========================
def main():
    df = get_data()

    df = filter_stock(df)
    df = score(df)

    df = df.sort_values("final_score", ascending=False).head(CFG.TOP_N)

    print("\n🔥 V3.0 因子模型选股结果\n")
    for _, r in df.iterrows():
        print(
            f"{r['name']}({r['code']}) | "
            f"Score={r['final_score']:.2f} | "
            f"涨幅={r['pct']:.2f}% | "
            f"成交额={r['amount']/1e8:.2f}亿"
        )


if __name__ == "__main__":
    main()
