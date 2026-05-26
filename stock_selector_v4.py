#!/usr/bin/env python3
# =========================
# A股因子模型 V4.0（GitHub Actions 自动版）
# =========================

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
    # 如果设置 TEST_MODE=1，只跑少量股票用于测试
    TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
    TEST_SYMBOLS = 200      # 测试时的股票数量
    ALL_SYMBOLS = None      # None 表示全市场

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
        raise ValueError("没有获取到任何历史数据")
    hist = pd.concat(frames, ignore_index=True)
    hist.columns = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    hist["date"] = pd.to_datetime(hist["date"])
    return hist

def calc_hist_factors(hist):
    hist = hist.sort_values(["code", "date"]).copy()
    hist["ret"] = hist.groupby("code")["close"].pct_change()
    hist["log_amount"] = np.log(hist["amount"].clip(lower=1))

    # 动量
    hist["momentum"] = hist.groupby("code")["close"].transform(
        lambda x: x.pct_change(periods=CFG.MOMENTUM_PERIOD)
    )

    # 波动率（取负，偏好低波）
    hist["volatility"] = hist.groupby("code")["ret"].transform(
        lambda x: x.rolling(CFG.VOL_PERIOD, min_periods=10).std().shift(1)
    )
    hist["volatility"] = -hist["volatility"]

    # 资金流（5日量比）
    hist["avg_amount_5"] = hist.groupby("code")["amount"].transform(
        lambda x: x.rolling(CFG.MF_PERIOD, min_periods=3).mean().shift(1)
    )
    hist["money_flow"] = hist["amount"] / (hist["avg_amount_5"] + 1e-9) - 1

    # 技术结构（均线多头排列）
    hist["ma5"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
    hist["ma10"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(10).mean())
    hist["ma20"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    hist["tech"] = 0.0
    cond = (hist["ma5"] > hist["ma10"]) & (hist["ma10"] > hist["ma20"])
    hist.loc[cond, "tech"] = 1.0
    hist.loc[(hist["ma5"] > hist["ma10"]) & ~cond, "tech"] = 0.5

    latest_date = hist["date"].max()
    latest = hist[hist["date"] == latest_date].copy()
    return latest

def process_factors(df, factor_cols):
    df = df.copy()
    # 去极值
    for col in factor_cols:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        upper = median + 5 * mad
        lower = median - 5 * mad
        df[col] = df[col].clip(lower, upper)

    # 市值中性化（用成交额对数代理）
    if "log_amount" in df.columns:
        for col in factor_cols:
            beta = np.polyfit(df["log_amount"], df[col], 1)[0]
            df[col] = df[col] - beta * df["log_amount"]

    # 标准化
    for col in factor_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-9)

    return df

def get_final_score(df):
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
    df = df.copy()
    df = df[df["amount"] > CFG.MIN_AMOUNT]
    df = df[df["close"] > CFG.PRICE_LOW]
    # 简单排除ST（如果名称可用则更准）
    if "name" in df.columns:
        df = df[~df["name"].str.contains("ST|退", na=False)]
    return df

def main():
    print(">>> 获取股票列表")
    spot = ak.stock_zh_a_spot_em()
    # 保留一部分用于产出快照文件
    spot.to_csv("last_spot_snapshot.csv", index=False)

    if CFG.TEST_MODE:
        codes = spot["代码"].tolist()[:CFG.TEST_SYMBOLS]
        print(f"TEST MODE：只处理前 {CFG.TEST_SYMBOLS} 只股票")
    else:
        codes = spot["代码"].tolist()

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=CFG.LOOKBACK_DAYS)).strftime("%Y%m%d")

    print(f">>> 获取历史数据（{len(codes)} 只股票，{start_date} - {end_date}）")
    hist = get_hist_data(codes, start_date, end_date)

    print(">>> 计算因子")
    latest = calc_hist_factors(hist)

    # 合并名称
    name_map = spot.set_index("代码")["名称"]
    latest["name"] = latest["code"].map(name_map)

    print(">>> 过滤与打分")
    latest = filter_stocks(latest)
    latest = get_final_score(latest)

    result = latest.sort_values("final_score", ascending=False).head(CFG.TOP_N)
    print("\n🔥 最终选股结果：")
    for _, row in result.iterrows():
        print(f"{row['code']} {row['name']}  总分={row['final_score']:.2f}")

    # 保存结果
    result[["code", "name", "close", "final_score",
            "momentum", "money_flow", "volatility", "tech"]].to_csv(
        "selected_stocks.csv", index=False, encoding="utf-8-sig"
    )
    print("\n结果已保存至 selected_stocks.csv")

if __name__ == "__main__":
    main()
