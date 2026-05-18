# ================================
# A股短线实战选股系统 V2.0
# 强化版（非自动交易）
# 作者：ChatGPT Quant Upgrade
# ================================

import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================================
# 配置参数
# ================================

TEST_MODE = True

TOP_N_CANDIDATES = 20
FINAL_HOLDINGS = 5

PRICE_MIN = 2
PRICE_MAX = 150

MIN_AMOUNT = 5e7

MA_PERIOD = 20

BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 1

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")

now = datetime.now(TZ_SHANGHAI)

today = now.strftime("%Y%m%d")

# ================================
# 工具函数
# ================================

def safe_float(v, default=0):

    try:
        if pd.isna(v):
            return default
        return float(v)

    except:
        return default


def calc_open_pct(row):

    prev_close = safe_float(row.get("prev_close"))

    open_price = safe_float(row.get("open"))

    if prev_close <= 0:
        return 0

    return (
        (open_price / prev_close - 1) * 100
    )


# ================================
# 市场情绪系统
# ================================

def get_market_emotion():

    try:

        spot = ak.stock_zh_a_spot_em()

        pct = pd.to_numeric(
            spot["涨跌幅"],
            errors="coerce"
        )

        up_limit = len(
            pct[pct >= 9.7]
        )

        down_limit = len(
            pct[pct <= -9.5]
        )

        strong = len(
            pct[pct >= 5]
        )

        weak = len(
            pct[pct <= -5]
        )

        score = (
            up_limit * 3
            + strong
            - down_limit * 4
            - weak
        )

        if score >= 200:
            emotion = "hot"

        elif score <= -50:
            emotion = "cold"

        else:
            emotion = "neutral"

        return {
            "emotion": emotion,
            "score": score,
            "up_limit": up_limit,
            "down_limit": down_limit
        }

    except Exception as e:

        logging.warning(f"情绪系统失败: {e}")

        return {
            "emotion": "neutral",
            "score": 0
        }


# ================================
# 热点板块系统
# ================================

def get_hot_sector_score():

    try:

        sector_df = ak.stock_board_industry_name_em()

        sector_df["涨跌幅"] = pd.to_numeric(
            sector_df["涨跌幅"],
            errors="coerce"
        )

        sector_df = sector_df.sort_values(
            "涨跌幅",
            ascending=False
        )

        score_map = {}

        for idx, row in sector_df.iterrows():

            score_map[
                row["板块名称"]
            ] = max(0, 100 - idx)

        return score_map

    except Exception as e:

        logging.warning(f"热点板块失败: {e}")

        return {}


# ================================
# MA均线过滤
# ================================

def is_above_ma(code, period=20):

    try:

        end = today

        start = (
            now - timedelta(days=60)
        ).strftime("%Y%m%d")

        hist = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        if hist is None or hist.empty:
            return False

        close = hist["收盘"].astype(float)

        ma = close.rolling(period).mean().iloc[-1]

        return close.iloc[-1] > ma

    except:

        return False


# ================================
# 龙头评分系统
# ================================

def calc_leader_score(row):

    score = 0

    score += row["pct"] * 2.5

    score += (
        row["amount"] / 1e8
    ) * 0.8

    score += row["turnover"] * 1.2

    score += row["lb"] * 1.5

    score -= row["amplitude"] * 0.4

    score -= abs(
        row["open_pct"]
    ) * 0.5

    return score


# ================================
# 波动率
# ================================

def calc_volatility_ratio(code):

    try:

        end = today

        start = (
            now - timedelta(days=40)
        ).strftime("%Y%m%d")

        hist = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        if hist is None or hist.empty:
            return np.nan

        close = hist["收盘"].astype(float)

        if len(close) < 20:
            return np.nan

        return (
            close.tail(20).std()
            /
            close.tail(20).mean()
        )

    except:

        return np.nan


# ================================
# 历史回测评分
# ================================

def evaluate_stock_history(code):

    try:

        end = today

        start = (
            now - timedelta(
                days=BACKTEST_LOOKBACK_DAYS
            )
        ).strftime("%Y%m%d")

        hist = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )

        if hist is None or hist.empty:
            return {
                "signals": 0,
                "history_score": -999
            }

        hist = hist.sort_values("日期")

        pct = hist["涨跌幅"].astype(float)

        strong_days = pct[pct >= 3]

        signals = len(strong_days)

        history_score = (
            pct.mean() * 5
            +
            pct.max() * 2
            -
            abs(pct.min())
        )

        return {
            "signals": signals,
            "history_score": history_score
        }

    except Exception as e:

        logging.warning(f"{code} 历史评分失败: {e}")

        return {
            "signals": 0,
            "history_score": -999
        }


# ================================
# 获取实时行情
# ================================

def fetch_spot_data():

    logging.info("获取实时行情...")

    raw = ak.stock_zh_a_spot_em()

    if raw is None or raw.empty:

        logging.error("行情为空")

        return pd.DataFrame()

    df = pd.DataFrame()

    df["code"] = raw["代码"]

    df["name"] = raw["名称"]

    df["price"] = pd.to_numeric(
        raw["最新价"],
        errors="coerce"
    )

    df["pct"] = pd.to_numeric(
        raw["涨跌幅"],
        errors="coerce"
    )

    df["amount"] = pd.to_numeric(
        raw["成交额"],
        errors="coerce"
    )

    df["lb"] = pd.to_numeric(
        raw.get("量比", 1),
        errors="coerce"
    ).fillna(1)

    df["turnover"] = pd.to_numeric(
        raw["换手率"],
        errors="coerce"
    )

    df["amplitude"] = pd.to_numeric(
        raw["振幅"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        raw["今开"],
        errors="coerce"
    )

    df["prev_close"] = pd.to_numeric(
        raw["昨收"],
        errors="coerce"
    )

    if "行业" in raw.columns:
        df["行业"] = raw["行业"]

    return df


# ================================
# 主程序
# ================================

logging.info("启动实战选股系统")

df = fetch_spot_data()

if df.empty:

    logging.error("无行情数据")

    sys.exit(0)

# 开盘涨幅

df["open_pct"] = df.apply(
    calc_open_pct,
    axis=1
)

# 波动率

logging.info("计算波动率...")

df["volatility_ratio"] = df["code"].apply(
    calc_volatility_ratio
)

# 删除ST

ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"

df = df[
    ~df["name"].str.contains(
        ban_pattern,
        na=False,
        regex=True
    )
]

# 主板

df = df[
    df["code"].astype(str).str.startswith(
        ("60", "00")
    )
]

# 价格过滤

df = df[
    (df["price"] >= PRICE_MIN)
    &
    (df["price"] <= PRICE_MAX)
]

# 成交额过滤

df = df[
    df["amount"] >= MIN_AMOUNT
]

logging.info(
    f"基础过滤后: {len(df)}"
)

# ================================
# 初始化评分
# ================================

df["score"] = 0

# 龙头评分

logging.info("计算龙头评分...")

df["score"] += df.apply(
    calc_leader_score,
    axis=1
)

# MA加分

logging.info("计算MA趋势...")

df["score"] += df["code"].apply(
    lambda x: 6 if is_above_ma(x, MA_PERIOD) else -3
)

# 波动率扣分

df["score"] -= (
    df["volatility_ratio"]
    .fillna(0)
    * 100
    * 0.5
)

# 强势龙头保护

strong_stock = (
    (df["pct"] >= 7)
    &
    (df["turnover"] >= 8)
)

df.loc[
    strong_stock,
    "score"
] += 15

# 涨停龙头

df.loc[
    df["pct"] >= 9,
    "score"
] += 20

# 放量突破

df.loc[
    (
        (df["turnover"] >= 10)
        &
        (df["lb"] >= 2)
    ),
    "score"
] += 10

# 超跌反弹

df.loc[
    (
        (df["pct"] >= 2)
        &
        (df["open_pct"] < 0)
    ),
    "score"
] += 6

# ================================
# 热点板块
# ================================

logging.info("计算热点板块...")

sector_map = get_hot_sector_score()

if "行业" in df.columns:

    df["sector_score"] = df["行业"].map(
        sector_map
    ).fillna(0)

    df["score"] += (
        df["sector_score"] * 0.5
    )

# ================================
# 市场情绪
# ================================

emotion = get_market_emotion()

logging.info(f"市场情绪: {emotion}")

if emotion["emotion"] == "hot":

    df["score"] += (
        df["pct"] * 1.5
    )

elif emotion["emotion"] == "cold":

    df["score"] -= (
        df["amplitude"] * 0.8
    )

# ================================
# 排序
# ================================

filtered = df.sort_values(
    "score",
    ascending=False
).head(80)

logging.info(
    f"评分后股票数量: {len(filtered)}"
)

# ================================
# 历史评分
# ================================

history_rows = []

valid_rows = []

logging.info("开始历史回测评分...")

for idx, row in filtered.iterrows():

    code = row["code"]

    logging.info(f"评估: {code}")

    hist = evaluate_stock_history(code)

    if hist["signals"] < BACKTEST_MIN_SIGNALS:
        continue

    history_rows.append(hist)

    valid_rows.append(idx)

candidates = filtered.loc[
    valid_rows
].reset_index(drop=True)

hist_df = pd.DataFrame(history_rows)

candidates = pd.concat(
    [candidates, hist_df],
    axis=1
)

# ================================
# 最终评分
# ================================

candidates["final_score"] = (
    candidates["score"] * 0.55
    +
    candidates["history_score"] * 0.45
)

# 最终排序

candidates = candidates.sort_values(
    "final_score",
    ascending=False
)

final_candidates = candidates.head(
    FINAL_HOLDINGS
)

# ================================
# 输出
# ================================

print("\n")
print("=" * 60)
print("A股实战短线选股结果")
print("=" * 60)

for idx, row in final_candidates.iterrows():

    print(f"""
股票: {row['name']} ({row['code']})

现价: {row['price']:.2f}

涨跌幅: {row['pct']:.2f}%

换手率: {row['turnover']:.2f}%

量比: {row['lb']:.2f}

行业: {row.get('行业', '未知')}

实时评分: {row['score']:.2f}

历史评分: {row['history_score']:.2f}

最终评分: {row['final_score']:.2f}

----------------------------------------
""")

# ================================
# 保存CSV
# ================================

save_cols = [
    "code",
    "name",
    "price",
    "pct",
    "turnover",
    "lb",
    "score",
    "history_score",
    "final_score"
]

final_candidates[
    save_cols
].to_csv(
    "selected_stocks.csv",
    index=False,
    encoding="utf-8-sig"
)

logging.info(
    "结果已保存 selected_stocks.csv"
)

print("\n完成。")