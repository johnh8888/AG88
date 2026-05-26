# =========================
# A股因子模型 V4.0（修正版）
# 多因子评分系统（带历史因子计算）
# =========================

import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from scipy.stats import mstats

# =========================
# CONFIG
# =========================
class CFG:
    TOP_N = 10
    MOMENTUM_PERIOD = 20          # 动量看过去20日
    VOL_PERIOD = 20               # 波动率看20日
    MF_PERIOD = 5                 # 资金流看5日
    MIN_AMOUNT = 2e8              # 提高门槛
    LOOKBACK_DAYS = 60            # 取历史行情的天数
    PRICE_LOW = 5                 # 最低价格
    EXCLUDE_ST = True


# =========================
# 数据获取：历史日行情
# =========================
def get_hist_data(all_codes, start_date, end_date):
    """
    批量获取股票历史日线（复权）
    返回 DataFrame，列包含：code, date, open, high, low, close, volume, amount, turnover
    """
    frames = []
    for i, code in enumerate(all_codes):
        try:
            # 获取前复权日线
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")
            if df.empty:
                continue
            df["code"] = code
            frames.append(df[["code", "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]])
        except Exception as e:
            continue

    if not frames:
        raise ValueError("没有获取到任何股票数据")
    hist = pd.concat(frames, ignore_index=True)
    hist.columns = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    hist["date"] = pd.to_datetime(hist["date"])
    return hist


# =========================
# 计算历史因子（在时间序列上）
# =========================
def calc_hist_factors(hist):
    """
    对每只股票的时间序列计算因子值，返回最后一天的因子截面
    """
    hist = hist.sort_values(["code", "date"]).copy()
    hist["ret"] = hist.groupby("code")["close"].pct_change()           # 日收益率
    hist["log_amount"] = np.log(hist["amount"].clip(lower=1))

    # 动量因子：过去MOMENTUM_PERIOD日的累计收益（去掉最近1天避免反转）
    hist["momentum"] = hist.groupby("code")["close"].transform(
        lambda x: x.pct_change(periods=CFG.MOMENTUM_PERIOD)
    )

    # 波动率因子：过去VOL_PERIOD日收益率的标准差（取负，偏好低波）
    hist["volatility"] = hist.groupby("code")["ret"].transform(
        lambda x: x.rolling(CFG.VOL_PERIOD, min_periods=10).std().shift(1)
    )
    hist["volatility"] = -hist["volatility"]  # 取负号，数值越大越好

    # 资金流因子：近5日日均成交额变化率（替代大单净流入，简化为量比）
    hist["avg_amount_5"] = hist.groupby("code")["amount"].transform(
        lambda x: x.rolling(CFG.MF_PERIOD, min_periods=3).mean().shift(1)
    )
    hist["money_flow"] = hist["amount"] / (hist["avg_amount_5"] + 1e-9) - 1

    # 技术结构因子：均线多头排列得分（MA5>MA10>MA20 得1，部分满足得0.5）
    hist["ma5"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
    hist["ma10"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(10).mean())
    hist["ma20"] = hist.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    hist["tech"] = 0
    cond = (hist["ma5"] > hist["ma10"]) & (hist["ma10"] > hist["ma20"])
    hist.loc[cond, "tech"] = 1.0
    hist.loc[(hist["ma5"] > hist["ma10"]) & ~cond, "tech"] = 0.5

    # 只取最新一天的数据
    latest_date = hist["date"].max()
    latest = hist[hist["date"] == latest_date].copy()
    return latest


# =========================
# 因子处理：去极值、中性化、标准化
# =========================
def process_factors(df, factor_cols):
    df = df.copy()
    # 1. 去极值：MAD法，5倍绝对中位差
    for col in factor_cols:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        upper = median + 5 * mad
        lower = median - 5 * mad
        df[col] = df[col].clip(lower, upper)

    # 2. 市值中性化（用成交额的对数代理市值，真实环境需用总市值）
    if "log_amount" in df.columns:
        for col in factor_cols:
            # 简单线性回归残差
            beta = np.polyfit(df["log_amount"], df[col], 1)[0]
            df[col] = df[col] - beta * df["log_amount"]

    # 3. 标准化
    for col in factor_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-9)

    return df


# =========================
# 最终得分与选股
# =========================
def get_final_score(df):
    factor_cols = ["momentum", "money_flow", "volatility", "tech"]
    df = process_factors(df, factor_cols)

    # 因子权重（可根据回测优化）
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
    # 剔除ST、退市等（需要名称字段，这里用代码前缀简单模拟）
    df = df[~df["code"].str.contains("000002|000003|300|688").fillna(False)]  # 示例，实际需名称
    return df


# =========================
# 主流程
# =========================
def main():
    # 1. 获取股票列表
    spot = ak.stock_zh_a_spot_em()
    all_codes = spot["代码"].tolist()[:500]  # 示例用500只，全市场运行时间较长
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=CFG.LOOKBACK_DAYS)).strftime("%Y%m%d")
    print(f"开始获取{len(all_codes)}只股票历史数据...")
    hist = get_hist_data(all_codes, start_date, end_date)

    # 2. 计算因子截面
    latest = calc_hist_factors(hist)

    # 3. 过滤与打分
    latest = filter_stocks(latest)
    latest = get_final_score(latest)

    # 4. 选TOP N
    result = latest.sort_values("final_score", ascending=False).head(CFG.TOP_N)

    print("\n🔥 V4.0 因子模型选股结果（基于历史因子）\n")
    for _, row in result.iterrows():
        print(
            f"{row['code']} | 收盘={row['close']:.2f} | "
            f"动量={row['momentum']:.3f} | 波动={row['volatility']:.3f} | "
            f"资金流={row['money_flow']:.3f} | 技术={row['tech']:.3f} | "
            f"总分={row['final_score']:.2f}"
        )

if __name__ == "__main__":
    main()
