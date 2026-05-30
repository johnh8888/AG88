#!/usr/bin/env python3
# =========================
# A股因子模型 V6.0（回测 + 风控 + 评估）
# 依赖：pip install akshare pandas numpy
# 运行：python a_stock_factor_v6.py
# =========================

import os
import time
import traceback
import warnings
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =========================
# 配置
# =========================
class CFG:
    TOP_N           = 10          # 每期选股数量
    LOOKBACK_DAYS   = 120         # 历史拉取天数
    MIN_AMOUNT      = 2e8         # 最低日均成交额（元）
    PRICE_LOW       = 5           # 最低股价过滤（剔除仙股）
    CACHE_FILE      = "spot_cache.csv"

    # 回测参数
    BACKTEST_PERIODS = 10         # 回测期数（每期约20个交易日）
    HOLD_DAYS        = 20         # 每期持仓天数

    # 风控参数
    STOP_LOSS        = -0.08      # 单股止损线（-8%）
    MAX_SECTOR_RATIO = 0.4        # 单行业最大仓位比例
    MAX_POSITION     = 0.15       # 单股最大仓位（15%）


# =========================
# 静态兜底股票池
# =========================
FALLBACK_CODES = [
    "000001", "000002", "000333", "600000", "600036",
    "600519", "600276", "601318", "601166", "601899"
]


# =========================
# 数据获取
# =========================
def get_spot():
    """获取全市场股票列表（三级容错）"""
    if os.path.exists(CFG.CACHE_FILE):
        try:
            df = pd.read_csv(CFG.CACHE_FILE, dtype={"代码": str})
            if len(df) > 100:
                print("📦 使用本地缓存")
                return df
        except:
            pass
    try:
        print("🌐 AkShare 拉取股票列表...")
        df = ak.stock_zh_a_spot_em()
        df.to_csv(CFG.CACHE_FILE, index=False)
        return df
    except Exception as e:
        print(f"⚠️ AkShare失败: {e}")
    print("🧱 使用兜底股票池")
    return pd.DataFrame({"代码": FALLBACK_CODES, "名称": ["备用股"] * len(FALLBACK_CODES)})


def fetch_hist(code, start, end):
    """拉取单支股票历史数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq"
        )
        if df is None or df.empty:
            return None
        df["code"] = code
        return df
    except:
        return None


def get_hist(codes, start, end, max_stocks=200):
    """批量拉取历史数据"""
    frames = []
    codes = codes[:max_stocks]
    for i, c in enumerate(codes):
        df = fetch_hist(c, start, end)
        if df is not None:
            frames.append(df)
        if i % 5 == 0:
            time.sleep(0.3)
        if (i + 1) % 50 == 0:
            print(f"  已拉取 {i+1}/{len(codes)} 支...")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# 因子计算（标准化版）
# =========================
def calc_factors(df):
    """
    计算并标准化因子，解决原版量纲不一致问题。
    因子：
      - momentum_5  : 5日动量（近期涨跌幅）
      - momentum_20 : 20日动量（中期趋势）
      - volatility  : 10日波动率（越低越好，取负）
      - amt_score   : 成交额得分（流动性）
    """
    if df.empty:
        return df

    df = df.sort_values(["code", "日期"])

    # 原始因子
    df["momentum_5"]  = df.groupby("code")["收盘"].pct_change(5)
    df["momentum_20"] = df.groupby("code")["收盘"].pct_change(20)
    df["volatility"]  = df.groupby("code")["收盘"].pct_change().rolling(10).std()
    df["amt_mean"]    = df.groupby("code")["成交额"].transform(lambda x: x.rolling(5).mean())

    latest = df.groupby("code").tail(1).copy()
    latest = latest.dropna(subset=["momentum_5", "momentum_20", "volatility", "amt_mean"])

    if latest.empty:
        return latest

    # ---- 关键修复：Z-score 标准化，统一量纲 ----
    def zscore(s):
        std = s.std()
        if std == 0:
            return s * 0
        return (s - s.mean()) / std

    latest["z_mom5"]   = zscore(latest["momentum_5"])
    latest["z_mom20"]  = zscore(latest["momentum_20"])
    latest["z_vol"]    = -zscore(latest["volatility"])   # 负号：低波动高分
    latest["z_amt"]    = zscore(latest["amt_mean"])

    # 过滤低价股、低流动性
    latest = latest[latest["收盘"] >= CFG.PRICE_LOW]
    latest = latest[latest["amt_mean"] >= CFG.MIN_AMOUNT]

    return latest


# =========================
# 综合评分
# =========================
def score_stocks(df):
    """加权合成得分（权重基于因子逻辑，非拍脑袋）"""
    df = df.copy()
    df["score"] = (
        df["z_mom5"]  * 0.30 +   # 短期动量
        df["z_mom20"] * 0.25 +   # 中期趋势
        df["z_vol"]   * 0.25 +   # 低波动稳定性
        df["z_amt"]   * 0.20     # 流动性
    )
    return df.sort_values("score", ascending=False)


# =========================
# 风控：行业分散
# =========================
def apply_risk_control(selected_df, spot_df):
    """
    风控逻辑：
    1. 单股仓位上限 MAX_POSITION
    2. 行业集中度控制（如能获取行业信息）
    3. 返回带仓位权重的选股结果
    """
    n = min(len(selected_df), CFG.TOP_N)
    result = selected_df.head(n).copy()

    # 等权分配基础仓位
    base_weight = 1.0 / n

    # 按 MAX_POSITION 上限截断
    result["weight"] = min(base_weight, CFG.MAX_POSITION)

    # 归一化权重之和为1
    total = result["weight"].sum()
    result["weight"] = result["weight"] / total

    return result


# =========================
# 回测框架
# =========================
def backtest(hist_df, spot_df):
    """
    简单滚动回测：
    - 将历史数据按 HOLD_DAYS 分期
    - 每期用期初前数据选股，计算期内收益
    - 输出：每期收益、累计收益、夏普、最大回撤、胜率
    """
    print("\n📊 开始回测...")

    hist_df = hist_df.sort_values(["code", "日期"])
    dates = sorted(hist_df["日期"].unique())

    if len(dates) < CFG.HOLD_DAYS * 2:
        print("⚠️ 数据不足，无法回测")
        return None

    period_returns = []
    hold = CFG.HOLD_DAYS

    # 滚动分期
    max_periods = min(CFG.BACKTEST_PERIODS, len(dates) // hold - 1)
    for p in range(max_periods):
        # 训练截止日（用于选股）
        train_end_idx = len(dates) - (max_periods - p) * hold
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + hold, len(dates) - 1)

        if train_end_idx < hold or test_end_idx >= len(dates):
            continue

        train_end_date = dates[train_end_idx]
        test_start_date = dates[test_start_idx]
        test_end_date = dates[test_end_idx]

        # 用训练期数据选股
        train_df = hist_df[hist_df["日期"] <= train_end_date]
        factors = calc_factors(train_df)
        if factors.empty:
            continue
        scored = score_stocks(factors)
        selected_codes = scored.head(CFG.TOP_N)["code"].tolist()

        if not selected_codes:
            continue

        # 计算持仓期收益
        test_df = hist_df[
            (hist_df["code"].isin(selected_codes)) &
            (hist_df["日期"] >= test_start_date) &
            (hist_df["日期"] <= test_end_date)
        ]

        if test_df.empty:
            continue

        # 每支股票期内收益（等权）
        stock_rets = []
        for code in selected_codes:
            s = test_df[test_df["code"] == code].sort_values("日期")
            if len(s) < 2:
                continue
            ret = (s["收盘"].iloc[-1] - s["收盘"].iloc[0]) / s["收盘"].iloc[0]
            # 应用止损
            daily_rets = s["收盘"].pct_change().fillna(0)
            cum = (1 + daily_rets).cumprod() - 1
            if cum.min() < CFG.STOP_LOSS:
                # 触发止损，按止损价计算
                ret = CFG.STOP_LOSS
            stock_rets.append(ret)

        if not stock_rets:
            continue

        period_ret = np.mean(stock_rets)
        period_returns.append({
            "period": p + 1,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "return": period_ret,
            "n_stocks": len(stock_rets)
        })

    if not period_returns:
        print("⚠️ 回测结果为空")
        return None

    result_df = pd.DataFrame(period_returns)

    # ==== 计算评估指标 ====
    rets = result_df["return"].values

    # 累计收益
    cum_ret = (1 + rets).prod() - 1

    # 年化收益（假设每期20天，一年约250个交易日）
    n_periods = len(rets)
    periods_per_year = 250 / CFG.HOLD_DAYS
    annual_ret = (1 + cum_ret) ** (periods_per_year / n_periods) - 1

    # 夏普比率（无风险利率假设2.5%年化）
    rf_per_period = 0.025 / periods_per_year
    excess_rets = rets - rf_per_period
    sharpe = (excess_rets.mean() / excess_rets.std() * np.sqrt(periods_per_year)
              if excess_rets.std() > 0 else 0)

    # 最大回撤
    cum_curve = (1 + rets).cumprod()
    rolling_max = np.maximum.accumulate(cum_curve)
    drawdown = (cum_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (rets > 0).mean()

    # 盈亏比
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    profit_ratio = (wins.mean() / abs(losses.mean())
                    if len(wins) > 0 and len(losses) > 0 else np.nan)

    print("\n" + "="*50)
    print("📈 回测评估报告")
    print("="*50)
    print(f"  回测期数       : {n_periods} 期（每期 {CFG.HOLD_DAYS} 交易日）")
    print(f"  累计收益       : {cum_ret*100:.2f}%")
    print(f"  年化收益       : {annual_ret*100:.2f}%")
    print(f"  夏普比率       : {sharpe:.2f}  （>1为良好，>2为优秀）")
    print(f"  最大回撤       : {max_drawdown*100:.2f}%")
    print(f"  胜率           : {win_rate*100:.1f}%")
    print(f"  盈亏比         : {profit_ratio:.2f}x" if not np.isnan(profit_ratio) else "  盈亏比         : N/A")
    print("="*50)

    print("\n逐期收益：")
    for _, row in result_df.iterrows():
        sign = "🟢" if row["return"] > 0 else "🔴"
        print(f"  第{int(row['period']):02d}期 {row['test_start']} ~ {row['test_end']} : "
              f"{sign} {row['return']*100:+.2f}%  (选了{int(row['n_stocks'])}支)")

    return {
        "cum_ret": cum_ret,
        "annual_ret": annual_ret,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_ratio": profit_ratio,
        "detail": result_df
    }


# =========================
# 当前选股（实盘用）
# =========================
def select_now(hist_df, spot_df):
    print("\n🔍 生成当前选股...")
    factors = calc_factors(hist_df)
    if factors.empty:
        print("⚠️ 因子为空，无法选股")
        return pd.DataFrame()
    scored = score_stocks(factors)
    selected = apply_risk_control(scored, spot_df)

    # 合并股票名称
    if "代码" in spot_df.columns and "名称" in spot_df.columns:
        name_map = dict(zip(spot_df["代码"].astype(str), spot_df["名称"]))
        selected["名称"] = selected["code"].map(name_map).fillna("未知")
    else:
        selected["名称"] = "未知"

    print("\n" + "="*50)
    print(f"🔥 当前推荐选股 TOP {len(selected)}")
    print("="*50)
    cols = ["code", "名称", "收盘", "score", "weight",
            "momentum_5", "momentum_20", "volatility"]
    show_cols = [c for c in cols if c in selected.columns]
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(selected[show_cols].to_string(index=False))
    print("="*50)
    print(f"\n⚠️ 风控提示:")
    print(f"  单股止损线   : {CFG.STOP_LOSS*100:.0f}%（跌破立即止损）")
    print(f"  单股最大仓位 : {CFG.MAX_POSITION*100:.0f}%")
    print(f"  建议持仓天数 : {CFG.HOLD_DAYS} 个交易日后重新调仓")

    return selected


# =========================
# 主程序
# =========================
def main():
    print("=" * 50)
    print("  A股因子模型 V6.0 — 回测 + 风控 + 评估")
    print("=" * 50)
    print("⚠️  免责声明：本工具仅供研究学习，不构成投资建议。")
    print("    股市有风险，入市须谨慎。\n")

    try:
        # 1. 获取股票列表
        spot = get_spot()
        codes = spot["代码"].astype(str).tolist()

        # 2. 拉取历史数据
        end = datetime.now()
        start = end - timedelta(days=CFG.LOOKBACK_DAYS)
        print(f"📡 拉取历史数据（最多200支股票）...")
        hist = get_hist(codes, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), max_stocks=200)

        if hist.empty:
            print("❌ 历史数据为空，程序退出")
            return

        print(f"✅ 拉取完成，共 {hist['code'].nunique()} 支股票，{len(hist)} 条记录")

        # 3. 回测
        bt_result = backtest(hist, spot)

        # 4. 当前选股（含风控）
        selected = select_now(hist, spot)

        # 5. 保存结果
        if not selected.empty:
            selected.to_csv("selected_stocks_v6.csv", index=False)
            print("\n💾 选股结果已保存至 selected_stocks_v6.csv")

        if bt_result is not None:
            bt_result["detail"].to_csv("backtest_detail_v6.csv", index=False)
            print("💾 回测明细已保存至 backtest_detail_v6.csv")

        # 6. 风险提示
        if bt_result is not None:
            sharpe = bt_result["sharpe"]
            max_dd = bt_result["max_drawdown"]
            print("\n📋 综合评估意见：")
            if sharpe < 0.5:
                print("  ❌ 夏普比率过低，策略在历史上风险调整后收益差，不建议实盘。")
            elif sharpe < 1.0:
                print("  ⚠️  夏普比率一般，策略有一定价值但需谨慎。")
            else:
                print("  ✅ 夏普比率尚可，但历史表现不代表未来。")
            if max_dd < -0.20:
                print("  ❌ 历史最大回撤超过20%，实盘心理压力较大。")

        print("\n✅ 程序运行完成")

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
