#!/usr/bin/env python3
# =========================
# A股每日单股精选 V1.0
# 数据源：Baostock（境外IP可用，免费）
# 过滤：创业板(300xxx)、北交所(8/4开头)、ST/*ST
# 策略：动量 + 低波动 + 资金流入，给出持有天数建议
# 依赖：pip install baostock pandas numpy
# =========================

import time
import traceback
import warnings
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =========================
# 配置
# =========================
class CFG:
    TOTAL_CAPITAL  = 20000      # 总资金（元），建议2万起
    LOOKBACK_DAYS  = 90         # 历史拉取天数
    MAX_STOCKS     = 300        # 最多分析股票数量
    PRICE_LOW      = 5          # 最低股价（剔除仙股）
    PRICE_HIGH     = 150        # 最高股价（1手不超过1.5万）
    MIN_AMOUNT     = 1e8        # 最低日均成交额1亿（保证流动性）
    COMMISSION     = 0.0003     # 手续费（买卖各0.03%，印花税卖方0.1%）
    SELL_TAX       = 0.001      # 印花税（卖方）
    STOP_LOSS      = -0.05      # 止损线 -5%
    TARGET_PROFIT  = 0.03       # 目标涨幅 +3%（1万本金赚300元）


# =========================
# Baostock 登录
# =========================
def bs_login():
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"Baostock登录失败: {result.error_msg}")
    print("✅ Baostock 登录成功")

def bs_logout():
    try:
        bs.logout()
    except:
        pass


# =========================
# 获取沪深主板股票列表
# 过滤：创业板(sz.3)、北交所(bj.)、ST
# =========================
def get_stock_list():
    print("🌐 获取股票列表（过滤创业板/北交所/ST）...")
    rs = bs.query_stock_basic(code_name="")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise RuntimeError("无法获取股票列表")

    df = pd.DataFrame(rows, columns=rs.fields)

    # 只保留上市状态的股票
    df = df[df["type"] == "1"]       # 股票类型
    df = df[df["status"] == "1"]     # 上市中

    # 过滤创业板（sz.300xxx / sz.301xxx）
    df = df[~df["code"].str.startswith("sz.3")]

    # 过滤北交所（bj. 开头）
    df = df[~df["code"].str.startswith("bj.")]

    # 过滤科创板（sh.688xxx）- 风险较高，可按需开放
    df = df[~df["code"].str.startswith("sh.688")]

    # 过滤ST / *ST（名称含ST）
    df = df[~df["code_name"].str.contains("ST", na=False)]

    print(f"✅ 过滤后剩余 {len(df)} 支主板股票")
    return df.reset_index(drop=True)


# =========================
# 拉取历史日K数据
# =========================
def fetch_hist(code, start_date, end_date):
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        return df
    except:
        return None


def get_hist(stock_list, start_date, end_date, max_stocks=300):
    codes = stock_list["code"].tolist()[:max_stocks]
    frames = []
    print(f"📡 拉取历史数据（{len(codes)} 支）...")
    for i, code in enumerate(codes):
        df = fetch_hist(code, start_date, end_date)
        if df is not None and len(df) >= 20:
            frames.append(df)
        if i % 10 == 0:
            time.sleep(0.15)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(codes)}，已获取 {len(frames)} 支...")
    print(f"✅ 完成，有效数据 {len(frames)} 支")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# 因子计算（针对短线优化）
# =========================
def calc_factors(df, name_map):
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["code", "date"])

    # 短线动量（3日、5日）
    df["mom_3"]  = df.groupby("code")["close"].pct_change(3)
    df["mom_5"]  = df.groupby("code")["close"].pct_change(5)

    # 波动率（10日）
    df["vol_10"] = df.groupby("code")["close"].pct_change().rolling(10).std()

    # 成交额（5日均值）
    df["amt_5"]  = df.groupby("code")["amount"].transform(lambda x: x.rolling(5).mean())

    # 量比：今日成交量 / 5日均量（>1说明放量）
    df["vol_ma5"]   = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)

    # 今日涨跌幅
    df["today_pct"] = df.groupby("code")["close"].pct_change(1)

    # 价格趋势：收盘价 / 20日均线
    df["ma20"]      = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["above_ma20"]= (df["close"] > df["ma20"]).astype(float)

    # 取最新一行
    latest = df.groupby("code").tail(1).copy()
    latest["名称"] = latest["code"].map(name_map).fillna("未知")
    latest["code_simple"] = latest["code"].str.replace("sh.", "").str.replace("sz.", "")

    latest = latest.dropna(subset=["mom_3", "mom_5", "vol_10", "amt_5", "vol_ratio"])

    # 过滤价格范围
    latest = latest[latest["close"] >= CFG.PRICE_LOW]
    latest = latest[latest["close"] <= CFG.PRICE_HIGH]

    # 过滤低流动性
    latest = latest[latest["amt_5"] >= CFG.MIN_AMOUNT]

    # 过滤今日跌停附近（跌幅超过9%不选，可能有异常）
    latest = latest[latest["today_pct"] > -0.09]

    # 过滤在均线下方太多的（跌势中的股票）
    latest = latest[latest["above_ma20"] == 1]

    return latest.reset_index(drop=True)


# =========================
# 综合评分（短线）
# =========================
def score_short(df):
    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    df = df.copy()
    df["z_mom3"]      = zscore(df["mom_3"])
    df["z_mom5"]      = zscore(df["mom_5"])
    df["z_vol"]       = -zscore(df["vol_10"])     # 低波动高分
    df["z_amt"]       = zscore(df["amt_5"])
    df["z_vol_ratio"] = zscore(df["vol_ratio"])   # 放量高分

    # 权重：短期动量最重要，放量次之，低波动保稳定
    df["score"] = (
        df["z_mom3"]      * 0.30 +
        df["z_mom5"]      * 0.20 +
        df["z_vol"]       * 0.20 +
        df["z_amt"]       * 0.15 +
        df["z_vol_ratio"] * 0.15
    )
    return df.sort_values("score", ascending=False)


# =========================
# 持有天数建议（基于动量强度）
# =========================
def suggest_hold_days(row):
    mom3 = row.get("mom_3", 0)
    vol_r = row.get("vol_ratio", 1)
    vol10 = row.get("vol_10", 0.02)

    # 强势突破：短线1-2天
    if mom3 > 0.04 and vol_r > 1.5:
        return 1, "强势突破，短线1天目标止盈"

    # 温和上涨 + 放量：持有2-3天
    if mom3 > 0.01 and vol_r > 1.2:
        return 2, "温和放量，建议持有2天"

    # 低波动稳健型：持有3-5天
    if vol10 < 0.015 and mom3 > 0:
        return 4, "低波动稳健，建议持有3-5天"

    # 默认2天
    return 2, "综合评估，建议持有2天"


# =========================
# 计算买入/卖出金额
# =========================
def calc_trade(row, capital):
    price     = row["close"]
    # 按资金80%买入（留20%备用）
    budget    = capital * 0.80
    shares    = int(budget / price / 100) * 100  # 整手
    shares    = max(shares, 100)                  # 至少1手

    # 实际买入
    buy_amt   = shares * price
    buy_fee   = buy_amt * CFG.COMMISSION
    buy_total = buy_amt + buy_fee

    # 止损卖出
    sl_price  = round(price * (1 + CFG.STOP_LOSS), 2)
    sl_amt    = shares * sl_price
    sl_fee    = sl_amt * (CFG.COMMISSION + CFG.SELL_TAX)
    sl_net    = sl_amt - sl_fee
    sl_loss   = sl_net - buy_total

    # 目标卖出
    tgt_price = round(price * (1 + CFG.TARGET_PROFIT), 2)
    tgt_amt   = shares * tgt_price
    tgt_fee   = tgt_amt * (CFG.COMMISSION + CFG.SELL_TAX)
    tgt_net   = tgt_amt - tgt_fee
    tgt_profit= tgt_net - buy_total

    return {
        "shares": shares,
        "buy_price": price,
        "buy_amt": round(buy_amt, 2),
        "buy_fee": round(buy_fee, 2),
        "buy_total": round(buy_total, 2),
        "sl_price": sl_price,
        "sl_net": round(sl_net, 2),
        "sl_loss": round(sl_loss, 2),
        "tgt_price": tgt_price,
        "tgt_net": round(tgt_net, 2),
        "tgt_profit": round(tgt_profit, 2),
    }


# =========================
# 主输出：每日单股推荐
# =========================
def print_daily_pick(best_row, trade, hold_days, hold_reason, top5_df):
    today = datetime.now().strftime("%Y-%m-%d")
    print("\n" + "="*60)
    print(f"  📅 {today}  每日精选股票")
    print("="*60)
    print(f"\n  🏆 今日推荐：【{best_row['code_simple']} {best_row['名称']}】")
    print(f"\n  📊 股票指标")
    print(f"    当前价格   : {best_row['close']:.2f} 元")
    print(f"    今日涨跌   : {best_row['today_pct']*100:+.2f}%")
    print(f"    3日动量    : {best_row['mom_3']*100:+.2f}%")
    print(f"    量比       : {best_row['vol_ratio']:.2f}x  {'🔥放量' if best_row['vol_ratio']>1.3 else '正常'}")
    print(f"    综合评分   : {best_row['score']:.3f}")

    print(f"\n  💰 交易建议（总资金 {CFG.TOTAL_CAPITAL:,.0f} 元）")
    print(f"    ✅ 买入    : {trade['shares']} 股 @ {trade['buy_price']:.2f} 元")
    print(f"               实付金额: {trade['buy_total']:,.0f} 元（含手续费 {trade['buy_fee']:.1f} 元）")
    print(f"    🕐 持有    : {hold_days} 天  ─  {hold_reason}")
    print(f"    🟢 目标卖出: {trade['tgt_price']:.2f} 元（+{CFG.TARGET_PROFIT*100:.0f}%）")
    print(f"               到手金额: {trade['tgt_net']:,.0f} 元  预期盈利: +{trade['tgt_profit']:,.0f} 元")
    print(f"    🔴 止损卖出: {trade['sl_price']:.2f} 元（{CFG.STOP_LOSS*100:.0f}%）")
    print(f"               到手金额: {trade['sl_net']:,.0f} 元  最大亏损: {trade['sl_loss']:,.0f} 元")

    profit_pct = trade['tgt_profit'] / trade['buy_total'] * 100
    loss_pct   = trade['sl_loss'] / trade['buy_total'] * 100
    print(f"\n  📐 盈亏比   : {abs(trade['tgt_profit'])/abs(trade['sl_loss']):.1f}:1  "
          f"（盈{profit_pct:.1f}% / 亏{abs(loss_pct):.1f}%）")

    print(f"\n  ⚠️  操作提示:")
    print(f"    · 明日开盘后观察，若高开超过1%建议等回调再买")
    print(f"    · 若开盘即跌破 {trade['sl_price']:.2f} 元，放弃当日操作")
    print(f"    · 达到目标价 {trade['tgt_price']:.2f} 元果断卖出，不贪")
    print(f"    · A股T+1，今日买入最快明日卖出")

    print(f"\n  📋 备选股票 TOP5（可在今日推荐无法操作时使用）")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>7} {'3日涨幅':>8} {'量比':>6} {'评分':>7}")
    print(f"  {'-'*52}")
    for _, r in top5_df.iterrows():
        print(f"  {r['code_simple']:<8} {r['名称']:<10} "
              f"{r['close']:>7.2f} {r['mom_3']*100:>+7.2f}% "
              f"{r['vol_ratio']:>6.2f}x {r['score']:>7.3f}")

    print("\n" + "="*60)
    print("  ⚠️  免责：以上为量化模型输出，不构成投资建议。")
    print("           股市有风险，亏损自负，请量力而为。")
    print("="*60)


# =========================
# 主程序
# =========================
def main():
    print("=" * 60)
    print("  A股每日单股精选 V1.0")
    print("  过滤：创业板 / 北交所 / 科创板 / ST")
    print("=" * 60)

    try:
        bs_login()

        # 1. 获取股票列表
        stock_list = get_stock_list()
        name_map   = dict(zip(stock_list["code"], stock_list["code_name"]))

        # 2. 拉取历史数据
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=CFG.LOOKBACK_DAYS)
        hist       = get_hist(
            stock_list,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            max_stocks=CFG.MAX_STOCKS
        )

        if hist.empty:
            print("❌ 历史数据为空，退出")
            return

        # 3. 计算因子
        factors = calc_factors(hist, name_map)
        if factors.empty:
            print("❌ 因子计算结果为空（可能今日无满足条件股票）")
            return

        print(f"✅ 满足条件股票: {len(factors)} 支")

        # 4. 评分排序
        scored = score_short(factors)

        # 5. 价格过滤（确保1手买得起）
        budget = CFG.TOTAL_CAPITAL * 0.80
        scored = scored[scored["close"] * 100 <= budget]

        if scored.empty:
            print(f"❌ 资金不足，当前资金 {CFG.TOTAL_CAPITAL} 元无法买入任何符合条件股票")
            print(f"   建议提高 TOTAL_CAPITAL 或降低 PRICE_HIGH")
            return

        # 6. 取最优1只 + 备选5只
        best      = scored.iloc[0]
        top5      = scored.iloc[1:6]

        # 7. 持有建议
        hold_days, hold_reason = suggest_hold_days(best)

        # 8. 计算交易金额
        trade = calc_trade(best, CFG.TOTAL_CAPITAL)

        # 9. 输出结果
        print_daily_pick(best, trade, hold_days, hold_reason, top5)

        # 10. 保存CSV
        output = scored.head(10)[["code_simple", "名称", "close", "mom_3",
                                   "mom_5", "vol_ratio", "vol_10", "score"]].copy()
        output.columns = ["代码", "名称", "现价", "3日涨幅", "5日涨幅", "量比", "波动率", "评分"]
        output["3日涨幅"] = output["3日涨幅"].map(lambda x: f"{x*100:+.2f}%")
        output["5日涨幅"] = output["5日涨幅"].map(lambda x: f"{x*100:+.2f}%")
        output.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
        print("\n💾 结果已保存至 selected_stocks.csv")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        pd.DataFrame(columns=["代码", "评分"]).to_csv("selected_stocks.csv", index=False)

    finally:
        bs_logout()
        print("✅ 完成")


if __name__ == "__main__":
    main()