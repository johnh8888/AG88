#!/usr/bin/env python3
# =========================
# A股短线选股系统 V3.0
# 数据源：Baostock
# 策略：多因子+大盘择时+分仓+动态止盈止损
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
    TOTAL_CAPITAL   = 50000      # 总资金
    MAX_STOCKS      = 500        # 分析股票数
    PRICE_LOW       = 8          # 最低股价（提高门槛）
    PRICE_HIGH      = 80         # 最高股价
    MIN_AMOUNT      = 2e8        # 日均成交额2亿（更高流动性要求）
    COMMISSION      = 0.00025    # 手续费 0.025%（买卖各万2.5）
    SELL_TAX        = 0.001      # 印花税 0.1%（卖方）
    STOP_LOSS       = -0.03      # 止损收紧到-3%
    TARGET_PROFIT   = 0.04       # 止盈放宽到+4%
    TRAILING_STOP   = 0.02       # 移动止盈：回撤2%就卖
    MAX_HOLD_DAYS   = 10         # 最长持有10天
    POSITION_PCT    = 0.20       # 单票20%仓位
    MAX_POSITIONS   = 3          # 最多同时持有3只
    MARKET_FILTER   = True       # 开启大盘择时


# =========================
# Baostock 登录/登出
# =========================
def bs_login():
    for attempt in range(3):
        try:
            result = bs.login()
            if result.error_code == "0":
                print("✅ Baostock 登录成功")
                return True
        except:
            pass
        print(f"⚠️ 登录重试 {attempt+1}/3...")
        time.sleep(2)
    raise RuntimeError("登录失败")

def bs_logout():
    try:
        bs.logout()
    except:
        pass


# =========================
# 获取股票列表
# =========================
def get_stock_list():
    print("🌐 获取股票列表...")
    rs = bs.query_stock_basic(code_name="")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["type"] == "1"]
    df = df[df["status"] == "1"]
    df = df[~df["code"].str.startswith("sz.3")]
    df = df[~df["code"].str.startswith("bj.")]
    df = df[~df["code"].str.startswith("sh.688")]
    df = df[~df["code_name"].str.contains("ST", na=False)]
    print(f"✅ 过滤后 {len(df)} 支股票")
    return df.reset_index(drop=True)


# =========================
# 获取大盘指数数据
# =========================
def get_index_data(start_date, end_date):
    """获取上证指数数据用于择时"""
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000001",
            fields="date,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna()
        return df
    except:
        return pd.DataFrame()


# =========================
# 拉取历史数据
# =========================
def fetch_hist(code, start_date, end_date):
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
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


def get_all_hist(stock_list, start_date, end_date, max_stocks=500):
    codes = stock_list["code"].tolist()[:max_stocks]
    frames = []
    print(f"📡 拉取历史数据（{len(codes)} 支）...")
    for i, code in enumerate(codes):
        df = fetch_hist(code, start_date, end_date)
        if df is not None and len(df) >= 60:
            frames.append(df)
        if i % 10 == 0:
            time.sleep(0.15)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(codes)}，已获取 {len(frames)} 支")
    print(f"✅ 有效数据 {len(frames)} 支")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# 大盘择时
# =========================
def market_timing(index_df, target_date):
    """
    判断当前是否适合交易
    条件：指数在20日均线上方且20日均线向上
    """
    if index_df.empty:
        return True

    idx = index_df[index_df["date"] <= target_date].copy()
    if len(idx) < 30:
        return True

    idx["ma20"] = idx["close"].rolling(20).mean()
    idx["ma20_slope"] = idx["ma20"].diff(5)

    latest = idx.iloc[-1]
    return latest["close"] > latest["ma20"] and latest["ma20_slope"] > 0


# =========================
# 因子计算
# =========================
def calc_factors_cross_section(df_full, target_date):
    """计算 target_date 当天的因子截面"""
    df = df_full[df_full["date"] <= target_date].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["code", "date"])

    # 动量（多重周期）
    df["mom_3"]   = df.groupby("code")["close"].pct_change(3)
    df["mom_5"]   = df.groupby("code")["close"].pct_change(5)
    df["mom_10"]  = df.groupby("code")["close"].pct_change(10)

    # 波动率
    df["ret"]     = df.groupby("code")["close"].pct_change()
    df["vol_10"]  = df.groupby("code")["ret"].transform(lambda x: x.rolling(10).std())
    df["vol_20"]  = df.groupby("code")["ret"].transform(lambda x: x.rolling(20).std())

    # 成交额
    df["amt_5"]   = df.groupby("code")["amount"].transform(lambda x: x.rolling(5).mean())
    df["amt_20"]  = df.groupby("code")["amount"].transform(lambda x: x.rolling(20).mean())

    # 量比
    df["vol_ma5"]   = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)

    # 均线系统
    df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["ma60"] = df.groupby("code")["close"].transform(lambda x: x.rolling(60).mean())
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)
    df["above_ma60"] = (df["close"] > df["ma60"]).astype(float)

    # 今日涨跌幅
    df["today_pct"] = df["ret"]

    # 换手率
    df["turn_ma5"] = df.groupby("code")["turn"].transform(lambda x: x.rolling(5).mean())

    # 取目标日期
    latest = df[df["date"] == target_date].copy()
    latest = latest.dropna(subset=["mom_3", "mom_5", "mom_10", "vol_10", "amt_5", "vol_ratio"])

    # 过滤
    latest = latest[latest["close"] >= CFG.PRICE_LOW]
    latest = latest[latest["close"] <= CFG.PRICE_HIGH]
    latest = latest[latest["amt_5"] >= CFG.MIN_AMOUNT]
    latest = latest[latest["today_pct"] > -0.07]
    latest = latest[latest["today_pct"] < 0.08]
    latest = latest[latest["above_ma20"] == 1]
    latest = latest[latest["vol_ratio"] > 0.8]
    latest = latest[latest["turn_ma5"] > 0.5]

    if latest.empty:
        return latest

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    latest = latest.copy()

    latest["z_mom"] = zscore(latest["mom_3"]) * 0.5 + zscore(latest["mom_5"]) * 0.3 + zscore(latest["mom_10"]) * 0.2
    latest["z_vol"] = -zscore(latest["vol_10"])
    latest["z_amt"] = zscore(latest["amt_5"] / latest["amt_20"])

    latest["vol_ratio_score"] = latest["vol_ratio"].apply(
        lambda x: 1.0 if 1.2 <= x <= 2.5 else (0.5 if 0.8 <= x <= 3.0 else 0)
    )

    latest["score"] = (
        latest["z_mom"]  * 0.35 +
        latest["z_vol"]  * 0.25 +
        latest["z_amt"]  * 0.20 +
        latest["vol_ratio_score"] * 0.20
    )

    return latest.sort_values("score", ascending=False)


# =========================
# 回测引擎（改进版）
# =========================
class BacktestEngine:
    def __init__(self, df_full, index_df, start_date, end_date):
        self.df_full = df_full
        self.index_df = index_df
        self.start_date = start_date
        self.end_date = end_date
        self.trades = []
        self.equity_curve = None

    def get_trading_dates(self):
        dates = self.df_full["date"].unique()
        dates = sorted([d for d in dates if self.start_date <= d <= self.end_date])
        return dates

    def run(self, capital=50000):
        dates = self.get_trading_dates()
        print(f"\n🔬 回测期间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")

        cash = capital
        holdings = []
        equity_curve = []

        for i, today in enumerate(dates):
            market_ok = True
            if CFG.MARKET_FILTER:
                market_ok = market_timing(self.index_df, today)

            # 处理持仓
            new_holdings = []
            for h in holdings:
                stock_data = self.df_full[
                    (self.df_full["code"] == h["code"]) &
                    (self.df_full["date"] >= h["buy_date"]) &
                    (self.df_full["date"] <= today)
                ].sort_values("date")

                if len(stock_data) == 0:
                    new_holdings.append(h)
                    continue

                last = stock_data.iloc[-1]
                hold_days = len(stock_data) - 1

                if last["high"] > h["highest_close"]:
                    h["highest_close"] = last["high"]

                sell_now = False
                sell_price = last["close"]
                sell_reason = ""

                if last["low"] <= h["buy_price"] * (1 + CFG.STOP_LOSS):
                    sell_now = True
                    sell_price = h["buy_price"] * (1 + CFG.STOP_LOSS)
                    sell_reason = "止损"
                elif last["high"] >= h["buy_price"] * (1 + CFG.TARGET_PROFIT):
                    sell_now = True
                    sell_price = h["buy_price"] * (1 + CFG.TARGET_PROFIT)
                    sell_reason = "止盈"
                elif h["highest_close"] > h["buy_price"] * 1.02:
                    if last["close"] <= h["highest_close"] * (1 - CFG.TRAILING_STOP):
                        sell_now = True
                        sell_price = last["close"]
                        sell_reason = "移动止盈"
                elif hold_days >= CFG.MAX_HOLD_DAYS:
                    sell_now = True
                    sell_price = last["close"]
                    sell_reason = "到期"

                if sell_now:
                    actual_sell = sell_price * 0.999
                    revenue = h["shares"] * actual_sell * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                    profit = revenue - h["cost"]
                    self.trades.append({
                        "code": h["code"],
                        "buy_date": h["buy_date"],
                        "sell_date": today,
                        "hold_days": hold_days,
                        "buy_price": h["buy_price"],
                        "sell_price": actual_sell,
                        "shares": h["shares"],
                        "cost": h["cost"],
                        "revenue": revenue,
                        "profit": profit,
                        "profit_pct": profit / h["cost"],
                        "exit_reason": sell_reason
                    })
                    cash += revenue
                else:
                    new_holdings.append(h)

            holdings = new_holdings

            # 选股买入
            if market_ok and len(holdings) < CFG.MAX_POSITIONS:
                candidates = calc_factors_cross_section(self.df_full, today)
                if not candidates.empty:
                    held_codes = {h["code"] for h in holdings}
                    candidates = candidates[~candidates["code"].isin(held_codes)]

                    for _, row in candidates.iterrows():
                        if len(holdings) >= CFG.MAX_POSITIONS:
                            break

                        budget_per_stock = capital * CFG.POSITION_PCT
                        actual_budget = min(budget_per_stock, cash)
                        if actual_budget < row["close"] * 100:
                            continue

                        shares = int(actual_budget / (row["close"] * 1.001) / 100) * 100
                        shares = max(shares, 100)
                        buy_price = row["close"] * 1.001
                        cost = shares * buy_price * (1 + CFG.COMMISSION)

                        if cost <= cash and cost <= budget_per_stock * 1.05:
                            cash -= cost
                            holdings.append({
                                "code": row["code"],
                                "shares": shares,
                                "buy_date": today,
                                "buy_price": buy_price,
                                "cost": cost,
                                "highest_close": row["close"]
                            })

            # 计算当日净值
            equity = cash
            for h in holdings:
                stock_data = self.df_full[
                    (self.df_full["code"] == h["code"]) &
                    (self.df_full["date"] <= today)
                ]
                if not stock_data.empty:
                    equity += h["shares"] * stock_data.iloc[-1]["close"]

            equity_curve.append({"date": today, "equity": equity, "holdings": len(holdings)})

            if (i + 1) % 50 == 0:
                print(f"  进度 {i+1}/{len(dates)}，净值: {equity:.0f}，持仓: {len(holdings)}")

        # 清仓
        for h in holdings:
            last_data = self.df_full[self.df_full["code"] == h["code"]]
            if not last_data.empty:
                last_close = last_data.iloc[-1]["close"]
                sell_price = last_close * 0.999
                revenue = h["shares"] * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                profit = revenue - h["cost"]
                self.trades.append({
                    "code": h["code"],
                    "buy_date": h["buy_date"],
                    "sell_date": equity_curve[-1]["date"],
                    "hold_days": 0,
                    "buy_price": h["buy_price"],
                    "sell_price": sell_price,
                    "shares": h["shares"],
                    "cost": h["cost"],
                    "revenue": revenue,
                    "profit": profit,
                    "profit_pct": profit / h["cost"],
                    "exit_reason": "回测结束清仓"
                })
                cash += revenue

        self.equity_curve = pd.DataFrame(equity_curve)
        return self.generate_report(capital)

    def generate_report(self, initial_capital):
        trades_df = pd.DataFrame(self.trades)
        equity_df = self.equity_curve

        if trades_df.empty:
            return {"error": "无交易记录"}

        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df["profit"] > 0])
        loss_trades = len(trades_df[trades_df["profit"] <= 0])
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        total_profit = trades_df["profit"].sum()
        avg_profit = trades_df["profit"].mean()
        avg_hold = trades_df["hold_days"].mean()

        avg_win = trades_df[trades_df["profit"] > 0]["profit"].mean() if win_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df["profit"] <= 0]["profit"].mean()) if loss_trades > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        final_equity = equity_df.iloc[-1]["equity"]
        total_return = (final_equity - initial_capital) / initial_capital * 100

        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["cummax"]) / equity_df["cummax"]
        max_drawdown = equity_df["drawdown"].min() * 100

        equity_df["daily_return"] = equity_df["equity"].pct_change()
        sharpe = (equity_df["daily_return"].mean() / equity_df["daily_return"].std() * np.sqrt(252)
                  if equity_df["daily_return"].std() > 0 else 0)

        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()

        return {
            "初始资金": initial_capital,
            "最终资金": round(final_equity, 2),
            "总收益率": f"{total_return:.2f}%",
            "最大回撤": f"{max_drawdown:.2f}%",
            "夏普比率": f"{sharpe:.2f}",
            "总交易次数": total_trades,
            "盈利次数": win_trades,
            "亏损次数": loss_trades,
            "胜率": f"{win_rate:.2f}%",
            "总盈亏": round(total_profit, 2),
            "平均盈亏": round(avg_profit, 2),
            "盈亏比": f"{profit_factor:.2f}",
            "平均持有天数": f"{avg_hold:.1f}",
            "退出原因分布": exit_reasons
        }


# =========================
# 主程序
# =========================
def main():
    print("=" * 60)
    print("  A股短线选股系统 V3.0（改进版）")
    print("=" * 60)

    try:
        bs_login()

        stock_list = get_stock_list()

        backtest_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        backtest_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        data_start = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")

        print(f"\n📊 数据区间: {data_start} ~ {backtest_end}")
        print(f"📊 回测区间: {backtest_start} ~ {backtest_end}")

        hist_full = get_all_hist(stock_list, data_start, backtest_end, CFG.MAX_STOCKS)
        index_df = get_index_data(data_start, backtest_end)

        if hist_full.empty:
            print("❌ 无数据")
            bs_logout()
            return

        print(f"✅ 股票数据: {hist_full['code'].nunique()} 支")
        print(f"✅ 指数数据: {len(index_df)} 天")

        engine = BacktestEngine(hist_full, index_df, backtest_start, backtest_end)
        report = engine.run(capital=CFG.TOTAL_CAPITAL)

        print("\n" + "="*60)
        print("  📊 回测报告")
        print("="*60)
        for k, v in report.items():
            if k != "退出原因分布":
                print(f"  {k:<14}: {v}")
            else:
                print(f"  {k:<14}:")
                for reason, count in v.items():
                    print(f"    {reason}: {count}次")
        print("="*60)

        pd.DataFrame(engine.trades).to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
        engine.equity_curve.to_csv("backtest_equity.csv", index=False, encoding="utf-8-sig")
        print("\n💾 回测记录已保存")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()

    finally:
        bs_logout()
        print("✅ 完成")


if __name__ == "__main__":
    main()