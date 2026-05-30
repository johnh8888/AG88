#!/usr/bin/env python3
# =========================
# A股每日单股精选 V2.0（带回测功能）
# 数据源：Baostock（免费，境外IP可用）
# 策略：动量 + 低波动 + 资金流入
# 新增：完整回测引擎 + 参数优化建议
# 依赖：pip install baostock pandas numpy
# =========================

import time
import traceback
import warnings
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os

warnings.filterwarnings("ignore")

# =========================
# 配置
# =========================
class CFG:
    TOTAL_CAPITAL  = 20000      # 总资金
    LOOKBACK_DAYS  = 365        # 回测时拉取更多历史数据
    MAX_STOCKS     = 500        # 回测用更多股票
    PRICE_LOW      = 5
    PRICE_HIGH     = 150
    MIN_AMOUNT     = 1e8
    COMMISSION     = 0.0003
    SELL_TAX       = 0.001
    STOP_LOSS      = -0.05      # 止损
    TARGET_PROFIT  = 0.03       # 止盈
    MAX_HOLD_DAYS  = 5          # 最大持有天数
    SLIPPAGE       = 0.001      # 滑点（0.1%）
    POSITION_PCT   = 0.8        # 仓位比例


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
    raise RuntimeError("Baostock 登录失败")

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

    if not rows:
        raise RuntimeError("无法获取股票列表")

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["type"] == "1"]
    df = df[df["status"] == "1"]
    df = df[~df["code"].str.startswith("sz.3")]   # 创业板
    df = df[~df["code"].str.startswith("bj.")]     # 北交所
    df = df[~df["code"].str.startswith("sh.688")]  # 科创板
    df = df[~df["code_name"].str.contains("ST", na=False)]

    print(f"✅ 过滤后 {len(df)} 支股票")
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


def get_all_hist(stock_list, start_date, end_date, max_stocks=300):
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
# 因子计算（给定日期截面的因子值）
# =========================
def calc_factors_cross_section(df_full, target_date):
    """计算 target_date 当天所有股票的因子截面"""
    df = df_full[df_full["date"] <= target_date].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["code", "date"])

    # 动量
    df["mom_3"]  = df.groupby("code")["close"].pct_change(3)
    df["mom_5"]  = df.groupby("code")["close"].pct_change(5)

    # 波动率（10日）
    df["vol_10"] = df.groupby("code")["close"].pct_change().rolling(10).std().reset_index(level=0, drop=True)

    # 成交额5日均值
    df["amt_5"]  = df.groupby("code")["amount"].transform(lambda x: x.rolling(5).mean())

    # 量比
    df["vol_ma5"]   = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)

    # 今日涨跌幅
    df["today_pct"] = df.groupby("code")["close"].pct_change(1)

    # 20日均线
    df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(float)

    # 取目标日期数据
    latest = df[df["date"] == target_date].copy()
    latest = latest.dropna(subset=["mom_3", "mom_5", "vol_10", "amt_5", "vol_ratio", "today_pct"])

    # 过滤
    latest = latest[latest["close"] >= CFG.PRICE_LOW]
    latest = latest[latest["close"] <= CFG.PRICE_HIGH]
    latest = latest[latest["amt_5"] >= CFG.MIN_AMOUNT]
    latest = latest[latest["today_pct"] > -0.09]
    latest = latest[latest["above_ma20"] == 1]

    if latest.empty:
        return latest

    # 评分
    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    latest = latest.copy()
    latest["z_mom3"]      = zscore(latest["mom_3"])
    latest["z_mom5"]      = zscore(latest["mom_5"])
    latest["z_vol"]       = -zscore(latest["vol_10"])
    latest["z_amt"]       = zscore(latest["amt_5"])
    latest["z_vol_ratio"] = zscore(latest["vol_ratio"])

    latest["score"] = (
        latest["z_mom3"]      * 0.30 +
        latest["z_mom5"]      * 0.20 +
        latest["z_vol"]       * 0.20 +
        latest["z_amt"]       * 0.15 +
        latest["z_vol_ratio"] * 0.15
    )

    return latest.sort_values("score", ascending=False)


# =========================
# 回测引擎
# =========================
class BacktestEngine:
    def __init__(self, df_full, start_date, end_date):
        self.df_full = df_full
        self.start_date = start_date
        self.end_date = end_date
        self.trades = []           # 每笔交易记录
        self.daily_equity = []     # 每日净值

    def get_trading_dates(self):
        """获取回测区间内的所有交易日"""
        dates = self.df_full["date"].unique()
        dates = sorted(dates)
        dates = [d for d in dates if self.start_date <= d <= self.end_date]
        return dates

    def simulate_trade(self, code, buy_date, buy_price, shares, capital):
        """模拟单笔交易：从买入日起跟踪到卖出日"""
        stock_data = self.df_full[
            (self.df_full["code"] == code) &
            (self.df_full["date"] >= buy_date)
        ].sort_values("date")

        if stock_data.empty or len(stock_data) < 2:
            return None

        buy_row = stock_data.iloc[0]
        actual_buy_price = buy_price * (1 + CFG.SLIPPAGE)  # 买入滑点
        buy_cost = shares * actual_buy_price * (1 + CFG.COMMISSION)

        for i in range(1, len(stock_data)):
            row = stock_data.iloc[i]
            hold_days = i

            # 达到最大持有天数，强制卖出
            if hold_days >= CFG.MAX_HOLD_DAYS:
                sell_price = row["close"] * (1 - CFG.SLIPPAGE)
                sell_revenue = shares * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                return {
                    "code": code,
                    "buy_date": buy_date,
                    "sell_date": row["date"],
                    "hold_days": hold_days,
                    "buy_price": actual_buy_price,
                    "sell_price": sell_price,
                    "shares": shares,
                    "cost": buy_cost,
                    "revenue": sell_revenue,
                    "profit": sell_revenue - buy_cost,
                    "profit_pct": (sell_revenue - buy_cost) / buy_cost,
                    "exit_reason": "到期卖出"
                }

            # 止损检查
            if row["low"] <= buy_price * (1 + CFG.STOP_LOSS):
                stop_price = buy_price * (1 + CFG.STOP_LOSS)
                actual_stop = min(stop_price, row["open"]) * (1 - CFG.SLIPPAGE)
                sell_revenue = shares * actual_stop * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                return {
                    "code": code,
                    "buy_date": buy_date,
                    "sell_date": row["date"],
                    "hold_days": hold_days,
                    "buy_price": actual_buy_price,
                    "sell_price": actual_stop,
                    "shares": shares,
                    "cost": buy_cost,
                    "revenue": sell_revenue,
                    "profit": sell_revenue - buy_cost,
                    "profit_pct": (sell_revenue - buy_cost) / buy_cost,
                    "exit_reason": "止损"
                }

            # 止盈检查
            if row["high"] >= buy_price * (1 + CFG.TARGET_PROFIT):
                target_price = buy_price * (1 + CFG.TARGET_PROFIT)
                actual_target = max(target_price, row["open"]) * (1 - CFG.SLIPPAGE)
                sell_revenue = shares * actual_target * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                return {
                    "code": code,
                    "buy_date": buy_date,
                    "sell_date": row["date"],
                    "hold_days": hold_days,
                    "buy_price": actual_buy_price,
                    "sell_price": actual_target,
                    "shares": shares,
                    "cost": buy_cost,
                    "revenue": sell_revenue,
                    "profit": sell_revenue - buy_cost,
                    "profit_pct": (sell_revenue - buy_cost) / buy_cost,
                    "exit_reason": "止盈"
                }

        # 数据走完未触发条件，以最后一天收盘价卖出
        last_row = stock_data.iloc[-1]
        sell_price = last_row["close"] * (1 - CFG.SLIPPAGE)
        sell_revenue = shares * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
        return {
            "code": code,
            "buy_date": buy_date,
            "sell_date": last_row["date"],
            "hold_days": len(stock_data) - 1,
            "buy_price": actual_buy_price,
            "sell_price": sell_price,
            "shares": shares,
            "cost": buy_cost,
            "revenue": sell_revenue,
            "profit": sell_revenue - buy_cost,
            "profit_pct": (sell_revenue - buy_cost) / buy_cost,
            "exit_reason": "数据结束"
        }

    def run(self, top_k=1, capital=20000):
        """执行回测"""
        dates = self.get_trading_dates()
        print(f"\n🔬 回测期间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")

        cash = capital
        holding = None  # 当前持仓: {"code", "shares", "buy_date", "buy_price"}
        equity_curve = []

        for i, today in enumerate(dates):
            # 先检查是否有持仓需要处理
            if holding is not None:
                stock_data = self.df_full[
                    (self.df_full["code"] == holding["code"]) &
                    (self.df_full["date"] >= holding["buy_date"]) &
                    (self.df_full["date"] <= today)
                ].sort_values("date")

                sell_now = False
                sell_reason = ""

                if len(stock_data) > 0:
                    last = stock_data.iloc[-1]
                    hold_days = len(stock_data) - 1

                    # 止损
                    if last["low"] <= holding["buy_price"] * (1 + CFG.STOP_LOSS):
                        sell_now = True
                        sell_reason = "止损"
                        sell_price = holding["buy_price"] * (1 + CFG.STOP_LOSS)
                    # 止盈
                    elif last["high"] >= holding["buy_price"] * (1 + CFG.TARGET_PROFIT):
                        sell_now = True
                        sell_reason = "止盈"
                        sell_price = holding["buy_price"] * (1 + CFG.TARGET_PROFIT)
                    # 到期
                    elif hold_days >= CFG.MAX_HOLD_DAYS:
                        sell_now = True
                        sell_reason = "到期"
                        sell_price = last["close"]

                if sell_now:
                    actual_sell = sell_price * (1 - CFG.SLIPPAGE)
                    revenue = holding["shares"] * actual_sell * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                    profit = revenue - holding["cost"]
                    self.trades.append({
                        "code": holding["code"],
                        "buy_date": holding["buy_date"],
                        "sell_date": today,
                        "hold_days": len(stock_data) - 1 if len(stock_data) > 0 else 0,
                        "buy_price": holding["buy_price"],
                        "sell_price": actual_sell,
                        "shares": holding["shares"],
                        "cost": holding["cost"],
                        "revenue": revenue,
                        "profit": profit,
                        "profit_pct": profit / holding["cost"],
                        "exit_reason": sell_reason
                    })
                    cash += revenue
                    holding = None

            # 无持仓时选股买入
            if holding is None:
                candidates = calc_factors_cross_section(self.df_full, today)
                if not candidates.empty and len(candidates) >= top_k:
                    best = candidates.iloc[0]
                    budget = cash * CFG.POSITION_PCT
                    shares = int(budget / (best["close"] * (1 + CFG.SLIPPAGE)) / 100) * 100
                    shares = max(shares, 100)

                    if shares * best["close"] <= budget:
                        buy_price = best["close"] * (1 + CFG.SLIPPAGE)
                        cost = shares * buy_price * (1 + CFG.COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            holding = {
                                "code": best["code"],
                                "shares": shares,
                                "buy_date": today,
                                "buy_price": buy_price,
                                "cost": cost
                            }

            # 记录每日净值
            if holding is not None:
                stock_data_today = self.df_full[
                    (self.df_full["code"] == holding["code"]) &
                    (self.df_full["date"] <= today)
                ]
                if not stock_data_today.empty:
                    unrealized_value = holding["shares"] * stock_data_today.iloc[-1]["close"]
                else:
                    unrealized_value = holding["cost"]
                equity = cash + unrealized_value
            else:
                equity = cash

            equity_curve.append({"date": today, "equity": equity})

            if (i + 1) % 50 == 0:
                print(f"  回测进度: {i+1}/{len(dates)}")

        # 最终清仓
        if holding is not None:
            last_data = self.df_full[self.df_full["code"] == holding["code"]]
            if not last_data.empty:
                last_close = last_data.iloc[-1]["close"]
                sell_price = last_close * (1 - CFG.SLIPPAGE)
                revenue = holding["shares"] * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                profit = revenue - holding["cost"]
                self.trades.append({
                    "code": holding["code"],
                    "buy_date": holding["buy_date"],
                    "sell_date": equity_curve[-1]["date"],
                    "hold_days": 0,
                    "buy_price": holding["buy_price"],
                    "sell_price": sell_price,
                    "shares": holding["shares"],
                    "cost": holding["cost"],
                    "revenue": revenue,
                    "profit": profit,
                    "profit_pct": profit / holding["cost"],
                    "exit_reason": "回测结束清仓"
                })
                cash += revenue

        self.equity_curve = pd.DataFrame(equity_curve)
        return self.generate_report(capital)

    def generate_report(self, initial_capital):
        """生成回测报告"""
        trades_df = pd.DataFrame(self.trades)
        equity_df = self.equity_curve

        if trades_df.empty:
            return {"error": "无交易记录"}

        # 基本统计
        total_trades = len(trades_df)
        win_trades = len(trades_df[trades_df["profit"] > 0])
        loss_trades = len(trades_df[trades_df["profit"] <= 0])
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        total_profit = trades_df["profit"].sum()
        avg_profit = trades_df["profit"].mean()
        max_profit = trades_df["profit"].max()
        max_loss = trades_df["profit"].min()
        avg_hold_days = trades_df["hold_days"].mean()

        # 盈亏比
        avg_win = trades_df[trades_df["profit"] > 0]["profit"].mean() if win_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df["profit"] <= 0]["profit"].mean()) if loss_trades > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

        # 收益率
        final_equity = equity_df.iloc[-1]["equity"]
        total_return = (final_equity - initial_capital) / initial_capital * 100

        # 最大回撤
        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["cummax"]) / equity_df["cummax"]
        max_drawdown = equity_df["drawdown"].min() * 100

        # 夏普比率（简化版，按日计算）
        equity_df["daily_return"] = equity_df["equity"].pct_change()
        sharpe_ratio = (equity_df["daily_return"].mean() / equity_df["daily_return"].std() * np.sqrt(252)
                        if equity_df["daily_return"].std() > 0 else 0)

        # 退出原因统计
        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()

        report = {
            "初始资金": initial_capital,
            "最终资金": round(final_equity, 2),
            "总收益率": f"{total_return:.2f}%",
            "年化收益率": f"{total_return / (len(equity_df)/252):.2f}%" if len(equity_df) > 0 else "N/A",
            "最大回撤": f"{max_drawdown:.2f}%",
            "夏普比率": f"{sharpe_ratio:.2f}",
            "总交易次数": total_trades,
            "盈利次数": win_trades,
            "亏损次数": loss_trades,
            "胜率": f"{win_rate:.2f}%",
            "总盈亏": round(total_profit, 2),
            "平均盈亏": round(avg_profit, 2),
            "最大单笔盈利": round(max_profit, 2),
            "最大单笔亏损": round(max_loss, 2),
            "盈亏比": f"{profit_factor:.2f}",
            "平均持有天数": f"{avg_hold_days:.1f}",
            "退出原因分布": exit_reasons
        }

        return report


# =========================
# 实盘选股（保留原功能）
# =========================
def daily_selection(stock_list, name_map, end_date):
    """在给定日期进行选股"""
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=CFG.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    hist = get_all_hist(stock_list, start_date, end_date, CFG.MAX_STOCKS)

    if hist.empty:
        return None

    candidates = calc_factors_cross_section(hist, end_date)
    if candidates.empty:
        return None

    # 资金过滤
    budget = CFG.TOTAL_CAPITAL * CFG.POSITION_PCT
    candidates = candidates[candidates["close"] * 100 <= budget]

    if candidates.empty:
        return None

    candidates["名称"] = candidates["code"].map(name_map).fillna("未知")
    candidates["code_simple"] = candidates["code"].str.replace("sh.", "").str.replace("sz.", "")

    return candidates.head(10)


# =========================
# 打印回测报告
# =========================
def print_report(report):
    print("\n" + "="*60)
    print("  📊 回测报告")
    print("="*60)
    if "error" in report:
        print(f"  ❌ {report['error']}")
        return

    for key, value in report.items():
        if key != "退出原因分布":
            print(f"  {key:<16}: {value}")
        else:
            print(f"  {key:<16}:")
            for reason, count in value.items():
                print(f"    {reason}: {count}次")

    print("="*60)


# =========================
# 主程序
# =========================
def main():
    print("=" * 60)
    print("  A股短线选股系统 V2.0（带回测功能）")
    print("=" * 60)

    try:
        bs_login()

        # 获取股票列表
        stock_list = get_stock_list()
        name_map = dict(zip(stock_list["code"], stock_list["code_name"]))

        # 回测模式
        print("\n" + "-"*60)
        print("  🔬 回测模式")
        print("-"*60)

        # 回测参数
        backtest_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        backtest_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        data_start = (datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")

        print(f"  数据拉取区间: {data_start} ~ {backtest_end}")
        print(f"  回测区间: {backtest_start} ~ {backtest_end}")

        # 拉取完整历史数据
        hist_full = get_all_hist(stock_list, data_start, backtest_end, CFG.MAX_STOCKS)

        if hist_full.empty:
            print("❌ 历史数据为空")
            bs_logout()
            return

        print(f"✅ 成功获取 {hist_full['code'].nunique()} 支股票的历史数据")

        # 运行回测
        engine = BacktestEngine(hist_full, backtest_start, backtest_end)
        report = engine.run(top_k=1, capital=CFG.TOTAL_CAPITAL)
        print_report(report)

        # 实盘选股（最新一个交易日）
        print("\n" + "-"*60)
        print("  📅 最新交易日选股")
        print("-"*60)

        latest_date = hist_full["date"].max()
        print(f"  数据截止日期: {latest_date}")

        top10 = daily_selection(stock_list, name_map, latest_date)

        if top10 is not None and not top10.empty:
            print(f"\n  📋 TOP10 选股结果")
            print(f"  {'排名':<5} {'代码':<10} {'名称':<10} {'现价':>7} {'3日动量':>8} {'量比':>6} {'评分':>7}")
            print(f"  {'-'*60}")
            for i, (_, row) in enumerate(top10.iterrows()):
                print(f"  {i+1:<5} {row['code_simple']:<10} {row['名称']:<10} "
                      f"{row['close']:>7.2f} {row['mom_3']*100:>+7.2f}% "
                      f"{row['vol_ratio']:>6.2f}x {row['score']:>7.3f}")

            # 保存结果
            output = top10[["code_simple", "名称", "close", "mom_3",
                           "mom_5", "vol_ratio", "vol_10", "score"]].copy()
            output.columns = ["代码", "名称", "现价", "3日涨幅", "5日涨幅", "量比", "波动率", "评分"]
            output["3日涨幅"] = output["3日涨幅"].map(lambda x: f"{x*100:+.2f}%")
            output["5日涨幅"] = output["5日涨幅"].map(lambda x: f"{x*100:+.2f}%")
            output.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
            print(f"\n💾 结果已保存至 selected_stocks.csv")
        else:
            print("  ❌ 当日无满足条件的股票")
            pd.DataFrame(columns=["代码", "评分"]).to_csv("selected_stocks.csv", index=False)

        # 保存回测交易记录
        if engine.trades:
            trades_df = pd.DataFrame(engine.trades)
            trades_df.to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
            print("💾 回测交易记录已保存至 backtest_trades.csv")

        # 保存净值曲线
        if engine.equity_curve is not None and not engine.equity_curve.empty:
            engine.equity_curve.to_csv("backtest_equity.csv", index=False, encoding="utf-8-sig")
            print("💾 回测净值曲线已保存至 backtest_equity.csv")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        pd.DataFrame(columns=["代码", "评分"]).to_csv("selected_stocks.csv", index=False)

    finally:
        bs_logout()
        print("\n✅ 完成")


if __name__ == "__main__":
    main()