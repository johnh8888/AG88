#!/usr/bin/env python3
# =========================
# A股短线选股系统 V5.0（完整修复版）
# 修复：get_stock_list / get_index_data / generate_report
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
    TOTAL_CAPITAL  = 20000       # 总资金（元），建议2万起
    MAX_STOCKS     = 400         # 最多分析股票数
    PRICE_LOW      = 6           # 最低股价
    PRICE_HIGH     = 90          # 最高股价（保证1手买得起）
    MIN_AMOUNT     = 1.5e8       # 日均成交额下限（流动性）
    COMMISSION     = 0.00025     # 佣金（买卖各）
    SELL_TAX       = 0.001       # 印花税（卖方）
    STOP_LOSS      = -0.05       # 止损 -5%
    TARGET_PROFIT  = 0.08        # 止盈 +8%
    TRAILING_STOP  = 0.06        # 移动止损（从最高点回落6%）
    MAX_HOLD_DAYS  = 15          # 最长持有天数
    POSITION_PCT   = 0.45        # 单票仓位（2万资金用45%≈9000元/笔）
    MAX_POSITIONS  = 2           # 最多同时持仓2只
    MARKET_FILTER  = True        # 开启大盘择时


# =========================
# 登录/登出
# =========================
def bs_login():
    for attempt in range(3):
        try:
            result = bs.login()
            if result.error_code == "0":
                print("✅ Baostock 登录成功")
                return
        except:
            pass
        time.sleep(2)
    raise RuntimeError("Baostock 登录失败，请检查网络")

def bs_logout():
    try:
        bs.logout()
    except:
        pass


# =========================
# 获取股票列表（修复版：用 rs.next() 逐行读取）
# 过滤：创业板(sz.3) / 北交所(bj.) / 科创板(sh.688) / ST
# =========================
def get_stock_list():
    print("🌐 获取沪深主板股票列表...")
    rs = bs.query_stock_basic(code_name="")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        raise RuntimeError("股票列表为空，Baostock可能未正常连接")

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    df = df[~df["code"].str.startswith(("sz.3", "bj.", "sh.688"))]
    df = df[~df["code_name"].str.contains("ST", na=False)]

    print(f"✅ 过滤后主板股票 {len(df)} 支")
    return df.reset_index(drop=True)


# =========================
# 获取大盘指数（修复版：补全缺失函数）
# 使用沪深300（sh.000300）作为市场基准
# =========================
def get_index_data(start_date, end_date):
    print("📈 获取大盘指数（沪深300）...")
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            fields="date,close,high,low,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "close", "high", "low", "volume"])
        for col in ["close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"✅ 大盘数据 {len(df)} 条")
        return df
    except Exception as e:
        print(f"⚠️ 大盘数据获取失败: {e}，跳过择时过滤")
        return pd.DataFrame()


# =========================
# 拉取个股历史日K（逐行读取）
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


def get_all_hist(stock_list, start_date, end_date):
    codes = stock_list["code"].tolist()[:CFG.MAX_STOCKS]
    frames = []
    print(f"📡 拉取历史数据（{len(codes)} 支）...")
    for i, code in enumerate(codes):
        df = fetch_hist(code, start_date, end_date)
        if df is not None and len(df) >= 60:
            frames.append(df)
        if i % 10 == 0:
            time.sleep(0.15)
        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(codes)}，有效 {len(frames)} 支")
    print(f"✅ 拉取完成，有效 {len(frames)} 支")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# 因子预计算
# =========================
def precompute_factors(df):
    print("⚙️  计算因子...")
    df = df.sort_values(["code", "date"]).copy()

    df["ret"]      = df.groupby("code")["close"].pct_change()
    df["mom_5"]    = df.groupby("code")["close"].pct_change(5)
    df["mom_10"]   = df.groupby("code")["close"].pct_change(10)
    df["mom_20"]   = df.groupby("code")["close"].pct_change(20)
    df["vol_10"]   = df.groupby("code")["ret"].transform(lambda x: x.rolling(10).std())
    df["amt_10"]   = df.groupby("code")["amount"].transform(lambda x: x.rolling(10).mean())
    df["amt_30"]   = df.groupby("code")["amount"].transform(lambda x: x.rolling(30).mean())
    df["vol_ma5"]  = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    df["vol_ratio"]= df["volume"] / df["vol_ma5"].replace(0, np.nan)
    df["turn_ma5"] = df.groupby("code")["turn"].transform(lambda x: x.rolling(5).mean())
    df["ma20"]     = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["above_ma20"] = (df["close"] > df["ma20"]).astype(int)

    print("✅ 因子计算完成")
    return df


# =========================
# 大盘择时
# =========================
def market_timing(index_df, target_date):
    if index_df.empty:
        return True
    idx = index_df[index_df["date"] <= target_date].copy()
    if len(idx) < 25:
        return True
    idx["ma20"]    = idx["close"].rolling(20).mean()
    idx["ma20_up"] = idx["ma20"].diff(3) > 0
    latest = idx.iloc[-1]
    return bool(latest["close"] > latest["ma20"] and latest["ma20_up"])


# =========================
# 每日选股
# =========================
def select_stocks(df_factors, target_date, held_codes):
    today = df_factors[df_factors["date"] == target_date].copy()
    if today.empty:
        return pd.DataFrame()

    today = today[
        (today["close"] >= CFG.PRICE_LOW) &
        (today["close"] <= CFG.PRICE_HIGH) &
        (today["amt_10"] >= CFG.MIN_AMOUNT) &
        (today["above_ma20"] == 1) &
        (today["vol_ratio"] > 0.85) &
        (today["turn_ma5"] > 0.6)
    ].copy()

    if today.empty:
        return today

    today["z_mom"]  = (today["mom_5"].rank(pct=True) * 0.40 +
                       today["mom_10"].rank(pct=True) * 0.35 +
                       today["mom_20"].rank(pct=True) * 0.25)
    today["z_vol"]  = -today["vol_10"].rank(pct=True)
    today["z_amt"]  = (today["amt_10"] / today["amt_30"].replace(0, np.nan)).rank(pct=True)
    today["z_vr"]   = today["vol_ratio"].clip(0, 3).rank(pct=True)

    today["score"]  = (today["z_mom"] * 0.45 +
                       today["z_vol"] * 0.25 +
                       today["z_amt"] * 0.20 +
                       today["z_vr"]  * 0.10)

    today = today[~today["code"].isin(held_codes)]
    return today.sort_values("score", ascending=False)


# =========================
# 回测引擎
# =========================
class BacktestEngine:
    def __init__(self, df_factors, index_df, start_date, end_date):
        self.df_factors  = df_factors
        self.index_df    = index_df
        self.start_date  = start_date
        self.end_date    = end_date
        self.trades      = []
        self.equity_curve= []

    def run(self):
        dates = sorted(self.df_factors["date"].unique())
        dates = [d for d in dates if self.start_date <= d <= self.end_date]
        if not dates:
            print("⚠️ 回测区间无数据")
            return

        cash     = float(CFG.TOTAL_CAPITAL)
        holdings = []
        print(f"🔬 回测区间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")

        for i, today in enumerate(dates):
            # ---- 处理现有持仓 ----
            new_holdings = []
            for h in holdings:
                sd = self.df_factors[
                    (self.df_factors["code"] == h["code"]) &
                    (self.df_factors["date"] <= today)
                ].sort_values("date")

                if len(sd) < 2:
                    new_holdings.append(h)
                    continue

                last      = sd.iloc[-1]
                hold_days = len(sd[sd["date"] >= h["buy_date"]])
                h["highest"] = max(h["highest"], float(last["high"]))

                sell_now  = False
                sell_price= float(last["close"])
                reason    = ""

                if float(last["low"]) <= h["buy_price"] * (1 + CFG.STOP_LOSS):
                    sell_now   = True
                    sell_price = h["buy_price"] * (1 + CFG.STOP_LOSS)
                    reason     = "止损"
                elif float(last["high"]) >= h["buy_price"] * (1 + CFG.TARGET_PROFIT):
                    sell_now   = True
                    sell_price = h["buy_price"] * (1 + CFG.TARGET_PROFIT * 0.85)
                    reason     = "止盈"
                elif (h["highest"] > h["buy_price"] * 1.06 and
                      float(last["close"]) <= h["highest"] * (1 - CFG.TRAILING_STOP)):
                    sell_now   = True
                    sell_price = float(last["close"])
                    reason     = "移动止盈"
                elif hold_days >= CFG.MAX_HOLD_DAYS:
                    sell_now   = True
                    reason     = "到期清仓"

                if sell_now:
                    revenue = h["shares"] * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
                    profit  = revenue - h["cost"]
                    self.trades.append({
                        "code": h["code"], "buy_date": h["buy_date"],
                        "sell_date": today, "buy_price": round(h["buy_price"], 2),
                        "sell_price": round(sell_price, 2), "shares": h["shares"],
                        "cost": round(h["cost"], 2), "revenue": round(revenue, 2),
                        "profit": round(profit, 2), "reason": reason
                    })
                    cash += revenue
                else:
                    new_holdings.append(h)

            holdings = new_holdings

            # ---- 买入新股 ----
            can_buy = (not CFG.MARKET_FILTER or market_timing(self.index_df, today))
            if can_buy and len(holdings) < CFG.MAX_POSITIONS:
                candidates = select_stocks(
                    self.df_factors, today, {h["code"] for h in holdings}
                )
                for _, row in candidates.iterrows():
                    if len(holdings) >= CFG.MAX_POSITIONS:
                        break
                    budget = min(cash * 0.95, CFG.TOTAL_CAPITAL * CFG.POSITION_PCT)
                    if budget < float(row["close"]) * 100:
                        continue
                    shares    = int(budget / (float(row["close"]) * 1.002) / 100) * 100
                    buy_price = float(row["close"]) * 1.002
                    cost      = shares * buy_price * (1 + CFG.COMMISSION)
                    if cost <= cash:
                        cash -= cost
                        holdings.append({
                            "code": row["code"], "shares": shares,
                            "buy_date": today, "buy_price": buy_price,
                            "cost": cost, "highest": float(row["close"])
                        })

            # ---- 记录净值 ----
            equity = cash
            for h in holdings:
                last_px = self.df_factors[
                    (self.df_factors["code"] == h["code"]) &
                    (self.df_factors["date"] <= today)
                ]["close"].iloc[-1]
                equity += h["shares"] * float(last_px)

            self.equity_curve.append({
                "date": today, "equity": round(equity, 2),
                "cash": round(cash, 2), "positions": len(holdings)
            })

            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(dates)}] {today} | 净值: {equity:,.0f} | 持仓: {len(holdings)}")

        self.generate_report()

    # =========================
    # 完整报告（修复版）
    # =========================
    def generate_report(self):
        print("\n" + "="*60)
        print("  📈 回测报告")
        print("="*60)

        if not self.equity_curve:
            print("⚠️ 无净值数据")
            return

        eq_df  = pd.DataFrame(self.equity_curve)
        final  = eq_df["equity"].iloc[-1]
        init   = float(CFG.TOTAL_CAPITAL)
        total_ret = (final - init) / init

        # 年化收益
        n_days    = len(eq_df)
        annual_ret= (1 + total_ret) ** (250 / n_days) - 1

        # 最大回撤
        cum_max   = eq_df["equity"].cummax()
        drawdown  = (eq_df["equity"] - cum_max) / cum_max
        max_dd    = drawdown.min()

        # 夏普
        eq_df["daily_ret"] = eq_df["equity"].pct_change()
        rf_daily  = 0.025 / 250
        excess    = eq_df["daily_ret"] - rf_daily
        sharpe    = (excess.mean() / excess.std() * np.sqrt(250)
                     if excess.std() > 0 else 0)

        print(f"  初始资金     : {init:>12,.0f} 元")
        print(f"  最终净值     : {final:>12,.0f} 元")
        print(f"  总收益       : {total_ret*100:>+11.2f}%")
        print(f"  年化收益     : {annual_ret*100:>+11.2f}%")
        print(f"  最大回撤     : {max_dd*100:>11.2f}%")
        print(f"  夏普比率     : {sharpe:>11.2f}")

        if self.trades:
            tr_df    = pd.DataFrame(self.trades)
            wins     = tr_df[tr_df["profit"] > 0]
            losses   = tr_df[tr_df["profit"] <= 0]
            win_rate = len(wins) / len(tr_df)
            avg_win  = wins["profit"].mean() if len(wins) else 0
            avg_loss = losses["profit"].mean() if len(losses) else 0
            pr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            print(f"\n  总交易次数   : {len(tr_df)}")
            print(f"  胜率         : {win_rate*100:.1f}%")
            print(f"  平均盈利     : +{avg_win:,.0f} 元/笔")
            print(f"  平均亏损     : {avg_loss:,.0f} 元/笔")
            print(f"  盈亏比       : {pr:.2f}:1")

            print(f"\n  离场原因分布:")
            for reason, cnt in tr_df["reason"].value_counts().items():
                pct = cnt / len(tr_df) * 100
                print(f"    {reason:<10}: {cnt} 次 ({pct:.1f}%)")

            print(f"\n  最近10笔交易:")
            print(f"  {'代码':<10} {'买入日':<12} {'卖出日':<12} "
                  f"{'买价':>7} {'卖价':>7} {'盈亏':>8} {'原因'}")
            print(f"  {'-'*65}")
            for _, t in tr_df.tail(10).iterrows():
                sign = "🟢" if t["profit"] > 0 else "🔴"
                print(f"  {sign}{t['code']:<9} {t['buy_date']:<12} {t['sell_date']:<12} "
                      f"{t['buy_price']:>7.2f} {t['sell_price']:>7.2f} "
                      f"{t['profit']:>+8.0f} {t['reason']}")

            # 综合评估
            print(f"\n  📋 综合评估:")
            if sharpe >= 1.5 and max_dd > -0.15 and win_rate >= 0.5:
                print("  ✅ 策略表现良好，可考虑谨慎实盘")
            elif sharpe >= 0.8:
                print("  ⚠️  策略一般，建议继续优化后实盘")
            else:
                print("  ❌ 策略表现差，不建议实盘")

            # 保存交易记录
            tr_df.to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
            eq_df.to_csv("backtest_equity.csv", index=False, encoding="utf-8-sig")
            print(f"\n  💾 交易记录 → backtest_trades.csv")
            print(f"  💾 净值曲线 → backtest_equity.csv")
        else:
            print("\n  ⚠️ 回测期间无任何成交（条件可能过严或数据不足）")

        print("="*60)


# =========================
# 今日推荐（实盘用）
# =========================
def today_pick(df_factors, stock_list):
    today_str = df_factors["date"].max()
    name_map  = dict(zip(stock_list["code"], stock_list["code_name"]))
    candidates= select_stocks(df_factors, today_str, set())

    if candidates.empty:
        print("⚠️ 今日无符合条件股票")
        candidates.to_csv("selected_stocks.csv", index=False)
        return

    candidates["名称"]       = candidates["code"].map(name_map).fillna("未知")
    candidates["code_simple"] = candidates["code"].str.replace(r"(sh\.|sz\.)", "", regex=True)

    best  = candidates.iloc[0]
    top5  = candidates.iloc[:5]

    price  = float(best["close"])
    budget = CFG.TOTAL_CAPITAL * CFG.POSITION_PCT
    shares = int(budget / (price * 1.002) / 100) * 100
    shares = max(shares, 100)
    buy_px = round(price * 1.002, 2)
    cost   = shares * buy_px * (1 + CFG.COMMISSION)
    sl_px  = round(buy_px * (1 + CFG.STOP_LOSS), 2)
    tgt_px = round(buy_px * (1 + CFG.TARGET_PROFIT), 2)
    sl_rev = shares * sl_px * (1 - CFG.COMMISSION - CFG.SELL_TAX)
    tgt_rev= shares * tgt_px * (1 - CFG.COMMISSION - CFG.SELL_TAX)

    print("\n" + "="*60)
    print(f"  📅 {today_str}  今日精选")
    print("="*60)
    print(f"\n  🏆 推荐：【{best['code_simple']} {best['名称']}】")
    print(f"\n  📊 指标")
    print(f"    现价      : {price:.2f} 元")
    print(f"    5日涨幅   : {best['mom_5']*100:+.2f}%")
    print(f"    10日涨幅  : {best['mom_10']*100:+.2f}%")
    print(f"    量比      : {best['vol_ratio']:.2f}x")
    print(f"    综合评分  : {best['score']:.3f}")
    print(f"\n  💰 交易建议（资金 {CFG.TOTAL_CAPITAL:,.0f} 元）")
    print(f"    ✅ 买入   : {shares} 股 @ {buy_px:.2f} 元 = {cost:,.0f} 元")
    print(f"    🟢 止盈   : {tgt_px:.2f} 元 → 到手 {tgt_rev:,.0f} 元 "
          f"(+{tgt_rev-cost:,.0f} 元)")
    print(f"    🔴 止损   : {sl_px:.2f} 元 → 到手 {sl_rev:,.0f} 元 "
          f"({sl_rev-cost:,.0f} 元)")
    print(f"\n  📋 备选 TOP5")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>7} {'5日':>7} {'量比':>6} {'评分':>7}")
    print(f"  {'-'*50}")
    for _, r in top5.iterrows():
        n = r.get("名称", "未知")
        c = r.get("code_simple", r["code"])
        print(f"  {c:<8} {n:<10} {r['close']:>7.2f} "
              f"{r['mom_5']*100:>+6.2f}% {r['vol_ratio']:>6.2f}x {r['score']:>7.3f}")
    print("\n" + "="*60)
    print("  ⚠️ 仅供参考，不构成投资建议，股市有风险")
    print("="*60)

    # 保存
    out = top5[["code_simple","名称","close","mom_5","mom_10","vol_ratio","score"]].copy()
    out.columns = ["代码","名称","现价","5日涨幅","10日涨幅","量比","评分"]
    out.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    print("\n💾 结果已保存至 selected_stocks.csv")


# =========================
# 主程序
# =========================
def main():
    print("="*60)
    print("  A股选股系统 V5.0 — 完整修复版")
    print("="*60)

    try:
        bs_login()
        stock_list = get_stock_list()

        end_date    = datetime.now().strftime("%Y-%m-%d")
        start_date  = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        data_start  = (datetime.now() - timedelta(days=550)).strftime("%Y-%m-%d")

        hist = get_all_hist(stock_list, data_start, end_date)
        if hist.empty:
            print("❌ 历史数据为空")
            pd.DataFrame().to_csv("selected_stocks.csv", index=False)
            return

        df_factors = precompute_factors(hist)
        index_df   = get_index_data(data_start, end_date)

        # 回测
        bt_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        engine   = BacktestEngine(df_factors, index_df, bt_start, end_date)
        engine.run()

        # 今日推荐
        today_pick(df_factors, stock_list)

    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
        pd.DataFrame(columns=["代码","评分"]).to_csv("selected_stocks.csv", index=False)
    finally:
        bs_logout()
        print("\n✅ 完成")


if __name__ == "__main__":
    main()
