#!/usr/bin/env python3
# =========================
# A股因子模型 V8.0（efinance数据源 + 回测 + 风控 + 动态买卖建议）
# 依赖：pip install efinance pandas numpy
# 运行：python a_stock_factor_v8.py
# =========================

import os
import time
import traceback
import warnings
import pandas as pd
import numpy as np
import efinance as ef
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =========================
# 配置
# =========================
class CFG:
    TOTAL_CAPITAL    = 10000        # 总资金（元）
    TOP_N            = 10           # 最多选股数量
    LOOKBACK_DAYS    = 120          # 历史天数
    MIN_AMOUNT       = 2e8          # 最低日均成交额
    PRICE_LOW        = 5            # 最低股价
    CACHE_FILE       = "spot_cache.csv"

    BACKTEST_PERIODS = 10
    HOLD_DAYS        = 20

    STOP_LOSS        = -0.08        # 止损线 -8%
    MAX_POSITION     = 0.40         # 单股最大仓位
    MIN_POSITION     = 0.05         # 单股最小仓位
    COMMISSION       = 0.0015       # 手续费（双边）


# =========================
# 兜底股票池
# =========================
FALLBACK_CODES = [
    "000001", "000002", "000333", "600000", "600036",
    "600519", "600276", "601318", "601166", "601899"
]


# =========================
# 数据获取（efinance版）
# =========================
def get_spot():
    """获取全市场实时行情"""
    # 优先读本地缓存（1小时内有效）
    if os.path.exists(CFG.CACHE_FILE):
        try:
            mtime = os.path.getmtime(CFG.CACHE_FILE)
            if time.time() - mtime < 3600:
                df = pd.read_csv(CFG.CACHE_FILE, dtype={"股票代码": str})
                if len(df) > 100:
                    print("📦 使用本地缓存（1小时内）")
                    return df
        except:
            pass

    try:
        print("🌐 efinance 拉取实时行情...")
        # ef.stock.get_realtime_quotes() 返回全市场实时数据
        df = ef.stock.get_realtime_quotes()
        if df is not None and len(df) > 100:
            df.to_csv(CFG.CACHE_FILE, index=False)
            print(f"✅ 获取到 {len(df)} 支股票")
            return df
    except Exception as e:
        print(f"⚠️ efinance实时行情失败: {e}")

    # 兜底
    print("🧱 使用兜底股票池")
    return pd.DataFrame({"股票代码": FALLBACK_CODES, "股票名称": ["备用股"] * len(FALLBACK_CODES)})


def fetch_hist_ef(code, start, end):
    """用 efinance 拉取单支股票日K历史"""
    try:
        # ef.stock.get_quote_history 返回日K数据
        df = ef.stock.get_quote_history(
            code,
            beg=start,
            end=end,
            klt=101  # 101=日K
        )
        if df is None or df.empty:
            return None
        df["code"] = str(code)
        return df
    except:
        return None


def get_hist(codes, start, end, max_stocks=200):
    """批量拉取历史数据"""
    frames = []
    codes = [str(c) for c in codes[:max_stocks]]

    # efinance 支持批量拉取，先尝试批量
    try:
        print("  尝试批量拉取历史数据...")
        df = ef.stock.get_quote_history(
            codes[:50],   # 先拉前50支测试
            beg=start,
            end=end,
            klt=101
        )
        if df is not None and not df.empty:
            # 批量返回时列名含股票代码
            if isinstance(df, dict):
                for code, sub_df in df.items():
                    if sub_df is not None and not sub_df.empty:
                        sub_df["code"] = str(code)
                        frames.append(sub_df)
            elif isinstance(df, pd.DataFrame) and "股票代码" in df.columns:
                df["code"] = df["股票代码"].astype(str)
                frames.append(df)

            if frames:
                # 继续拉剩余股票
                remaining = codes[50:]
                for i in range(0, len(remaining), 50):
                    batch = remaining[i:i+50]
                    try:
                        sub = ef.stock.get_quote_history(batch, beg=start, end=end, klt=101)
                        if sub is not None and not sub.empty:
                            if isinstance(sub, dict):
                                for c, s in sub.items():
                                    if s is not None and not s.empty:
                                        s["code"] = str(c)
                                        frames.append(s)
                            elif isinstance(sub, pd.DataFrame) and "股票代码" in sub.columns:
                                sub["code"] = sub["股票代码"].astype(str)
                                frames.append(sub)
                        time.sleep(0.5)
                    except:
                        pass
                print(f"  批量拉取完成，共 {len(frames)} 批")
    except Exception as e:
        print(f"  批量拉取失败: {e}，改为逐支拉取...")
        frames = []

    # 如果批量失败，逐支拉取
    if not frames:
        print("  逐支拉取历史数据...")
        for i, c in enumerate(codes):
            df = fetch_hist_ef(c, start, end)
            if df is not None:
                frames.append(df)
            if i % 5 == 0:
                time.sleep(0.3)
            if (i + 1) % 50 == 0:
                print(f"  已拉取 {i+1}/{len(codes)} 支...")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # 统一列名（efinance 列名中文）
    col_map = {
        "日期": "日期", "开盘": "开盘", "收盘": "收盘",
        "最高": "最高", "最低": "最低", "成交量": "成交量",
        "成交额": "成交额", "振幅": "振幅", "涨跌幅": "涨跌幅",
        "涨跌额": "涨跌额", "换手率": "换手率"
    }
    # efinance列名已是中文，直接使用
    return result


def _get_code_col(df):
    """自动识别股票代码列名"""
    for col in ["code", "股票代码"]:
        if col in df.columns:
            return col
    return None


def _get_price_col(df):
    """自动识别收盘价列名"""
    for col in ["收盘", "最新价", "close"]:
        if col in df.columns:
            return col
    return None


def _get_amount_col(df):
    """自动识别成交额列名"""
    for col in ["成交额", "amount"]:
        if col in df.columns:
            return col
    return None


# =========================
# 因子计算
# =========================
def calc_factors(df):
    if df.empty:
        return df

    code_col   = _get_code_col(df)
    price_col  = _get_price_col(df)
    amount_col = _get_amount_col(df)

    if not code_col or not price_col:
        print(f"⚠️ 列名识别失败，当前列: {list(df.columns)}")
        return pd.DataFrame()

    df = df.copy()
    df = df.sort_values([code_col, "日期"])

    df["momentum_5"]  = df.groupby(code_col)[price_col].pct_change(5)
    df["momentum_20"] = df.groupby(code_col)[price_col].pct_change(20)
    df["volatility"]  = df.groupby(code_col)[price_col].pct_change().rolling(10).std()

    if amount_col:
        df["amt_mean"] = df.groupby(code_col)[amount_col].transform(lambda x: x.rolling(5).mean())
    else:
        df["amt_mean"] = 1e9  # 无成交额数据时给默认值

    latest = df.groupby(code_col).tail(1).copy()
    latest = latest.rename(columns={code_col: "code", price_col: "收盘"})
    latest = latest.dropna(subset=["momentum_5", "momentum_20", "volatility"])

    if latest.empty:
        return latest

    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    latest["z_mom5"]  = zscore(latest["momentum_5"])
    latest["z_mom20"] = zscore(latest["momentum_20"])
    latest["z_vol"]   = -zscore(latest["volatility"])
    latest["z_amt"]   = zscore(latest["amt_mean"]) if latest["amt_mean"].std() > 0 else 0

    latest = latest[latest["收盘"] >= CFG.PRICE_LOW]
    if "amt_mean" in latest.columns and latest["amt_mean"].max() > 1e9:
        latest = latest[latest["amt_mean"] >= CFG.MIN_AMOUNT]

    return latest


# =========================
# 综合评分
# =========================
def score_stocks(df):
    df = df.copy()
    df["score"] = (
        df["z_mom5"]  * 0.30 +
        df["z_mom20"] * 0.25 +
        df["z_vol"]   * 0.25 +
        df["z_amt"]   * 0.20
    )
    return df.sort_values("score", ascending=False)


# =========================
# 动态仓位 + 买卖建议
# =========================
def calc_position(selected_df, total_capital):
    df = selected_df.copy().reset_index(drop=True)

    df["vol_safe"]       = df["volatility"].clip(lower=0.005)
    df["risk_adj_score"] = df["score"] / df["vol_safe"]

    min_s = df["risk_adj_score"].min()
    max_s = df["risk_adj_score"].max()
    if max_s > min_s:
        df["raw_weight"] = (df["risk_adj_score"] - min_s) / (max_s - min_s)
    else:
        df["raw_weight"] = 1.0 / len(df)

    df["raw_weight"] = df["raw_weight"] * 0.7 + (1.0 / len(df)) * 0.3
    df["weight"]     = df["raw_weight"] / df["raw_weight"].sum()
    df["weight"]     = df["weight"].clip(lower=CFG.MIN_POSITION, upper=CFG.MAX_POSITION)
    df["weight"]     = df["weight"] / df["weight"].sum()
    df               = df[df["weight"] >= CFG.MIN_POSITION].copy()
    df["weight"]     = df["weight"] / df["weight"].sum()

    df["买入金额(元)"]     = (df["weight"] * total_capital).round(0)
    df["建议股数(股)"]     = ((df["买入金额(元)"] / df["收盘"]) // 100 * 100).astype(int)
    df["建议股数(股)"]     = df["建议股数(股)"].clip(lower=100)
    df["实际买入金额(元)"] = (df["建议股数(股)"] * df["收盘"]).round(2)
    df["买入手续费(元)"]   = (df["实际买入金额(元)"] * CFG.COMMISSION).round(2)
    df["止损价(元)"]       = (df["收盘"] * (1 + CFG.STOP_LOSS)).round(2)
    df["止损卖出金额(元)"] = (df["止损价(元)"] * df["建议股数(股)"] * (1 - CFG.COMMISSION)).round(2)
    df["最大亏损(元)"]     = (df["止损卖出金额(元)"] - df["实际买入金额(元)"] - df["买入手续费(元)"]).round(2)

    df["目标涨幅(%)"]      = (df["momentum_20"].clip(lower=0.02, upper=0.20) * 50).round(1)
    df["目标价(元)"]       = (df["收盘"] * (1 + df["目标涨幅(%)"] / 100)).round(2)
    df["目标卖出金额(元)"] = (df["目标价(元)"] * df["建议股数(股)"] * (1 - CFG.COMMISSION)).round(2)
    df["预期盈利(元)"]     = (df["目标卖出金额(元)"] - df["实际买入金额(元)"] - df["买入手续费(元)"]).round(2)

    return df


# =========================
# 回测
# =========================
def backtest(hist_df, spot_df):
    print("\n📊 开始回测...")

    code_col  = _get_code_col(hist_df)
    price_col = _get_price_col(hist_df)

    if not code_col or not price_col:
        print("⚠️ 列名识别失败，跳过回测")
        return None

    hist_df = hist_df.sort_values([code_col, "日期"])
    dates   = sorted(hist_df["日期"].unique())

    if len(dates) < CFG.HOLD_DAYS * 2:
        print("⚠️ 数据不足，无法回测")
        return None

    period_returns = []
    hold           = CFG.HOLD_DAYS
    max_periods    = min(CFG.BACKTEST_PERIODS, len(dates) // hold - 1)

    for p in range(max_periods):
        train_end_idx  = len(dates) - (max_periods - p) * hold
        test_start_idx = train_end_idx
        test_end_idx   = min(train_end_idx + hold, len(dates) - 1)

        if train_end_idx < hold or test_end_idx >= len(dates):
            continue

        train_end_date  = dates[train_end_idx]
        test_start_date = dates[test_start_idx]
        test_end_date   = dates[test_end_idx]

        train_df = hist_df[hist_df["日期"] <= train_end_date]
        factors  = calc_factors(train_df)
        if factors.empty:
            continue

        scored     = score_stocks(factors)
        top        = scored.head(CFG.TOP_N)
        positioned = calc_position(top, CFG.TOTAL_CAPITAL)
        sel_codes  = positioned["code"].tolist()
        weights    = dict(zip(positioned["code"], positioned["weight"]))

        test_df = hist_df[
            (hist_df[code_col].astype(str).isin(sel_codes)) &
            (hist_df["日期"] >= test_start_date) &
            (hist_df["日期"] <= test_end_date)
        ]
        if test_df.empty:
            continue

        stock_rets = []
        for code in sel_codes:
            s = test_df[test_df[code_col].astype(str) == code].sort_values("日期")
            if len(s) < 2:
                continue
            ret        = (s[price_col].iloc[-1] - s[price_col].iloc[0]) / s[price_col].iloc[0]
            daily_rets = s[price_col].pct_change().fillna(0)
            cum        = (1 + daily_rets).cumprod() - 1
            if cum.min() < CFG.STOP_LOSS:
                ret = CFG.STOP_LOSS
            w = weights.get(code, 1.0 / len(sel_codes))
            stock_rets.append(ret * w)

        if not stock_rets:
            continue

        period_returns.append({
            "period": p + 1,
            "train_end": train_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
            "return": sum(stock_rets),
            "n_stocks": len(stock_rets)
        })

    if not period_returns:
        print("⚠️ 回测结果为空")
        return None

    result_df      = pd.DataFrame(period_returns)
    rets           = result_df["return"].values
    cum_ret        = (1 + rets).prod() - 1
    n_periods      = len(rets)
    periods_per_yr = 250 / CFG.HOLD_DAYS
    annual_ret     = (1 + cum_ret) ** (periods_per_yr / n_periods) - 1
    rf_per_period  = 0.025 / periods_per_yr
    excess_rets    = rets - rf_per_period
    sharpe         = (excess_rets.mean() / excess_rets.std() * np.sqrt(periods_per_yr)
                      if excess_rets.std() > 0 else 0)
    cum_curve      = (1 + rets).cumprod()
    rolling_max    = np.maximum.accumulate(cum_curve)
    max_drawdown   = ((cum_curve - rolling_max) / rolling_max).min()
    win_rate       = (rets > 0).mean()
    wins           = rets[rets > 0]
    losses         = rets[rets < 0]
    profit_ratio   = (wins.mean() / abs(losses.mean())
                      if len(wins) > 0 and len(losses) > 0 else np.nan)

    print("\n" + "="*55)
    print("📈 回测评估报告")
    print("="*55)
    print(f"  回测期数     : {n_periods} 期（每期 {CFG.HOLD_DAYS} 交易日）")
    print(f"  累计收益     : {cum_ret*100:.2f}%")
    print(f"  年化收益     : {annual_ret*100:.2f}%")
    print(f"  夏普比率     : {sharpe:.2f}  （>1良好，>2优秀）")
    print(f"  最大回撤     : {max_drawdown*100:.2f}%")
    print(f"  胜率         : {win_rate*100:.1f}%")
    if not np.isnan(profit_ratio):
        print(f"  盈亏比       : {profit_ratio:.2f}x")
    print("="*55)

    print("\n逐期收益：")
    for _, row in result_df.iterrows():
        sign = "🟢" if row["return"] > 0 else "🔴"
        print(f"  第{int(row['period']):02d}期 {row['test_start']}~{row['test_end']} : "
              f"{sign} {row['return']*100:+.2f}%  ({int(row['n_stocks'])}支)")

    return {"cum_ret": cum_ret, "annual_ret": annual_ret, "sharpe": sharpe,
            "max_drawdown": max_drawdown, "win_rate": win_rate,
            "profit_ratio": profit_ratio, "detail": result_df}


# =========================
# 当前选股 + 买卖建议
# =========================
def select_now(hist_df, spot_df):
    print("\n🔍 生成当前选股及买卖建议...")
    factors = calc_factors(hist_df)
    if factors.empty:
        print("⚠️ 因子为空")
        return pd.DataFrame()

    scored = score_stocks(factors)
    top    = scored.head(CFG.TOP_N)

    # 合并股票名称
    for name_col in ["股票名称", "名称"]:
        for code_col in ["股票代码", "代码"]:
            if name_col in spot_df.columns and code_col in spot_df.columns:
                name_map    = dict(zip(spot_df[code_col].astype(str), spot_df[name_col]))
                top         = top.copy()
                top["名称"] = top["code"].map(name_map).fillna("未知")
                break

    result = calc_position(top, CFG.TOTAL_CAPITAL)

    total_buy    = result["实际买入金额(元)"].sum()
    total_fee    = result["买入手续费(元)"].sum()
    total_sl     = result["止损卖出金额(元)"].sum()
    total_loss   = result["最大亏损(元)"].sum()
    total_target = result["目标卖出金额(元)"].sum()
    total_profit = result["预期盈利(元)"].sum()

    print("\n" + "="*70)
    print(f"  💰 总资金: {CFG.TOTAL_CAPITAL:,.0f} 元  |  实际投入: {total_buy:,.0f} 元  |  手续费: {total_fee:.1f} 元")
    print("="*70)

    for _, row in result.iterrows():
        name = row.get("名称", "未知")
        print(f"\n  【{row['code']} {name}】")
        print(f"    当前价格   : {row['收盘']:.2f} 元")
        print(f"    仓位权重   : {row['weight']*100:.1f}%")
        print(f"    ✅ 建议买入 : {row['建议股数(股)']:.0f} 股  |  买入金额: {row['实际买入金额(元)']:,.0f} 元  |  手续费: {row['买入手续费(元)']:.1f} 元")
        print(f"    🔴 止损卖出 : 跌至 {row['止损价(元)']:.2f} 元  →  到手 {row['止损卖出金额(元)']:,.0f} 元  |  最大亏损: {row['最大亏损(元)']:,.0f} 元")
        print(f"    🟢 目标卖出 : 涨至 {row['目标价(元)']:.2f} 元（+{row['目标涨幅(%)']:.1f}%）  →  到手 {row['目标卖出金额(元)']:,.0f} 元  |  预期盈利: {row['预期盈利(元)']:,.0f} 元")

    print("\n" + "="*70)
    print(f"  📊 汇总（共 {len(result)} 支）")
    print(f"    总买入金额   : {total_buy:>10,.0f} 元")
    print(f"    总手续费     : {total_fee:>10.1f} 元")
    print(f"    止损全出金额 : {total_sl:>10,.0f} 元  （全止损最大亏损 {total_loss:,.0f} 元）")
    print(f"    目标全出金额 : {total_target:>10,.0f} 元  （全达标预期盈利 {total_profit:,.0f} 元）")
    print("="*70)
    print(f"\n⚠️  风控提示:")
    print(f"  · 单股跌破止损价立即卖出，不要侥幸等待")
    print(f"  · 建议持仓 {CFG.HOLD_DAYS} 个交易日后重新评估调仓")
    print(f"  · 剩余备用资金: {CFG.TOTAL_CAPITAL - total_buy:,.0f} 元")

    return result


# =========================
# 主程序
# =========================
def main():
    print("=" * 55)
    print("  A股因子模型 V8.0 — efinance + 回测 + 买卖建议")
    print("=" * 55)
    print("⚠️  免责声明：仅供研究学习，不构成投资建议。")
    print("    股市有风险，入市须谨慎。\n")

    try:
        spot  = get_spot()

        # 识别代码列
        code_col = None
        for c in ["股票代码", "代码", "code"]:
            if c in spot.columns:
                code_col = c
                break
        if not code_col:
            print(f"❌ 无法识别股票代码列，当前列: {list(spot.columns)}")
            return

        codes = spot[code_col].astype(str).tolist()

        end   = datetime.now()
        start = end - timedelta(days=CFG.LOOKBACK_DAYS)
        print(f"📡 拉取历史数据（最多200支）...")
        hist  = get_hist(codes, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), max_stocks=200)

        if hist.empty:
            print("❌ 历史数据为空，退出")
            # 生成空文件避免工作流报错
            pd.DataFrame(columns=["code", "score"]).to_csv("selected_stocks.csv", index=False)
            return

        print(f"✅ 拉取完成，{len(hist)} 条记录，列: {list(hist.columns)[:8]}")

        bt_result = backtest(hist, spot)
        selected  = select_now(hist, spot)

        if not selected.empty:
            selected.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
            print("\n💾 选股结果已保存至 selected_stocks.csv")
        else:
            pd.DataFrame(columns=["code", "score"]).to_csv("selected_stocks.csv", index=False)

        if bt_result is not None:
            bt_result["detail"].to_csv("backtest_detail.csv", index=False, encoding="utf-8-sig")
            print("💾 回测明细已保存至 backtest_detail.csv")

        if bt_result is not None:
            sharpe = bt_result["sharpe"]
            max_dd = bt_result["max_drawdown"]
            print("\n📋 综合评估意见：")
            if sharpe < 0.5:
                print("  ❌ 夏普比率过低，历史表现差，请勿轻易实盘。")
            elif sharpe < 1.0:
                print("  ⚠️  夏普比率一般，谨慎操作。")
            else:
                print("  ✅ 夏普比率尚可，但历史不代表未来。")
            if max_dd < -0.20:
                print("  ❌ 历史最大回撤超20%，实盘心理压力大。")

        print("\n✅ 运行完成")

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        traceback.print_exc()
        pd.DataFrame(columns=["code", "score"]).to_csv("selected_stocks.csv", index=False)


if __name__ == "__main__":
    main()
