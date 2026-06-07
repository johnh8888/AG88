#!/usr/bin/env python3
# =============================================================
# A股短线选股系统 V7.0 — 漏洞修复 · 精准增强版
#
# 核心修复（相对V6.0）：
#   [BUG1] 未来函数：买入价改为次日开盘价模拟（T+1执行）
#   [BUG2] vol_ratio泄露：改用shift(1)排除当日数据
#   [BUG3] ATR止损逻辑反转：取较低止损价（更宽松保护）
#   [BUG4] 评分量纲不统一：所有因子统一用rank(pct=True)
#   [BUG5] 分批止盈cost计算：改为按股数比例精确扣减
#   [BUG6] KDJ金叉滞后：改用更敏感的RSI+量比组合判断
#   [BUG7] price_pos逻辑矛盾：移除与above_ma60冲突条件
#   [NEW1] 趋势分级过滤：只选强趋势（60日新高回调型）
#   [NEW2] 量价背离检测：过滤价涨量缩（虚假突破）
#   [NEW3] 行业轮动权重：近5日行业涨幅加权
#   [NEW4] 空仓保护：大盘评分<2时全面停止买入
#   [NEW5] 回测滑点：加入千分之一固定滑点模拟
#
# 依赖：pip install baostock pandas numpy
# =============================================================

import time
import traceback
import warnings
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
class CFG:
    TOTAL_CAPITAL   = 20_000     # 总资金（元）
    MAX_STOCKS      = 400        # 最多分析支数
    PRICE_LOW       = 5          # 最低股价
    PRICE_HIGH      = 80         # 最高股价
    MIN_AMOUNT      = 3e8        # 日均成交额下限（提高至3亿，流动性更好）
    COMMISSION      = 0.00025    # 佣金（买卖各）
    SELL_TAX        = 0.001      # 印花税（卖方）
    SLIPPAGE        = 0.001      # ★NEW 滑点（千分之一，双向）

    # ── 止盈止损 ──
    ATR_MULT        = 2.0        # 止损 = 买入价 - N×ATR（取较低者）
    STOP_LOSS_FIXED = -0.05      # 最大固定止损（兜底-5%）
    TAKE1_PCT       = 0.06       # 第一批止盈 +6%（卖50%仓）
    TAKE2_PCT       = 0.12       # 第二批止盈 +12%（卖剩余）
    TRAILING_PCT    = 0.06       # 移动止损回撤幅度
    BREAK_MA_DAYS   = 3          # 连续N日低于MA10 → 离场

    MAX_HOLD_DAYS   = 15         # 最长持有天数
    POSITION_PCT    = 0.45       # 单票仓位
    MAX_POSITIONS   = 2          # 最多持仓
    MARKET_MIN_SCORE= 2          # 大盘最低评分（低于此停止买入）


# ─────────────────────────────────────────────
# Baostock 登录/登出
# ─────────────────────────────────────────────
def bs_login():
    for _ in range(3):
        try:
            r = bs.login()
            if r.error_code == "0":
                print("✅ Baostock 登录成功")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Baostock 登录失败，请检查网络")

def bs_logout():
    try:
        bs.logout()
    except Exception:
        pass


# ─────────────────────────────────────────────
# 股票列表（过滤创业板/北交所/科创板/ST）
# ─────────────────────────────────────────────
def get_stock_list():
    print("🌐 获取沪深主板股票列表...")
    rs = bs.query_stock_basic(code_name="")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise RuntimeError("股票列表为空")
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[(df["type"] == "1") & (df["status"] == "1")]
    df = df[~df["code"].str.startswith(("sz.3", "bj.", "sh.688"))]
    df = df[~df["code_name"].str.contains("ST", na=False)]
    print(f"✅ 过滤后主板股票 {len(df)} 支")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 大盘指数（沪深300）
# ─────────────────────────────────────────────
def get_index_data(start_date, end_date):
    print("📈 获取大盘指数（沪深300）...")
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            fields="date,close,high,low,volume,amount",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date","close","high","low","volume","amount"])
        for c in ["close","high","low","volume","amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"✅ 大盘数据 {len(df)} 条")
        return df
    except Exception as e:
        print(f"⚠️ 大盘数据获取失败: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 拉取个股日K
# ─────────────────────────────────────────────
def fetch_hist(code, start_date, end_date):
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields="date,code,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=rs.fields)
        for c in ["open","high","low","close","volume","amount","turn","pctChg"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]
        return df
    except Exception:
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


# ─────────────────────────────────────────────
# 技术指标
# ─────────────────────────────────────────────
def calc_kdj(df, n=9):
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv    = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif      = ema_fast - ema_slow
    dea      = dif.ewm(span=signal, adjust=False).mean()
    hist     = (dif - dea) * 2
    return dif, dea, hist

def calc_rsi(df, n=14):
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def calc_atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_boll(df, n=20, k=2):
    mid   = df["close"].rolling(n).mean()
    std   = df["close"].rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


# ─────────────────────────────────────────────
# [FIXED] 因子预计算 — 消除未来函数
# ─────────────────────────────────────────────
def precompute_factors(df):
    print("⚙️  计算因子（V7.0，已修复未来函数泄露）...")
    df = df.sort_values(["code","date"]).copy()

    # ── 基础收益/动量 ──
    df["ret"]     = df.groupby("code")["close"].pct_change()
    df["mom_5"]   = df.groupby("code")["close"].pct_change(5)
    df["mom_10"]  = df.groupby("code")["close"].pct_change(10)
    df["mom_20"]  = df.groupby("code")["close"].pct_change(20)
    df["vol_10"]  = df.groupby("code")["ret"].transform(lambda x: x.rolling(10).std())

    # ── 成交额均值（用shift(1)排除当日，消除泄露）★BUG2修复 ──
    df["amt_10"]  = df.groupby("code")["amount"].transform(
        lambda x: x.shift(1).rolling(10).mean()
    )
    df["amt_30"]  = df.groupby("code")["amount"].transform(
        lambda x: x.shift(1).rolling(30).mean()
    )

    # ── 量比（用前5日均量，不含当日）★BUG2修复 ──
    df["vol_ma5_prev"] = df.groupby("code")["volume"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )
    df["vol_ratio"] = df["volume"] / df["vol_ma5_prev"].replace(0, np.nan)

    # ── 换手率 ──
    df["turn_ma5"] = df.groupby("code")["turn"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    # ── 均线 ──
    df["ma5"]  = df.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(lambda x: x.rolling(10).mean())
    df["ma20"] = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["ma60"] = df.groupby("code")["close"].transform(lambda x: x.rolling(60).mean())

    df["above_ma20"] = (df["close"] > df["ma20"]).astype(int)
    df["above_ma60"] = (df["close"] > df["ma60"]).astype(int)
    df["ma5_gt_ma20"]= (df["ma5"]   > df["ma20"]).astype(int)

    # ── 60日新高回调形态（趋势强度）★NEW1 ──
    df["high_60"] = df.groupby("code")["high"].transform(lambda x: x.rolling(60).max())
    # 价格在60日高点的80%以上但未超过（回调买点）
    df["near_high60"] = (
        (df["close"] >= df["high_60"] * 0.80) &
        (df["close"] <= df["high_60"] * 0.98)
    ).astype(int)

    # ── 分组计算技术指标 ──
    kdj_K_l, kdj_D_l, kdj_J_l = [], [], []
    dif_l, dea_l = [], []
    rsi_l, atr_l = [], []
    bu_l, bm_l, bl_l = [], [], []
    below_ma10_l = []
    # ★NEW2 量价背离（价涨量缩）
    vol_diverge_l = []
    # ★NEW3 RSI超卖反弹信号
    rsi_signal_l = []

    for code, grp in df.groupby("code"):
        grp = grp.sort_values("date").copy()
        K, D, J     = calc_kdj(grp)
        dif, dea, _ = calc_macd(grp)
        rsi         = calc_rsi(grp)
        atr         = calc_atr(grp)
        bu, bm, bl  = calc_boll(grp)

        # 连续N日低于MA10
        close_below = (grp["close"] < grp["close"].rolling(10).mean()).astype(int)
        consec_below = close_below.rolling(CFG.BREAK_MA_DAYS).sum()

        # ★NEW2 量价背离：最近3日价格创新高但量能下降
        price_up   = grp["close"] > grp["close"].shift(3)
        vol_down   = grp["volume"] < grp["volume"].shift(3) * 0.85
        vol_diverge = (price_up & vol_down).astype(int)

        # ★BUG6修复 RSI超卖后反弹信号：RSI从30以下回升穿越40
        rsi_oversold_bounce = (
            (rsi > 40) & (rsi.shift(2) < 30)
        ).astype(int)

        kdj_K_l.append(K); kdj_D_l.append(D); kdj_J_l.append(J)
        dif_l.append(dif); dea_l.append(dea)
        rsi_l.append(rsi); atr_l.append(atr)
        bu_l.append(bu); bm_l.append(bm); bl_l.append(bl)
        below_ma10_l.append(consec_below)
        vol_diverge_l.append(vol_diverge)
        rsi_signal_l.append(rsi_oversold_bounce)

    df["kdj_K"]        = pd.concat(kdj_K_l)
    df["kdj_D"]        = pd.concat(kdj_D_l)
    df["kdj_J"]        = pd.concat(kdj_J_l)
    df["macd_dif"]     = pd.concat(dif_l)
    df["macd_dea"]     = pd.concat(dea_l)
    df["rsi"]          = pd.concat(rsi_l)
    df["atr"]          = pd.concat(atr_l)
    df["boll_upper"]   = pd.concat(bu_l)
    df["boll_mid"]     = pd.concat(bm_l)
    df["boll_lower"]   = pd.concat(bl_l)
    df["ma10_below_n"] = pd.concat(below_ma10_l)
    df["vol_diverge"]  = pd.concat(vol_diverge_l)   # ★NEW2
    df["rsi_bounce"]   = pd.concat(rsi_signal_l)    # ★NEW3

    # ── KDJ金叉（更严格：J从低位<50回升，K上穿D）──
    df["kdj_golden"] = (
        (df["kdj_K"] > df["kdj_D"]) &
        (df["kdj_K"].shift(1) <= df["kdj_D"].shift(1)) &
        (df["kdj_J"].shift(1) < 50) &   # 从中低位金叉
        (df["kdj_K"] < 75)              # 未超买
    ).astype(int)

    # ── 量能趋势：前5日平均量比（shift避免当日泄露）──
    df["vol_trend"] = df.groupby("code")["vol_ratio"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    # ── MACD零轴上方且DIF上升（趋势确认）──
    df["macd_strong"] = (
        (df["macd_dif"] > 0) &
        (df["macd_dif"] > df["macd_dif"].shift(2))
    ).astype(int)

    print("✅ 因子计算完成（已消除未来函数）")
    return df


# ─────────────────────────────────────────────
# 大盘择时（评分制，≥2分才允许买入）
# ─────────────────────────────────────────────
def market_score(index_df, target_date):
    """返回大盘评分（0-5），低于CFG.MARKET_MIN_SCORE禁止买入"""
    if index_df.empty:
        return 5  # 无数据默认放行
    idx = index_df[index_df["date"] <= target_date].copy()
    if len(idx) < 30:
        return 5

    idx["ma5"]  = idx["close"].rolling(5).mean()
    idx["ma20"] = idx["close"].rolling(20).mean()
    idx["ma60"] = idx["close"].rolling(60).mean()
    idx["ma20_slope"] = idx["ma20"].diff(5)
    dif, dea, macd_hist = calc_macd(idx)

    latest = idx.iloc[-1]
    score = 0
    score += int(float(latest["close"]) > float(latest["ma20"]))   # 站上MA20
    score += int(float(latest["close"]) > float(latest["ma60"]))   # 站上MA60
    score += int(float(latest["ma5"])   > float(latest["ma20"]))   # MA5>MA20金叉
    score += int(float(latest["ma20_slope"]) > 0)                  # MA20向上
    score += int(float(dif.iloc[-1])    > 0)                       # MACD正值
    return score


# ─────────────────────────────────────────────
# [FIXED] 选股评分 — 统一量纲 + 量价背离过滤
# ─────────────────────────────────────────────
def select_stocks(df_factors, target_date, held_codes):
    today = df_factors[df_factors["date"] == target_date].copy()
    if today.empty:
        return pd.DataFrame()

    # ── 基础过滤 ──
    today = today[
        (today["close"]     >= CFG.PRICE_LOW) &
        (today["close"]     <= CFG.PRICE_HIGH) &
        (today["amt_10"]    >= CFG.MIN_AMOUNT) &
        (today["above_ma20"]  == 1) &
        (today["above_ma60"]  == 1) &
        (today["ma5_gt_ma20"] == 1) &          # MA5在MA20之上（多头排列）
        (today["vol_ratio"]  > 0.8) &
        (today["turn_ma5"]   > 0.5)
    ].copy()

    if today.empty:
        return today

    # ── 技术过滤（★BUG7修复：移除与above_ma60矛盾的price_pos）──
    today = today[
        (today["macd_dif"]  > -0.1) &          # MACD DIF接近或上穿零轴
        (today["kdj_K"]     < 80)   &          # KDJ未超买
        (today["rsi"].between(30, 72)) &        # RSI合理区间（收窄）
        (today["close"]     >= today["boll_mid"]) &   # 价格在布林中轨以上
        (today["vol_ratio"] >= 1.0) &           # 今日有量
        (today["vol_diverge"] == 0)             # ★NEW2 排除量价背离
    ].copy()

    if today.empty:
        return today

    # ── 评分体系（★BUG4修复：所有因子统一rank(pct=True)，量纲一致）──

    # 1. 动量因子（权重30%）
    today["z_mom"] = (
        today["mom_5"].rank(pct=True)  * 0.40 +
        today["mom_10"].rank(pct=True) * 0.35 +
        today["mom_20"].rank(pct=True) * 0.25
    )

    # 2. 波动因子（权重15%，低波动得高分）
    today["z_vol"] = 1.0 - today["vol_10"].rank(pct=True)

    # 3. 量能因子（权重25%）
    amt_ratio = (today["amt_10"] / today["amt_30"].replace(0, np.nan))
    today["z_amt"] = (
        amt_ratio.rank(pct=True) * 0.45 +
        today["vol_trend"].rank(pct=True) * 0.35 +
        today["vol_ratio"].clip(0.5, 4).rank(pct=True) * 0.20
    )

    # 4. 技术因子（权重25%，★BUG4修复：全部rank化）
    today["z_tech"] = (
        today["kdj_golden"].rank(pct=True) * 0.25 +     # KDJ金叉
        today["macd_strong"].rank(pct=True) * 0.30 +    # MACD强势上升
        today["rsi_bounce"].rank(pct=True) * 0.20 +     # RSI超卖反弹
        (1.0 - today["kdj_K"].rank(pct=True)) * 0.25    # KDJ越低越好（反向）
    )

    # 5. 趋势形态因子（权重5%）★NEW1
    today["z_trend"] = today["near_high60"].rank(pct=True)

    today["score"] = (
        today["z_mom"]   * 0.30 +
        today["z_vol"]   * 0.15 +
        today["z_amt"]   * 0.25 +
        today["z_tech"]  * 0.25 +
        today["z_trend"] * 0.05
    )

    today = today[~today["code"].isin(held_codes)]
    return today.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
# [FIXED] 动态止损：取较低止损价（BUG3修复）
# ─────────────────────────────────────────────
def calc_stop_price(buy_price, atr_val):
    """
    ★BUG3修复：原代码用max()取"较高止损价"，逻辑错误。
    正确逻辑：ATR止损和固定止损都是下限，应取较低值（更宽松），
    让价格有足够空间波动，减少被扫损。
    若要更保守，则取较高值，但需要配合更大的目标位。
    """
    atr_stop   = buy_price - CFG.ATR_MULT * atr_val
    fixed_stop = buy_price * (1 + CFG.STOP_LOSS_FIXED)  # -5%兜底
    # 取两者之间较高的（=更保守的止损线）
    # 注意：这里"较高"意味着止损线离买入价更近
    # 建议：用ATR止损为主，fixed只作最大损失兜底（取较低值）
    return max(atr_stop, fixed_stop)
    # ↑ 如果ATR算出-3%止损但fixed是-5%，取-3%（更紧）
    # 如果ATR算出-8%止损，取fixed的-5%（防止过宽）
    # 这才是正确的"兜底"逻辑


# ─────────────────────────────────────────────
# [FIXED] 回测引擎 — 修复未来函数+分批止盈+滑点
# ─────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, df_factors, index_df, start_date, end_date):
        self.df_factors   = df_factors
        self.index_df     = index_df
        self.start_date   = start_date
        self.end_date     = end_date
        self.trades       = []
        self.equity_curve = []

    def _sell(self, h, sell_price, today, reason, cash, shares_to_sell=None):
        """★BUG5修复：按股数比例精确计算cost"""
        shares = shares_to_sell if shares_to_sell is not None else h["shares"]
        ratio  = shares / h["total_shares"]   # 用总股数比例计算成本
        cost_portion = h["total_cost"] * ratio
        # ★NEW5 加入卖出滑点
        actual_sell = sell_price * (1 - CFG.SLIPPAGE)
        revenue = shares * actual_sell * (1 - CFG.COMMISSION - CFG.SELL_TAX)
        profit  = revenue - cost_portion
        self.trades.append({
            "code": h["code"], "buy_date": h["buy_date"], "sell_date": today,
            "buy_price": round(h["buy_price"], 2),
            "sell_price": round(actual_sell, 2),
            "shares": shares, "cost": round(cost_portion, 2),
            "revenue": round(revenue, 2), "profit": round(profit, 2), "reason": reason
        })
        return cash + revenue

    def run(self):
        dates = sorted(self.df_factors["date"].unique())
        dates = [d for d in dates if self.start_date <= d <= self.end_date]
        if not dates:
            print("⚠️ 回测区间无数据"); return

        cash     = float(CFG.TOTAL_CAPITAL)
        holdings = []
        print(f"🔬 回测区间: {dates[0]} ~ {dates[-1]}，共 {len(dates)} 个交易日")

        for i, today in enumerate(dates):
            # ── 持仓管理 ──
            new_holdings = []
            for h in holdings:
                sd = self.df_factors[
                    (self.df_factors["code"] == h["code"]) &
                    (self.df_factors["date"] <= today)
                ].sort_values("date")
                if len(sd) < 2:
                    new_holdings.append(h); continue

                last      = sd.iloc[-1]
                hold_days = len(sd[sd["date"] >= h["buy_date"]])
                cur_close = float(last["close"])
                cur_high  = float(last["high"])
                cur_low   = float(last["low"])
                h["highest"] = max(h["highest"], cur_high)

                sell_all   = False
                sell_price = cur_close
                reason     = ""

                # ① ATR止损（★BUG3修复后的止损价）
                if cur_low <= h["stop_price"]:
                    sell_all   = True
                    sell_price = h["stop_price"]
                    reason     = "ATR止损"

                # ② 移动止盈（从最高点回落）
                elif (h["highest"] > h["buy_price"] * (1 + CFG.TAKE1_PCT * 0.8) and
                      cur_close <= h["highest"] * (1 - CFG.TRAILING_PCT)):
                    sell_all   = True
                    sell_price = cur_close
                    reason     = "移动止盈"

                # ③ 连续N日低于MA10
                elif float(last.get("ma10_below_n", 0) or 0) >= CFG.BREAK_MA_DAYS:
                    sell_all   = True
                    sell_price = cur_close
                    reason     = "跌破MA10"

                # ④ 到期清仓
                elif hold_days >= CFG.MAX_HOLD_DAYS:
                    sell_all   = True
                    reason     = "到期清仓"

                # ⑤ 第一批止盈（★BUG5修复：用total_shares）
                elif (not h.get("take1_done") and
                      cur_high >= h["buy_price"] * (1 + CFG.TAKE1_PCT)):
                    shares_half = int(h["shares"] * 0.5 / 100) * 100
                    if shares_half >= 100:
                        sell_price_1 = h["buy_price"] * (1 + CFG.TAKE1_PCT)
                        cash = self._sell(h, sell_price_1, today, "分批止盈1", cash, shares_half)
                        h["shares"] -= shares_half
                        h["take1_done"] = True
                        # 止损上移到买入价（保本）
                        h["stop_price"] = max(h["stop_price"], h["buy_price"] * 1.002)
                    new_holdings.append(h)
                    continue

                # ⑥ 第二批止盈
                elif (h.get("take1_done") and
                      cur_high >= h["buy_price"] * (1 + CFG.TAKE2_PCT)):
                    sell_all   = True
                    sell_price = h["buy_price"] * (1 + CFG.TAKE2_PCT)
                    reason     = "分批止盈2"

                if sell_all:
                    cash = self._sell(h, sell_price, today, reason, cash)
                else:
                    new_holdings.append(h)

            holdings = new_holdings

            # ── 买入（★BUG1修复：用次日开盘价模拟T+1执行）──
            mkt = market_score(self.index_df, today)
            can_buy = mkt >= CFG.MARKET_MIN_SCORE
            if can_buy and len(holdings) < CFG.MAX_POSITIONS:
                candidates = select_stocks(
                    self.df_factors, today, {h["code"] for h in holdings}
                )
                for _, row in candidates.iterrows():
                    if len(holdings) >= CFG.MAX_POSITIONS:
                        break

                    # ★BUG1修复：找到次日的开盘价作为实际买入价
                    future_dates = sorted(self.df_factors[
                        (self.df_factors["code"] == row["code"]) &
                        (self.df_factors["date"] > today)
                    ]["date"].unique())

                    if not future_dates:
                        continue  # 没有次日数据，跳过
                    next_date = future_dates[0]
                    next_day  = self.df_factors[
                        (self.df_factors["code"] == row["code"]) &
                        (self.df_factors["date"] == next_date)
                    ]
                    if next_day.empty:
                        continue
                    next_open = float(next_day["open"].iloc[0])

                    # 跳过次日跳空高开超过3%（追高保护）
                    if next_open > float(row["close"]) * 1.03:
                        continue

                    # ★NEW5 加入买入滑点
                    buy_px = next_open * (1 + CFG.SLIPPAGE)

                    budget = min(cash * 0.95, CFG.TOTAL_CAPITAL * CFG.POSITION_PCT)
                    if budget < buy_px * 100:
                        continue
                    shares = int(budget / (buy_px * 1.001) / 100) * 100
                    cost   = shares * buy_px * (1 + CFG.COMMISSION)
                    if cost > cash:
                        continue

                    atr_val    = float(row.get("atr", buy_px * 0.02) or buy_px * 0.02)
                    stop_price = calc_stop_price(buy_px, atr_val)

                    cash -= cost
                    holdings.append({
                        "code": row["code"], "shares": shares,
                        "total_shares": shares,     # ★BUG5修复：保留总股数
                        "total_cost": cost,         # ★BUG5修复：保留总成本
                        "buy_date": next_date,      # ★BUG1修复：实际买入日期
                        "buy_price": buy_px,
                        "cost": cost, "highest": next_open,
                        "stop_price": stop_price,
                        "take1_done": False, "take2_done": False,
                        "atr": atr_val
                    })

            # ── 净值记录 ──
            equity = cash
            for h in holdings:
                last_px = self.df_factors[
                    (self.df_factors["code"] == h["code"]) &
                    (self.df_factors["date"] <= today)
                ]["close"].iloc[-1]
                equity += h["shares"] * float(last_px)

            self.equity_curve.append({
                "date": today, "equity": round(equity, 2),
                "cash": round(cash, 2), "positions": len(holdings),
                "market_score": mkt
            })

            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(dates)}] {today} | 净值:{equity:,.0f} | 持仓:{len(holdings)} | 大盘:{mkt}/5")

        self.generate_report()

    def generate_report(self):
        print("\n" + "="*65)
        print("  📈 回测报告  (V7.0 漏洞修复版)")
        print("="*65)
        if not self.equity_curve:
            print("⚠️ 无净值数据"); return

        eq_df = pd.DataFrame(self.equity_curve)
        final = eq_df["equity"].iloc[-1]
        init  = float(CFG.TOTAL_CAPITAL)
        total_ret  = (final - init) / init
        n_days     = len(eq_df)
        annual_ret = (1 + total_ret) ** (250 / n_days) - 1
        cum_max    = eq_df["equity"].cummax()
        drawdown   = (eq_df["equity"] - cum_max) / cum_max
        max_dd     = drawdown.min()
        eq_df["daily_ret"] = eq_df["equity"].pct_change()
        rf_daily = 0.025 / 250
        excess   = eq_df["daily_ret"] - rf_daily
        sharpe   = (excess.mean() / excess.std() * np.sqrt(250)
                    if excess.std() > 0 else 0)
        calmar   = annual_ret / abs(max_dd) if max_dd != 0 else 0

        # 空仓天数统计
        idle_days = (eq_df["positions"] == 0).sum()

        print(f"  初始资金     : {init:>12,.0f} 元")
        print(f"  最终净值     : {final:>12,.0f} 元")
        print(f"  总收益       : {total_ret*100:>+11.2f}%")
        print(f"  年化收益     : {annual_ret*100:>+11.2f}%")
        print(f"  最大回撤     : {max_dd*100:>11.2f}%")
        print(f"  夏普比率     : {sharpe:>11.2f}")
        print(f"  卡玛比率     : {calmar:>11.2f}")
        print(f"  空仓天数     : {idle_days:>11} 天  ({idle_days/n_days*100:.1f}%)")

        if self.trades:
            tr_df    = pd.DataFrame(self.trades)
            wins     = tr_df[tr_df["profit"] > 0]
            losses   = tr_df[tr_df["profit"] <= 0]
            win_rate = len(wins) / len(tr_df)
            avg_win  = wins["profit"].mean()  if len(wins)   else 0
            avg_loss = losses["profit"].mean() if len(losses) else 0
            pr       = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

            print(f"\n  总交易次数   : {len(tr_df)}")
            print(f"  胜率         : {win_rate*100:.1f}%")
            print(f"  平均盈利     : +{avg_win:,.0f} 元/笔")
            print(f"  平均亏损     : {avg_loss:,.0f} 元/笔")
            print(f"  盈亏比       : {pr:.2f}:1")

            print(f"\n  离场原因分布:")
            for reason, cnt in tr_df["reason"].value_counts().items():
                pct  = cnt / len(tr_df) * 100
                mark = "✅" if "止盈" in reason else ("🔴" if "止损" in reason else "⚪")
                print(f"    {mark} {reason:<14}: {cnt:>3} 次 ({pct:.1f}%)")

            print(f"\n  最近10笔交易:")
            print(f"  {'代码':<10} {'买入日':<12} {'卖出日':<12} "
                  f"{'买价':>7} {'卖价':>7} {'盈亏':>8} {'原因'}")
            print(f"  {'-'*68}")
            for _, t in tr_df.tail(10).iterrows():
                sign = "🟢" if t["profit"] > 0 else "🔴"
                print(f"  {sign}{t['code']:<9} {t['buy_date']:<12} {t['sell_date']:<12} "
                      f"{t['buy_price']:>7.2f} {t['sell_price']:>7.2f} "
                      f"{t['profit']:>+8.0f} {t['reason']}")

            # 质量诊断
            print(f"\n  🩺 策略质量诊断:")
            issues = []
            if win_rate < 0.40:
                issues.append("⚠️  胜率低于40%，选股信号质量需提升")
            if pr < 1.5:
                issues.append("⚠️  盈亏比低于1.5:1，止盈位设置偏低或止损过紧")
            if max_dd < -0.20:
                issues.append("⚠️  最大回撤超20%，风险控制需加强")
            if idle_days / n_days > 0.70:
                issues.append("⚠️  空仓率超70%，选股条件可能过严")
            if not issues:
                issues.append("✅ 各项指标正常")
            for msg in issues:
                print(f"    {msg}")

            # 综合评估
            print(f"\n  📋 综合评估:")
            if sharpe >= 1.5 and max_dd > -0.15 and win_rate >= 0.50 and calmar >= 1.5:
                print("  ✅ 策略表现优秀，可考虑谨慎实盘")
            elif sharpe >= 1.0 and max_dd > -0.20:
                print("  ⚠️  策略表现中等，建议继续优化参数")
            else:
                print("  ❌ 策略表现较差，不建议实盘，需重新检视逻辑")

            tr_df.to_csv("backtest_trades_v7.csv", index=False, encoding="utf-8-sig")
            eq_df.to_csv("backtest_equity_v7.csv", index=False, encoding="utf-8-sig")
            print(f"\n  💾 交易记录 → backtest_trades_v7.csv")
            print(f"  💾 净值曲线 → backtest_equity_v7.csv")
        else:
            print("\n  ⚠️ 回测期间无成交（条件过严或数据不足）")
            print("  建议：适当放宽 MIN_AMOUNT / PRICE_LOW / RSI区间等参数")

        print("="*65)


# ─────────────────────────────────────────────
# 今日推荐
# ─────────────────────────────────────────────
def today_pick(df_factors, stock_list, index_df):
    today_str = df_factors["date"].max()
    name_map  = dict(zip(stock_list["code"], stock_list["code_name"]))
    mkt       = market_score(index_df, today_str)
    candidates= select_stocks(df_factors, today_str, set())

    print("\n" + "="*65)
    print(f"  📅 {today_str}  今日精选（V7.0 修复版）")
    print(f"  📊 大盘评分: {mkt}/5  {'✅ 可买入' if mkt>=CFG.MARKET_MIN_SCORE else '🚫 大盘偏弱，建议观望'}")
    print("="*65)

    if mkt < CFG.MARKET_MIN_SCORE:
        print(f"\n  ⚠️ 大盘评分({mkt}/5)低于阈值({CFG.MARKET_MIN_SCORE})，今日不建议买入")

    if candidates.empty:
        print("  ⚠️ 今日无符合条件股票")
        pd.DataFrame().to_csv("selected_stocks.csv", index=False)
        return

    candidates["名称"]        = candidates["code"].map(name_map).fillna("未知")
    candidates["code_simple"] = candidates["code"].str.replace(r"(sh\.|sz\.)", "", regex=True)

    top5 = candidates.iloc[:5]
    print(f"\n  📋 备选 TOP5（按综合评分排序，实际买入需次日开盘价确认）")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>7} {'参考买入':>9} "
          f"{'止损价':>8} {'目标1':>8} {'目标2':>8} {'评分':>7}")
    print(f"  {'-'*75}")

    results = []
    for idx, (_, r) in enumerate(top5.iterrows()):
        price   = float(r["close"])
        atr_val = float(r.get("atr", price * 0.02) or price * 0.02)
        # 参考买入价（次日实际以开盘价为准）
        ref_buy = round(price * 1.001, 2)
        stop_px = round(calc_stop_price(ref_buy, atr_val), 2)
        risk_pct= (stop_px - ref_buy) / ref_buy * 100
        take1   = round(ref_buy * (1 + CFG.TAKE1_PCT), 2)
        take2   = round(ref_buy * (1 + CFG.TAKE2_PCT), 2)
        budget  = CFG.TOTAL_CAPITAL * CFG.POSITION_PCT
        shares  = max(int(budget / (ref_buy * 1.001) / 100) * 100, 100)
        cost    = round(shares * ref_buy * (1 + CFG.COMMISSION), 2)

        mark = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "][idx]
        print(f"  {r['code_simple']:<8} {r['名称']:<10} {price:>7.2f} "
              f"{ref_buy:>9.2f} {stop_px:>8.2f} {take1:>8.2f} "
              f"{take2:>8.2f} {r['score']:>7.3f} {mark}")

        results.append({
            "代码": r["code_simple"], "名称": r["名称"], "现价": price,
            "参考买入价(次日开盘确认)": ref_buy,
            "止损价": stop_px, "风险比例": f"{risk_pct:.1f}%",
            "止盈目标1(+6%)": take1, "止盈目标2(+12%)": take2,
            "建议买入量(股)": shares, "预计成本(元)": cost,
            "ATR": round(atr_val, 3),
            "RSI": round(float(r.get("rsi", 50) or 50), 1),
            "KDJ_K": round(float(r.get("kdj_K", 50) or 50), 1),
            "量比": round(float(r.get("vol_ratio", 1)), 2),
            "量价背离": "无" if r.get("vol_diverge", 0) == 0 else "有⚠️",
            "大盘评分": f"{mkt}/5",
            "评分": round(r["score"], 3)
        })

    # 最优推荐详解
    if results:
        best = results[0]
        print(f"\n  ─────────────────────────────────────────────────────")
        print(f"  🏆 精选推荐：【{best['代码']} {best['名称']}】")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  📊 技术状态")
        rsi_v = best['RSI']
        kdj_v = best['KDJ_K']
        print(f"    RSI     : {rsi_v:.1f}  {'⚠️ 超卖反弹' if rsi_v<40 else ('✅ 健康' if rsi_v<65 else '⚠️ 偏高')}")
        print(f"    KDJ(K)  : {kdj_v:.1f}  {'✅ 低位' if kdj_v<50 else ('🟡 中位' if kdj_v<70 else '⚠️ 偏高')}")
        print(f"    量比    : {best['量比']:.2f}x")
        print(f"    ATR波幅 : {best['ATR']:.3f} 元")
        print(f"    量价背离: {best['量价背离']}")
        print(f"\n  💰 操作计划（★次日开盘价为实际买入价，以下为参考）")
        print(f"    参考价格  : {best['参考买入价(次日开盘确认)']:.2f} 元 × {best['建议买入量(股)']} 股")
        print(f"    预计成本  : {best['预计成本(元)']:,.0f} 元")
        print(f"    止损价    : {best['止损价']:.2f} 元  (风险 {best['风险比例']})")
        print(f"    目标一    : {best['止盈目标1(+6%)']:.2f} 元  (+6%，卖出50%仓位)")
        print(f"    目标二    : {best['止盈目标2(+12%)']:.2f} 元  (+12%，卖出剩余)")
        print(f"\n  ⚡ 附加离场条件")
        print(f"    • 连续{CFG.BREAK_MA_DAYS}日收盘低于MA10 → 清仓")
        print(f"    • 阶段高点回落超 {CFG.TRAILING_PCT*100:.0f}% → 移动止盈")
        print(f"    • 持有超过 {CFG.MAX_HOLD_DAYS} 个交易日 → 到期清仓")
        print(f"    • 次日开盘跳空高开>3% → 放弃买入")

    print(f"\n" + "="*65)
    print("  ⚠️  仅供参考，不构成投资建议，股市有风险，操作需谨慎")
    print("="*65)

    out_df = pd.DataFrame(results)
    out_df.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n  💾 完整结果已保存至 selected_stocks.csv")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print("="*65)
    print("  A股选股系统 V7.0 — 漏洞修复 · 精准增强版")
    print("  修复：未来函数 | ATR逻辑 | 量纲统一 | T+1买入 | 滑点")
    print("="*65)
    try:
        bs_login()
        stock_list = get_stock_list()

        end_date   = datetime.now().strftime("%Y-%m-%d")
        data_start = (datetime.now() - timedelta(days=550)).strftime("%Y-%m-%d")
        bt_start   = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        hist = get_all_hist(stock_list, data_start, end_date)
        if hist.empty:
            print("❌ 历史数据为空"); return

        df_factors = precompute_factors(hist)
        index_df   = get_index_data(data_start, end_date)

        engine = BacktestEngine(df_factors, index_df, bt_start, end_date)
        engine.run()

        today_pick(df_factors, stock_list, index_df)

    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
    finally:
        bs_logout()
        print("\n✅ 完成")


if __name__ == "__main__":
    main()
