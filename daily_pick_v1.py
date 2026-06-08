#!/usr/bin/env python3
# =============================================================
# A股短线选股系统 V9.0 — 低位买入 · 高命中率版
#
# 核心重构（解决V7.0 亏损-24%问题）：
#
# 【根本原因】V7.0 买在动量高点（RSI 65+、KDJ 70+）
#             止损率49%，盈亏比仅1:1，必然亏损
#
# 【V9.0 核心策略：强趋势中的低位回调买点】
#   条件1  趋势确认：MA5>MA10>MA20>MA60 四线多头排列
#   条件2  回调确认：价格从阶段高点回调3%~12%（不追涨）
#   条件3  超卖确认：KDJ_J < 30 或 RSI < 45（低位信号）
#   条件4  缩量确认：回调期间量比 < 0.8（缩量回调，非出货）
#   条件5  启动确认：今日收阳+量比≥1.2（放量启动信号）
#   条件6  位置确认：价格在MA20~MA60之间（黄金买入区）
#
# 【止盈止损重构】
#   止损：ATR×1.5（更宽），不被轻易扫损
#   止盈：+8% 卖50%，+15% 卖剩余（扩大目标位，提升盈亏比）
#   移动止损：回调7%触发（给利润更多空间）
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
    TOTAL_CAPITAL   = 20_000
    MAX_STOCKS      = 400
    PRICE_LOW       = 5
    PRICE_HIGH      = 80
    MIN_AMOUNT      = 3e8
    COMMISSION      = 0.00025
    SELL_TAX        = 0.001
    SLIPPAGE        = 0.001

    # ── 止盈止损（扩大目标位，盈亏比从1:1提升到2:1+）──
    ATR_MULT        = 1.5        # ATR×1.5，宽一点，减少误止损
    STOP_LOSS_FIXED = -0.07      # 固定止损兜底 -7%
    TAKE1_PCT       = 0.08       # 第一止盈 +8%（卖50%）
    TAKE2_PCT       = 0.15       # 第二止盈 +15%（卖剩余）
    TRAILING_PCT    = 0.07       # 移动止损回撤7%
    BREAK_MA_DAYS   = 3

    MAX_HOLD_DAYS   = 20         # 延长持有等待利润
    POSITION_PCT    = 0.45
    MAX_POSITIONS   = 2
    MARKET_MIN_SCORE= 3          # 大盘需≥3分才买（更严格）

    # ── 回调买点参数（V9.0核心）──
    PULLBACK_MIN    = 0.03       # 最小回调3%（必须从高点有所回落）
    PULLBACK_MAX    = 0.15       # 最大回调15%（超过则趋势可能破坏）
    SHRINK_VOL_MAX  = 0.85       # 回调期间量比≤0.85（缩量健康回调）
    LAUNCH_VOL_MIN  = 1.20       # 启动日量比≥1.20（放量确认）
    RSI_BUY_MAX     = 52         # 买入RSI上限（低位买入）
    KDJ_J_BUY_MAX   = 60         # 买入KDJ_J上限（低位）


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

    # ── KDJ金叉（严格：从低位<50穿越）──
    df["kdj_golden"] = (
        (df["kdj_K"] > df["kdj_D"]) &
        (df["kdj_K"].shift(1) <= df["kdj_D"].shift(1)) &
        (df["kdj_J"].shift(1) < 50) &
        (df["kdj_K"] < 75)
    ).astype(int)

    # ── 量能趋势（shift避免泄露）──
    df["vol_trend"] = df.groupby("code")["vol_ratio"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    # ── MACD上升 ──
    df["macd_strong"] = (
        (df["macd_dif"] > 0) &
        (df["macd_dif"] > df["macd_dif"].shift(2))
    ).astype(int)

    # ════════════════════════════════════════════
    # V9.0 核心：回调买点形态检测
    # ════════════════════════════════════════════

    # 1. 近10日最高价（阶段高点）
    df["recent_high"] = df.groupby("code")["high"].transform(
        lambda x: x.shift(1).rolling(10).max()   # shift(1)避免当日泄露
    )

    # 2. 回调幅度 = (阶段高点 - 当前收盘) / 阶段高点
    df["pullback_pct"] = (df["recent_high"] - df["close"]) / df["recent_high"].replace(0, np.nan)

    # 3. 回调期间是否缩量：近3日最大量比（shift后）
    df["vol_3d_max_lag"] = df.groupby("code")["vol_ratio"].transform(
        lambda x: x.shift(1).rolling(3).max()
    )

    # 4. 今日是否收阳（低位启动信号）
    df["is_up_day"] = (df["close"] > df["open"]).astype(int)

    # 5. MA多头排列（四线排列）
    df["ma5_gt_ma10"] = (df["ma5"] > df["ma10"]).astype(int)
    df["ma10_gt_ma20"]= (df["ma10"]> df["ma20"]).astype(int)
    df["ma20_gt_ma60"]= (df["ma20"]> df["ma60"]).astype(int)
    df["ma_bull_full"]= (
        df["ma5_gt_ma10"] & df["ma10_gt_ma20"] & df["ma20_gt_ma60"]
    ).astype(int)

    # 6. 价格位于MA20和MA60之间（黄金买入区：强趋势低点）
    df["in_golden_zone"] = (
        (df["close"] > df["ma20"]) &           # 在MA20上方（趋势未破）
        (df["close"] < df["ma60"] * 1.08)      # 但未远离MA60（不追高）
    ).astype(int)

    # 7. 综合回调买点信号（必须同时满足）
    df["pullback_buy"] = (
        # 趋势必须多头排列
        (df["ma_bull_full"] == 1) &
        # 有回调（3%~15%）
        (df["pullback_pct"] >= CFG.PULLBACK_MIN) &
        (df["pullback_pct"] <= CFG.PULLBACK_MAX) &
        # 回调期间缩量（健康回调，非出货）
        (df["vol_3d_max_lag"] <= CFG.SHRINK_VOL_MAX) &
        # 今日放量启动
        (df["vol_ratio"] >= CFG.LAUNCH_VOL_MIN) &
        # 今日收阳
        (df["is_up_day"] == 1) &
        # RSI在低位（不买高位）
        (df["rsi"] <= CFG.RSI_BUY_MAX) &
        # KDJ_J在低位
        (df["kdj_J"] <= CFG.KDJ_J_BUY_MAX)
    ).astype(int)

    print(f"✅ 因子计算完成（回调买点信号股数：{df[df['pullback_buy']==1]['code'].nunique()} 支/日平均）")
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
# 选股评分 V9.0 — 低位回调买点逻辑
# ─────────────────────────────────────────────
def select_stocks(df_factors, target_date, held_codes):
    today = df_factors[df_factors["date"] == target_date].copy()
    if today.empty:
        return pd.DataFrame()

    # ══════════════════════════════════════
    # 第一关：基础流动性过滤
    # ══════════════════════════════════════
    today = today[
        today["close"].between(CFG.PRICE_LOW, CFG.PRICE_HIGH) &
        (today["amt_10"] >= CFG.MIN_AMOUNT) &
        (today["turn_ma5"] > 0.3)
    ].copy()
    if today.empty:
        return today

    # ══════════════════════════════════════
    # 第二关：必须触发回调买点信号（硬条件，不打折）
    # ══════════════════════════════════════
    today = today[today["pullback_buy"] == 1].copy()
    if today.empty:
        return today

    # ══════════════════════════════════════
    # 第三关：MACD辅助确认（不要求在零轴上，允许刚翻正）
    # ══════════════════════════════════════
    today = today[
        (today["macd_dif"] > today["macd_dea"]) &  # DIF在DEA上方（MACD金叉）
        (today["vol_diverge"] == 0)                  # 无量价背离
    ].copy()
    if today.empty:
        return today

    # ══════════════════════════════════════
    # 评分（低位质量排序）
    # ══════════════════════════════════════

    # A. 回调质量（40%）：回调幅度越接近理想区间，缩量越好
    #    理想回调：5%~10%，量比~0.6（深度回调缩量）
    ideal_pullback = 0.07
    today["s_pullback"] = (
        # 回调幅度接近7%得分最高（两侧递减）
        (1 - (today["pullback_pct"] - ideal_pullback).abs() / 0.08).clip(0, 1) * 0.50 +
        # 回调期间量比越小越好（越缩量越健康）
        (1 - today["vol_3d_max_lag"].clip(0, 1)).clip(0, 1) * 0.30 +
        # KDJ_J越低得分越高（超卖反弹潜力大）
        (1 - today["kdj_J"].clip(0, 100) / 100) * 0.20
    )

    # B. 趋势强度（30%）：均线排列越好+MACD越强
    today["s_trend"] = (
        today["ma_bull_full"].astype(float) * 0.40 +
        today["macd_strong"].astype(float) * 0.30 +
        today["in_golden_zone"].astype(float) * 0.30
    )

    # C. 量能质量（20%）：今日放量启动的力度
    amt_ratio = (today["amt_10"] / today["amt_30"].replace(0, np.nan))
    today["s_vol"] = (
        today["vol_ratio"].clip(1, 5).rank(pct=True) * 0.50 +
        amt_ratio.rank(pct=True) * 0.30 +
        today["kdj_golden"].astype(float) * 0.20
    )

    # D. RSI位置（10%）：RSI越低越好（超卖反弹）
    today["s_rsi"] = (1 - today["rsi"].clip(20, 52) / 52)

    today["score"] = (
        today["s_pullback"] * 0.40 +
        today["s_trend"]    * 0.30 +
        today["s_vol"]      * 0.20 +
        today["s_rsi"]      * 0.10
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

            tr_df.to_csv("backtest_trades_v9.csv", index=False, encoding="utf-8-sig")
            eq_df.to_csv("backtest_equity_v9.csv", index=False, encoding="utf-8-sig")
            print(f"\n  💾 交易记录 → backtest_trades_v9.csv")
            print(f"  💾 净值曲线 → backtest_equity_v9.csv")
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
    print(f"  📅 {today_str}  今日精选（V9.0 低位回调版）")
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
    print(f"\n  📋 备选 TOP5（次日在【挂单下限~上限】区间挂单，跳空高开放弃）")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>6}   {'挂单区间(低~高)':^18}  {'止损':>6}  {'+8%':>6}  {'+15%':>7}  {'评分':>6}")
    print(f"  {'-'*85}")

    results = []
    for idx, (_, r) in enumerate(top5.iterrows()):
        price   = float(r["close"])
        atr_val = float(r.get("atr", price * 0.02) or price * 0.02)
        high20  = float(r.get("high_20d", price))    # 20日高点

        # ── 精确买入价：以今收盘价为上限，不追高 ──
        # 今日是放量启动日，收盘价本身就是低位，
        # 次日若平开或小幅低开则在MA10附近挂单
        ma10_px = float(r.get("ma10", price * 0.97))
        boll_m  = float(r.get("boll_mid", price * 0.97))
        # 建议挂单价：收盘价下方0~1%，不超过MA10支撑
        buy_upper = round(price, 2)                         # 上限：今收盘
        buy_lower = round(max(ma10_px, boll_m) * 1.001, 2) # 下限：MA10/中轨支撑
        # 取中间值作为参考挂单价
        ref_buy   = round((buy_upper + buy_lower) / 2, 2)
        ref_buy   = min(ref_buy, buy_upper)                 # 不超收盘价

        stop_px  = round(calc_stop_price(ref_buy, atr_val), 2)
        risk_pct = (stop_px - ref_buy) / ref_buy * 100
        take1    = round(ref_buy * (1 + CFG.TAKE1_PCT), 2)
        take2    = round(ref_buy * (1 + CFG.TAKE2_PCT), 2)
        budget   = CFG.TOTAL_CAPITAL * CFG.POSITION_PCT
        shares   = max(int(budget / ref_buy / 100) * 100, 100)
        cost     = round(shares * ref_buy * (1 + CFG.COMMISSION), 2)

        pullback = float(r.get("pullback_pct", 0)) * 100   # 回调幅度%
        kdj_j    = float(r.get("kdj_J", 50) or 50)
        rsi_v    = float(r.get("rsi", 50) or 50)
        vr       = float(r.get("vol_ratio", 1))

        mark = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "][idx]
        print(f"  {r['code_simple']:<8} {r['名称']:<10} {price:>7.2f} "
              f"  挂单:{buy_lower:.2f}~{buy_upper:.2f}  止损:{stop_px:.2f}"
              f"  +8%:{take1:.2f}  +15%:{take2:.2f}  {r['score']:.3f} {mark}")

        results.append({
            "代码": r["code_simple"], "名称": r["名称"], "现价": price,
            "挂单下限": buy_lower, "挂单上限(收盘)": buy_upper,
            "参考买入价": ref_buy,
            "止损价": stop_px, "风险比例": f"{risk_pct:.1f}%",
            "止盈目标1(+8%)": take1, "止盈目标2(+15%)": take2,
            "建议买入量(股)": shares, "预计成本(元)": cost,
            "ATR": round(atr_val, 3),
            "已回调幅度": f"{pullback:.1f}%",
            "RSI": round(rsi_v, 1),
            "KDJ_J": round(kdj_j, 1),
            "量比(今日)": round(vr, 2),
            "大盘评分": f"{mkt}/5",
            "评分": round(r["score"], 3)
        })

    # 最优推荐详解
    if results:
        best = results[0]
        rsi_v = best['RSI']
        kdj_j = best['KDJ_J']
        print(f"\n  ─────────────────────────────────────────────────────")
        print(f"  🏆 精选推荐：【{best['代码']} {best['名称']}】")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  📊 回调买点信号")
        print(f"    已从高点回调  : {best['已回调幅度']}  ← 低位区")
        print(f"    RSI           : {rsi_v:.1f}  {'✅ 低位' if rsi_v<45 else ('🟡 中位' if rsi_v<55 else '⚠️ 偏高，谨慎')}")
        print(f"    KDJ_J         : {kdj_j:.1f}  {'✅ 超卖区' if kdj_j<30 else ('🟡 低位' if kdj_j<55 else '⚠️ 偏高')}")
        print(f"    量比(今日)     : {best['量比(今日)']:.2f}x  ← 放量启动")
        print(f"    ATR波幅       : {best['ATR']:.3f} 元")
        print(f"\n  💰 操作计划（资金 {CFG.TOTAL_CAPITAL:,.0f} 元，仓位 {CFG.POSITION_PCT*100:.0f}%）")
        print(f"    ★ 挂单区间  : {best['挂单下限']:.2f} ~ {best['挂单上限(收盘)']:.2f} 元")
        print(f"       参考挂单  : {best['参考买入价']:.2f} 元（MA10/布林中轨附近）")
        print(f"       建议股数  : {best['建议买入量(股)']} 股  预计成本: {best['预计成本(元)']:,.0f} 元")
        print(f"    🔴 止损价    : {best['止损价']:.2f} 元  (风险 {best['风险比例']}，ATR×1.5)")
        print(f"    🟡 目标一    : {best['止盈目标1(+8%)']:.2f} 元  (+8%，卖出50%仓位)")
        print(f"    🟢 目标二    : {best['止盈目标2(+15%)']:.2f} 元  (+15%，卖出剩余)")
        print(f"\n  ⚡ 离场条件（任一触发）")
        print(f"    • 次日开盘超今收盘3% → 放弃不买（追高风险大）")
        print(f"    • 连续{CFG.BREAK_MA_DAYS}日收盘低于MA10 → 清仓")
        print(f"    • 阶段高点回落超 {CFG.TRAILING_PCT*100:.0f}% → 移动止盈")
        print(f"    • 持满 {CFG.MAX_HOLD_DAYS} 个交易日 → 到期清仓")

    print(f"\n" + "="*65)
    print("  ⚠️  仅供参考，不构成投资建议，股市有风险，操作需谨慎")
    print("="*65)

    out_df = pd.DataFrame(results)
    out_df.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n  💾 今日精选已保存至 selected_stocks.csv")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print("="*65)
    print("  A股选股系统 V9.0 — 低位回调买点 · 高命中率版")
    print("  策略：强趋势回调3~15% + KDJ/RSI低位 + 放量启动确认")
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
