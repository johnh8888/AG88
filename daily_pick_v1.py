#!/usr/bin/env python3
# =============================================================
# A股短线选股系统 V6.0 — 稳健增强版
#
# 核心升级：
#   1. 买入时机：KDJ金叉 + MACD零轴上方 + 布林带中轨支撑
#              + 缩量回调再放量突破形态
#   2. 卖出策略：ATR动态止损 + 分批止盈（50%@+5%, 50%@+10%）
#              + 均线跌破强制离场
#   3. 选股评分：加入RSI超卖反弹、缩量回调成本区因子
#   4. 筹码/量能：5日量能趋势 + 放量突破确认
#   5. 买入价位：不追高，限定"今日最高价×1.005内"挂单
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
# 配置（稳健风格）
# ─────────────────────────────────────────────
class CFG:
    TOTAL_CAPITAL   = 20_000     # 总资金（元）
    MAX_STOCKS      = 400        # 最多分析支数
    PRICE_LOW       = 6          # 最低股价
    PRICE_HIGH      = 90         # 最高股价
    MIN_AMOUNT      = 2e8        # 日均成交额下限（稳健提高至2亿）
    COMMISSION      = 0.00025    # 佣金（买卖各）
    SELL_TAX        = 0.001      # 印花税（卖方）

    # ── 止盈止损（稳健分批）──
    STOP_LOSS_ATR   = 2.0        # 止损 = 买入价 - N×ATR
    STOP_LOSS_FIXED = -0.045     # 最大固定止损（兜底）
    TAKE1_PCT       = 0.05       # 第一批止盈 +5%（卖50%仓）
    TAKE2_PCT       = 0.10       # 第二批止盈 +10%（卖剩余全部）
    TRAILING_STOP   = 0.05       # 移动止损（从最高点回落5%）
    BREAK_MA_DAYS   = 3          # 连续N日收盘低于MA10 → 离场

    MAX_HOLD_DAYS   = 12         # 最长持有天数
    POSITION_PCT    = 0.45       # 单票仓位比例
    MAX_POSITIONS   = 2          # 最多持仓2只
    MARKET_FILTER   = True       # 大盘择时


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
            fields="date,close,high,low,volume",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="3"
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date","close","high","low","volume"])
        for c in ["close","high","low","volume"]:
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
# 技术指标计算工具
# ─────────────────────────────────────────────
def calc_kdj(df, n=9):
    """KDJ（随机指标）"""
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv    = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J

def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif      = ema_fast - ema_slow
    dea      = dif.ewm(span=signal, adjust=False).mean()
    hist     = (dif - dea) * 2
    return dif, dea, hist

def calc_rsi(df, n=14):
    """RSI"""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def calc_atr(df, n=14):
    """ATR（平均真实波幅）"""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_boll(df, n=20, k=2):
    """布林带"""
    mid  = df["close"].rolling(n).mean()
    std  = df["close"].rolling(n).std()
    upper= mid + k * std
    lower= mid - k * std
    return upper, mid, lower


# ─────────────────────────────────────────────
# 因子预计算（升级版）
# ─────────────────────────────────────────────
def precompute_factors(df):
    print("⚙️  计算因子（含KDJ/MACD/RSI/ATR/BOLL）...")
    df = df.sort_values(["code","date"]).copy()

    # ── 基础因子 ──
    df["ret"]       = df.groupby("code")["close"].pct_change()
    df["mom_5"]     = df.groupby("code")["close"].pct_change(5)
    df["mom_10"]    = df.groupby("code")["close"].pct_change(10)
    df["mom_20"]    = df.groupby("code")["close"].pct_change(20)
    df["vol_10"]    = df.groupby("code")["ret"].transform(lambda x: x.rolling(10).std())
    df["amt_10"]    = df.groupby("code")["amount"].transform(lambda x: x.rolling(10).mean())
    df["amt_30"]    = df.groupby("code")["amount"].transform(lambda x: x.rolling(30).mean())
    df["vol_ma5"]   = df.groupby("code")["volume"].transform(lambda x: x.rolling(5).mean())
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)
    df["turn_ma5"]  = df.groupby("code")["turn"].transform(lambda x: x.rolling(5).mean())
    df["ma10"]      = df.groupby("code")["close"].transform(lambda x: x.rolling(10).mean())
    df["ma20"]      = df.groupby("code")["close"].transform(lambda x: x.rolling(20).mean())
    df["ma60"]      = df.groupby("code")["close"].transform(lambda x: x.rolling(60).mean())
    df["above_ma20"]= (df["close"] > df["ma20"]).astype(int)
    df["above_ma60"]= (df["close"] > df["ma60"]).astype(int)

    # ── 技术指标（按股票分组计算）──
    kdj_K, kdj_D, kdj_J   = [], [], []
    macd_dif, macd_dea    = [], []
    rsi_list, atr_list     = [], []
    boll_upper, boll_mid, boll_lower = [], [], []
    kdj_golden, ma10_below = [], []

    for code, grp in df.groupby("code"):
        grp = grp.sort_values("date")
        K, D, J     = calc_kdj(grp)
        dif, dea, _ = calc_macd(grp)
        rsi         = calc_rsi(grp)
        atr         = calc_atr(grp)
        bu, bm, bl  = calc_boll(grp)

        # KDJ金叉（K上穿D，且J从低位回升）
        golden = ((K > D) & (K.shift(1) <= D.shift(1)) & (J < 80)).astype(int)

        # 连续N日低于MA10（离场信号用）
        close_below_ma10 = (grp["close"] < grp["close"].rolling(10).mean()).astype(int)
        consecutive_below = close_below_ma10.rolling(CFG.BREAK_MA_DAYS).sum()

        kdj_K.append(K); kdj_D.append(D); kdj_J.append(J)
        macd_dif.append(dif); macd_dea.append(dea)
        rsi_list.append(rsi); atr_list.append(atr)
        boll_upper.append(bu); boll_mid.append(bm); boll_lower.append(bl)
        kdj_golden.append(golden)
        ma10_below.append(consecutive_below)

    df["kdj_K"]        = pd.concat(kdj_K)
    df["kdj_D"]        = pd.concat(kdj_D)
    df["kdj_J"]        = pd.concat(kdj_J)
    df["macd_dif"]     = pd.concat(macd_dif)
    df["macd_dea"]     = pd.concat(macd_dea)
    df["rsi"]          = pd.concat(rsi_list)
    df["atr"]          = pd.concat(atr_list)
    df["boll_upper"]   = pd.concat(boll_upper)
    df["boll_mid"]     = pd.concat(boll_mid)
    df["boll_lower"]   = pd.concat(boll_lower)
    df["kdj_golden"]   = pd.concat(kdj_golden)
    df["ma10_below_n"] = pd.concat(ma10_below)

    # ── 量能趋势（近5日量比均值）──
    df["vol_trend"]    = df.groupby("code")["vol_ratio"].transform(lambda x: x.rolling(5).mean())

    # ── 缩量回调形态（近3日量比<1.0，但今日放量>1.2）──
    df["vol_3d_min"]   = df.groupby("code")["vol_ratio"].transform(lambda x: x.rolling(3).min())
    df["pullback_vol"] = ((df["vol_3d_min"] < 1.0) & (df["vol_ratio"] > 1.2)).astype(int)

    # ── 位置分数（在MA20和MA60之间，得分更高）──
    df["price_pos"]    = np.where(
        (df["close"] > df["ma20"]) & (df["close"] < df["ma60"] * 1.15),
        1, np.where(df["close"] > df["ma20"], 0.5, 0)
    )

    print("✅ 因子计算完成")
    return df


# ─────────────────────────────────────────────
# 大盘择时（升级：加MACD+量能）
# ─────────────────────────────────────────────
def market_timing(index_df, target_date):
    if index_df.empty:
        return True
    idx = index_df[index_df["date"] <= target_date].copy()
    if len(idx) < 30:
        return True
    idx["ma20"]     = idx["close"].rolling(20).mean()
    idx["ma5"]      = idx["close"].rolling(5).mean()
    idx["vol_ma5"]  = idx["volume"].rolling(5).mean()
    idx["ma20_up"]  = idx["ma20"].diff(3) > 0
    dif, dea, _     = calc_macd(idx)
    idx["macd_pos"] = dif > 0
    latest = idx.iloc[-1]
    score  = sum([
        float(latest["close"]) > float(latest["ma20"]),   # 站上MA20
        float(latest["ma5"])   > float(latest["ma20"]),   # MA5>MA20
        bool(latest["ma20_up"]),                           # MA20向上
        bool(latest["macd_pos"]),                          # MACD零轴上方
    ])
    return score >= 3   # 至少满足3条才入场


# ─────────────────────────────────────────────
# 选股评分（升级版）
# ─────────────────────────────────────────────
def select_stocks(df_factors, target_date, held_codes):
    today = df_factors[df_factors["date"] == target_date].copy()
    if today.empty:
        return pd.DataFrame()

    # ── 基础过滤 ──
    today = today[
        (today["close"]    >= CFG.PRICE_LOW) &
        (today["close"]    <= CFG.PRICE_HIGH) &
        (today["amt_10"]   >= CFG.MIN_AMOUNT) &
        (today["above_ma20"] == 1) &           # 站上MA20
        (today["above_ma60"] == 1) &           # 站上MA60（稳健必要）
        (today["vol_ratio"] > 0.8) &
        (today["turn_ma5"]  > 0.5)
    ].copy()

    if today.empty:
        return today

    # ── 技术形态过滤（买入时机精确化）──
    today = today[
        # MACD DIF在零轴附近或上方（-0.05到+∞，允许刚翻正）
        (today["macd_dif"] >= -0.05) &
        # KDJ不超买（K<85）
        (today["kdj_K"] < 85) &
        # RSI不超买（<75）且不极端超卖（>25）
        (today["rsi"].between(25, 75)) &
        # 股价在布林带中轨以上（趋势向上确认）
        (today["close"] >= today["boll_mid"]) &
        # 今日量比>1.0（有量）
        (today["vol_ratio"] >= 1.0)
    ].copy()

    if today.empty:
        return today

    # ── 评分体系（5个维度）──
    # 1. 动量因子（35%）
    today["z_mom"] = (
        today["mom_5"].rank(pct=True)  * 0.40 +
        today["mom_10"].rank(pct=True) * 0.35 +
        today["mom_20"].rank(pct=True) * 0.25
    )
    # 2. 波动因子（20%，越低越好）
    today["z_vol"] = -today["vol_10"].rank(pct=True)

    # 3. 成交量能因子（25%）
    today["z_amt"] = (
        (today["amt_10"] / today["amt_30"].replace(0, np.nan)).rank(pct=True) * 0.50 +
        today["vol_trend"].rank(pct=True) * 0.30 +
        today["vol_ratio"].clip(0, 3).rank(pct=True) * 0.20
    )
    # 4. 技术指标因子（15%）
    today["z_tech"] = (
        today["kdj_golden"].astype(float) * 0.40 +    # KDJ金叉加分
        today["pullback_vol"].astype(float) * 0.30 +  # 缩量回调放量突破
        today["macd_dif"].rank(pct=True) * 0.30        # MACD强度
    )
    # 5. 位置因子（5%）
    today["z_pos"] = today["price_pos"].rank(pct=True)

    today["score"] = (
        today["z_mom"]  * 0.35 +
        today["z_vol"]  * 0.20 +
        today["z_amt"]  * 0.25 +
        today["z_tech"] * 0.15 +
        today["z_pos"]  * 0.05
    )

    today = today[~today["code"].isin(held_codes)]
    return today.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
# 动态止损价计算（ATR为主，固定为兜底）
# ─────────────────────────────────────────────
def calc_stop_price(buy_price, atr_val):
    atr_stop   = buy_price - CFG.STOP_LOSS_ATR * atr_val
    fixed_stop = buy_price * (1 + CFG.STOP_LOSS_FIXED)
    return max(atr_stop, fixed_stop)   # 取较高的（更保守）


# ─────────────────────────────────────────────
# 回测引擎（升级：分批止盈 + ATR动态止损）
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
        shares = shares_to_sell if shares_to_sell else h["shares"]
        revenue = shares * sell_price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
        cost_partial = h["cost"] * (shares / h["shares"])
        profit = revenue - cost_partial
        self.trades.append({
            "code": h["code"], "buy_date": h["buy_date"], "sell_date": today,
            "buy_price": round(h["buy_price"], 2), "sell_price": round(sell_price, 2),
            "shares": shares, "cost": round(cost_partial, 2),
            "revenue": round(revenue, 2), "profit": round(profit, 2), "reason": reason
        })
        return cash + revenue

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
            # ── 持仓管理 ──
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
                cur_close = float(last["close"])
                cur_high  = float(last["high"])
                cur_low   = float(last["low"])
                h["highest"] = max(h["highest"], cur_high)

                sell_all   = False
                sell_price = cur_close
                reason     = ""

                # ① ATR动态止损
                if cur_low <= h["stop_price"]:
                    sell_all   = True
                    sell_price = h["stop_price"]
                    reason     = "ATR止损"

                # ② 移动止盈（从最高点回落5%）
                elif (h["highest"] > h["buy_price"] * 1.06 and
                      cur_close <= h["highest"] * (1 - CFG.TRAILING_STOP)):
                    sell_all   = True
                    sell_price = cur_close
                    reason     = "移动止盈"

                # ③ 均线跌破（连续N日低于MA10）
                elif float(last.get("ma10_below_n", 0) or 0) >= CFG.BREAK_MA_DAYS:
                    sell_all   = True
                    sell_price = cur_close
                    reason     = "跌破MA10"

                # ④ 到期清仓
                elif hold_days >= CFG.MAX_HOLD_DAYS:
                    sell_all   = True
                    reason     = "到期清仓"

                # ⑤ 第一批止盈（+5%，卖50%仓位）
                elif (not h.get("take1_done") and
                      cur_high >= h["buy_price"] * (1 + CFG.TAKE1_PCT)):
                    shares_half = int(h["shares"] * 0.5 / 100) * 100
                    if shares_half >= 100:
                        sell_price = h["buy_price"] * (1 + CFG.TAKE1_PCT)
                        cash = self._sell(h, sell_price, today, "分批止盈1", cash, shares_half)
                        h["shares"] -= shares_half
                        h["cost"]   *= 0.5
                        h["take1_done"] = True
                        # 止损上移至成本价
                        h["stop_price"] = max(h["stop_price"], h["buy_price"] * 1.005)
                    new_holdings.append(h)
                    continue

                # ⑥ 第二批止盈（+10%）
                elif (h.get("take1_done") and not h.get("take2_done") and
                      cur_high >= h["buy_price"] * (1 + CFG.TAKE2_PCT)):
                    sell_all   = True
                    sell_price = h["buy_price"] * (1 + CFG.TAKE2_PCT)
                    reason     = "分批止盈2"

                if sell_all:
                    cash = self._sell(h, sell_price, today, reason, cash)
                else:
                    new_holdings.append(h)

            holdings = new_holdings

            # ── 买入新股 ──
            can_buy = (not CFG.MARKET_FILTER or market_timing(self.index_df, today))
            if can_buy and len(holdings) < CFG.MAX_POSITIONS:
                candidates = select_stocks(
                    self.df_factors, today, {h["code"] for h in holdings}
                )
                for _, row in candidates.iterrows():
                    if len(holdings) >= CFG.MAX_POSITIONS:
                        break
                    # ★ 买入价控制：限当日最高价×1.005以内，避免追高
                    ref_price = float(row["close"])
                    limit_px  = float(row["high"]) * 1.005
                    buy_px    = min(ref_price * 1.002, limit_px)

                    budget = min(cash * 0.95, CFG.TOTAL_CAPITAL * CFG.POSITION_PCT)
                    if budget < buy_px * 100:
                        continue
                    shares = int(budget / (buy_px * 1.002) / 100) * 100
                    cost   = shares * buy_px * (1 + CFG.COMMISSION)
                    if cost > cash:
                        continue

                    atr_val    = float(row.get("atr", ref_price * 0.02) or ref_price * 0.02)
                    stop_price = calc_stop_price(buy_px, atr_val)

                    cash -= cost
                    holdings.append({
                        "code": row["code"], "shares": shares,
                        "buy_date": today, "buy_price": buy_px,
                        "cost": cost, "highest": ref_price,
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
                "cash": round(cash, 2), "positions": len(holdings)
            })

            if (i + 1) % 40 == 0:
                print(f"  [{i+1}/{len(dates)}] {today} | 净值: {equity:,.0f} | 持仓: {len(holdings)}")

        self.generate_report()

    def generate_report(self):
        print("\n" + "="*62)
        print("  📈 回测报告  (V6.0 稳健版)")
        print("="*62)
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
        rf_daily   = 0.025 / 250
        excess     = eq_df["daily_ret"] - rf_daily
        sharpe     = (excess.mean() / excess.std() * np.sqrt(250)
                      if excess.std() > 0 else 0)
        calmar     = annual_ret / abs(max_dd) if max_dd != 0 else 0

        print(f"  初始资金     : {init:>12,.0f} 元")
        print(f"  最终净值     : {final:>12,.0f} 元")
        print(f"  总收益       : {total_ret*100:>+11.2f}%")
        print(f"  年化收益     : {annual_ret*100:>+11.2f}%")
        print(f"  最大回撤     : {max_dd*100:>11.2f}%")
        print(f"  夏普比率     : {sharpe:>11.2f}")
        print(f"  卡玛比率     : {calmar:>11.2f}  ← 年化/最大回撤")

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
                mark = "✅" if "止盈" in reason else ("🔴" if "止损" in reason else "⚪")
                print(f"    {mark} {reason:<12}: {cnt:>3} 次 ({pct:.1f}%)")

            print(f"\n  最近10笔交易:")
            print(f"  {'代码':<10} {'买入日':<12} {'卖出日':<12} "
                  f"{'买价':>7} {'卖价':>7} {'盈亏':>8} {'原因'}")
            print(f"  {'-'*66}")
            for _, t in tr_df.tail(10).iterrows():
                sign = "🟢" if t["profit"] > 0 else "🔴"
                print(f"  {sign}{t['code']:<9} {t['buy_date']:<12} {t['sell_date']:<12} "
                      f"{t['buy_price']:>7.2f} {t['sell_price']:>7.2f} "
                      f"{t['profit']:>+8.0f} {t['reason']}")

            # 综合评估
            print(f"\n  📋 综合评估:")
            if sharpe >= 1.5 and max_dd > -0.15 and win_rate >= 0.5 and calmar >= 1.5:
                print("  ✅ 策略表现优秀，可考虑谨慎实盘")
            elif sharpe >= 1.0 and max_dd > -0.20:
                print("  ⚠️  策略表现中等，建议继续优化")
            else:
                print("  ❌ 策略表现较差，不建议实盘")

            tr_df.to_csv("backtest_trades_v6.csv", index=False, encoding="utf-8-sig")
            eq_df.to_csv("backtest_equity_v6.csv", index=False, encoding="utf-8-sig")
            print(f"\n  💾 交易记录 → backtest_trades_v6.csv")
            print(f"  💾 净值曲线 → backtest_equity_v6.csv")
        else:
            print("\n  ⚠️ 回测期间无成交（条件过严或数据不足）")

        print("="*62)


# ─────────────────────────────────────────────
# 今日推荐（精准价位）
# ─────────────────────────────────────────────
def today_pick(df_factors, stock_list):
    today_str = df_factors["date"].max()
    name_map  = dict(zip(stock_list["code"], stock_list["code_name"]))
    candidates= select_stocks(df_factors, today_str, set())

    print("\n" + "="*62)
    print(f"  📅 {today_str}  今日精选（V6.0 稳健版）")
    print("="*62)

    if candidates.empty:
        print("  ⚠️ 今日无符合条件股票（大盘偏弱或技术形态不佳）")
        pd.DataFrame().to_csv("selected_stocks.csv", index=False)
        return

    candidates["名称"]        = candidates["code"].map(name_map).fillna("未知")
    candidates["code_simple"] = candidates["code"].str.replace(r"(sh\.|sz\.)", "", regex=True)

    top5 = candidates.iloc[:5]

    print(f"\n  📋 备选 TOP5（按综合评分排序）")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>7} {'建议买入':>9} "
          f"{'止损价':>8} {'目标1':>8} {'目标2':>8} {'评分':>7}")
    print(f"  {'-'*73}")

    results = []
    for idx, (_, r) in enumerate(top5.iterrows()):
        price    = float(r["close"])
        atr_val  = float(r.get("atr", price * 0.02) or price * 0.02)

        # ── 精准买入价：收盘价或次日微涨开盘估算（不超最高×1.005）──
        buy_suggest = round(price * 1.002, 2)          # 略高于收盘挂单
        limit_px    = round(float(r["high"]) * 1.005, 2)
        buy_suggest = min(buy_suggest, limit_px)

        # ── 动态止损（ATR×2，不超-4.5%）──
        stop_px  = round(calc_stop_price(buy_suggest, atr_val), 2)
        risk_pct = (stop_px - buy_suggest) / buy_suggest * 100

        # ── 分批止盈目标 ──
        take1_px = round(buy_suggest * (1 + CFG.TAKE1_PCT),  2)
        take2_px = round(buy_suggest * (1 + CFG.TAKE2_PCT),  2)

        # ── 仓位计算 ──
        budget   = CFG.TOTAL_CAPITAL * CFG.POSITION_PCT
        shares   = max(int(budget / (buy_suggest * 1.002) / 100) * 100, 100)
        cost     = round(shares * buy_suggest * (1 + CFG.COMMISSION), 2)

        rank_mark = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "][idx]
        print(f"  {r['code_simple']:<8} {r['名称']:<10} {price:>7.2f} "
              f"{buy_suggest:>9.2f} {stop_px:>8.2f} {take1_px:>8.2f} "
              f"{take2_px:>8.2f} {r['score']:>7.3f} {rank_mark}")

        results.append({
            "代码": r["code_simple"], "名称": r["名称"], "现价": price,
            "建议买入价": buy_suggest, "止损价": stop_px,
            "止盈目标1(+5%)": take1_px, "止盈目标2(+10%)": take2_px,
            "风险比例": f"{risk_pct:.1f}%",
            "建议买入量(股)": shares, "预计成本(元)": cost,
            "ATR": round(atr_val, 3),
            "5日涨幅": f"{r['mom_5']*100:+.2f}%",
            "RSI": round(float(r.get("rsi", 50) or 50), 1),
            "KDJ_K": round(float(r.get("kdj_K", 50) or 50), 1),
            "量比": round(float(r.get("vol_ratio", 1)), 2),
            "评分": round(r["score"], 3)
        })

    # 最优推荐详细分析
    best = results[0]
    r0   = top5.iloc[0]
    print(f"\n  ─────────────────────────────────────────────────────")
    print(f"  🏆 精选推荐：【{best['代码']} {best['名称']}】")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  📊 技术状态")
    print(f"    RSI     : {best['RSI']:.1f}  {'⚠️ 接近超卖' if best['RSI']<35 else ('✅ 健康' if best['RSI']<65 else '⚠️ 偏高')}")
    print(f"    KDJ(K)  : {best['KDJ_K']:.1f}  {'🟡 金叉区间' if 20<best['KDJ_K']<60 else '⚠️ 注意'}")
    print(f"    量比    : {best['量比']:.2f}x")
    print(f"    ATR波幅 : {best['ATR']:.3f} 元")
    print(f"\n  💰 操作计划（资金 {CFG.TOTAL_CAPITAL:,.0f} 元，仓位 {CFG.POSITION_PCT*100:.0f}%）")
    print(f"    ✅ 建议买入 : {best['建议买入价']:.2f} 元 × {best['建议买入量(股)']} 股")
    print(f"       预计成本 : {best['预计成本(元)']:,.0f} 元")
    print(f"    🔴 止损价   : {best['止损价']:.2f} 元  (风险 {best['风险比例']}，ATR×2)")
    print(f"    🟡 目标一   : {best['止盈目标1(+5%)']:.2f} 元  (+5%，卖出50%仓位)")
    print(f"    🟢 目标二   : {best['止盈目标2(+10%)']:.2f} 元  (+10%，卖出剩余仓位)")
    print(f"\n  ⚡ 附加离场条件（任一触发即卖出）")
    print(f"    • 连续{CFG.BREAK_MA_DAYS}日收盘低于MA10")
    print(f"    • 从阶段高点回落超 {CFG.TRAILING_STOP*100:.0f}%（移动止盈）")
    print(f"    • 持有超过 {CFG.MAX_HOLD_DAYS} 个交易日")
    print(f"\n" + "="*62)
    print("  ⚠️  仅供参考，不构成投资建议，股市有风险，操作需谨慎")
    print("="*62)

    # 保存
    out_df = pd.DataFrame(results)
    out_df.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    print(f"\n  💾 完整结果已保存至 selected_stocks.csv")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print("="*62)
    print("  A股选股系统 V6.0 — 稳健增强版")
    print("  升级：精准买入价 | ATR动态止损 | 分批止盈 | KDJ/MACD过滤")
    print("="*62)

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

        today_pick(df_factors, stock_list)

    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
    finally:
        bs_logout()
        print("\n✅ 完成")


if __name__ == "__main__":
    main()
