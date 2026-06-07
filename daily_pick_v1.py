#!/usr/bin/env python3
# =============================================================
# A股短线选股系统 V8.0 — AkShare数据源 · 高命中率版
#
# 数据源: AkShare（稳定，免费，无需注册）
# 安装:   pip install akshare pandas numpy
#
# 核心选股逻辑（三重共振）:
#   1. 趋势过滤：MA多头排列 + 量能稳步放大
#   2. 形态识别：缩量回调到支撑位 + 放量启动确认
#   3. 技术共振：MACD零轴上方金叉 + KDJ低位启动 + RSI健康
#
# 高命中率设计原则:
#   - 宁可少选，不选错（严格过滤 > 广撒网）
#   - 必须有成交量配合，拒绝无量上涨
#   - 买在回调低点而非追涨高点
#   - 大盘弱势期完全空仓
# =============================================================

import time
import warnings
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

try:
    import akshare as ak
    DATASOURCE = "akshare"
    print("✅ 使用 AkShare 数据源")
except ImportError:
    print("❌ 未安装 AkShare，请运行: pip install akshare")
    print("   备用: pip install baostock  (再将DATASOURCE改为baostock)")
    exit(1)


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
class CFG:
    # 资金
    TOTAL_CAPITAL   = 20_000
    POSITION_PCT    = 0.45       # 单票仓位45%
    MAX_POSITIONS   = 2

    # 股票过滤
    PRICE_LOW       = 5
    PRICE_HIGH      = 80
    MIN_AMOUNT_B    = 2.0        # 最低日均成交额（亿元）
    MAX_STOCKS      = 500        # 最多分析支数

    # 止损止盈
    ATR_MULT        = 2.0
    STOP_LOSS_MAX   = -0.05      # 最大止损-5%
    TAKE1_PCT       = 0.06       # 第一止盈+6%（卖50%）
    TAKE2_PCT       = 0.12       # 第二止盈+12%（卖剩余）
    TRAILING_PCT    = 0.06       # 移动止损回撤6%
    MAX_HOLD_DAYS   = 15

    # 回测
    COMMISSION      = 0.00025
    SELL_TAX        = 0.001
    SLIPPAGE        = 0.001      # 滑点0.1%

    # 大盘
    MARKET_MIN_SCORE= 3          # 大盘评分低于3禁止买入（满分5）

    # 选股严格度（越高越严格，命中率越高但候选越少）
    MIN_SCORE       = 0.55       # 评分低于此不输出


# ─────────────────────────────────────────────
# AkShare 数据接口
# ─────────────────────────────────────────────
class AkShareData:

    @staticmethod
    def get_stock_list():
        """获取A股主板股票列表（过滤ST/创业板/科创板/北交所）"""
        print("🌐 获取股票列表...")
        try:
            df = ak.stock_info_a_code_name()
            df.columns = ["code", "name"]
            # 过滤：只要6开头（沪主板）和0/3... 只要0开头（深主板）
            df = df[
                df["code"].str.startswith(("60", "00"))  # 沪深主板
            ]
            df = df[~df["name"].str.contains("ST|退", na=False)]
            print(f"✅ 主板股票 {len(df)} 支")
            return df
        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_hist(code, start_date, end_date):
        """获取个股日K线（后复权）"""
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="hfq"
            )
            if df is None or df.empty:
                return None
            # 统一列名
            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "换手率": "turn", "涨跌幅": "pctChg"
            }
            df = df.rename(columns=col_map)
            df["date"] = df["date"].astype(str)
            df["code"] = code
            for c in ["open", "close", "high", "low", "volume", "amount"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"])
            df = df[df["close"] > 0]
            return df.reset_index(drop=True)
        except Exception:
            return None

    @staticmethod
    def get_index_hist(start_date, end_date):
        """获取沪深300指数"""
        print("📈 获取大盘指数（沪深300）...")
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            df.columns = ["date", "open", "close", "high", "low", "volume"]
            df["date"] = df["date"].astype(str)
            for c in ["open", "close", "high", "low", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[
                (df["date"] >= start_date) &
                (df["date"] <= end_date)
            ].reset_index(drop=True)
            print(f"✅ 大盘数据 {len(df)} 条")
            return df
        except Exception as e:
            print(f"⚠️ 大盘数据失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_all_hist(stock_list, start_date, end_date, max_stocks=500):
        codes = stock_list["code"].tolist()[:max_stocks]
        frames = []
        print(f"📡 拉取历史数据（{len(codes)} 支，预计需要几分钟）...")
        for i, code in enumerate(codes):
            df = AkShareData.get_hist(code, start_date, end_date)
            if df is not None and len(df) >= 60:
                frames.append(df)
            # AkShare需要限速，否则会被封IP
            time.sleep(0.12)
            if (i + 1) % 50 == 0:
                print(f"  进度 {i+1}/{len(codes)}，有效 {len(frames)} 支")
        print(f"✅ 数据拉取完成，有效 {len(frames)} 支")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────
# 技术指标
# ─────────────────────────────────────────────
def calc_kdj(df, n=9):
    low_n  = df["low"].rolling(n, min_periods=1).min()
    high_n = df["high"].rolling(n, min_periods=1).max()
    rsv    = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    dif   = ema_f - ema_s
    dea   = dif.ewm(span=signal, adjust=False).mean()
    hist  = (dif - dea) * 2
    return dif, dea, hist

def calc_rsi(series, n=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(n).mean()
    loss  = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))

def calc_atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_boll(series, n=20, k=2):
    mid   = series.rolling(n).mean()
    std   = series.rolling(n).std()
    return mid + k*std, mid, mid - k*std


# ─────────────────────────────────────────────
# 因子计算（无未来函数）
# ─────────────────────────────────────────────
def precompute_factors(df):
    print("⚙️  计算技术因子...")
    df = df.sort_values(["code", "date"]).copy()

    g = df.groupby("code")

    # 收益率
    df["ret"]   = g["close"].pct_change()
    df["mom5"]  = g["close"].pct_change(5)
    df["mom10"] = g["close"].pct_change(10)
    df["mom20"] = g["close"].pct_change(20)

    # 波动率
    df["vol10"] = g["ret"].transform(lambda x: x.rolling(10).std())

    # 成交额（shift消除当日泄露）
    df["amt10"] = g["amount"].transform(lambda x: x.shift(1).rolling(10).mean())
    df["amt30"] = g["amount"].transform(lambda x: x.shift(1).rolling(30).mean())

    # 量比（用前5日均量）
    df["vol_ma5_lag"] = g["volume"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["vol_ratio"]   = df["volume"] / df["vol_ma5_lag"].replace(0, np.nan)

    # 换手率均值
    if "turn" in df.columns:
        df["turn_ma5"] = g["turn"].transform(lambda x: x.shift(1).rolling(5).mean())
    else:
        df["turn_ma5"] = 1.0

    # 均线
    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = g["close"].transform(lambda x: x.rolling(n).mean())

    # 均线趋势
    df["ma_bull"] = (
        (df["ma5"] > df["ma10"]) &
        (df["ma10"] > df["ma20"]) &
        (df["ma20"] > df["ma60"])
    ).astype(int)

    # 60日新高（近期强势回调位）
    df["high60"] = g["high"].transform(lambda x: x.rolling(60).max())
    df["near_high60"] = (
        (df["close"] >= df["high60"] * 0.82) &
        (df["close"] <= df["high60"] * 0.97)
    ).astype(int)

    # 分组计算技术指标
    K_l, D_l, J_l   = [], [], []
    dif_l, dea_l     = [], []
    rsi_l, atr_l     = [], []
    bu_l, bm_l, bl_l = [], [], []
    below_ma10_l      = []
    vol_div_l         = []
    # 缩量回调再放量突破
    shrink_rebound_l  = []
    # MA10连续下方天数
    consec_below_l    = []

    for code, grp in df.groupby("code"):
        grp = grp.sort_values("date").copy()

        K, D, J     = calc_kdj(grp)
        dif, dea, _ = calc_macd(grp["close"])
        rsi         = calc_rsi(grp["close"])
        atr         = calc_atr(grp)
        bu, bm, bl  = calc_boll(grp["close"])

        # 连续N日低于MA10
        ma10 = grp["close"].rolling(10).mean()
        below = (grp["close"] < ma10).astype(int)
        consec_below = below.rolling(3).sum()

        # 量价背离（价涨量缩）
        price_up = grp["close"] > grp["close"].shift(3)
        vol_down = grp["volume"] < grp["volume"].shift(3) * 0.80
        vol_div  = (price_up & vol_down).astype(int)

        # 缩量回调再放量（近5日最小量比<0.8，今日量比>1.5）
        vr = grp["volume"] / grp["volume"].shift(1).rolling(5).mean().replace(0, np.nan)
        min_vr_5 = vr.shift(1).rolling(5).min()
        shrink_rebound = ((min_vr_5 < 0.8) & (vr > 1.5)).astype(int)

        K_l.append(K); D_l.append(D); J_l.append(J)
        dif_l.append(dif); dea_l.append(dea)
        rsi_l.append(rsi); atr_l.append(atr)
        bu_l.append(bu); bm_l.append(bm); bl_l.append(bl)
        consec_below_l.append(consec_below)
        vol_div_l.append(vol_div)
        shrink_rebound_l.append(shrink_rebound)

    df["kdj_K"]       = pd.concat(K_l)
    df["kdj_D"]       = pd.concat(D_l)
    df["kdj_J"]       = pd.concat(J_l)
    df["macd_dif"]    = pd.concat(dif_l)
    df["macd_dea"]    = pd.concat(dea_l)
    df["rsi"]         = pd.concat(rsi_l)
    df["atr"]         = pd.concat(atr_l)
    df["boll_u"]      = pd.concat(bu_l)
    df["boll_m"]      = pd.concat(bm_l)
    df["boll_l"]      = pd.concat(bl_l)
    df["consec_below"]= pd.concat(consec_below_l)
    df["vol_div"]     = pd.concat(vol_div_l)
    df["shrink_rb"]   = pd.concat(shrink_rebound_l)

    # KDJ金叉（严格：从低位<55穿越，K<75未超买）
    df["kdj_cross"] = (
        (df["kdj_K"] > df["kdj_D"]) &
        (df["kdj_K"].shift(1) <= df["kdj_D"].shift(1)) &
        (df["kdj_J"].shift(1) < 55) &
        (df["kdj_K"] < 75)
    ).astype(int)

    # MACD：DIF在零轴上方且上升
    df["macd_up"] = (
        (df["macd_dif"] > 0) &
        (df["macd_dif"] > df["macd_dif"].shift(3))
    ).astype(int)

    # MACD：DIF刚上穿零轴（-0.05到+0.1之间，金叉区）
    df["macd_cross0"] = (
        (df["macd_dif"] > 0) &
        (df["macd_dif"].shift(2) < 0)
    ).astype(int)

    # 量能趋势（近5日量比均值，shift避免泄露）
    df["vol_trend"] = df.groupby("code")["vol_ratio"].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )

    # 次日开盘价（用于回测T+1买入）
    df["next_open"] = df.groupby("code")["open"].shift(-1)
    df["next_date"] = df.groupby("code")["date"].shift(-1)

    print("✅ 因子计算完成")
    return df


# ─────────────────────────────────────────────
# 大盘评分（0-5）
# ─────────────────────────────────────────────
def get_market_score(index_df, target_date):
    if index_df.empty:
        return 5
    idx = index_df[index_df["date"] <= target_date].copy()
    if len(idx) < 30:
        return 5
    idx["ma5"]  = idx["close"].rolling(5).mean()
    idx["ma20"] = idx["close"].rolling(20).mean()
    idx["ma60"] = idx["close"].rolling(60).mean()
    idx["ma20_slope"] = idx["ma20"].diff(5)
    dif, _, _   = calc_macd(idx["close"])

    r = idx.iloc[-1]
    score = 0
    score += int(float(r["close"]) > float(r["ma20"]))       # 站上MA20
    score += int(float(r["close"]) > float(r["ma60"]))       # 站上MA60
    score += int(float(r["ma5"])   > float(r["ma20"]))       # MA5>MA20
    score += int(float(r["ma20_slope"]) > 0)                 # MA20向上
    score += int(float(dif.iloc[-1]) > 0)                    # MACD正值
    return score


# ─────────────────────────────────────────────
# 核心选股（三重共振）
# ─────────────────────────────────────────────
def select_stocks(df_factors, target_date, held_codes):
    today = df_factors[df_factors["date"] == target_date].copy()
    if today.empty:
        return pd.DataFrame()

    # ════════════════════════════════
    # 第一关：基础质量过滤
    # ════════════════════════════════
    today = today[
        today["close"].between(CFG.PRICE_LOW, CFG.PRICE_HIGH) &
        (today["amt10"] >= CFG.MIN_AMOUNT_B * 1e8) &   # 日均成交≥2亿
        (today["vol_ratio"] >= 1.0)                     # 今日有量（不缩量）
    ].copy()
    if today.empty:
        return today

    # ════════════════════════════════
    # 第二关：趋势质量过滤
    # ════════════════════════════════
    today = today[
        (today["ma_bull"] == 1) &           # MA多头排列（5>10>20>60）
        (today["close"] > today["ma20"]) &  # 价格在MA20上
        (today["close"] > today["ma60"]) &  # 价格在MA60上
        (today["vol_div"] == 0) &           # 无量价背离
        (today["consec_below"] == 0)        # 没有连续跌破MA10
    ].copy()
    if today.empty:
        return today

    # ════════════════════════════════
    # 第三关：技术形态过滤（精确买点）
    # ════════════════════════════════
    today = today[
        (today["macd_dif"] > -0.05) &                # MACD DIF接近或上穿零轴
        (today["kdj_K"] < 80) &                      # KDJ未超买
        (today["rsi"].between(32, 70)) &             # RSI健康区间
        (today["close"] >= today["boll_m"]) &        # 价格在布林中轨以上
        (today["close"] <= today["boll_u"] * 0.97)   # 未碰上轨（避免追顶）
    ].copy()
    if today.empty:
        return today

    # ════════════════════════════════
    # 评分（五维，量纲统一rank）
    # ════════════════════════════════

    # A. 动量质量（权重25%）：近期涨幅稳健
    today["s_mom"] = (
        today["mom5"].rank(pct=True)  * 0.35 +
        today["mom10"].rank(pct=True) * 0.35 +
        today["mom20"].rank(pct=True) * 0.30
    )

    # B. 量能质量（权重30%）：成交量稳步放大
    amt_growth = (today["amt10"] / today["amt30"].replace(0, np.nan))
    today["s_vol"] = (
        amt_growth.rank(pct=True) * 0.40 +
        today["vol_trend"].rank(pct=True) * 0.35 +
        today["vol_ratio"].clip(1, 5).rank(pct=True) * 0.25
    )

    # C. 技术共振（权重30%）：多指标同向
    today["s_tech"] = (
        today["kdj_cross"].rank(pct=True) * 0.25 +     # KDJ金叉
        today["macd_up"].rank(pct=True)   * 0.25 +     # MACD上升
        today["macd_cross0"].rank(pct=True) * 0.20 +   # MACD刚上穿零轴
        today["shrink_rb"].rank(pct=True)  * 0.15 +    # 缩量回调放量启动
        (1 - today["kdj_K"].rank(pct=True)) * 0.15     # KDJ低位（反向）
    )

    # D. 趋势位置（权重10%）：在强势回调位
    today["s_pos"] = today["near_high60"].rank(pct=True)

    # E. 低波动（权重5%）
    today["s_stab"] = 1.0 - today["vol10"].rank(pct=True)

    today["score"] = (
        today["s_mom"]  * 0.25 +
        today["s_vol"]  * 0.30 +
        today["s_tech"] * 0.30 +
        today["s_pos"]  * 0.10 +
        today["s_stab"] * 0.05
    )

    # 过滤低分股（命中率保证）
    today = today[today["score"] >= CFG.MIN_SCORE]
    today = today[~today["code"].isin(held_codes)]
    return today.sort_values("score", ascending=False)


# ─────────────────────────────────────────────
# 止损计算
# ─────────────────────────────────────────────
def calc_stop(buy_price, atr):
    atr_stop   = buy_price - CFG.ATR_MULT * atr
    fixed_stop = buy_price * (1 + CFG.STOP_LOSS_MAX)
    return max(atr_stop, fixed_stop)   # 取较高（更紧），fixed作最大亏损兜底


# ─────────────────────────────────────────────
# 回测引擎（T+1买入，双向滑点）
# ─────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, df, index_df, start_date, end_date):
        self.df       = df
        self.idx      = index_df
        self.t_start  = start_date
        self.t_end    = end_date
        self.trades   = []
        self.equity   = []

    def _do_sell(self, h, raw_price, date, reason, cash, shares=None):
        shares = shares if shares is not None else h["shares"]
        ratio  = shares / h["tot_shares"]
        cost_p = h["tot_cost"] * ratio
        price  = raw_price * (1 - CFG.SLIPPAGE)
        rev    = shares * price * (1 - CFG.COMMISSION - CFG.SELL_TAX)
        profit = rev - cost_p
        self.trades.append(dict(
            code=h["code"], buy_date=h["buy_date"], sell_date=date,
            buy_price=round(h["buy_price"],2), sell_price=round(price,2),
            shares=shares, profit=round(profit,2), reason=reason
        ))
        return cash + rev

    def run(self):
        dates = sorted(self.df["date"].unique())
        dates = [d for d in dates if self.t_start <= d <= self.t_end]
        if not dates:
            print("⚠️ 回测区间无数据")
            return

        cash, holdings = float(CFG.TOTAL_CAPITAL), []
        print(f"🔬 回测: {dates[0]} → {dates[-1]} ({len(dates)}日)")

        for i, today in enumerate(dates):
            # ── 持仓处理 ──
            keep = []
            for h in holdings:
                hist = self.df[
                    (self.df["code"]==h["code"]) &
                    (self.df["date"]<=today)
                ].sort_values("date")
                if len(hist) < 2:
                    keep.append(h); continue

                last   = hist.iloc[-1]
                hdays  = len(hist[hist["date"] >= h["buy_date"]])
                c_cl   = float(last["close"])
                c_hi   = float(last["high"])
                c_lo   = float(last["low"])
                h["peak"] = max(h["peak"], c_hi)

                sold = False

                if c_lo <= h["stop"]:                             # ① 止损
                    cash = self._do_sell(h, h["stop"], today, "止损", cash)
                    sold = True
                elif (h["peak"] > h["buy_price"]*(1+CFG.TAKE1_PCT*0.7) and
                      c_cl <= h["peak"]*(1-CFG.TRAILING_PCT)):   # ② 移动止盈
                    cash = self._do_sell(h, c_cl, today, "移动止盈", cash)
                    sold = True
                elif float(last.get("consec_below",0) or 0) >= 3: # ③ 跌破MA10
                    cash = self._do_sell(h, c_cl, today, "跌破MA10", cash)
                    sold = True
                elif hdays >= CFG.MAX_HOLD_DAYS:                  # ④ 到期
                    cash = self._do_sell(h, c_cl, today, "到期清仓", cash)
                    sold = True
                elif (not h.get("t1") and
                      c_hi >= h["buy_price"]*(1+CFG.TAKE1_PCT)):  # ⑤ 止盈1
                    half = int(h["shares"]*0.5/100)*100
                    if half >= 100:
                        cash = self._do_sell(h, h["buy_price"]*(1+CFG.TAKE1_PCT),
                                             today, "止盈+6%", cash, half)
                        h["shares"] -= half
                        h["t1"] = True
                        h["stop"] = max(h["stop"], h["buy_price"]*1.005)  # 保本止损
                    keep.append(h); continue
                elif (h.get("t1") and
                      c_hi >= h["buy_price"]*(1+CFG.TAKE2_PCT)):  # ⑥ 止盈2
                    cash = self._do_sell(h, h["buy_price"]*(1+CFG.TAKE2_PCT),
                                         today, "止盈+12%", cash)
                    sold = True

                if not sold:
                    keep.append(h)
            holdings = keep

            # ── 买入（T+1开盘价）──
            mkt = get_market_score(self.idx, today)
            if mkt >= CFG.MARKET_MIN_SCORE and len(holdings) < CFG.MAX_POSITIONS:
                cands = select_stocks(self.df, today, {h["code"] for h in holdings})
                for _, row in cands.iterrows():
                    if len(holdings) >= CFG.MAX_POSITIONS:
                        break
                    # T+1：用next_open（已预计算，无未来函数问题）
                    if pd.isna(row.get("next_open")) or pd.isna(row.get("next_date")):
                        continue
                    buy_px = float(row["next_open"]) * (1 + CFG.SLIPPAGE)
                    # 跳空高开>3%放弃
                    if buy_px > float(row["close"]) * 1.035:
                        continue
                    budget = min(cash * 0.95, CFG.TOTAL_CAPITAL * CFG.POSITION_PCT)
                    shares = int(budget / buy_px / 100) * 100
                    if shares < 100:
                        continue
                    cost = shares * buy_px * (1 + CFG.COMMISSION)
                    if cost > cash:
                        continue
                    atr_v = float(row.get("atr", buy_px * 0.02) or buy_px * 0.02)
                    cash -= cost
                    holdings.append(dict(
                        code=row["code"],
                        buy_date=str(row["next_date"]),
                        buy_price=buy_px,
                        shares=shares,
                        tot_shares=shares,
                        tot_cost=cost,
                        peak=buy_px,
                        stop=calc_stop(buy_px, atr_v),
                        t1=False
                    ))

            # ── 净值 ──
            eq = cash
            for h in holdings:
                px = self.df[
                    (self.df["code"]==h["code"]) & (self.df["date"]<=today)
                ]["close"].iloc[-1]
                eq += h["shares"] * float(px)
            self.equity.append(dict(date=today, equity=round(eq,2),
                                    cash=round(cash,2), pos=len(holdings), mkt=mkt))
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(dates)}] {today} 净值:{eq:,.0f} 持仓:{len(holdings)} 大盘:{mkt}/5")

        self._report()

    def _report(self):
        sep = "=" * 62
        print(f"\n{sep}")
        print("  📈 回测报告  V8.0 AkShare版")
        print(sep)
        if not self.equity:
            print("  ⚠️ 无数据"); return

        eq  = pd.DataFrame(self.equity)
        fin = eq["equity"].iloc[-1]
        ini = float(CFG.TOTAL_CAPITAL)
        ret = (fin - ini) / ini
        n   = len(eq)
        ann = (1 + ret) ** (250/n) - 1
        dd  = ((eq["equity"] - eq["equity"].cummax()) / eq["equity"].cummax()).min()
        eq["dr"] = eq["equity"].pct_change()
        ex  = eq["dr"] - 0.025/250
        sh  = ex.mean() / ex.std() * 250**0.5 if ex.std()>0 else 0
        cal = ann / abs(dd) if dd != 0 else 0
        idle= (eq["pos"]==0).sum()

        print(f"  初始资金  : {ini:>12,.0f} 元")
        print(f"  最终净值  : {fin:>12,.0f} 元")
        print(f"  总收益    : {ret*100:>+11.2f}%")
        print(f"  年化收益  : {ann*100:>+11.2f}%")
        print(f"  最大回撤  : {dd*100:>11.2f}%")
        print(f"  夏普比率  : {sh:>11.2f}")
        print(f"  卡玛比率  : {cal:>11.2f}")
        print(f"  空仓天数  : {idle:>4} / {n} 天 ({idle/n*100:.0f}%)")

        if self.trades:
            tr  = pd.DataFrame(self.trades)
            win = tr[tr["profit"]>0]
            los = tr[tr["profit"]<=0]
            wr  = len(win)/len(tr)
            aw  = win["profit"].mean() if len(win) else 0
            al  = los["profit"].mean() if len(los) else 0
            pr  = abs(aw/al) if al!=0 else 9.99

            print(f"\n  总交易次数: {len(tr)}")
            print(f"  胜率      : {wr*100:.1f}%")
            print(f"  平均盈利  : +{aw:,.0f} 元")
            print(f"  平均亏损  : {al:,.0f} 元")
            print(f"  盈亏比    : {pr:.2f}:1")
            print(f"\n  离场原因:")
            for r, c in tr["reason"].value_counts().items():
                mk = "✅" if "止盈" in r else ("🔴" if "止损" in r else "⚪")
                print(f"    {mk} {r:<12} {c:>3}次 ({c/len(tr)*100:.0f}%)")

            # 诊断
            print(f"\n  🩺 诊断:")
            if wr < 0.45:
                print("    ⚠️  胜率低：检查选股条件是否过于宽松")
            if pr < 1.5:
                print("    ⚠️  盈亏比低：止盈位太小或止损太紧")
            if dd < -0.20:
                print("    ⚠️  回撤过大：大盘过滤分数建议提高到4")
            if idle/n > 0.75:
                print("    ⚠️  空仓率过高：可适当降低MIN_SCORE至0.50")
            if wr >= 0.50 and pr >= 1.5 and sh >= 1.2:
                print("    ✅  策略质量良好")

            if sh>=1.5 and dd>-0.15 and wr>=0.50:
                print(f"\n  ✅ 综合评估：表现优秀，可谨慎试盘")
            elif sh>=1.0 and dd>-0.20:
                print(f"\n  ⚠️  综合评估：中等，建议继续优化")
            else:
                print(f"\n  ❌ 综合评估：较差，不建议实盘")

            tr.to_csv("trades_v8.csv", index=False, encoding="utf-8-sig")
            eq.to_csv("equity_v8.csv", index=False, encoding="utf-8-sig")
            print(f"\n  💾 trades_v8.csv / equity_v8.csv")
        else:
            print("\n  ⚠️ 无成交（建议降低 MIN_SCORE 或 MIN_AMOUNT_B）")
        print(sep)


# ─────────────────────────────────────────────
# 今日推荐
# ─────────────────────────────────────────────
def today_pick(df, stock_list, index_df):
    today_str = df["date"].max()
    mkt       = get_market_score(index_df, today_str)
    cands     = select_stocks(df, today_str, set())

    name_map  = {}
    if "code" in stock_list.columns and "name" in stock_list.columns:
        name_map = dict(zip(stock_list["code"], stock_list["name"]))

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  📅 {today_str}  今日精选 V8.0")
    mkt_str = f"{mkt}/5  {'✅可买' if mkt>=CFG.MARKET_MIN_SCORE else '🚫建议观望'}"
    print(f"  📊 大盘评分: {mkt_str}")
    print(sep)

    if cands.empty:
        print("  ⚠️ 今日无候选（大盘偏弱 或 无符合三重共振形态的股票）")
        pd.DataFrame().to_csv("picks_today.csv", index=False)
        return

    top = cands.head(5)
    print(f"\n  ※ 建议次日开盘挂单买入，跳空高开>3%放弃")
    print(f"  {'代码':<8} {'名称':<10} {'现价':>6} {'参考买入':>8} {'止损':>7} "
          f"{'目标1':>7} {'目标2':>7} {'评分':>6}")
    print(f"  {'-'*72}")

    rows = []
    for rank, (_, r) in enumerate(top.iterrows()):
        px   = float(r["close"])
        atr  = float(r.get("atr", px*0.02) or px*0.02)
        buy  = round(px*1.001, 2)
        stop = round(calc_stop(buy, atr), 2)
        t1   = round(buy*(1+CFG.TAKE1_PCT), 2)
        t2   = round(buy*(1+CFG.TAKE2_PCT), 2)
        risk = (stop-buy)/buy*100
        shrs = max(int(CFG.TOTAL_CAPITAL*CFG.POSITION_PCT/buy/100)*100, 100)
        cost = round(shrs*buy*(1+CFG.COMMISSION), 2)
        name = name_map.get(r["code"], "")
        medals = ["🥇","🥈","🥉","4️⃣ ","5️⃣ "]

        print(f"  {r['code']:<8} {name:<10} {px:>6.2f} {buy:>8.2f} {stop:>7.2f} "
              f"{t1:>7.2f} {t2:>7.2f} {r['score']:>6.3f} {medals[rank]}")

        rows.append({
            "排名": rank+1, "代码": r["code"], "名称": name,
            "现价": px, "参考买入价": buy, "止损价": stop,
            "风险": f"{risk:.1f}%",
            "目标1(+6%)": t1, "目标2(+12%)": t2,
            "建议股数": shrs, "预计成本": cost,
            "RSI": round(float(r.get("rsi",50) or 50), 1),
            "KDJ_K": round(float(r.get("kdj_K",50) or 50), 1),
            "量比": round(float(r.get("vol_ratio",1)), 2),
            "MACD_DIF": round(float(r.get("macd_dif",0) or 0), 4),
            "KDJ金叉": "是" if r.get("kdj_cross",0) else "否",
            "缩量回调放量": "是" if r.get("shrink_rb",0) else "否",
            "MA多头": "是" if r.get("ma_bull",0) else "否",
            "综合评分": round(r["score"], 3),
            "大盘评分": f"{mkt}/5"
        })

    # 首选详情
    if rows:
        b = rows[0]
        print(f"\n  🏆 首选详解：【{b['代码']} {b['名称']}】")
        print(f"  ┌─ 技术信号 {'─'*35}")
        print(f"  │  RSI      {b['RSI']:.1f}  KDJ_K {b['KDJ_K']:.1f}  量比 {b['量比']:.2f}x")
        print(f"  │  MACD DIF {b['MACD_DIF']:.4f}  KDJ金叉:{b['KDJ金叉']}  MA多头:{b['MA多头']}")
        print(f"  │  缩量回调放量: {b['缩量回调放量']}")
        print(f"  ├─ 操作计划 {'─'*35}")
        print(f"  │  买入价   {b['参考买入价']:.2f} 元 × {b['建议股数']} 股 ≈ {b['预计成本']:,.0f} 元")
        print(f"  │  止损价   {b['止损价']:.2f} 元  (风险 {b['风险']})")
        print(f"  │  目标一   {b['目标1(+6%)']:.2f} 元  (+6%，卖出50%)")
        print(f"  │  目标二   {b['目标2(+12%)']:.2f} 元  (+12%，卖出剩余)")
        print(f"  └─ 离场条件 {'─'*35}")
        print(f"     连续3日低于MA10 / 最高点回撤6% / 持满{CFG.MAX_HOLD_DAYS}日")

    print(f"\n  ⚠️  仅供参考，不构成投资建议")
    print(sep)

    pd.DataFrame(rows).to_csv("picks_today.csv", index=False, encoding="utf-8-sig")
    print(f"  💾 picks_today.csv")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    sep = "=" * 62
    print(sep)
    print("  A股选股系统 V8.0 — AkShare · 三重共振 · 高命中率")
    print("  数据源: AkShare（稳定免费）  选股: 趋势+量能+技术共振")
    print(sep)

    end_date   = datetime.now().strftime("%Y-%m-%d")
    data_start = (datetime.now() - timedelta(days=550)).strftime("%Y-%m-%d")
    bt_start   = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        stock_list = AkShareData.get_stock_list()
        if stock_list.empty:
            print("❌ 股票列表为空"); return

        hist = AkShareData.get_all_hist(stock_list, data_start, end_date, CFG.MAX_STOCKS)
        if hist.empty:
            print("❌ 历史数据为空"); return

        df = precompute_factors(hist)
        idx_df = AkShareData.get_index_hist(data_start, end_date)

        # 回测
        bt = BacktestEngine(df, idx_df, bt_start, end_date)
        bt.run()

        # 今日推荐
        today_pick(df, stock_list, idx_df)

    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
    finally:
        print("\n✅ 完成")


if __name__ == "__main__":
    main()
