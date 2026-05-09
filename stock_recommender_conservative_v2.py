import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 可调节配置（重要） ====================
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN")
TEST_MODE = os.getenv("TEST_MODE", "1") == "1"

TOTAL_CAPITAL = 20000
TRADE_RATIO = 0.6
FIX_AMOUNT = int(TOTAL_CAPITAL * TRADE_RATIO)

# 交易时间规则（北京时间）
MORNING_START, MORNING_END = 10, 10.67
AFTERNOON_START, AFTERNOON_END = 14.67, 14.92
TRADE_WEEKDAYS = {0, 1, 2, 3}  # 周一～周四

# 风控
LOW_BUY_RATIO = 0.997          # 不再用于回测模拟买入价（实盘可保留）
HARD_STOP_RATIO = -0.02
MAX_ACCEPTABLE_MARKET_DROP = -0.35

# 手续费
BUY_FEE_RATE = 0.0003
SELL_FEE_RATE = 0.0003
SELL_TAX_RATE = 0.0005
ROUND_TRIP_FEE_RATE = BUY_FEE_RATE + SELL_FEE_RATE + SELL_TAX_RATE
NET_PROFIT_TARGET_MIN = 250
NET_PROFIT_TARGET_MAX = 350

# ---------- 过滤开关 ----------
MA20_FILTER = False               # 大盘20日线过滤（近期建议关闭）
SECTOR_FILTER_ENABLED = False     # 板块排名过滤（关闭可大幅增加候选）
CONSECUTIVE_UP_ENABLED = False    # 连续小阳过滤（关闭可避免误杀）

# 筛选数值参数（微量放宽）
MIN_PRICE, MAX_PRICE = 8, 30
EARLY_MIN_PCT = 0.8
MIN_AMOUNT = 1.5e8
MIN_LB, MAX_LB = 1.0, 2.5
MIN_TURNOVER, MAX_TURNOVER = 2.0, 9.0
MIN_AMPLITUDE, MAX_AMPLITUDE = 1.8, 7.0
MAX_OPEN_PCT = 2.0
MAX_PCT = 5.5

# 尾盘条件
EOD_MAX_PCT = 1.8
EOD_MIN_PCT = -0.5
EOD_MAX_TURNOVER = 4.5
EOD_MIN_LB, EOD_MAX_LB = 0.9, 1.5
EOD_MAX_AMPLITUDE = 3.5

# 技术 & 评分
MIN_SCORE_THRESHOLD = 7.0        # 归一化后再评估的门槛（需观察调整）
TOP_N_CANDIDATES = 5
BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 3
MIN_CONSECUTIVE_UP = 3
FUNDAMENTAL_CHECK = True

# 北京时间
TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)
today = now.strftime("%Y%m%d")
week_num = now.weekday()
current_hour = now.hour + now.minute / 60.0


# ---------- 工具函数 ----------
def push(title, content):
    """推送消息"""
    if PUSHPLUS_TOKEN:
        try:
            requests.post("http://www.pushplus.plus/send",
                          json={"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "markdown"},
                          timeout=10)
        except Exception as e:
            logging.warning(f"PushPlus 推送失败: {e}")


def safe_float(value, default=0.0):
    """安全转为浮点数"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_col(df, col, default=np.nan):
    """获取列，若不存在则返回默认值序列"""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def calc_open_pct(row):
    """开盘涨幅"""
    prev = safe_float(row.get("prev_close", row.get("昨收")), 0.0)
    opn = safe_float(row.get("open", row.get("今开")), 0.0)
    if prev <= 0 or opn <= 0:
        return np.nan
    return (opn / prev - 1) * 100


def get_next_trade_day_text(base_dt):
    """获取下一交易日（yyyyMMdd），基于交易日历"""
    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        if trade_cal is not None and not trade_cal.empty:
            dates = sorted(trade_cal["trade_date"].astype(str).tolist())
            base_str = base_dt.strftime("%Y-%m-%d")
            for d in dates:
                if d > base_str:
                    return d.replace("-", "")
    except Exception as e:
        logging.warning(f"交易日历获取失败，使用简单推算: {e}")
    # 简单兜底：跳过周末
    candidate = base_dt + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.strftime("%Y%m%d")


def market_is_weak(market_pct):
    """市场是否过弱"""
    return market_pct <= MAX_ACCEPTABLE_MARKET_DROP


def calc_int_shares(capital, price):
    """计算整数手股数（100股/手）"""
    if price <= 0 or capital <= 0:
        return 0
    shares_per_lot = 100
    max_shares = int(capital // (price * shares_per_lot)) * shares_per_lot
    return max_shares


def calc_net_profit(sell_price, buy_price, capital):
    """按整数手计算净利润（扣除所有费用）"""
    shares = calc_int_shares(capital, buy_price)
    if shares == 0 or buy_price <= 0 or sell_price <= 0:
        return 0.0
    cost = shares * buy_price
    gross = (sell_price - buy_price) * shares
    fees = cost * BUY_FEE_RATE + (shares * sell_price) * (SELL_FEE_RATE + SELL_TAX_RATE)
    return gross - fees


def calc_target_sell_price(buy_price, capital, net_profit_target):
    """计算达到目标净利润所需的卖出价（近似，基于整数手）"""
    shares = calc_int_shares(capital, buy_price)
    if shares == 0 or buy_price <= 0:
        return 0.0
    cost = shares * buy_price
    # 解方程: (P - buy_price)*shares - cost*BUY_FEE_RATE - shares*P*(SELL_FEE+TAX) = net_profit
    # 化简得 P = (net_profit + cost*BUY_FEE_RATE + shares*buy_price) / (shares * (1 - SELL_FEE_RATE - SELL_TAX_RATE))
    denominator = shares * (1 - SELL_FEE_RATE - SELL_TAX_RATE)
    if denominator == 0:
        return 0.0
    numerator = net_profit_target + cost * BUY_FEE_RATE + shares * buy_price
    return round(numerator / denominator, 2)


def get_market_ma20_safe():
    """安全获取大盘20日均线状态"""
    try:
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        index_df = index_df.sort_values("date").tail(30)
        close = float(index_df["close"].iloc[-1])
        ma20 = float(index_df["close"].rolling(20).mean().iloc[-1])
        ma20_prev = float(index_df["close"].rolling(20).mean().iloc[-2])
        return close, ma20, (close > ma20 and ma20 > ma20_prev)
    except Exception as e:
        logging.warning(f"获取大盘均线失败: {e}")
        return 0, 0, True


def get_sector_rank_map():
    """获取板块涨跌幅映射"""
    try:
        sector_df = ak.stock_board_industry_name_em()
        return dict(zip(sector_df["板块名称"], sector_df["涨跌幅"]))
    except Exception as e:
        logging.warning(f"板块排名获取失败: {e}")
        return {}


def has_consecutive_mild_up(code, days=MIN_CONSECUTIVE_UP):
    """连续小阳线过滤（需开启）"""
    if not CONSECUTIVE_UP_ENABLED:
        return True
    try:
        end = (now - timedelta(days=1)).strftime("%Y%m%d")
        start = (now - timedelta(days=30)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if hist is None or hist.empty or len(hist) < days:
            return False
        recent = hist.tail(days + 5)
        pct_col = get_col(recent, "涨跌幅")
        tail = pct_col.tail(days)
        if tail.isna().any():
            return False
        if not tail.between(0.5, 4.5).all():
            return False
        if (pct_col.tail(20) < -5).any():
            return False
        return True
    except Exception as e:
        logging.warning(f"连续小阳检查失败 {code}: {e}")
        return False


def has_safe_fundamentals(code):
    """基础面过滤（净利润>0）"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if info is None or info.empty:
            return True
        info_dict = dict(zip(info["item"], info["value"]))
        return safe_float(info_dict.get("归属母公司股东的净利润", 0)) > 0
    except Exception as e:
        logging.warning(f"基本面检查失败 {code}: {e}")
        return True


def is_limit_up_down(next_open, next_high, next_low, prev_close):
    """
    判断次日是否一字涨停或跌停（无法买入/卖出）。
    返回 True 表示属于一字板，应剔除。
    """
    if prev_close <= 0 or next_open <= 0:
        return False
    limit_up = round(prev_close * 1.10, 2)
    limit_down = round(prev_close * 0.90, 2)
    # 一字涨停：开盘=涨停价，且最高=最低=开盘（无波动）
    if (abs(next_open - limit_up) < 0.01 and
        abs(next_high - limit_up) < 0.01 and
        abs(next_low - limit_up) < 0.01):
        return True
    # 一字跌停同理
    if (abs(next_open - limit_down) < 0.01 and
        abs(next_high - limit_down) < 0.01 and
        abs(next_low - limit_down) < 0.01):
        return True
    return False


def evaluate_stock_history(symbol):
    """
    历史回测：以次日开盘价作为模拟买入价，统计信号表现。
    排除一字涨停/跌停的样本。
    """
    start_date = (now - timedelta(days=BACKTEST_LOOKBACK_DAYS + 40)).strftime("%Y%m%d")
    end_date = today
    try:
        hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        logging.warning(f"获取历史数据失败 {symbol}: {e}")
        return {"signals": 0, "win_rate": 0.0, "target_250_hit_rate": 0.0,
                "target_350_hit_rate": 0.0, "avg_next_close": 0.0,
                "avg_next_high": 0.0, "avg_worst_drawdown": 0.0, "history_score": -999}

    if hist is None or hist.empty:
        return {"signals": 0, "win_rate": 0.0, "target_250_hit_rate": 0.0,
                "target_350_hit_rate": 0.0, "avg_next_close": 0.0,
                "avg_next_high": 0.0, "avg_worst_drawdown": 0.0, "history_score": -999}

    # 统一列名
    hist = hist.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "涨跌幅": "pct", "成交额": "amount",
        "换手率": "turnover", "振幅": "amplitude"
    }).copy()
    hist = hist.sort_values("date").tail(BACKTEST_LOOKBACK_DAYS).reset_index(drop=True)
    hist["open_pct"] = (hist["open"] / hist["close"].shift(1) - 1) * 100

    signals = []
    for i in range(1, len(hist) - 1):
        row = hist.iloc[i]
        nxt = hist.iloc[i + 1]

        # 当日筛选条件
        if not (EARLY_MIN_PCT <= safe_float(row["pct"]) <= MAX_PCT and
                safe_float(row["amount"]) >= MIN_AMOUNT and
                MIN_TURNOVER <= safe_float(row.get("turnover"), 0) <= MAX_TURNOVER and
                MIN_AMPLITUDE <= safe_float(row.get("amplitude"), 0) <= MAX_AMPLITUDE and
                MIN_PRICE <= safe_float(row["close"]) <= MAX_PRICE and
                safe_float(row.get("open_pct"), 999) <= MAX_OPEN_PCT):
            continue

        # 模拟买入价：次日开盘价（更贴近实盘）
        buy_price = safe_float(nxt["open"])
        if buy_price <= 0:
            continue

        next_high = safe_float(nxt["high"])
        next_low = safe_float(nxt["low"])
        next_close = safe_float(nxt["close"])
        prev_close = safe_float(row["close"])  # 当日收盘作为前收

        # 剔除一字涨跌停
        if is_limit_up_down(buy_price, next_high, next_low, prev_close):
            continue

        t250 = calc_target_sell_price(buy_price, FIX_AMOUNT, NET_PROFIT_TARGET_MIN)
        t350 = calc_target_sell_price(buy_price, FIX_AMOUNT, NET_PROFIT_TARGET_MAX)

        signals.append({
            "win": 1 if next_close > buy_price else 0,
            "target_250_hit": 1 if next_high >= t250 else 0,
            "target_350_hit": 1 if next_high >= t350 else 0,
            "next_close_ret": (next_close / buy_price - 1) * 100,
            "next_high_ret": (next_high / buy_price - 1) * 100,
            "next_low_ret": (next_low / buy_price - 1) * 100,
        })

    if not signals:
        return {"signals": 0, "win_rate": 0.0, "target_250_hit_rate": 0.0,
                "target_350_hit_rate": 0.0, "avg_next_close": 0.0,
                "avg_next_high": 0.0, "avg_worst_drawdown": 0.0, "history_score": -999}

    s = pd.DataFrame(signals)
    n_sig = len(s)
    win_r = float(s["win"].mean() * 100)
    hit250 = float(s["target_250_hit"].mean() * 100)
    hit350 = float(s["target_350_hit"].mean() * 100)
    avg_c = float(s["next_close_ret"].mean())
    avg_h = float(s["next_high_ret"].mean())
    avg_l = float(s["next_low_ret"].mean())

    # 信号数量惩罚（最多按15个满分）
    penalty = min(n_sig, 15) / 15
    score = (win_r * 0.22 + hit250 * 0.38 + hit350 * 0.22 +
             avg_c * 9.0 + avg_h * 4.5 + avg_l * 2.0) * penalty

    return {"signals": n_sig, "win_rate": win_r, "target_250_hit_rate": hit250,
            "target_350_hit_rate": hit350, "avg_next_close": avg_c,
            "avg_next_high": avg_h, "avg_worst_drawdown": avg_l, "history_score": score}


# ---------- 行情获取（双源容错） ----------
def fetch_spot_data():
    """获取实时行情，带必要列检查"""
    required_cols = ["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率", "振幅", "今开", "昨收"]
    for attempt in range(1, 3):
        try:
            logging.info(f"东方财富行情，第{attempt}次尝试...")
            raw = ak.stock_zh_a_spot_em()
            if raw is None or raw.empty:
                continue
            # 检查必要列
            missing = [c for c in required_cols if c not in raw.columns]
            if missing:
                logging.error(f"东方财富缺失必要列: {missing}")
                continue
            standard = pd.DataFrame()
            standard["code"] = raw["代码"]
            standard["name"] = raw["名称"]
            standard["price"] = raw["最新价"].astype(float)
            standard["pct"] = raw["涨跌幅"].astype(float)
            standard["amount"] = raw["成交额"].astype(float)
            standard["lb"] = raw.get("量比", pd.Series([1.0]*len(raw))).astype(float)
            standard["turnover"] = raw["换手率"].astype(float)
            standard["amplitude"] = raw["振幅"].astype(float)
            standard["open"] = raw["今开"].astype(float)
            standard["prev_close"] = raw["昨收"].astype(float)
            # 行业列
            for col in ["行业", "所属行业"]:
                if col in raw.columns:
                    standard[col] = raw[col]
            logging.info("东方财富行情获取成功")
            return standard
        except Exception as e:
            logging.error(f"东方财富行情失败: {e}")
            time.sleep(3)

    # 新浪回退
    try:
        logging.info("尝试新浪行情...")
        raw = ak.stock_zh_a_spot()
        if raw is None or raw.empty:
            return pd.DataFrame()
        standard = pd.DataFrame()
        standard["code"] = raw["代码"]
        standard["name"] = raw["名称"]
        standard["price"] = pd.to_numeric(raw["最新价"], errors="coerce")
        standard["pct"] = pd.to_numeric(raw["涨跌幅"], errors="coerce")
        standard["amount"] = pd.to_numeric(raw["成交额"], errors="coerce")
        standard["lb"] = 1.0
        standard["turnover"] = pd.to_numeric(raw.get("换手率", pd.Series([0]*len(raw))), errors="coerce")
        standard["amplitude"] = pd.to_numeric(raw.get("振幅", pd.Series([0]*len(raw))), errors="coerce")
        standard["open"] = pd.to_numeric(raw.get("今开", raw["最新价"]), errors="coerce")
        standard["prev_close"] = pd.to_numeric(raw.get("昨收", raw["最新价"]), errors="coerce")
        logging.info("新浪行情获取成功")
        return standard
    except Exception as e:
        logging.error(f"新浪行情获取失败: {e}")
        return pd.DataFrame()


# ==================== 主流程 ====================
if not TEST_MODE:
    in_morning = (week_num in TRADE_WEEKDAYS) and (MORNING_START <= current_hour < MORNING_END)
    in_afternoon = (week_num in TRADE_WEEKDAYS) and (AFTERNOON_START <= current_hour < AFTERNOON_END)
    if not (in_morning or in_afternoon):
        logging.info("非允许交易时段，退出")
        sys.exit(0)
else:
    in_morning = True
    in_afternoon = False

if MA20_FILTER:
    _, _, ma_safe = get_market_ma20_safe()
    if not ma_safe and not TEST_MODE:
        logging.info("大盘不在20日线上方或均线未向上，暂停开仓")
        sys.exit(0)
    if not ma_safe and TEST_MODE:
        logging.warning("⚠️ 测试模式：大盘均线不满足，但仍继续运行")

raw_df = fetch_spot_data()
if raw_df.empty:
    logging.error("所有行情接口均不可用，退出")
    sys.exit(0)

market_pct = 0.0
name_col = "name"
if name_col in raw_df.columns:
    sh_mask = raw_df[name_col].str.contains("上证指数|上证综合指数", na=False)
    if sh_mask.any():
        market_pct = safe_float(raw_df.loc[sh_mask, "pct"].iloc[0], 0.0)

if market_is_weak(market_pct):
    logging.info(f"市场跌幅{market_pct:.2f}%过深，空仓")
    sys.exit(0)

df = raw_df.copy()
df["open_pct"] = df.apply(calc_open_pct, axis=1)
for col_name in ["turnover", "amplitude", "open_pct"]:
    df[col_name] = get_col(df, col_name, np.nan)

# 剔除 ST/新股等
ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"
df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]
df = df[(df["code"].astype(str).str.startswith(("60", "00")))]

# 板块过滤（可关闭）
if SECTOR_FILTER_ENABLED:
    sector_map = get_sector_rank_map()
    if sector_map:
        sector_pcts = sorted(sector_map.values(), reverse=True)
        cutoff_idx = int(len(sector_pcts) * 0.4)
        cutoff_pct = sector_pcts[cutoff_idx] if sector_pcts else -100
        sector_col = None
        for col_name in ["行业", "所属行业"]:
            if col_name in df.columns:
                sector_col = col_name
                break
        if sector_col:
            df["sector_pct"] = df[sector_col].map(sector_map)
            df = df[df["sector_pct"].notna() & (df["sector_pct"] >= cutoff_pct)]
    else:
        logging.warning("板块排名数据为空，跳过板块过滤")

# 早盘筛选
if in_morning:
    filtered = df[
        (df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE) &
        (df["pct"] >= EARLY_MIN_PCT) & (df["pct"] <= MAX_PCT) &
        (df["amount"] >= MIN_AMOUNT) &
        (df["lb"] >= MIN_LB) & (df["lb"] <= MAX_LB) &
        (df["turnover"] >= MIN_TURNOVER) & (df["turnover"] <= MAX_TURNOVER) &
        (df["amplitude"] >= MIN_AMPLITUDE) & (df["amplitude"] <= MAX_AMPLITUDE) &
        (df["open_pct"] <= MAX_OPEN_PCT)
    ].copy()
    if CONSECUTIVE_UP_ENABLED:
        filtered = filtered[filtered["code"].apply(has_consecutive_mild_up)]
    logging.info(f"早盘初步筛选出 {len(filtered)} 只")
else:
    filtered = pd.DataFrame()

if (filtered.empty and not in_morning) or in_afternoon:
    logging.info("切换到尾盘防御模式...")
    filtered_eod = df[
        (df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE) &
        (df["pct"] >= EOD_MIN_PCT) & (df["pct"] <= EOD_MAX_PCT) &
        (df["amount"] >= MIN_AMOUNT) &
        (df["lb"] >= EOD_MIN_LB) & (df["lb"] <= EOD_MAX_LB) &
        (df["turnover"] <= EOD_MAX_TURNOVER) &
        (df["amplitude"] <= EOD_MAX_AMPLITUDE)
    ].copy()
    if CONSECUTIVE_UP_ENABLED:
        filtered_eod = filtered_eod[filtered_eod["code"].apply(has_consecutive_mild_up)]
    filtered = filtered_eod
    logging.info(f"尾盘初步筛选出 {len(filtered)} 只")

if filtered.empty:
    logging.info("今日无标的，空仓")
    sys.exit(0)

# 实时评分
filtered["realtime_score"] = (
    filtered["pct"] * 1.3 +
    filtered["lb"] * 2.0 +
    (filtered["amount"] / 1e8) * 0.7 +
    filtered["turnover"] * 0.7 -
    filtered["amplitude"] * 0.4 -
    filtered["open_pct"].fillna(0) * 0.5
)

candidates = filtered.sort_values("realtime_score", ascending=False).head(TOP_N_CANDIDATES).copy()

# 逐个回测
history_rows, valid_idx = [], []
for idx, row in candidates.iterrows():
    code = str(row["code"])
    if FUNDAMENTAL_CHECK and not has_safe_fundamentals(code):
        continue
    time.sleep(0.3)  # 防止请求过快
    hist_res = evaluate_stock_history(code)
    history_rows.append(hist_res)
    valid_idx.append(idx)

if not valid_idx:
    logging.info("基本面或历史样本不足，空仓")
    sys.exit(0)

candidates = candidates.loc[valid_idx].reset_index(drop=True)
candidates = pd.concat([candidates, pd.DataFrame(history_rows)], axis=1)
candidates = candidates[candidates["signals"] >= BACKTEST_MIN_SIGNALS].copy()
if candidates.empty:
    logging.info("历史样本不足，空仓")
    sys.exit(0)

# ---------- 评分归一化（min-max） ----------
def min_max_norm(series):
    """min-max 归一化到 [0,1]，若常数则返回 0.5"""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)

candidates["norm_real"] = min_max_norm(candidates["realtime_score"])
candidates["norm_hist"] = min_max_norm(candidates["history_score"])
candidates["final_score"] = candidates["norm_real"] * 0.28 + candidates["norm_hist"] * 0.72 * 100  # 放大到百分制以便阈值

candidates = candidates[candidates["final_score"] >= MIN_SCORE_THRESHOLD]
candidates = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)

if candidates.empty:
    logging.info("评分不足，空仓")
    sys.exit(0)

stock = candidates.iloc[0]
p = safe_float(stock["price"])
buy_ref = round(p * LOW_BUY_RATIO, 2)  # 实盘低吸参考价（仍保留）
stop = round(buy_ref * (1 + HARD_STOP_RATIO), 2)
next_sell_day = get_next_trade_day_text(now)

target_sell_min = calc_target_sell_price(buy_ref, FIX_AMOUNT, NET_PROFIT_TARGET_MIN)
target_sell_max = calc_target_sell_price(buy_ref, FIX_AMOUNT, NET_PROFIT_TARGET_MAX)
net_profit_min = round(calc_net_profit(target_sell_min, buy_ref, FIX_AMOUNT), 2)
net_profit_max = round(calc_net_profit(target_sell_max, buy_ref, FIX_AMOUNT), 2)
net_stop_loss = round(calc_net_profit(stop, buy_ref, FIX_AMOUNT), 2)

# 准备推送内容
best = candidates.head(3)[["code", "name", "pct", "signals", "win_rate",
                           "target_250_hit_rate", "target_350_hit_rate", "final_score"]].copy()
lines = []
for _, row in best.iterrows():
    lines.append(
        f"- {row['name']}({row['code']})｜现涨 {safe_float(row['pct']):.2f}%｜"
        f"胜率 {safe_float(row['win_rate']):.1f}%｜"
        f"250命中 {safe_float(row['target_250_hit_rate']):.1f}%｜"
        f"350命中 {safe_float(row['target_350_hit_rate']):.1f}%"
    )

content = f"""
## {today} 低吸稳赢候选
- 股票：{stock['name']}({stock['code']})
- 现价：{p:.2f}
- 计划低吸买入参考：{buy_ref}
- 止盈区间：{target_sell_min} ~ {target_sell_max}
- 硬止损价格：{stop}
- 卖出窗口：{next_sell_day} 起
- 预估净利：{net_profit_min} ~ {net_profit_max}
- 止损预估亏损：{net_stop_loss}
- 历史样本：{int(stock['signals'])}
- 次日收盘胜率：{safe_float(stock['win_rate']):.1f}%
- 目标250命中率：{safe_float(stock['target_250_hit_rate']):.1f}%
- 目标350命中率：{safe_float(stock['target_350_hit_rate']):.1f}%

### 前3候选
{os.linesep.join(lines)}
""".strip()

push("低吸稳赢候选", content)
print(f"今日推荐：{stock['name']}({stock['code']})")

# 写入日志
log_file = "trade_log.csv"
log_row = {
    "date": today, "time_window": "morning" if in_morning else "afternoon",
    "code": str(stock["code"]), "name": str(stock["name"]),
    "price_at_signal": p, "buy_ref": buy_ref, "stop": stop,
    "target_min": target_sell_min, "target_max": target_sell_max,
    "net_profit_min": net_profit_min, "net_profit_max": net_profit_max,
    "signals": int(stock["signals"]), "win_rate": round(safe_float(stock["win_rate"]), 2),
    "hit_250": round(safe_float(stock["target_250_hit_rate"]), 2),
    "hit_350": round(safe_float(stock["target_350_hit_rate"]), 2),
    "final_score": round(safe_float(stock["final_score"]), 2),
    "market_pct": market_pct, "weekday": week_num
}
file_exists = os.path.isfile(log_file)
try:
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_row)
    logging.info(f"交易日志已写入：{log_file}")
except Exception as e:
    logging.error(f"日志写入失败: {e}")