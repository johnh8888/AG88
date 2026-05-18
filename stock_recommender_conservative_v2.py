import csv
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 配置 ====================
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN")
TEST_MODE = os.getenv("TEST_MODE", "1") == "1"
RUN_MODE = os.getenv("RUN_MODE", "diagnostic").lower()  # diagnostic / paper / live

TOTAL_CAPITAL = 20000
TRADE_RATIO = 0.6
FIX_AMOUNT = int(TOTAL_CAPITAL * TRADE_RATIO)

MORNING_START, MORNING_END = 10, 10.67
AFTERNOON_START, AFTERNOON_END = 14.67, 14.92
TRADE_WEEKDAYS = {0, 1, 2, 3}

LOW_BUY_RATIO = 0.997
HARD_STOP_RATIO = -0.02
MAX_ACCEPTABLE_MARKET_DROP = -0.35

BUY_FEE_RATE = 0.0003
SELL_FEE_RATE = 0.0003
SELL_TAX_RATE = 0.0005
ROUND_TRIP_FEE_RATE = BUY_FEE_RATE + SELL_FEE_RATE + SELL_TAX_RATE
NET_PROFIT_TARGET_MIN = 250
NET_PROFIT_TARGET_MAX = 350

# ---------- 过滤开关 ----------
MA20_FILTER = False
SECTOR_FILTER_ENABLED = False
CONSECUTIVE_UP_ENABLED = False

INDIVIDUAL_MA_FILTER = True
MA_PERIOD = 20
MAIN_INFLOW_FILTER = True
RECENT_LIMIT_DOWN_FILTER = True
LIMIT_DOWN_LOOKBACK = 5

# ---------- 放宽版筛选参数 ----------
PRICE_MIN_CAP = 3.0
PRICE_MAX_CAP = 120.0
EARLY_MIN_PCT = 0.3
MIN_AMOUNT = 1.0e8
MIN_LB, MAX_LB = 0.8, 3.0
MIN_TURNOVER, MAX_TURNOVER = 1.0, 9.0
MIN_AMPLITUDE, MAX_AMPLITUDE = 1.0, 7.0
MAX_OPEN_PCT = 2.5
MAX_PCT = 5.5
VOL_LOOKBACK_DAYS = 20
MAX_VOLATILITY_RATIO = 0.065
MAX_GAP_OPEN_RATIO = 0.045

# 尾盘条件放宽
EOD_MAX_PCT = 2.5
EOD_MIN_PCT = -1.5
EOD_MAX_TURNOVER = 5.5
EOD_MIN_LB, EOD_MAX_LB = 0.7, 2.0
EOD_MAX_AMPLITUDE = 4.0

# 评分与样本放宽
MIN_SCORE_THRESHOLD = 5.5
TOP_N_CANDIDATES = 5
FINAL_HOLDINGS = 3
BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 2
MIN_CONSECUTIVE_UP = 3
FUNDAMENTAL_CHECK = True

MAX_PORTFOLIO_RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS_RATIO = 0.02
MIN_EXPECTED_NET_PROFIT = 80
TRAILING_STOP_RATIO = 0.01

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)
today = now.strftime("%Y%m%d")
week_num = now.weekday()
current_hour = now.hour + now.minute / 60.0

MIN_LB_DYNAMIC, MAX_LB_DYNAMIC = MIN_LB, MAX_LB
EOD_MIN_LB_DYNAMIC, EOD_MAX_LB_DYNAMIC = EOD_MIN_LB, EOD_MAX_LB

# ---------- 工具函数 ----------
def push(title, content):
    if PUSHPLUS_TOKEN:
        try:
            requests.post("http://www.pushplus.plus/send",
                          json={"token": PUSHPLUS_TOKEN, "title": title,
                                "content": content, "template": "markdown"}, timeout=10)
        except Exception as e:
            logging.warning(f"推送失败: {e}")

def safe_float(value, default=0.0):
    try:
        if pd.isna(value): return default
        return float(value)
    except: return default

def get_col(df, col, default=np.nan):
    return df[col] if col in df.columns else pd.Series([default]*len(df), index=df.index)

def calc_open_pct(row):
    prev = safe_float(row.get("prev_close", row.get("昨收")), 0.0)
    opn = safe_float(row.get("open", row.get("今开")), 0.0)
    if prev <= 0 or opn <= 0: return np.nan
    return (opn / prev - 1) * 100


def calc_volatility_ratio(code, lookback_days=VOL_LOOKBACK_DAYS):
    try:
        end = now.strftime("%Y%m%d")
        start = (now - timedelta(days=lookback_days + 40)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if hist is None or hist.empty or len(hist) < lookback_days:
            return np.nan
        close = hist.sort_values("日期")["收盘"].astype(float).tail(lookback_days)
        if close.mean() <= 0:
            return np.nan
        return float(close.std(ddof=0) / close.mean())
    except Exception as e:
        logging.debug(f"波动率计算失败 {code}: {e}")
        return np.nan


def get_recent_market_state():
    try:
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        if index_df is None or index_df.empty:
            return {"market_trend": 0.0, "market_volatility": np.nan, "market_bias": "unknown"}
        index_df = index_df.sort_values("date").tail(30).copy()
        close = index_df["close"].astype(float)
        trend_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0.0
        volatility = float(close.tail(20).std(ddof=0) / close.tail(20).mean()) if len(close) >= 20 and close.tail(20).mean() > 0 else np.nan
        if trend_5 >= 1.0:
            bias = "bull"
        elif trend_5 <= -1.0:
            bias = "bear"
        else:
            bias = "neutral"
        return {"market_trend": trend_5, "market_volatility": volatility, "market_bias": bias}
    except Exception as e:
        logging.warning(f"市场状态获取失败: {e}")
        return {"market_trend": 0.0, "market_volatility": np.nan, "market_bias": "unknown"}


def get_dynamic_price_bounds(market_state):
    if market_state.get("market_bias") == "bull":
        return 5.0, 160.0
    if market_state.get("market_bias") == "bear":
        return 8.0, 35.0
    return PRICE_MIN_CAP, PRICE_MAX_CAP


def get_dynamic_volatility_cap(market_state):
    vol = market_state.get("market_volatility")
    if pd.isna(vol):
        return MAX_VOLATILITY_RATIO
    if market_state.get("market_bias") == "bull":
        return min(MAX_VOLATILITY_RATIO * 1.15, 0.085)
    if market_state.get("market_bias") == "bear":
        return MAX_VOLATILITY_RATIO * 0.8
    return MAX_VOLATILITY_RATIO


def get_dynamic_gap_cap(market_state):
    if market_state.get("market_bias") == "bull":
        return 0.06
    if market_state.get("market_bias") == "bear":
        return 0.03
    return MAX_GAP_OPEN_RATIO


def calculate_position_size(capital, stop_price, entry_price):
    if capital <= 0 or entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
        return 0
    risk_per_share = entry_price - stop_price
    max_risk_cash = capital * MAX_PORTFOLIO_RISK_PER_TRADE
    shares = int(max_risk_cash // risk_per_share)
    return max((shares // 100) * 100, 0)


def get_dynamic_exit_prices(entry_price, capital):
    hard_stop = round(entry_price * 0.98, 2)
    trailing_stop = round(entry_price * (1 - TRAILING_STOP_RATIO), 2)
    stop_price = min(hard_stop, trailing_stop)
    target_min = round(entry_price * 1.03, 2)
    target_max = round(entry_price * 1.05, 2)
    size = calculate_position_size(capital, stop_price, entry_price)
    return {
        "stop_price": stop_price,
        "target_min": target_min,
        "target_max": target_max,
        "position_size": size,
    }


def record_filter_diag(diag, stage, before, after, note=""):
    diag.append({"stage": stage, "before": before, "after": after, "drop": before - after, "note": note})


def log_filter_diag(diag):
    if not diag:
        return
    logging.info("===== 逐层筛选诊断 =====")
    for item in diag:
        msg = f"{item['stage']}: {item['before']} -> {item['after']} (减少 {item['drop']})"
        if item.get("note"):
            msg += f" | {item['note']}"
        logging.info(msg)
    logging.info("========================")

def get_next_trade_day_text(base_dt):
    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        if trade_cal is not None and not trade_cal.empty:
            dates = sorted(trade_cal["trade_date"].astype(str).tolist())
            base_str = base_dt.strftime("%Y-%m-%d")
            for d in dates:
                if d > base_str: return d.replace("-", "")
    except Exception as e:
        logging.warning(f"交易日历获取失败: {e}")
    candidate = base_dt + timedelta(days=1)
    while candidate.weekday() >= 5: candidate += timedelta(days=1)
    return candidate.strftime("%Y%m%d")

def market_is_weak(market_pct):
    return market_pct <= MAX_ACCEPTABLE_MARKET_DROP

def calc_int_shares(capital, price):
    if price <= 0 or capital <= 0: return 0
    return int(capital // (price * 100)) * 100

def calc_net_profit(sell_price, buy_price, capital):
    shares = calc_int_shares(capital, buy_price)
    if shares == 0 or buy_price <= 0 or sell_price <= 0: return 0.0
    cost = shares * buy_price
    gross = (sell_price - buy_price) * shares
    fees = cost * BUY_FEE_RATE + (shares * sell_price) * (SELL_FEE_RATE + SELL_TAX_RATE)
    return gross - fees

def calc_target_sell_price(buy_price, capital, net_profit_target):
    shares = calc_int_shares(capital, buy_price)
    if shares == 0 or buy_price <= 0: return 0.0
    cost = shares * buy_price
    denom = shares * (1 - SELL_FEE_RATE - SELL_TAX_RATE)
    if denom == 0: return 0.0
    return round((net_profit_target + cost * BUY_FEE_RATE + shares * buy_price) / denom, 2)

def get_market_ma20_safe():
    try:
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        index_df = index_df.sort_values("date").tail(30)
        close = float(index_df["close"].iloc[-1])
        ma20 = float(index_df["close"].rolling(20).mean().iloc[-1])
        return close, ma20, close > ma20
    except Exception as e:
        logging.warning(f"大盘均线失败: {e}")
        return 0, 0, True

def get_sector_rank_map():
    try:
        sector_df = ak.stock_board_industry_name_em()
        return dict(zip(sector_df["板块名称"], sector_df["涨跌幅"]))
    except Exception as e:
        logging.warning(f"板块排名获取失败: {e}")
        return {}

def has_consecutive_mild_up(code, days=MIN_CONSECUTIVE_UP):
    if not CONSECUTIVE_UP_ENABLED: return True
    try:
        end = (now - timedelta(days=1)).strftime("%Y%m%d")
        start = (now - timedelta(days=30)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if hist is None or hist.empty or len(hist) < days: return False
        recent = hist.tail(days + 5)
        pct_col = get_col(recent, "涨跌幅")
        tail = pct_col.tail(days)
        if tail.isna().any(): return False
        if not tail.between(0.5, 4.5).all(): return False
        if (pct_col.tail(20) < -5).any(): return False
        return True
    except Exception as e:
        logging.debug(f"连续小阳检查失败 {code}: {e}")
        return False

def has_safe_fundamentals(code):
    try:
        info = ak.stock_individual_info_em(symbol=code)
        if info is None or info.empty: return True
        info_dict = dict(zip(info["item"], info["value"]))
        return safe_float(info_dict.get("归属母公司股东的净利润", 0)) > 0
    except Exception as e:
        logging.debug(f"基本面检查失败 {code}: {e}")
        return True

def is_above_ma(code, period=MA_PERIOD):
    if not INDIVIDUAL_MA_FILTER: return True
    try:
        end = now.strftime("%Y%m%d")
        start = (now - timedelta(days=80)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if hist is None or hist.empty or len(hist) < period: return True
        close = hist["收盘"].astype(float)
        ma = close.rolling(period).mean().iloc[-1]
        return close.iloc[-1] > ma
    except Exception as e:
        logging.debug(f"MA过滤失败 {code}: {e}")
        return True

def has_main_inflow(code):
    if not MAIN_INFLOW_FILTER: return True
    try:
        market = "sh" if code.startswith("6") else "sz"
        flow = ak.stock_individual_fund_flow(stock=code, market=market)
        if flow is None or flow.empty: return True
        return float(flow["主力净流入"].iloc[-1]) > 0
    except Exception as e:
        logging.debug(f"资金流过滤失败 {code}: {e}")
        return True

def has_recent_limit_down(code, days=LIMIT_DOWN_LOOKBACK):
    if not RECENT_LIMIT_DOWN_FILTER: return False
    try:
        end = now.strftime("%Y%m%d")
        start = (now - timedelta(days=60)).strftime("%Y%m%d")
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
        if hist is None or hist.empty: return False
        hist = hist.sort_values("日期").tail(days)
        close = hist["收盘"].astype(float)
        preclose = hist["昨收"].astype(float) if "昨收" in hist.columns else close.shift(1)
        for c, ld in zip(close, preclose):
            if abs(c - round(ld * 0.9, 2)) < 0.01: return True
        return False
    except Exception as e:
        logging.debug(f"跌停检查失败 {code}: {e}")
        return False

def is_suspended(row):
    amount = safe_float(row.get("amount", row.get("成交额")), 0)
    turnover = safe_float(row.get("turnover", row.get("换手率")), 0)
    return amount <= 0 and turnover <= 0

def is_limit_up_down(next_open, next_high, next_low, prev_close):
    if prev_close <= 0 or next_open <= 0: return False
    up = round(prev_close * 1.10, 2)
    dn = round(prev_close * 0.90, 2)
    if (abs(next_open - up) < 0.01 and abs(next_high - up) < 0.01 and abs(next_low - up) < 0.01): return True
    if (abs(next_open - dn) < 0.01 and abs(next_high - dn) < 0.01 and abs(next_low - dn) < 0.01): return True
    return False

def evaluate_stock_history(symbol):
    start_date = (now - timedelta(days=BACKTEST_LOOKBACK_DAYS + 40)).strftime("%Y%m%d")
    end_date = today
    try:
        hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        logging.warning(f"历史数据失败 {symbol}: {e}")
        return default_history_result()
    if hist is None or hist.empty: return default_history_result()
    hist = hist.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
                                "最低": "low", "涨跌幅": "pct", "成交额": "amount",
                                "换手率": "turnover", "振幅": "amplitude"}).copy()
    hist = hist.sort_values("date").tail(BACKTEST_LOOKBACK_DAYS).reset_index(drop=True)
    if len(hist) < 2: return default_history_result()
    hist["open_pct"] = (hist["open"] / hist["close"].shift(1) - 1) * 100

    signals = []
    for i in range(1, len(hist) - 1):
        row = hist.iloc[i]; nxt = hist.iloc[i+1]
        if is_suspended(row) or is_suspended(nxt): continue
        if not (EARLY_MIN_PCT <= safe_float(row["pct"]) <= MAX_PCT and
                safe_float(row["amount"]) >= MIN_AMOUNT and
                MIN_TURNOVER <= safe_float(row.get("turnover"), 0) <= MAX_TURNOVER and
                MIN_AMPLITUDE <= safe_float(row.get("amplitude"), 0) <= MAX_AMPLITUDE and
                PRICE_MIN_CAP <= safe_float(row["close"]) <= PRICE_MAX_CAP and
                safe_float(row.get("open_pct"), 999) <= MAX_OPEN_PCT):
            continue
        buy_price = safe_float(nxt["open"])
        if buy_price <= 0: continue
        if is_limit_up_down(buy_price, safe_float(nxt["high"]), safe_float(nxt["low"]), safe_float(row["close"])):
            continue
        next_high = safe_float(nxt["high"]); next_low = safe_float(nxt["low"]); next_close = safe_float(nxt["close"])
        t250 = calc_target_sell_price(buy_price, FIX_AMOUNT, NET_PROFIT_TARGET_MIN)
        t350 = calc_target_sell_price(buy_price, FIX_AMOUNT, NET_PROFIT_TARGET_MAX)
        signals.append({
            "win": 1 if next_close > buy_price else 0,
            "target_250_hit": 1 if next_high >= t250 else 0,
            "target_350_hit": 1 if next_high >= t350 else 0,
            "next_close_ret": (next_close/buy_price-1)*100,
            "next_high_ret": (next_high/buy_price-1)*100,
            "next_low_ret": (next_low/buy_price-1)*100,
        })
    if not signals: return default_history_result()
    s = pd.DataFrame(signals)
    n_sig = len(s)
    win_r = float(s["win"].mean()*100)
    hit250 = float(s["target_250_hit"].mean()*100)
    hit350 = float(s["target_350_hit"].mean()*100)
    avg_c = float(s["next_close_ret"].mean())
    avg_h = float(s["next_high_ret"].mean())
    avg_l = float(s["next_low_ret"].mean())
    penalty = min(n_sig, 15) / 15
    score = (win_r*0.22 + hit250*0.38 + hit350*0.22 + avg_c*9.0 + avg_h*4.5 + avg_l*2.0) * penalty
    return {"signals": n_sig, "win_rate": win_r, "target_250_hit_rate": hit250,
            "target_350_hit_rate": hit350, "avg_next_close": avg_c, "avg_next_high": avg_h,
            "avg_worst_drawdown": avg_l, "history_score": score}

def default_history_result():
    return {"signals": 0, "win_rate": 0.0, "target_250_hit_rate": 0.0,
            "target_350_hit_rate": 0.0, "avg_next_close": 0.0, "avg_next_high": 0.0,
            "avg_worst_drawdown": 0.0, "history_score": -999}

def quantile_norm(series, n_quantiles=5):
    if series.nunique() <= 1: return pd.Series(0.5, index=series.index)
    try:
        qs = [series.quantile(i/n_quantiles) for i in range(n_quantiles+1)]
        def map_q(x):
            for i, q in enumerate(qs):
                if x <= q: return i/n_quantiles
            return 1.0
        return series.apply(map_q)
    except: return (series-series.min())/(series.max()-series.min()+1e-9)

# ---------- 行情获取 ----------
def fetch_spot_data():
    for attempt in range(1, 4):
        try:
            logging.info(f"东方财富实时行情，第{attempt}次尝试...")
            raw = ak.stock_zh_a_spot_em()
            if raw is not None and not raw.empty:
                required_cols = ["代码","名称","最新价","涨跌幅","成交额","换手率","振幅","今开","昨收"]
                if all(c in raw.columns for c in required_cols):
                    df = pd.DataFrame()
                    df["code"] = raw["代码"]; df["name"] = raw["名称"]
                    df["price"] = raw["最新价"].astype(float); df["pct"] = raw["涨跌幅"].astype(float)
                    df["amount"] = raw["成交额"].astype(float)
                    df["lb"] = raw.get("量比", pd.Series([1.0]*len(raw))).astype(float)
                    df["turnover"] = raw["换手率"].astype(float); df["amplitude"] = raw["振幅"].astype(float)
                    df["open"] = raw["今开"].astype(float); df["prev_close"] = raw["昨收"].astype(float)
                    for col in ["行业","所属行业"]:
                        if col in raw.columns: df[col] = raw[col]
                    logging.info("✅ 东方财富实时数据获取成功")
                    return df
                else:
                    missing = [c for c in required_cols if c not in raw.columns]
                    logging.warning(f"东方财富缺失字段: {missing}")
            time.sleep(2)
        except Exception as e:
            logging.error(f"东方财富异常: {e}")
            time.sleep(3)

    try:
        logging.info("尝试新浪实时行情...")
        raw = ak.stock_zh_a_spot()
        if raw is not None and not raw.empty:
            df = pd.DataFrame()
            df["code"] = raw["代码"]; df["name"] = raw["名称"]
            df["price"] = pd.to_numeric(raw["最新价"], errors="coerce")
            df["pct"] = pd.to_numeric(raw["涨跌幅"], errors="coerce")
            df["amount"] = pd.to_numeric(raw["成交额"], errors="coerce")
            df["lb"] = 1.0
            df["turnover"] = pd.to_numeric(raw.get("换手率", pd.Series([0]*len(raw))), errors="coerce")
            df["amplitude"] = pd.to_numeric(raw.get("振幅", pd.Series([0]*len(raw))), errors="coerce")
            df["open"] = pd.to_numeric(raw.get("今开", raw["最新价"]), errors="coerce")
            df["prev_close"] = pd.to_numeric(raw.get("昨收", raw["最新价"]), errors="coerce")
            logging.info("✅ 新浪实时数据获取成功（量比默认为1.0）")
            return df
    except Exception as e:
        logging.error(f"新浪行情失败: {e}")

    if TEST_MODE:
        logging.warning("实时数据源均失败，启用历史日线模拟...")
        return _generate_historical_snapshot()
    else:
        logging.error("所有行情源失败，且非测试模式，退出")
        return pd.DataFrame()

def _generate_historical_snapshot():
    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        if trade_cal is None or trade_cal.empty: raise Exception("交易日历不可用")
        all_dates = sorted(trade_cal["trade_date"].astype(str).tolist())
        today_str = now.strftime("%Y-%m-%d")
        past_dates = [d for d in all_dates if d <= today_str]
        if not past_dates: raise Exception("无历史交易日")
        last_trade_day = past_dates[-1].replace("-", "")
        logging.info(f"模拟日期: {last_trade_day}")
        try:
            raw = ak.stock_zh_a_spot_em()
            if raw is not None and not raw.empty:
                df = pd.DataFrame()
                df["code"] = raw["代码"]; df["name"] = raw["名称"]
                df["price"] = raw["最新价"].astype(float); df["pct"] = raw["涨跌幅"].astype(float)
                df["amount"] = raw["成交额"].astype(float)
                df["lb"] = raw.get("量比", pd.Series([1.0]*len(raw))).astype(float)
                df["turnover"] = raw["换手率"].astype(float); df["amplitude"] = raw["振幅"].astype(float)
                df["open"] = raw["今开"].astype(float); df["prev_close"] = raw["昨收"].astype(float)
                logging.info("历史快照模拟成功")
                return df
        except: pass
        logging.error("无法生成历史快照")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"历史快照异常: {e}")
        return pd.DataFrame()

# ==================== 主流程 ====================
if not TEST_MODE:
    in_morning = (week_num in TRADE_WEEKDAYS) and (MORNING_START <= current_hour < MORNING_END)
    in_afternoon = (week_num in TRADE_WEEKDAYS) and (AFTERNOON_START <= current_hour < AFTERNOON_END)
    if not (in_morning or in_afternoon):
        logging.info("非允许交易时段，退出")
        sys.exit(0)
else:
    in_morning = (week_num in TRADE_WEEKDAYS) and (MORNING_START <= current_hour < MORNING_END)
    in_afternoon = (week_num in TRADE_WEEKDAYS) and (AFTERNOON_START <= current_hour < AFTERNOON_END)
    if not (in_morning or in_afternoon):
        in_morning = False
        in_afternoon = True
        logging.info("测试模式：非交易时段，强制启用尾盘防御模式")

suggested_position_ratio = 0.6
market_state = get_recent_market_state()
dynamic_price_min, dynamic_price_max = get_dynamic_price_bounds(market_state)
dynamic_vol_cap = get_dynamic_volatility_cap(market_state)
dynamic_gap_cap = get_dynamic_gap_cap(market_state)
logging.info(
    f"市场状态：趋势{market_state.get('market_trend', 0.0):+.2f}% / "
    f"波动{safe_float(market_state.get('market_volatility'), 0.0) * 100:.2f}% / "
    f"{market_state.get('market_bias', 'unknown')}"
)
if RUN_MODE == "diagnostic":
    logging.info("运行模式：诊断模式，仅输出筛选结果与诊断信息")
elif RUN_MODE == "paper":
    logging.info("运行模式：模拟盘模式，输出推荐但不做外部推送依赖")
else:
    logging.info("运行模式：实盘模式")
if MA20_FILTER:
    _, _, ma_safe = get_market_ma20_safe()
    if not ma_safe and not TEST_MODE:
        logging.info("大盘不在20日线上，暂停开仓")
        sys.exit(0)
    suggested_position_ratio = 0.3 if not ma_safe else 0.6
else:
    try:
        close, ma20, _ = get_market_ma20_safe()
        if close > 0 and ma20 > 0:
            suggested_position_ratio = 0.6 if close > ma20 else 0.3
    except: pass

raw_df = fetch_spot_data()
if raw_df.empty:
    logging.error("行情获取失败，退出")
    sys.exit(0)

# 新浪数据检测
is_sina_data = False
if raw_df is not None and not raw_df.empty and 'lb' in raw_df.columns:
    if raw_df['lb'].nunique() == 1 and raw_df['lb'].iloc[0] == 1.0:
        is_sina_data = True
        logging.warning("检测到量比数据全部为1.0（新浪数据），自动放宽量比限制")
        MIN_LB_DYNAMIC, MAX_LB_DYNAMIC = 0.5, 5.0
        EOD_MIN_LB_DYNAMIC, EOD_MAX_LB_DYNAMIC = 0.5, 5.0
    else:
        MIN_LB_DYNAMIC, MAX_LB_DYNAMIC = MIN_LB, MAX_LB
        EOD_MIN_LB_DYNAMIC, EOD_MAX_LB_DYNAMIC = EOD_MIN_LB, EOD_MAX_LB
else:
    MIN_LB_DYNAMIC, MAX_LB_DYNAMIC = MIN_LB, MAX_LB
    EOD_MIN_LB_DYNAMIC, EOD_MAX_LB_DYNAMIC = EOD_MIN_LB, EOD_MAX_LB

market_pct = 0.0
if "name" in raw_df.columns:
    sh_mask = raw_df["name"].str.contains("上证指数|上证综合指数", na=False)
    if sh_mask.any(): market_pct = safe_float(raw_df.loc[sh_mask, "pct"].iloc[0], 0.0)
if market_is_weak(market_pct):
    logging.info(f"市场跌幅 {market_pct:.2f}% 过深，空仓")
    sys.exit(0)

if dynamic_price_min > 0 and dynamic_price_max > 0:
    logging.info(f"动态价格区间：{dynamic_price_min:.2f} ~ {dynamic_price_max:.2f}")


df = raw_df.copy()
df["open_pct"] = df.apply(calc_open_pct, axis=1)
for col in ["turnover","amplitude","open_pct"]: df[col] = get_col(df, col, np.nan)
df["volatility_ratio"] = df["code"].apply(calc_volatility_ratio)
df["gap_open_ratio"] = df["open_pct"].fillna(999).abs() / 100.0

diag = []
record_filter_diag(diag, "原始行情", len(df), len(df), "实时行情数据")

ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"
before = len(df)
df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]
record_filter_diag(diag, "剔除ST/退市/异常名称", before, len(df))

before = len(df)
df = df[(df["code"].astype(str).str.startswith(("60","00")))]
record_filter_diag(diag, "仅保留沪深主板", before, len(df))

before = len(df)
df = df[(df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max)]
record_filter_diag(diag, "动态价格过滤", before, len(df), f"{dynamic_price_min:.2f}~{dynamic_price_max:.2f}")

before = len(df)
df = df[(df["volatility_ratio"].isna()) | (df["volatility_ratio"] <= dynamic_vol_cap)]
record_filter_diag(diag, "近20日波动率过滤", before, len(df), f"上限 {dynamic_vol_cap:.2%}")

before = len(df)
df = df[(df["gap_open_ratio"].isna()) | (df["gap_open_ratio"] <= dynamic_gap_cap)]
record_filter_diag(diag, "开盘跳空过滤", before, len(df), f"上限 {dynamic_gap_cap:.2%}")

if SECTOR_FILTER_ENABLED:
    try:
        sector_map = get_sector_rank_map()
        if sector_map:
            sector_pcts = sorted(sector_map.values(), reverse=True)
            cutoff_idx = int(len(sector_pcts)*0.4)
            cutoff_pct = sector_pcts[cutoff_idx] if sector_pcts else -100
            if "行业" in df.columns or "所属行业" in df.columns:
                col = "行业" if "行业" in df.columns else "所属行业"
                df["sector_pct"] = df[col].map(sector_map)
                df = df[df["sector_pct"].notna() & (df["sector_pct"] >= cutoff_pct)]
    except Exception as e:
        logging.warning(f"板块过滤失败: {e}")

# 早盘筛选
if in_morning:
    before = len(df)
    filtered = df[
        (df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max) &
        (df["pct"] >= EARLY_MIN_PCT) & (df["pct"] <= MAX_PCT) &
        (df["amount"] >= MIN_AMOUNT) &
        (df["lb"] >= MIN_LB_DYNAMIC) & (df["lb"] <= MAX_LB_DYNAMIC) &
        (df["turnover"] >= MIN_TURNOVER) & (df["turnover"] <= MAX_TURNOVER) &
        (df["amplitude"] >= MIN_AMPLITUDE) & (df["amplitude"] <= MAX_AMPLITUDE) &
        (df["open_pct"] <= MAX_OPEN_PCT) &
        (df["volatility_ratio"].isna() | (df["volatility_ratio"] <= dynamic_vol_cap))
    ].copy()
    if CONSECUTIVE_UP_ENABLED:
        filtered = filtered[filtered["code"].apply(has_consecutive_mild_up)]
    record_filter_diag(diag, "早盘初筛", before, len(filtered), "强势/温和放量/非高开")
else:
    filtered = pd.DataFrame()

# 尾盘 + 增强 fallback
if (filtered.empty and not in_morning) or in_afternoon:
    logging.info("切换到尾盘防御模式...")
    before = len(df)
    filtered = df[
        (df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max) &
        (df["pct"] >= EOD_MIN_PCT) & (df["pct"] <= EOD_MAX_PCT) &
        (df["amount"] >= MIN_AMOUNT) &
        (df["lb"] >= EOD_MIN_LB_DYNAMIC) & (df["lb"] <= EOD_MAX_LB_DYNAMIC) &
        (df["turnover"] <= EOD_MAX_TURNOVER) &
        (df["amplitude"] <= EOD_MAX_AMPLITUDE) &
        (df["volatility_ratio"].isna() | (df["volatility_ratio"] <= dynamic_vol_cap))
    ].copy()
    if CONSECUTIVE_UP_ENABLED:
        filtered = filtered[filtered["code"].apply(has_consecutive_mild_up)]
    record_filter_diag(diag, "尾盘筛选", before, len(filtered), "低波动/低跳空")

    if filtered.empty and is_sina_data:
        logging.warning("尾盘空，开始数据诊断及fallback")
        price_ok = len(df[(df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max)])
        amount_ok = len(df[df["amount"] >= MIN_AMOUNT])
        pct_range = len(df[(df["pct"] >= -3) & (df["pct"] <= 3)])
        diag_msg = (f"## 数据诊断 ({today})\n"
                    f"- 数据源：新浪\n"
                    f"- 动态价格区间：{dynamic_price_min:.2f}~{dynamic_price_max:.2f}\n"
                    f"- 成交额≥1亿：{amount_ok} 只\n"
                    f"- 涨跌幅-3%~3%：{pct_range} 只\n"
                    f"- 价格min/max：{df['price'].min():.2f}/{df['price'].max():.2f}\n"
                    f"- 成交额min/max：{df['amount'].min():.0f}/{df['amount'].max():.0f}\n"
                    f"- 涨跌幅min/max：{df['pct'].min():.2f}%/{df['pct'].max():.2f}%")
        push("选股系统数据诊断", diag_msg)

        logging.warning("启用新浪fallback：涨跌幅-3%~3%")
        before = len(df)
        filtered = df[
            (df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max) &
            (df["pct"] >= -3) & (df["pct"] <= 3) &
            (df["amount"] >= MIN_AMOUNT) &
            (df["lb"] >= 0.5) & (df["lb"] <= 5.0) &
            (df["volatility_ratio"].isna() | (df["volatility_ratio"] <= dynamic_vol_cap))
        ].copy()
        record_filter_diag(diag, "新浪fallback", before, len(filtered), "放宽涨跌幅、保留成交额")
        logging.info(f"fallback筛选 {len(filtered)} 只")

        if filtered.empty:
            logging.warning("最终兜底：取消成交额限制，仅价格与涨跌幅")
            before = len(df)
            filtered = df[
                (df["price"] >= dynamic_price_min) & (df["price"] <= dynamic_price_max) &
                (df["pct"] >= -10) & (df["pct"] <= 10) &
                (df["amount"] > 0)
            ].copy()
            record_filter_diag(diag, "最终兜底", before, len(filtered), "仅保留价格/涨跌幅/成交额>0")
            logging.info(f"最终兜底筛选 {len(filtered)} 只")

log_filter_diag(diag)

if filtered.empty:
    logging.info("初筛无标的，空仓退出")
    sys.exit(0)

if 'code' not in filtered.columns:
    logging.error("filtered 缺失 'code' 列")
    sys.exit(1)

# 增强过滤
before = len(filtered)
if INDIVIDUAL_MA_FILTER:
    filtered = filtered[filtered["code"].apply(lambda x: is_above_ma(x, MA_PERIOD))]
    record_filter_diag(diag, "均线过滤", before, len(filtered), f"MA{MA_PERIOD} 上方")
    if filtered.empty: logging.info("退出"); log_filter_diag(diag); sys.exit(0)

before = len(filtered)
if MAIN_INFLOW_FILTER:
    filtered = filtered[filtered["code"].apply(has_main_inflow)]
    record_filter_diag(diag, "主力资金过滤", before, len(filtered), "主力净流入为正")
    if filtered.empty: logging.info("退出"); log_filter_diag(diag); sys.exit(0)

before = len(filtered)
if RECENT_LIMIT_DOWN_FILTER:
    filtered = filtered[~filtered["code"].apply(has_recent_limit_down)]
    record_filter_diag(diag, "近跌停过滤", before, len(filtered), f"近{LIMIT_DOWN_LOOKBACK}日无跌停")
    if filtered.empty: logging.info("退出"); log_filter_diag(diag); sys.exit(0)

log_filter_diag(diag)

# 优化后的实时评分
filtered["realtime_score"] = (
    filtered["pct"] * 1.8 + filtered["lb"] * 1.5 + (filtered["amount"] / 1e8) * 1.0 +
    filtered["turnover"] * 0.8 - filtered["amplitude"] * 0.3 - filtered["open_pct"].fillna(0) * 0.3
)
candidates = filtered.sort_values("realtime_score", ascending=False).head(TOP_N_CANDIDATES).copy()

history_rows, valid_idx = [], []
for idx, row in candidates.iterrows():
    code = str(row["code"])
    if FUNDAMENTAL_CHECK and not has_safe_fundamentals(code): continue
    time.sleep(0.3)
    hist_res = evaluate_stock_history(code)
    history_rows.append(hist_res)
    valid_idx.append(idx)

if not valid_idx: logging.info("基本面/历史样本不足，退出"); sys.exit(0)

candidates = candidates.loc[valid_idx].reset_index(drop=True)
candidates = pd.concat([candidates, pd.DataFrame(history_rows)], axis=1)
candidates = candidates[candidates["signals"] >= BACKTEST_MIN_SIGNALS].copy()
if candidates.empty: logging.info("历史样本不足，退出"); sys.exit(0)

candidates["norm_real"] = quantile_norm(candidates["realtime_score"])
candidates["norm_hist"] = quantile_norm(candidates["history_score"])
candidates["final_score"] = candidates["norm_real"] * 0.28 + candidates["norm_hist"] * 0.72 * 100
candidates = candidates[candidates["final_score"] >= MIN_SCORE_THRESHOLD]
candidates = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
if candidates.empty: logging.info("评分不足，退出"); sys.exit(0)

final_candidates = candidates.head(FINAL_HOLDINGS).copy()
logging.info(f"最终入选 {len(final_candidates)} 只")

market_state = get_recent_market_state()
market_bias = market_state.get("market_bias", "unknown")
market_trend = safe_float(market_state.get("market_trend"), 0.0)
market_volatility = safe_float(market_state.get("market_volatility"), 0.0)

defensive_mode = market_bias == "bear" or market_trend < -1.0
capital_per_trade = FIX_AMOUNT if not defensive_mode else max(int(FIX_AMOUNT * 0.5), 1000)

trade_plans = []
for _, stock in final_candidates.iterrows():
    p = safe_float(stock["price"])
    buy_ref = round(p * LOW_BUY_RATIO, 2)
    exit_plan = get_dynamic_exit_prices(buy_ref, capital_per_trade)
    stop = exit_plan["stop_price"]
    target_min = exit_plan["target_min"]
    target_max = exit_plan["target_max"]
    position_size = exit_plan["position_size"]
    net_min = round(calc_net_profit(target_min, buy_ref, capital_per_trade), 2)
    net_max = round(calc_net_profit(target_max, buy_ref, capital_per_trade), 2)
    net_stop = round(calc_net_profit(stop, buy_ref, capital_per_trade), 2)
    atr_stop = None
    try:
        hist_atr = ak.stock_zh_a_hist(symbol=stock["code"], period="daily",
                                      start_date=(now - timedelta(days=30)).strftime("%Y%m%d"),
                                      end_date=today, adjust="qfq")
        if hist_atr is not None and not hist_atr.empty:
            high = hist_atr["最高"].astype(float); low = hist_atr["最低"].astype(float)
            close_atr = hist_atr["收盘"].astype(float)
            tr = np.maximum(high - low, np.abs(high - close_atr.shift(1)), np.abs(low - close_atr.shift(1)))
            atr14 = tr.tail(14).mean()
            if not np.isnan(atr14): atr_stop = round(buy_ref - 1.5 * atr14, 2)
    except: pass
    trade_plans.append({
        "name": stock["name"], "code": stock["code"], "price": p,
        "buy_ref": buy_ref, "stop_hard": stop, "stop_atr": atr_stop,
        "target_min": target_min, "target_max": target_max,
        "net_min": net_min, "net_max": net_max, "net_stop": net_stop,
        "position_size": position_size,
        "signals": int(stock["signals"]), "win_rate": safe_float(stock["win_rate"]),
        "hit_250": safe_float(stock["target_250_hit_rate"]),
        "hit_350": safe_float(stock["target_350_hit_rate"]),
        "final_score": safe_float(stock["final_score"]), "market_pct": market_pct,
        "market_bias": market_bias, "market_trend": market_trend, "market_volatility": market_volatility
    })

push_lines = [f"## {today} 低吸风控 · 组合推荐", ""]
push_lines.append(f"- **大盘涨跌**：{market_pct:+.2f}%")
push_lines.append(f"- **市场趋势（5日）**：{market_trend:+.2f}%")
push_lines.append(f"- **市场波动率**：{market_volatility:.2%}" if market_volatility else "- **市场波动率**：N/A")
push_lines.append(f"- **市场偏向**：{market_bias}")
push_lines.append(f"- **建议总仓位**：{suggested_position_ratio*100:.0f}%")
push_lines.append(f"- **单票风险上限**：{MAX_PORTFOLIO_RISK_PER_TRADE*100:.1f}%")
push_lines.append(f"- **单票预估净利门槛**：≥{MIN_EXPECTED_NET_PROFIT} 元")
push_lines.append("- **止盈策略**：分批止盈 / 移动止盈（回撤1%离场）")
if defensive_mode:
    push_lines.append("- ⚠️ 当前进入防御模式，仓位自动减半")
if is_sina_data:
    push_lines.append("- ⚠️ 数据源：新浪（部分字段缺失，条件已放宽）")
push_lines.append("")
for plan in trade_plans:
    if plan["net_min"] < MIN_EXPECTED_NET_PROFIT:
        continue
    push_lines.append(f"### {plan['name']}({plan['code']})")
    push_lines.append(f"- 现价：{plan['price']:.2f}")
    push_lines.append(f"- 买入参考：{plan['buy_ref']}")
    push_lines.append(f"- 止盈区间：{plan['target_min']} ~ {plan['target_max']}")
    stop_info = f"- 硬止损：{plan['stop_hard']}" + (f" / ATR止损：{plan['stop_atr']}" if plan['stop_atr'] else "")
    push_lines.append(stop_info)
    push_lines.append(f"- 建议股数：{plan['position_size']} 股")
    push_lines.append(f"- 预估净利：{plan['net_min']} ~ {plan['net_max']} 元")
    push_lines.append(f"- 止损亏损：{plan['net_stop']} 元")
    push_lines.append(f"- 信号：{plan['signals']}次 | 胜率：{plan['win_rate']:.1f}% | "
                     f"250命中：{plan['hit_250']:.1f}% | 350命中：{plan['hit_350']:.1f}%\n")
push("\n".join(push_lines).strip())
for plan in trade_plans:
    if plan["net_min"] < MIN_EXPECTED_NET_PROFIT:
        continue
    print(f"推荐 {plan['name']}({plan['code']}) 买入参考 {plan['buy_ref']}，建议股数 {plan['position_size']}，预估净利 {plan['net_min']}~{plan['net_max']} 元")

log_file = "trade_log.csv"
file_exists = os.path.isfile(log_file)
try:
    with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
        writer = None
        for plan in trade_plans:
            log_row = {"date": today, "time_window": "morning" if in_morning else "afternoon",
                       "code": plan["code"], "name": plan["name"], "price": plan["price"],
                       "buy_ref": plan["buy_ref"], "stop_hard": plan["stop_hard"], "stop_atr": plan["stop_atr"],
                       "target_min": plan["target_min"], "target_max": plan["target_max"],
                       "net_min": plan["net_min"], "net_max": plan["net_max"], "signals": plan["signals"],
                       "win_rate": plan["win_rate"], "hit_250": plan["hit_250"], "hit_350": plan["hit_350"],
                       "final_score": plan["final_score"], "market_pct": market_pct,
                       "position_ratio": suggested_position_ratio,
                       "market_bias": market_bias,
                       "market_trend": market_trend,
                       "market_volatility": market_volatility,
                       "position_size": plan["position_size"],
                       "defensive_mode": defensive_mode}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
                if not file_exists: writer.writeheader()
            writer.writerow(log_row)
    logging.info(f"日志已写入 {log_file}")
except Exception as e:
    logging.error(f"日志写入失败: {e}")
