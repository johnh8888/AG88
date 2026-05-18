# ================================
# A股短线实战选股系统 V2.2
# 新浪主力 + 腾讯备用 + 强化缓存
# 作者：ChatGPT Quant Upgrade（Grok优化版）
# ================================

import csv
import logging
import os
import random
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================================
# 配置参数
# ================================

TEST_MODE = True
TOP_N_CANDIDATES = 20
FINAL_HOLDINGS = 5

PRICE_MIN = 2
PRICE_MAX = 150
MIN_AMOUNT = 5e7
MA_PERIOD = 20
BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 1

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)
today = now.strftime("%Y%m%d")

# ================================
# 工具函数
# ================================

def safe_float(v, default=0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except:
        return default

def calc_open_pct(row):
    prev_close = safe_float(row.get("prev_close"))
    open_price = safe_float(row.get("open"))
    if prev_close <= 0:
        return 0
    return (open_price / prev_close - 1) * 100

# ================================
# 网络请求 + 重试
# ================================

def http_get_with_retry(url, max_tries=3, timeout=8):
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < max_tries:
                delay = (2 ** (attempt - 1)) + random.uniform(0, 1)
                logging.warning(f"HTTP请求失败，第{attempt}次重试，等待{delay:.1f}秒: {e}")
                time.sleep(delay)
    if last_exc:
        raise last_exc
    return None

# ================================
# 新浪行情接口（主力）
# ================================

def parse_sina_spot(raw_text: str):
    results = []
    lines = [line.strip() for line in raw_text.strip().split('\n') if line.strip()]
    for line in lines:
        if not line.startswith('var hq_str_'):
            continue
        try:
            code_part = line.split('=')[0].replace('var hq_str_', '').strip()
            value_str = line.split('=')[1].strip('"')
            fields = value_str.split(',')
            
            if len(fields) < 30:
                continue
                
            code = code_part[2:] if len(code_part) > 2 else code_part
            name = fields[0]
            open_price = safe_float(fields[1])
            prev_close = safe_float(fields[2])
            price = safe_float(fields[3])
            high = safe_float(fields[4])
            low = safe_float(fields[5])
            volume = safe_float(fields[8])
            amount = safe_float(fields[9])
            
            pct = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0
            amplitude = high - low if high > 0 and low > 0 else 0
            
            results.append({
                "code": code,
                "name": name,
                "price": price,
                "pct": round(pct, 2),
                "amount": amount,
                "lb": 1.0,
                "turnover": 0,
                "amplitude": round(amplitude, 2),
                "open": open_price,
                "prev_close": prev_close,
                "行业": "",
            })
        except:
            continue
    return results


def fetch_spot_data_sina():
    codes = _get_all_codes()
    if not codes:
        return pd.DataFrame()
    
    all_data = []
    batch_size = 700
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        sina_codes = [f"sh{c}" if str(c).startswith('6') else f"sz{c}" for c in batch]
        url = f"https://hq.sinajs.cn/list={','.join(sina_codes)}"
        
        try:
            resp = http_get_with_retry(url, timeout=10)
            if resp:
                data = parse_sina_spot(resp.text)
                all_data.extend(data)
            time.sleep(0.15)
        except Exception as e:
            logging.warning(f"新浪批次失败: {e}")
            continue
    
    df = pd.DataFrame(all_data)
    logging.info(f"新浪接口返回 {len(df)} 条行情")
    return df


# ================================
# 腾讯接口（备用）
# ================================

def parse_tencent_spot(raw_text: str):
    results = []
    pattern = r'v_(\w+)="([^"]+)"'
    matches = re.findall(pattern, raw_text)
    for full_code, value_str in matches:
        fields = value_str.split("~")
        if len(fields) < 30:
            continue
        try:
            code = full_code[2:] if len(full_code) > 2 else full_code
            name = fields[1]
            price = safe_float(fields[3])
            prev_close = safe_float(fields[4])
            open_price = safe_float(fields[5])
            volume = safe_float(fields[6])
            amount_idx = 36 if len(fields) > 36 else 37
            amount = safe_float(fields[amount_idx] if len(fields) > amount_idx else 0) * 10000
            if amount == 0 and volume > 0 and price > 0:
                amount = volume * price * 100
            pct = safe_float(fields[32] if len(fields) > 32 else 0)
            turnover = safe_float(fields[38] if len(fields) > 38 else 0)
            amplitude = safe_float(fields[43] if len(fields) > 43 else 0)
            lb = safe_float(fields[48] if len(fields) > 48 else 1.0)

            results.append({
                "code": code,
                "name": name,
                "price": price,
                "pct": pct,
                "amount": amount,
                "lb": lb,
                "turnover": turnover,
                "amplitude": amplitude,
                "open": open_price,
                "prev_close": prev_close,
                "行业": "",
            })
        except:
            continue
    return results


def fetch_spot_data_tencent(codes: list):
    if not codes:
        return pd.DataFrame()
    tencent_codes = [f"sh{c}" if str(c).startswith('6') else f"sz{c}" for c in codes]
    all_data = []
    batch_size = 40
    for i in range(0, len(tencent_codes), batch_size):
        batch = tencent_codes[i:i+batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)
        try:
            resp = http_get_with_retry(url, timeout=8)
            if resp:
                all_data.extend(parse_tencent_spot(resp.text))
            time.sleep(0.1)
        except:
            continue
    return pd.DataFrame(all_data)


# ================================
# 腾讯历史K线
# ================================

def get_tencent_kline(code, start_date=None, end_date=None, period="daily", adjust="qfq"):
    if code.startswith("60"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"
    klt = {"daily": "day", "weekly": "week", "monthly": "month"}.get(period, "day")
    param = f"{symbol},{klt},,,320," + ("qfq" if adjust == "qfq" else "")
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={param}"
    
    try:
        resp = http_get_with_retry(url, timeout=10)
        data = resp.json()
        klines = data.get("data", {}).get(symbol, {}).get(klt, [])
        if not klines:
            klines = data.get("data", {}).get(symbol, {}).get("qfq" + klt, [])
        if not klines:
            return None

        df = pd.DataFrame(klines, columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])
        if len(df.columns) == 6:
            df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
            if (df["收盘"] > df["最高"]).any():
                df.columns = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]

        for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["日期"] = pd.to_datetime(df["日期"])

        if start_date:
            df = df[df["日期"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["日期"] <= pd.to_datetime(end_date)]

        df = df.sort_values("日期").reset_index(drop=True)
        df["涨跌幅"] = df["收盘"].pct_change() * 100
        df["涨跌幅"] = df["涨跌幅"].fillna(0)
        return df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]
    except:
        return None


# ================================
# 历史数据缓存
# ================================

@lru_cache(maxsize=128)
def _get_stock_hist_cached(code, start_date, end_date):
    try:
        hist = get_tencent_kline(code, start_date, end_date, "daily", "qfq")
        if hist is not None and len(hist) > 5:
            return hist
    except:
        pass
    try:
        hist_ak = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start_date,
            end_date=end_date, adjust="qfq"
        )
        if not hist_ak.empty:
            return hist_ak[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]
    except:
        pass
    return None


# ================================
# 市场情绪 & 热点板块
# ================================

def get_market_emotion():
    try:
        spot = ak.stock_zh_a_spot_em()
        pct = pd.to_numeric(spot["涨跌幅"], errors="coerce")
        up_limit = len(pct[pct >= 9.7])
        down_limit = len(pct[pct <= -9.5])
        strong = len(pct[pct >= 5])
        weak = len(pct[pct <= -5])
        score = up_limit * 3 + strong - down_limit * 4 - weak
        emotion = "hot" if score >= 200 else "cold" if score <= -50 else "neutral"
        return {"emotion": emotion, "score": score, "up_limit": up_limit, "down_limit": down_limit}
    except:
        return {"emotion": "neutral", "score": 0}

def get_hot_sector_score():
    try:
        sector_df = ak.stock_board_industry_name_em()
        sector_df["涨跌幅"] = pd.to_numeric(sector_df["涨跌幅"], errors="coerce")
        sector_df = sector_df.sort_values("涨跌幅", ascending=False)
        return {row["板块名称"]: max(0, 100 - idx) for idx, row in sector_df.iterrows()}
    except:
        return {}

# ================================
# 其他核心函数
# ================================

def is_above_ma(code, period=MA_PERIOD):
    end = today
    start = (now - timedelta(days=60)).strftime("%Y%m%d")
    hist = _get_stock_hist_cached(code, start, end)
    if hist is None or hist.empty:
        return False
    close = hist["收盘"].astype(float)
    if len(close) < period:
        return False
    ma = close.rolling(period).mean().iloc[-1]
    return close.iloc[-1] > ma

def calc_leader_score(row):
    score = 0
    score += row["pct"] * 2.5
    score += (row["amount"] / 1e8) * 0.8
    score += row["turnover"] * 1.2
    score += row["lb"] * 1.5
    score -= row["amplitude"] * 0.4
    score -= abs(row.get("open_pct", 0)) * 0.5
    return score

def calc_volatility_ratio(code):
    end = today
    start = (now - timedelta(days=40)).strftime("%Y%m%d")
    hist = _get_stock_hist_cached(code, start, end)
    if hist is None or hist.empty:
        return np.nan
    close = hist["收盘"].astype(float)
    if len(close) < 20:
        return np.nan
    return close.tail(20).std() / close.tail(20).mean()

def evaluate_stock_history(code):
    end = today
    start = (now - timedelta(days=BACKTEST_LOOKBACK_DAYS)).strftime("%Y%m%d")
    hist = _get_stock_hist_cached(code, start, end)
    if hist is None or hist.empty:
        return {"signals": 0, "history_score": -999}
    pct = hist["涨跌幅"].astype(float)
    signals = len(pct[pct >= 3])
    history_score = pct.mean() * 5 + pct.max() * 2 - abs(pct.min())
    return {"signals": signals, "history_score": history_score}

# ================================
# 代码列表缓存
# ================================

_CODE_LIST_CACHE = None

def _get_all_codes():
    global _CODE_LIST_CACHE
    if _CODE_LIST_CACHE is None:
        try:
            raw = ak.stock_zh_a_spot_em(limit=None)
            _CODE_LIST_CACHE = raw["代码"].tolist()
        except:
            _CODE_LIST_CACHE = []
    return _CODE_LIST_CACHE


# ================================
# 统一行情获取（核心）
# ================================

def fetch_spot_data():
    if os.path.exists("last_spot_snapshot.csv"):
        file_time = datetime.fromtimestamp(os.path.getmtime("last_spot_snapshot.csv"), TZ_SHANGHAI)
        if (now.hour < 9 or now.hour >= 15) or (now - file_time).total_seconds() < 3600:
            logging.info("使用本地缓存行情")
            return pd.read_csv("last_spot_snapshot.csv")

    # 新浪主力
    logging.info("尝试新浪接口获取实时行情...")
    try:
        df = fetch_spot_data_sina()
        if len(df) > 1500:
            try:
                raw_ak = ak.stock_zh_a_spot_em(limit=None)
                industry_map = dict(zip(raw_ak["代码"].astype(str), raw_ak.get("行业", "未知")))
                df["行业"] = df["code"].map(industry_map).fillna("未知")
            except:
                df["行业"] = "未知"
            df.to_csv("last_spot_snapshot.csv", index=False, encoding="utf-8-sig")
            return df
    except Exception as e:
        logging.warning(f"新浪失败: {e}")

    # 腾讯备用
    logging.info("新浪失败，回退腾讯接口...")
    try:
        codes = _get_all_codes()
        df = fetch_spot_data_tencent(codes)
        if len(df) > 1500:
            try:
                raw_ak = ak.stock_zh_a_spot_em(limit=None)
                industry_map = dict(zip(raw_ak["代码"].astype(str), raw_ak.get("行业", "未知")))
                df["行业"] = df["code"].map(industry_map).fillna("未知")
            except:
                df["行业"] = "未知"
            df.to_csv("last_spot_snapshot.csv", index=False, encoding="utf-8-sig")
            return df
    except Exception as e:
        logging.warning(f"腾讯失败: {e}")

    if os.path.exists("last_spot_snapshot.csv"):
        logging.warning("使用本地缓存")
        return pd.read_csv("last_spot_snapshot.csv")

    raise RuntimeError("所有数据源均失败")


# ================================
# 主程序
# ================================

logging.info("启动A股短线选股系统 V2.2")

df = fetch_spot_data()
if df.empty:
    logging.error("无行情数据")
    sys.exit(0)

df["open_pct"] = df.apply(calc_open_pct, axis=1)
logging.info("计算波动率...")
df["volatility_ratio"] = df["code"].apply(calc_volatility_ratio)

# 过滤
ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"
df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]
df = df[df["code"].astype(str).str.startswith(("60", "00"))]
df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
df = df[df["amount"] >= MIN_AMOUNT]

logging.info(f"基础过滤后: {len(df)} 只")

# 评分
df["score"] = 0
logging.info("计算龙头评分...")
df["score"] += df.apply(calc_leader_score, axis=1)

logging.info("计算MA趋势...")
df["score"] += df["code"].apply(lambda x: 6 if is_above_ma(x, MA_PERIOD) else -3)

df["score"] -= df["volatility_ratio"].fillna(0) * 100 * 0.5

# 强势加分
strong_stock = (df["pct"] >= 7) & (df["turnover"] >= 8)
df.loc[strong_stock, "score"] += 15
df.loc[df["pct"] >= 9, "score"] += 20
df.loc[(df["turnover"] >= 10) & (df["lb"] >= 2), "score"] += 10
df.loc[(df["pct"] >= 2) & (df["open_pct"] < 0), "score"] += 6

# 板块 & 情绪
sector_map = get_hot_sector_score()
if "行业" in df.columns:
    df["sector_score"] = df["行业"].map(sector_map).fillna(0)
    df["score"] += df["sector_score"] * 0.5

emotion = get_market_emotion()
logging.info(f"市场情绪: {emotion}")
if emotion["emotion"] == "hot":
    df["score"] += df["pct"] * 1.5
elif emotion["emotion"] == "cold":
    df["score"] -= df["amplitude"] * 0.8

# 排序 + 历史回测
filtered = df.sort_values("score", ascending=False).head(80)
history_rows = []
valid_indices = []
for idx, row in filtered.iterrows():
    hist = evaluate_stock_history(row["code"])
    if hist["signals"] < BACKTEST_MIN_SIGNALS:
        continue
    history_rows.append(hist)
    valid_indices.append(idx)

candidates = filtered.loc[valid_indices].reset_index(drop=True)
hist_df = pd.DataFrame(history_rows)
candidates = pd.concat([candidates, hist_df], axis=1)

candidates["final_score"] = candidates["score"] * 0.55 + candidates["history_score"] * 0.45
final_candidates = candidates.sort_values("final_score", ascending=False).head(FINAL_HOLDINGS)

# 输出
print("\n" + "=" * 60)
print("A股实战短线选股结果 V2.2")
print("=" * 60)
for _, row in final_candidates.iterrows():
    print(f"""
股票: {row['name']} ({row['code']})
现价: {row['price']:.2f}  涨跌幅: {row['pct']:.2f}%
换手: {row.get('turnover', 0):.2f}%  量比: {row.get('lb', 1):.2f}
行业: {row.get('行业', '未知')}
实时评分: {row['score']:.1f}  历史评分: {row['history_score']:.1f}  最终: {row['final_score']:.1f}
----------------------------------------""")

final_candidates[["code", "name", "price", "pct", "turnover", "lb", "score", "history_score", "final_score"]].to_csv(
    "selected_stocks.csv", index=False, encoding="utf-8-sig"
)
logging.info("选股完成，结果已保存至 selected_stocks.csv")