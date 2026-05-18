# ================================
# A股短线实战选股系统 V3.1
# 完全双源容错 + 无AKShare实时行情仍可运行
# ================================

import csv
import logging
import os
import random
import re
import signal
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

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

MIN_AMOUNT = 5e7                # 成交额 > 5千万

MA_PERIOD = 20

BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 1

MAX_CONSECUTIVE_COLD_DAYS = 2   # 连续冰点天数则空仓

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)
today = now.strftime("%Y%m%d")

# 数据源优先级
DATA_SOURCE_ORDER = ["tencent", "akshare"]

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

def http_get_with_retry(url, max_tries=3, timeout=5):
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
                logging.warning(f"HTTP请求失败，第{attempt}次重试: {e}")
                time.sleep(delay)
    raise last_exc

# ================================
# 腾讯行情接口
# ================================

def parse_tencent_spot(raw_text):
    results = []
    pattern = r'v_(\w+)="([^"]+)"'
    matches = re.findall(pattern, raw_text)
    for match in matches:
        full_code, value_str = match
        fields = value_str.split("~")
        if len(fields) < 40:
            continue
        try:
            code = full_code[2:]
            name = fields[1]
            price = float(fields[3])
            prev_close = float(fields[4])
            open_price = float(fields[5])
            volume = float(fields[6])
            amount = float(fields[37]) * 10000 if fields[37] else 0
            if amount == 0 and volume > 0:
                amount = volume * price * 100
            pct = float(fields[32])
            turnover = float(fields[38])
            amplitude = float(fields[43])
            lb = float(fields[48]) if len(fields) > 48 and fields[48] else 1.0
            sector = ""
            results.append({
                "code": code, "name": name, "price": price, "pct": pct,
                "amount": amount, "lb": lb, "turnover": turnover,
                "amplitude": amplitude, "open": open_price,
                "prev_close": prev_close, "行业": sector,
            })
        except (ValueError, IndexError):
            continue
    return results

def fetch_spot_data_tencent(codes):
    if not codes:
        return pd.DataFrame()
    tencent_codes = [f"sh{c}" if c.startswith("60") else f"sz{c}" for c in codes]
    all_data = []
    batch_size = 50
    for i in range(0, len(tencent_codes), batch_size):
        batch = tencent_codes[i:i+batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)
        resp = http_get_with_retry(url, timeout=8)
        all_data.extend(parse_tencent_spot(resp.text))
        time.sleep(0.1)
    return pd.DataFrame(all_data)

def get_tencent_kline(code, start_date=None, end_date=None, period="daily", adjust="qfq"):
    symbol = f"sh{code}" if code.startswith("60") else f"sz{code}"
    klt = {"daily": "day", "weekly": "week", "monthly": "month"}.get(period, "day")
    param = f"{symbol},{klt},,,320," + ("qfq" if adjust == "qfq" else "")
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={param}"
    resp = http_get_with_retry(url, timeout=10)
    data = resp.json()
    klines = data.get("data", {}).get(symbol, {}).get(klt, [])
    if not klines and adjust == "qfq":
        klines = data.get("data", {}).get(symbol, {}).get("qfq" + klt, [])
    if not klines:
        return None
    df = pd.DataFrame(klines, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
    if len(df.columns) >= 6:
        df.columns = ["日期", "col1", "col2", "col3", "col4", "col5"][:len(df.columns)]
        if len(df.columns) == 6:
            if (pd.to_numeric(df["col2"], errors="coerce") > pd.to_numeric(df["col3"], errors="coerce")).any():
                df.columns = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
            else:
                df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
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

# ================================
# 历史数据统一缓存
# ================================

@lru_cache(maxsize=128)
def _get_stock_hist_cached(code, start_date, end_date):
    try:
        hist = get_tencent_kline(code, start_date, end_date, "daily", "qfq")
        if hist is not None and len(hist) > 5:
            return hist
    except Exception as e:
        logging.debug(f"腾讯K线失败 {code}: {e}")
    try:
        hist_ak = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start_date, end_date=end_date, adjust="qfq"
        )
        if hist_ak is not None and not hist_ak.empty:
            df = hist_ak.rename(columns={
                "日期": "日期", "开盘": "开盘", "收盘": "收盘",
                "最高": "最高", "最低": "最低", "成交量": "成交量", "涨跌幅": "涨跌幅"
            })
            for col in ["开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]
    except:
        pass
    return None

# ================================
# 市场情绪（基于已有行情数据）
# ================================

def get_market_emotion(df=None):
    """优先使用传入的DataFrame，否则使用本地快照，都失败返回中性"""
    if df is not None and not df.empty and "pct" in df.columns:
        pct = pd.to_numeric(df["pct"], errors="coerce")
    else:
        # 尝试从本地快照计算
        if os.path.exists("last_spot_snapshot.csv"):
            try:
                snap = pd.read_csv("last_spot_snapshot.csv")
                pct = pd.to_numeric(snap["pct"], errors="coerce")
            except:
                return {"emotion": "neutral", "score": 0}
        else:
            return {"emotion": "neutral", "score": 0}

    up_limit = len(pct[pct >= 9.7])
    down_limit = len(pct[pct <= -9.5])
    strong = len(pct[pct >= 5])
    weak = len(pct[pct <= -5])
    score = up_limit * 3 + strong - down_limit * 4 - weak
    if score >= 200:
        emotion = "hot"
    elif score <= -50:
        emotion = "cold"
    else:
        emotion = "neutral"
    return {"emotion": emotion, "score": score, "up_limit": up_limit, "down_limit": down_limit}

# ================================
# 热点板块系统（独立，网络失败不影响）
# ================================

def get_hot_sector_score():
    try:
        sector_df = ak.stock_board_industry_name_em()
        sector_df["涨跌幅"] = pd.to_numeric(sector_df["涨跌幅"], errors="coerce")
        sector_df = sector_df.sort_values("涨跌幅", ascending=False)
        score_map = {}
        for idx, row in sector_df.iterrows():
            score_map[row["板块名称"]] = max(0, 100 - idx)
        return score_map
    except:
        return {}

# ================================
# MA均线
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

# ================================
# 龙头评分
# ================================

def calc_leader_score(row):
    score = 0
    score += row["pct"] * 2.5
    score += (row["amount"] / 1e8) * 0.8
    score += row["turnover"] * 1.2
    score += row["lb"] * 1.5
    score -= row["amplitude"] * 0.4
    score -= abs(row["open_pct"]) * 0.5
    return score

# ================================
# 波动率
# ================================

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

# ================================
# 历史模式评分（去未来函数）
# ================================

def evaluate_stock_history(code):
    end = today
    start = (now - timedelta(days=BACKTEST_LOOKBACK_DAYS)).strftime("%Y%m%d")
    hist = _get_stock_hist_cached(code, start, end)
    if hist is None or hist.empty:
        return {"signals": 0, "history_score": -999}
    hist["ma20"] = hist["收盘"].rolling(20).mean()
    hist["ma_slope"] = hist["ma20"].diff(3)
    hist["vol_ma5"] = hist["成交量"].rolling(5).mean()
    hist["vol_ratio"] = hist["成交量"] / hist["vol_ma5"]
    bull_days = ((hist["ma_slope"] > 0) & (hist["vol_ratio"] > 1.2)).sum()
    history_score = bull_days * 2 + hist["涨跌幅"].std() * (-0.1)
    return {"signals": bull_days, "history_score": history_score}

# ================================
# 代码列表获取（四级容错）
# ================================

# 内置100只高流动性主板蓝筹股，确保在无任何网络时仍有足够候选池
_BUILTIN_CODES = [
    "600519", "000858", "601318", "000333", "600036", "601166", "600276", "600030", "000651", "002415",
    "601012", "600900", "000001", "600809", "002475", "601888", "600887", "601398", "601939", "601288",
    "600585", "600690", "000725", "002714", "601668", "600048", "600050", "601688", "600309", "600031",
    "000002", "601328", "600000", "601601", "600837", "000338", "002230", "601899", "600570", "000776",
    "300059", "601857", "600104", "600009", "601006", "600016", "600029", "601211", "601628", "601319",
    "601336", "600015", "601818", "601988", "600028", "601088", "600019", "601766", "000100", "002352",
    "601390", "600346", "000063", "600588", "600547", "600438", "300015", "002142", "600196", "600111",
    "002027", "601878", "600926", "600703", "000876", "000538", "002456", "601225", "600150", "601919",
    "002353", "600143", "002460", "300122", "000661", "300124", "002241", "600183", "600745", "000977",
    "002049", "603259", "601360", "600018", "000568", "000625", "002304", "600132", "000895", "002007",
    "600893", "600519", "000858", "002271", "600436", "600809", "000596", "000799", "002142", "000423",
]

_CODE_LIST_CACHE = None

def _get_all_codes():
    """四级容错：AKShare spot -> 本地快照 -> 内置列表，永不报空"""
    global _CODE_LIST_CACHE
    if _CODE_LIST_CACHE is not None:
        return _CODE_LIST_CACHE

    # 级别1: 尝试 AKShare（仅此一次，不重试避免超时）
    try:
        raw = ak.stock_zh_a_spot_em()
        codes = raw["代码"].tolist()
        _CODE_LIST_CACHE = codes
        return codes
    except Exception as e:
        logging.warning(f"AKShare 代码获取失败: {e}")

    # 级别2: 本地快照
    if os.path.exists("last_spot_snapshot.csv"):
        try:
            df = pd.read_csv("last_spot_snapshot.csv")
            if "code" in df.columns:
                codes = df["code"].tolist()
                _CODE_LIST_CACHE = codes
                return codes
        except Exception as e:
            logging.warning(f"读取本地快照代码失败: {e}")

    # 级别3: 内置100只蓝筹股
    logging.warning("使用内置100只蓝筹股列表作为股票池")
    _CODE_LIST_CACHE = _BUILTIN_CODES.copy()
    return _BUILTIN_CODES

# ================================
# 实时行情获取（完全重构）
# ================================

def fetch_spot_data():
    """主力数据源：腾讯接口（需要代码列表），失败则降级本地快照"""
    codes = _get_all_codes()

    # 优先腾讯
    if "tencent" in DATA_SOURCE_ORDER:
        try:
            df = fetch_spot_data_tencent(codes)
            if df is not None and not df.empty and len(df) > 10:
                # 尝试补充行业信息（若网络可用）
                try:
                    raw_ak = ak.stock_zh_a_spot_em()
                    industry_map = dict(zip(raw_ak["代码"], raw_ak.get("行业", "未知")))
                    df["行业"] = df["code"].map(industry_map).fillna("未知")
                except:
                    df["行业"] = "未知"
                df.to_csv("last_spot_snapshot.csv", index=False)
                return df
        except Exception as e:
            logging.warning(f"腾讯行情失败: {e}")

    # 降级：本地快照
    if os.path.exists("last_spot_snapshot.csv"):
        logging.warning("使用本地离线行情快照")
        return pd.read_csv("last_spot_snapshot.csv")

    # 最终尝试：用内置代码再请求一次腾讯
    try:
        df = fetch_spot_data_tencent(_BUILTIN_CODES)
        if df is not None and not df.empty:
            return df
    except:
        pass

    raise RuntimeError("所有行情数据源均不可用，请检查网络或运行一次本地生成快照")

# ================================
# 全局超时
# ================================

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("选股流程超过90秒，强制终止")

# ================================
# 主程序
# ================================

def main():
    # 90秒超时（涵盖多线程历史回测）
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(90)

    try:
        logging.info("启动实战选股系统 V3.1")

        # 获取行情数据（优先腾讯）
        df = fetch_spot_data()
        if df.empty:
            logging.error("无行情数据")
            return

        # 基于已有数据计算情绪（不再单独请求AKShare）
        emotion = get_market_emotion(df)
        logging.info(f"市场情绪: {emotion}")

        # 连续冰点检查
        cold_days_file = "cold_days.txt"
        previous_cold = 0
        if os.path.exists(cold_days_file):
            with open(cold_days_file, "r") as f:
                previous_cold = int(f.read().strip())
        if emotion["emotion"] == "cold":
            previous_cold += 1
        else:
            previous_cold = 0
        with open(cold_days_file, "w") as f:
            f.write(str(previous_cold))

        final_holdings = FINAL_HOLDINGS
        if previous_cold >= MAX_CONSECUTIVE_COLD_DAYS:
            logging.warning(f"连续冰点{previous_cold}天，系统建议空仓")
            final_holdings = 0

        # 剔除涨跌停
        df = df[df["pct"].abs() < 9.8]

        # 剔除高波动股
        logging.info("计算波动率...")
        df["volatility_ratio"] = df["code"].apply(calc_volatility_ratio)
        df = df[df["volatility_ratio"] < 0.08]

        # 开盘涨幅
        df["open_pct"] = df.apply(calc_open_pct, axis=1)

        # 基础过滤
        ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"
        df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]
        df = df[df["code"].astype(str).str.startswith(("60", "00"))]
        df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
        df = df[df["amount"] >= MIN_AMOUNT]

        logging.info(f"基础过滤后: {len(df)} 只")

        # 初始化评分
        df["score"] = 0

        # 龙头评分
        logging.info("计算龙头评分...")
        df["score"] += df.apply(calc_leader_score, axis=1)

        # MA加分
        logging.info("计算MA趋势...")
        df["score"] += df["code"].apply(lambda x: 6 if is_above_ma(x, MA_PERIOD) else -3)

        # 波动率扣分
        df["score"] -= df["volatility_ratio"].fillna(0) * 100 * 0.5

        # 资金流向（仅对初筛前20只计算，避免网络压力）
        top20 = df.sort_values("score", ascending=False).head(20)
        logging.info("计算资金流向（前20名）...")
        for idx, row in top20.iterrows():
            code = row["code"]
            try:
                flow = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("60") else "sz")
                if flow is not None and not flow.empty:
                    latest = flow.iloc[-1]
                    main_net = safe_float(latest.get("主力净流入", 0))
                    amount = safe_float(latest.get("成交额", 1))
                    flow_score = (main_net / amount) * 10
                    df.loc[idx, "score"] += flow_score
            except Exception as e:
                logging.debug(f"资金流向获取失败 {code}: {e}")
            time.sleep(0.02)

        # 强势龙头保护
        strong_stock = (df["pct"] >= 7) & (df["turnover"] >= 8)
        df.loc[strong_stock, "score"] += 15

        # 涨停龙头（虽然已剔除涨停，保留逻辑）
        df.loc[df["pct"] >= 9, "score"] += 20

        # 放量突破
        df.loc[(df["turnover"] >= 10) & (df["lb"] >= 2), "score"] += 10

        # 超跌反弹
        df.loc[(df["pct"] >= 2) & (df["open_pct"] < 0), "score"] += 6

        # 热点板块
        logging.info("计算热点板块...")
        sector_map = get_hot_sector_score()
        if "行业" in df.columns and sector_map:
            df["sector_score"] = df["行业"].map(sector_map).fillna(0)
            df["score"] += df["sector_score"] * 0.5

        # 市场情绪调节
        if emotion["emotion"] == "hot":
            df["score"] += df["pct"] * 1.5
        elif emotion["emotion"] == "cold":
            df["score"] -= df["amplitude"] * 0.8

        # 板块龙头过滤（同一行业只留前2名）
        if "行业" in df.columns and len(df) > 0:
            df['rank_in_sector'] = df.groupby('行业')['pct'].rank(ascending=False)
            df = df[df['rank_in_sector'] <= 2]

        # 初排序
        filtered = df.sort_values("score", ascending=False).head(80)
        logging.info(f"评分后候选: {len(filtered)}")

        # 历史回测评分（多线程）
        history_rows = []
        valid_indices = []

        def process_single(row):
            code = row["code"]
            hist = evaluate_stock_history(code)
            if hist["signals"] >= BACKTEST_MIN_SIGNALS:
                return (row.name, hist)
            return None

        logging.info("开始历史回测评分（多线程）...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_single, row) for _, row in filtered.iterrows()]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    idx, hist = res
                    valid_indices.append(idx)
                    history_rows.append(hist)

        candidates = filtered.loc[valid_indices].reset_index(drop=True)
        hist_df = pd.DataFrame(history_rows)
        candidates = pd.concat([candidates, hist_df], axis=1)

        if candidates.empty:
            logging.warning("无符合条件的股票")
            return

        # 最终评分
        candidates["final_score"] = (
            candidates["score"] * 0.55 + candidates["history_score"] * 0.45
        )
        candidates = candidates.sort_values("final_score", ascending=False)
        final_candidates = candidates.head(final_holdings)

        # 动态仓位
        total_score = final_candidates["final_score"].sum()
        if total_score > 0:
            final_candidates["建议仓位%"] = (final_candidates["final_score"] / total_score * 100).round(1)
        else:
            final_candidates["建议仓位%"] = 0

        # 输出
        print("\n" + "=" * 60)
        print("A股实战短线选股结果 (V3.1)")
        print("=" * 60)
        for _, row in final_candidates.iterrows():
            print(f"""
股票: {row['name']} ({row['code']})
现价: {row['price']:.2f}
涨跌幅: {row['pct']:.2f}%
换手率: {row['turnover']:.2f}%
量比: {row['lb']:.2f}
行业: {row.get('行业', '未知')}
实时评分: {row['score']:.2f}
历史评分: {row['history_score']:.2f}
最终评分: {row['final_score']:.2f}
建议仓位: {row['建议仓位%']}%
----------------------------------------
""")

        # 保存CSV
        save_cols = [
            "code", "name", "price", "pct", "turnover", "lb",
            "score", "history_score", "final_score", "建议仓位%"
        ]
        final_candidates[save_cols].to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
        logging.info("结果已保存 selected_stocks.csv")

    except TimeoutError:
        logging.error("流程超时（>90秒），请减少候选股数量或检查网络")
    except Exception as e:
        logging.exception(f"运行异常: {e}")
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    main()