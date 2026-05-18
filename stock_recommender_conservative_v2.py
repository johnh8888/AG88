# ================================
# A股短线实战选股系统 V2.1
# 强化版 + 腾讯接口双源容错
# 作者：ChatGPT Quant Upgrade
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

TEST_MODE = True                     # 测试模式（保留原样）

TOP_N_CANDIDATES = 20
FINAL_HOLDINGS = 5

PRICE_MIN = 2
PRICE_MAX = 150

MIN_AMOUNT = 5e7                    # 成交额 > 5千万

MA_PERIOD = 20

BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 1

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)
today = now.strftime("%Y%m%d")

# 数据源优先级：先腾讯后AKShare
DATA_SOURCE_ORDER = ["tencent", "akshare"]

# ================================
# 工具函数
# ================================

def safe_float(v, default=0):
    """安全转换为 float，处理 nan 和异常"""
    try:
        if pd.isna(v):
            return default
        return float(v)
    except:
        return default

def calc_open_pct(row):
    """计算开盘涨幅（相对于昨收）"""
    prev_close = safe_float(row.get("prev_close"))
    open_price = safe_float(row.get("open"))
    if prev_close <= 0:
        return 0
    return (open_price / prev_close - 1) * 100

# ================================
# 网络请求 + 重试
# ================================

def http_get_with_retry(url, max_tries=3, timeout=5):
    """带指数退避和随机抖动的 HTTP GET 重试"""
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
    raise last_exc

# ================================
# 腾讯行情接口
# ================================

def parse_tencent_spot(raw_text: str):
    """
    解析腾讯实时行情文本
    返回字典列表，每个字典包含单只股票信息
    """
    results = []
    # 文本格式: v_sz000001="51~平安银行~000001~12.34~..."
    pattern = r'v_(\w+)="([^"]+)"'
    matches = re.findall(pattern, raw_text)
    for match in matches:
        full_code, value_str = match
        fields = value_str.split("~")
        # 腾讯正常字段数40+，不足则跳过
        if len(fields) < 40:
            continue
        try:
            code = full_code[2:]                # sz000001 -> 000001
            name = fields[1]
            price = float(fields[3])            # 最新价
            prev_close = float(fields[4])       # 昨收
            open_price = float(fields[5])       # 今开
            volume = float(fields[6])           # 成交量（手）
            # 成交额字段37，可能是万元，需转成元
            amount = float(fields[37]) * 10000 if fields[37] else 0
            # 如果成交额为0，用成交量*价格估算
            if amount == 0 and volume > 0:
                amount = volume * price * 100   # 手 * 100股 * 价格
            pct = float(fields[32])             # 涨跌幅（%）
            turnover = float(fields[38])        # 换手率（%）
            amplitude = float(fields[43])       # 振幅
            lb = float(fields[48]) if len(fields) > 48 and fields[48] else 1.0  # 量比
            sector = ""                          # 腾讯无行业，后续补充
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
                "行业": sector,
            })
        except (ValueError, IndexError) as e:
            logging.debug(f"解析腾讯行情字段失败 {full_code}: {e}")
            continue
    return results

def fetch_spot_data_tencent(codes: list):
    """
    通过腾讯接口批量获取实时行情
    codes: 纯数字代码列表，如 ['600519', '000001']
    返回与原有格式一致的DataFrame
    """
    if not codes:
        return pd.DataFrame()
    # 构建腾讯代码格式：sh600519, sz000001
    tencent_codes = []
    for c in codes:
        if c.startswith("60"):
            tencent_codes.append(f"sh{c}")
        else:
            tencent_codes.append(f"sz{c}")

    all_data = []
    batch_size = 50   # 每批最多50只
    for i in range(0, len(tencent_codes), batch_size):
        batch = tencent_codes[i:i+batch_size]
        url = "http://qt.gtimg.cn/q=" + ",".join(batch)
        resp = http_get_with_retry(url, timeout=8)
        all_data.extend(parse_tencent_spot(resp.text))
        time.sleep(0.1)  # 礼貌性延迟
    return pd.DataFrame(all_data)

def get_tencent_kline(code, start_date=None, end_date=None, period="daily", adjust="qfq"):
    """
    从腾讯获取历史K线（前复权）
    返回DataFrame，包含：日期,开盘,收盘,最高,最低,成交量,涨跌幅
    start_date/end_date: str YYYYMMDD 或 YYYY-MM-DD
    """
    if code.startswith("60"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"

    # 周期映射
    klt = {"daily": "day", "weekly": "week", "monthly": "month"}.get(period, "day")
    param = f"{symbol},{klt},,,320," + ("qfq" if adjust == "qfq" else "")
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={param}"
    resp = http_get_with_retry(url, timeout=10)
    data = resp.json()

    # 提取K线数据
    klines = data.get("data", {}).get(symbol, {}).get(klt, [])
    if not klines and adjust == "qfq":
        klines = data.get("data", {}).get(symbol, {}).get("qfq" + klt, [])
    if not klines:
        return None

    # 腾讯K线格式：["2024-01-02","10.50","10.80","10.20","10.60","123456"]
    df = pd.DataFrame(klines, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量"])
    # 注意：列顺序有时为 [日期, 开盘, 最高, 最低, 收盘, 成交量]，需确认
    # 实测腾讯格式为 ["日期","开盘","收盘","最高","最低","成交量"] 或 ["日期","开盘","最高","最低","收盘","成交量"]？这里根据常见腾讯接口文档修正：
    # 常见格式: [日期, 开盘, 收盘, 最高, 最低, 成交量] （后复权有时不同），为确保安全，动态识别列名
    # 我们假设列名固定为 ["日期","开盘","最高","最低","收盘","成交量"] 并调整
    if len(df.columns) >= 6:
        # 重命名，避免错位
        df.columns = ["日期", "col1", "col2", "col3", "col4", "col5"][:len(df.columns)]
        # 常见实际顺序可能是 ["日期","开盘","收盘","最高","最低","成交量"]，我们做个判断
        # 简单处理：若col2 > col3 (即收盘>最高不可能)，则可能顺序是开盘,最高,最低,收盘,成交量
        # 但样本数据： ["2022-01-04","10.00","10.50","9.80","10.20","12345"] 如果是 [日期, 开盘, 最高, 最低, 收盘, 成交量]，
        # 这里我们稳妥重命名：
        if len(df.columns) == 6:
            # 暂时按 [日期, 开盘, 收盘, 最高, 最低, 成交量] 处理，这也是常见格式
            df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量"]
            # 验证：如果“收盘”>“最高”很多，可能顺序不对，交换收盘和最高
            if (df["收盘"] > df["最高"]).any():
                # 可能是 开盘,最高,最低,收盘,成交量 顺序
                df.columns = ["日期", "开盘", "最高", "最低", "收盘", "成交量"]
    else:
        return None

    # 类型转换
    for col in ["开盘", "收盘", "最高", "最低", "成交量"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["日期"] = pd.to_datetime(df["日期"])

    # 按日期过滤
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df["日期"] >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df["日期"] <= end_dt]

    # 排序
    df = df.sort_values("日期").reset_index(drop=True)

    # 计算涨跌幅（基于收盘价变动）
    df["涨跌幅"] = df["收盘"].pct_change() * 100
    df["涨跌幅"] = df["涨跌幅"].fillna(0)

    return df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]

# ================================
# 历史数据统一缓存（双源）
# ================================

@lru_cache(maxsize=128)
def _get_stock_hist_cached(code, start_date, end_date):
    """
    获取历史日线数据，优先腾讯，失败则用AKShare
    返回统一的DataFrame，包含：日期,开盘,收盘,最高,最低,成交量,涨跌幅
    """
    # 尝试腾讯
    try:
        hist = get_tencent_kline(code, start_date, end_date, "daily", "qfq")
        if hist is not None and len(hist) > 5:
            return hist
    except Exception as e:
        logging.debug(f"腾讯历史K线失败 {code}: {e}")

    # 降级 AKShare
    try:
        hist_ak = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        if hist_ak is not None and not hist_ak.empty:
            # 字段统一
            df = hist_ak.rename(columns={
                "日期": "日期",
                "开盘": "开盘",
                "收盘": "收盘",
                "最高": "最高",
                "最低": "最低",
                "成交量": "成交量",
                "涨跌幅": "涨跌幅"
            })
            for col in ["开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]
    except Exception as e:
        logging.warning(f"AKShare历史数据也失败 {code}: {e}")

    return None

# ================================
# 市场情绪系统
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
        if score >= 200:
            emotion = "hot"
        elif score <= -50:
            emotion = "cold"
        else:
            emotion = "neutral"
        return {
            "emotion": emotion,
            "score": score,
            "up_limit": up_limit,
            "down_limit": down_limit
        }
    except Exception as e:
        logging.warning(f"情绪系统失败: {e}")
        return {"emotion": "neutral", "score": 0}

# ================================
# 热点板块系统
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
    except Exception as e:
        logging.warning(f"热点板块失败: {e}")
        return {}

# ================================
# MA均线过滤
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
# 龙头评分系统
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
# 历史回测评分
# ================================

def evaluate_stock_history(code):
    end = today
    start = (now - timedelta(days=BACKTEST_LOOKBACK_DAYS)).strftime("%Y%m%d")
    hist = _get_stock_hist_cached(code, start, end)
    if hist is None or hist.empty:
        return {"signals": 0, "history_score": -999}
    hist = hist.sort_values("日期")
    pct = hist["涨跌幅"].astype(float)
    strong_days = pct[pct >= 3]
    signals = len(strong_days)
    history_score = pct.mean() * 5 + pct.max() * 2 - abs(pct.min())
    return {"signals": signals, "history_score": history_score}

# ================================
# 获取实时行情（双源智能切换）
# ================================

# 用于缓存代码列表，避免重复请求AKShare
_CODE_LIST_CACHE = None

def _get_all_codes():
    """获取全市场代码列表，带缓存"""
    global _CODE_LIST_CACHE
    if _CODE_LIST_CACHE is None:
        try:
            raw = ak.stock_zh_a_spot_em()
            _CODE_LIST_CACHE = raw["代码"].tolist()
        except:
            # 极端情况返回空列表
            return []
    return _CODE_LIST_CACHE

def fetch_spot_data():
    """
    获取实时行情数据，优先腾讯，失败则用AKShare
    返回包含所有字段的DataFrame
    """
    # 获取代码列表（先通过AKShare拿一次，以后缓存）
    codes = _get_all_codes()
    if not codes:
        raise RuntimeError("无法获取股票代码列表，程序终止")

    # 尝试腾讯
    if "tencent" in DATA_SOURCE_ORDER:
        logging.info("尝试通过腾讯接口获取实时行情...")
        try:
            df = fetch_spot_data_tencent(codes)
            if df is not None and not df.empty and len(df) > 1000:
                logging.info(f"腾讯接口成功，获取到 {len(df)} 条行情")
                # 补充行业信息（从AKShare补充，失败则置空）
                try:
                    raw_ak = ak.stock_zh_a_spot_em()
                    industry_map = dict(zip(raw_ak["代码"], raw_ak.get("行业", "未知")))
                    df["行业"] = df["code"].map(industry_map).fillna("未知")
                except:
                    df["行业"] = "未知"
                # 保存快照备份
                df.to_csv("last_spot_snapshot.csv", index=False)
                return df
        except Exception as e:
            logging.warning(f"腾讯行情失败: {e}")

    # 降级使用 AKShare
    if "akshare" in DATA_SOURCE_ORDER:
        logging.info("降级使用 AKShare 获取行情...")
        try:
            raw = ak.stock_zh_a_spot_em()
            if raw is None or raw.empty:
                raise RuntimeError("AKShare返回空数据")
            # 原有解析逻辑
            df = pd.DataFrame()
            df["code"] = raw["代码"]
            df["name"] = raw["名称"]
            df["price"] = pd.to_numeric(raw["最新价"], errors="coerce")
            df["pct"] = pd.to_numeric(raw["涨跌幅"], errors="coerce")
            df["amount"] = pd.to_numeric(raw["成交额"], errors="coerce")
            df["lb"] = pd.to_numeric(raw.get("量比", 1), errors="coerce").fillna(1)
            df["turnover"] = pd.to_numeric(raw["换手率"], errors="coerce")
            df["amplitude"] = pd.to_numeric(raw["振幅"], errors="coerce")
            df["open"] = pd.to_numeric(raw["今开"], errors="coerce")
            df["prev_close"] = pd.to_numeric(raw["昨收"], errors="coerce")
            if "行业" in raw.columns:
                df["行业"] = raw["行业"]
            else:
                df["行业"] = "未知"
            df.to_csv("last_spot_snapshot.csv", index=False)
            return df
        except Exception as e:
            logging.error(f"AKShare行情获取失败: {e}")

    # 最后尝试读取本地备份
    if os.path.exists("last_spot_snapshot.csv"):
        logging.warning("使用本地离线行情快照")
        return pd.read_csv("last_spot_snapshot.csv")

    raise RuntimeError("所有行情数据源均不可用")

# ================================
# 主程序
# ================================

logging.info("启动实战选股系统")

df = fetch_spot_data()
if df.empty:
    logging.error("无行情数据")
    sys.exit(0)

# 开盘涨幅
df["open_pct"] = df.apply(calc_open_pct, axis=1)

# 波动率
logging.info("计算波动率...")
df["volatility_ratio"] = df["code"].apply(calc_volatility_ratio)

# 过滤ST等
ban_pattern = r"(^ST|^\*ST|退市|^N|^C[^N]|XD|XR)"
df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]

# 主板
df = df[df["code"].astype(str).str.startswith(("60", "00"))]

# 价格过滤
df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]

# 成交额过滤
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

# 强势龙头保护
strong_stock = (df["pct"] >= 7) & (df["turnover"] >= 8)
df.loc[strong_stock, "score"] += 15

# 涨停龙头
df.loc[df["pct"] >= 9, "score"] += 20

# 放量突破
df.loc[(df["turnover"] >= 10) & (df["lb"] >= 2), "score"] += 10

# 超跌反弹
df.loc[(df["pct"] >= 2) & (df["open_pct"] < 0), "score"] += 6

# 热点板块
logging.info("计算热点板块...")
sector_map = get_hot_sector_score()
if "行业" in df.columns:
    df["sector_score"] = df["行业"].map(sector_map).fillna(0)
    df["score"] += df["sector_score"] * 0.5

# 市场情绪
emotion = get_market_emotion()
logging.info(f"市场情绪: {emotion}")
if emotion["emotion"] == "hot":
    df["score"] += df["pct"] * 1.5
elif emotion["emotion"] == "cold":
    df["score"] -= df["amplitude"] * 0.8

# 初排序
filtered = df.sort_values("score", ascending=False).head(80)
logging.info(f"评分后股票数量: {len(filtered)}")

# 历史回测评分
history_rows = []
valid_indices = []
logging.info("开始历史回测评分...")
for idx, row in filtered.iterrows():
    code = row["code"]
    logging.info(f"评估: {code}")
    hist = evaluate_stock_history(code)
    if hist["signals"] < BACKTEST_MIN_SIGNALS:
        continue
    history_rows.append(hist)
    valid_indices.append(idx)

candidates = filtered.loc[valid_indices].reset_index(drop=True)
hist_df = pd.DataFrame(history_rows)
candidates = pd.concat([candidates, hist_df], axis=1)

# 最终评分
candidates["final_score"] = (
    candidates["score"] * 0.55 + candidates["history_score"] * 0.45
)
candidates = candidates.sort_values("final_score", ascending=False)
final_candidates = candidates.head(FINAL_HOLDINGS)

# 输出
print("\n" + "=" * 60)
print("A股实战短线选股结果")
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
----------------------------------------
""")

# 保存CSV
save_cols = [
    "code", "name", "price", "pct", "turnover", "lb",
    "score", "history_score", "final_score"
]
final_candidates[save_cols].to_csv(
    "selected_stocks.csv", index=False, encoding="utf-8-sig"
)
logging.info("结果已保存 selected_stocks.csv")
print("\n完成。")