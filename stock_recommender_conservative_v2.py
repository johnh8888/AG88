# ================================
# A股短线实战选股系统 V2.4
# efinance 主力 + 本地缓存（适配 GitHub Actions）
# ================================

import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import efinance as ef
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================================
# 配置参数
# ================================

PRICE_MIN = 2
PRICE_MAX = 150
MIN_AMOUNT = 5e7          # 成交额 > 5000万
MA_PERIOD = 20
BACKTEST_LOOKBACK_DAYS = 180
BACKTEST_MIN_SIGNALS = 1
FINAL_HOLDINGS = 5

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
# efinance 数据获取（主力）
# ================================

def fetch_spot_data_efinance():
    """使用 efinance 获取全市场实时行情"""
    cache_file = "last_spot_snapshot.csv"
    
    # 非交易时间优先使用缓存
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file), TZ_SHANGHAI)
        if (now.hour < 9 or now.hour >= 15) or (now - file_time).total_seconds() < 7200:
            logging.info("使用本地缓存行情")
            return pd.read_csv(cache_file)

    logging.info("正在通过 efinance 获取实时行情...")
    try:
        # 获取沪深A股实时行情
        df = ef.stock.get_realtime_quotes()
        
        if df is None or df.empty:
            logging.error("efinance 返回空数据")
            return pd.DataFrame()

        logging.info(f"efinance 成功获取 {len(df)} 条行情")

        # 字段重命名和清洗
        rename_map = {
            '股票代码': 'code',
            '股票名称': 'name',
            '最新价': 'price',
            '涨跌幅': 'pct',
            '成交额': 'amount',
            '换手率': 'turnover',
            '量比': 'lb',
            '振幅': 'amplitude',
            '今开': 'open',
            '昨收': 'prev_close',
            '行业': '行业'
        }
        
        df = df.rename(columns=rename_map)
        
        # 确保数值列正确
        num_cols = ['price', 'pct', 'amount', 'turnover', 'lb', 'amplitude', 'open', 'prev_close']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 成交额单位处理（efinance 通常为元）
        if 'amount' in df.columns:
            df['amount'] = df['amount'].fillna(0)
        
        # 保存缓存
        df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        return df

    except Exception as e:
        logging.error(f"efinance 获取失败: {e}")
        # 失败时尝试读取缓存
        if os.path.exists(cache_file):
            logging.warning("使用本地缓存作为兜底")
            return pd.read_csv(cache_file)
        return pd.DataFrame()


# ================================
# 历史K线（efinance）
# ================================

@lru_cache(maxsize=100)
def get_stock_hist(code: str, days: int = 200):
    """获取前复权历史K线"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        df = ef.stock.get_quote_history(
            stock_code=code,
            klt=101,                    # 101=日线
            beg="", 
            end=end_date,
            adjust="qfq"                # 前复权
        )
        if df is None or df.empty:
            return None
            
        df = df.rename(columns={
            '日期': '日期',
            '开盘': '开盘',
            '收盘': '收盘',
            '最高': '最高',
            '最低': '最低',
            '成交量': '成交量',
            '涨跌幅': '涨跌幅'
        })
        df['日期'] = pd.to_datetime(df['日期'])
        return df[['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅']]
    except:
        return None


# ================================
# 其他核心函数（保持原有逻辑）
# ================================

def is_above_ma(code, period=MA_PERIOD):
    hist = get_stock_hist(code, 60)
    if hist is None or len(hist) < period:
        return False
    close = hist["收盘"].astype(float)
    ma = close.rolling(period).mean().iloc[-1]
    return close.iloc[-1] > ma


def calc_leader_score(row):
    score = 0
    score += row.get("pct", 0) * 2.5
    score += (row.get("amount", 0) / 1e8) * 0.8
    score += row.get("turnover", 0) * 1.2
    score += row.get("lb", 1) * 1.5
    score -= row.get("amplitude", 0) * 0.4
    score -= abs(row.get("open_pct", 0)) * 0.5
    return score


def calc_volatility_ratio(code):
    hist = get_stock_hist(code, 40)
    if hist is None or len(hist) < 20:
        return 0.0
    close = hist["收盘"].astype(float).tail(20)
    return close.std() / close.mean()


def evaluate_stock_history(code):
    hist = get_stock_hist(code, BACKTEST_LOOKBACK_DAYS)
    if hist is None or len(hist) < 30:
        return {"signals": 0, "history_score": -999}
    pct = hist["涨跌幅"].astype(float)
    signals = len(pct[pct >= 3])
    history_score = pct.mean() * 5 + pct.max() * 2 - abs(pct.min())
    return {"signals": signals, "history_score": history_score}


# ================================
# 主程序
# ================================

logging.info("启动A股短线选股系统 V2.4 (efinance版)")

df = fetch_spot_data_efinance()

if df.empty or len(df) < 1000:
    logging.error("未能获取有效行情数据")
    sys.exit(1)

# 数据处理
df["open_pct"] = df.apply(calc_open_pct, axis=1)
df["volatility_ratio"] = df["code"].apply(calc_volatility_ratio)

# 过滤
ban_pattern = r"(^ST|^\*ST|退市|^N|^C|^U|XD|XR)"
df = df[~df["name"].str.contains(ban_pattern, na=False, regex=True)]
df = df[df["code"].astype(str).str.startswith(("60", "00", "30"))]
df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
df = df[df["amount"] >= MIN_AMOUNT]

logging.info(f"基础过滤后剩余: {len(df)} 只")

# 评分系统
df["score"] = df.apply(calc_leader_score, axis=1)
df["score"] += df["code"].apply(lambda x: 6 if is_above_ma(x, MA_PERIOD) else -3)
df["score"] -= df["volatility_ratio"].fillna(0) * 50

# 加分项
df.loc[df["pct"] >= 9, "score"] += 20
df.loc[(df["turnover"] >= 10) & (df["lb"] >= 2), "score"] += 10
df.loc[(df["pct"] >= 2) & (df["open_pct"] < 0), "score"] += 6

# 排序
filtered = df.sort_values("score", ascending=False).head(80)

# 历史回测
candidates = []
for _, row in filtered.iterrows():
    hist_eval = evaluate_stock_history(row["code"])
    if hist_eval["signals"] >= BACKTEST_MIN_SIGNALS:
        row_dict = row.to_dict()
        row_dict.update(hist_eval)
        candidates.append(row_dict)

candidates_df = pd.DataFrame(candidates)
if not candidates_df.empty:
    candidates_df["final_score"] = candidates_df["score"] * 0.55 + candidates_df["history_score"] * 0.45
    final_candidates = candidates_df.sort_values("final_score", ascending=False).head(FINAL_HOLDINGS)
else:
    final_candidates = pd.DataFrame()

# 输出结果
print("\n" + "=" * 70)
print("A股短线实战选股结果 V2.4 (efinance)")
print("=" * 70)

for _, row in final_candidates.iterrows():
    print(f"""
股票: {row.get('name')} ({row.get('code')})
现价: {row.get('price',0):.2f}   涨跌幅: {row.get('pct',0):.2f}%
换手率: {row.get('turnover',0):.2f}%   量比: {row.get('lb',1):.2f}
行业: {row.get('行业','未知')}
最终评分: {row.get('final_score',0):.1f}
----------------------------------------""")

# 保存结果
save_cols = ["code", "name", "price", "pct", "turnover", "lb", "score", "history_score", "final_score"]
final_candidates[save_cols].to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")

logging.info("选股完成！结果已保存至 selected_stocks.csv")