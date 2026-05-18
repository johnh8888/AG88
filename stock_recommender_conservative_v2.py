# ================================
# A股短线实战选股系统 V3.2
# 多源容错（新浪 + efinance + AKShare）
# ================================

import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import akshare as ak
import efinance as ef
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================
# 配置
# ================================
FINAL_HOLDINGS = 5
PRICE_MIN = 2
PRICE_MAX = 150
MIN_AMOUNT = 5e7

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")
now = datetime.now(TZ_SHANGHAI)

# ================================
# 新浪实时行情（推荐备用）
# ================================
def fetch_spot_data_sina(codes=None):
    if not codes:
        codes = ["sh600519", "sz000001"]  # 测试用，实际会用全市场
    try:
        url = f"https://hq.sinajs.cn/list={','.join(codes[:800])}"  # 限制数量避免封禁
        headers = {"Referer": "https://finance.sina.com.cn"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = {}
        for line in resp.text.split('\n'):
            if not line.startswith('var hq_str_'):
                continue
            code_part = line.split('=')[0].replace('var hq_str_', '').strip()
            fields = line.split('=')[1].strip('"').split(',')
            if len(fields) < 10:
                continue
            code = code_part[2:] if len(code_part) > 2 else code_part
            data[code] = {
                "code": code,
                "name": fields[0],
                "open": float(fields[1]),
                "prev_close": float(fields[2]),
                "price": float(fields[3]),
                "high": float(fields[4]),
                "low": float(fields[5]),
                "volume": float(fields[8]),
                "amount": float(fields[9]),
                "pct": (float(fields[3]) - float(fields[2])) / float(fields[2]) * 100 if float(fields[2]) > 0 else 0
            }
        return pd.DataFrame.from_dict(data, orient='index')
    except Exception as e:
        logging.warning(f"新浪失败: {e}")
        return pd.DataFrame()


# ================================
# 主行情获取函数（多源）
# ================================
def fetch_spot_data():
    cache_file = "last_spot_snapshot.csv"
    
    if os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file), TZ_SHANGHAI)
        if (now.hour < 9 or now.hour >= 15) or (now - file_time).total_seconds() < 7200:
            logging.info("使用本地缓存")
            return pd.read_csv(cache_file)

    # 优先级：efinance → 新浪 → AKShare
    sources = [
        ("efinance", lambda: ef.stock.get_realtime_quotes()),
        ("sina", lambda: fetch_spot_data_sina()),
        ("akshare", lambda: ak.stock_zh_a_spot_em(limit=None))
    ]

    for name, func in sources:
        logging.info(f"尝试 {name} 接口...")
        try:
            df = func()
            if df is not None and not df.empty and len(df) > 800:
                logging.info(f"{name} 成功，返回 {len(df)} 条")
                
                # 统一字段名
                if name == "akshare":
                    rename_map = {'代码':'code','名称':'name','最新价':'price','涨跌幅':'pct',
                                  '成交额':'amount','换手率':'turnover','量比':'lb',
                                  '振幅':'amplitude','今开':'open','昨收':'prev_close'}
                    df = df.rename(columns=rename_map)
                
                df.to_csv(cache_file, index=False, encoding='utf-8-sig')
                return df
        except Exception as e:
            logging.warning(f"{name} 失败: {e}")

    # 最终缓存兜底
    if os.path.exists(cache_file):
        logging.warning("所有在线接口失败，使用缓存")
        return pd.read_csv(cache_file)

    raise RuntimeError("所有数据源均不可用")


# ================================
# 主程序（简化稳定版）
# ================================
if __name__ == "__main__":
    logging.info("启动A股短线选股系统 V3.2 多源版")

    df = fetch_spot_data()
    if len(df) < 1000:
        logging.error("行情数据不足，退出")
        sys.exit(1)

    # 基础处理
    df["pct"] = pd.to_numeric(df.get("pct"), errors='coerce')
    df["amount"] = pd.to_numeric(df.get("amount"), errors='coerce')
    df["price"] = pd.to_numeric(df.get("price"), errors='coerce')

    # 过滤
    ban = r"(^ST|^\*ST|退市|^N|^C)"
    df = df[~df["name"].str.contains(ban, na=False, regex=True)]
    df = df[df["code"].astype(str).str.startswith(("60", "00", "30"))]
    df = df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)]
    df = df[df["amount"] >= MIN_AMOUNT]

    logging.info(f"过滤后剩余 {len(df)} 只")

    # 简单评分（可后续扩展）
    df["score"] = df["pct"] * 2 + (df["amount"] / 1e8) * 0.5

    final = df.nlargest(FINAL_HOLDINGS, "score")

    print("\n" + "="*70)
    print("今日选股结果 V3.2")
    print("="*70)
    for _, r in final.iterrows():
        print(f"{r.get('name')} ({r.get('code')}) | 现价 {r.get('price'):.2f} | 涨跌 {r.get('pct'):.2f}% | 成交额 {r.get('amount')/1e8:.1f}亿")

    final.to_csv("selected_stocks.csv", index=False, encoding="utf-8-sig")
    logging.info("选股完成！")