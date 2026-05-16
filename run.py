import akshare as ak
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# =========================
# 1. 数据获取
# =========================
def get_data():
    df = ak.stock_zh_a_hist(symbol="000001", period="daily", adjust="qfq")

    df = df.rename(columns={
        "收盘": "close",
        "开盘": "open",
        "成交量": "volume"
    })

    df = df.sort_index()

    return df


# =========================
# 2. 特征工程
# =========================
def build_features(df):
    df = df.copy()

    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()

    df["momentum"] = df["close"] / df["ma_5"]

    df["volatility"] = df["ret_1"].rolling(10).std()

    df["volume_z"] = (
        df["volume"] - df["volume"].rolling(10).mean()
    ) / df["volume"].rolling(10).std()

    return df.dropna()


# =========================
# 3. 标签（未来收益）
# =========================
def build_label(df, horizon=5):
    df = df.copy()

    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (df["future_ret"] > 0).astype(int)

    return df.dropna()


# =========================
# 4. 训练模型
# =========================
def train_model(df):
    features = [
        "ret_1", "ret_5",
        "momentum",
        "volatility",
        "volume_z"
    ]

    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    print("\n=== MODEL RESULT ===")
    print("Accuracy:", round(acc, 4))

    return model, features


# =========================
# 5. 预测 + 回测
# =========================
def backtest(df, model, features):
    df = df.copy()

    df["prob"] = model.predict_proba(df[features])[:, 1]

    top = df.sort_values("prob", ascending=False).head(10)

    avg_ret = top["future_ret"].mean()
    win_rate = (top["future_ret"] > 0).mean()

    print("\n=== BACKTEST ===")
    print("Avg Return:", round(avg_ret, 4))
    print("Win Rate:", round(win_rate, 4))


# =========================
# 6. 主程序
# =========================
def main():
    print("V15 QUANT SYSTEM START")

    df = get_data()
    df = build_features(df)
    df = build_label(df)

    model, features = train_model(df)

    backtest(df, model, features)


if __name__ == "__main__":
    main()