import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame):
    closes = df["Close"]
    highs  = df["High"]
    lows   = df["Low"]
    volume = df["Volume"]

    ma5   = closes.rolling(5).mean()
    ma10  = closes.rolling(10).mean()
    ma20  = closes.rolling(20).mean()
    ma50  = closes.rolling(50).mean()

    roc5  = closes.pct_change(5)
    roc10 = closes.pct_change(10)
    roc20 = closes.pct_change(20)

    std10 = closes.rolling(10).std()
    std20 = closes.rolling(20).std()

    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    rsi   = 100 - (100 / (1 + rs))

    bb_mid = closes.rolling(20).mean()
    bb_std = closes.rolling(20).std()
    bb_pos = (closes - bb_mid) / (bb_std + 1e-9)

    vol_ma10  = volume.rolling(10).mean()
    vol_ratio = volume / (vol_ma10 + 1e-9)

    high_20     = highs.rolling(20).max()
    low_20      = lows.rolling(20).min()
    channel_pos = (closes - low_20) / (high_20 - low_20 + 1e-9)

    ma5_cross_ma20  = (ma5 - ma20) / (closes + 1e-9)
    ma10_cross_ma50 = (ma10 - ma50) / (closes + 1e-9)

    X = np.column_stack([
        ma5, ma10, ma20, ma50,
        roc5, roc10, roc20,
        std10, std20,
        rsi,
        bb_pos,
        vol_ratio,
        channel_pos,
        ma5_cross_ma20,
        ma10_cross_ma50,
    ])

    y    = (closes.shift(-1) > closes).astype(int).values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    return X, y, mask
