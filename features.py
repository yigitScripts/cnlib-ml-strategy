import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame):
    closes = df["Close"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(closes)

    def safe(arr):
        return np.where(np.isfinite(arr), arr, np.nan)

    def rolling_mean(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = arr[i - w + 1:i + 1].mean()
        return out

    def rolling_std(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = arr[i - w + 1:i + 1].std()
        return out

    def rolling_max(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = arr[i - w + 1:i + 1].max()
        return out

    def rolling_min(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = arr[i - w + 1:i + 1].min()
        return out

    # --- Trend ---
    ma5   = rolling_mean(closes, 5)
    ma10  = rolling_mean(closes, 10)
    ma20  = rolling_mean(closes, 20)
    ma50  = rolling_mean(closes, 50)
    ma200 = rolling_mean(closes, 200)

    ma5_cross_ma20   = safe((ma5  - ma20)  / (closes + 1e-9))
    ma10_cross_ma50  = safe((ma10 - ma50)  / (closes + 1e-9))
    ma50_cross_ma200 = safe((ma50 - ma200) / (closes + 1e-9))

    # --- Momentum ---
    roc3  = safe(np.diff(closes, prepend=closes[0]) / (closes + 1e-9))
    roc5  = safe(np.concatenate([np.full(5,  np.nan),
                  (closes[5:]  - closes[:-5])  / (closes[:-5]  + 1e-9)]))
    roc10 = safe(np.concatenate([np.full(10, np.nan),
                  (closes[10:] - closes[:-10]) / (closes[:-10] + 1e-9)]))
    roc20 = safe(np.concatenate([np.full(20, np.nan),
                  (closes[20:] - closes[:-20]) / (closes[:-20] + 1e-9)]))

    # --- Volatilite ---
    std10 = rolling_std(closes, 10)
    std20 = rolling_std(closes, 20)
    atr   = np.full(n, np.nan)
    for i in range(1, n):
        tr      = max(highs[i] - lows[i],
                      abs(highs[i] - closes[i - 1]),
                      abs(lows[i]  - closes[i - 1]))
        atr[i]  = tr
    atr14 = rolling_mean(atr, 14)
    atr_pct = safe(atr14 / (closes + 1e-9))

    # --- RSI ---
    delta = np.diff(closes, prepend=closes[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = rolling_mean(gain, 14)
    avg_loss = rolling_mean(loss, 14)
    rs   = safe(avg_gain / (avg_loss + 1e-9))
    rsi  = safe(100 - (100 / (1 + rs)))
    rsi_norm = safe((rsi - 50) / 50)  # -1 ile 1 arası normalize

    # --- Stochastic RSI ---
    rsi_min = rolling_min(rsi, 14)
    rsi_max = rolling_max(rsi, 14)
    stoch_rsi = safe((rsi - rsi_min) / (rsi_max - rsi_min + 1e-9))

    # --- MACD ---
    def ema(arr, span):
        out = np.full(n, np.nan)
        k   = 2 / (span + 1)
        start = span - 1
        if start >= n:
            return out
        out[start] = arr[:start + 1].mean()
        for i in range(start + 1, n):
            if np.isnan(arr[i]):
                out[i] = out[i - 1]
            else:
                out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    ema12      = ema(closes, 12)
    ema26      = ema(closes, 26)
    macd_line  = safe(ema12 - ema26)
    macd_valid = np.where(np.isfinite(macd_line), macd_line, 0.0)
    signal_line = ema(macd_valid, 9)
    macd_hist   = safe(macd_line - signal_line)
    macd_norm   = safe(macd_hist / (closes + 1e-9))

    # --- Bollinger Band ---
    bb_mid  = ma20
    bb_std  = rolling_std(closes, 20)
    bb_up   = safe(bb_mid + 2 * bb_std)
    bb_low  = safe(bb_mid - 2 * bb_std)
    bb_pos  = safe((closes - bb_mid) / (bb_std + 1e-9))   # -2 ile 2 arası
    bb_wid  = safe((bb_up - bb_low) / (bb_mid + 1e-9))    # bant genişliği

    # --- Volume ---
    vol_ma10  = rolling_mean(volume, 10)
    vol_ma20  = rolling_mean(volume, 20)
    vol_ratio = safe(volume / (vol_ma10 + 1e-9))
    vol_trend = safe((vol_ma10 - vol_ma20) / (vol_ma20 + 1e-9))

    # Volume price confirmation: fiyat + volume aynı yönde mi?
    price_dir  = np.sign(np.diff(closes, prepend=closes[0]))
    vol_dir    = np.sign(np.diff(volume, prepend=volume[0]))
    vol_confirm = (price_dir == vol_dir).astype(float)

    # --- Kanal pozisyonu ---
    high20      = rolling_max(highs, 20)
    low20       = rolling_min(lows,  20)
    channel_pos = safe((closes - low20) / (high20 - low20 + 1e-9))

    # --- Regime sinyali (ham, model için feature) ---
    above_ma50  = (closes > ma50).astype(float)
    above_ma200 = (closes > ma200).astype(float)

    # --- Hedef: yarın yükselecek mi? (long=1, short=-1 için ayrı model) ---
    next_ret = np.full(n, np.nan)
    next_ret[:-1] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-9)

    y_long  = (next_ret > 0).astype(float)   # long hedef
    y_short = (next_ret < 0).astype(float)   # short hedef

    X = np.column_stack([
        # Trend (6)
        ma5_cross_ma20, ma10_cross_ma50, ma50_cross_ma200,
        above_ma50, above_ma200,
        safe((closes - ma200) / (closes + 1e-9)),
        # Momentum (4)
        roc3, roc5, roc10, roc20,
        # Volatilite (3)
        std10 / (closes + 1e-9), std20 / (closes + 1e-9), atr_pct,
        # RSI grubu (3)
        rsi_norm, stoch_rsi, safe(rsi / 100),
        # MACD (2)
        macd_norm, safe(np.sign(macd_hist)),
        # Bollinger (2)
        bb_pos, bb_wid,
        # Volume (3)
        vol_ratio, vol_trend, vol_confirm,
        # Kanal (1)
        channel_pos,
    ])

    mask = (
        ~np.isnan(X).any(axis=1) &
        np.isfinite(X).all(axis=1) &
        ~np.isnan(y_long) &
        ~np.isnan(y_short)
    )

    return X, y_long, y_short, mask