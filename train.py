import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
from features import build_features

COINS = [
    "kapcoin-usd_train",
    "metucoin-usd_train",
    "tamcoin-usd_train",
]

TOTAL_DAYS      = 1570
PREDICT_DAYS    = 368
TRAIN_DAYS      = TOTAL_DAYS - PREDICT_DAYS   # 1202
START_CANDLE    = TRAIN_DAYS                   # 1202
CHUNK           = TRAIN_DAYS // 4             # 300 (her parça)

STOP_LOSS_PCT   = 0.05
TAKE_PROFIT_PCT = 0.15


def get_full_data():
    collected = {}

    class TempStrategy(BaseStrategy):
        def predict(self, data):
            for coin, df in data.items():
                if coin not in collected or len(df) > len(collected[coin]):
                    collected[coin] = df.copy()
            return [
                {"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1}
                for c in data
            ]

    t = TempStrategy()
    backtest.run(strategy=t, initial_capital=3000.0, silent=True)
    return collected


def split_by_chunk(df, n_chunks_train):
    """
    n_chunks_train: kaç chunk eğitimde kullanılacak (1,2,3,4)
    Train: ilk n_chunks_train * CHUNK gün
    Val:   sonraki CHUNK gün (parça 4 için kalan tüm eğitim verisi)
    """
    train_end = min(n_chunks_train * CHUNK, TRAIN_DAYS)
    val_end   = min(train_end + CHUNK, TRAIN_DAYS)
    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    return train, val


def train_models(coin_data, n_chunks_train):
    models = {}
    for coin in COINS:
        df = coin_data[coin]
        train_df, val_df = split_by_chunk(df, n_chunks_train)

        print(f"  [{coin}] train={len(train_df)} gun, val={len(val_df)} gun")

        X_train, y_train, mask_train = build_features(train_df)
        X_train, y_train = X_train[mask_train], y_train[mask_train]

        if len(X_train) < 10:
            print(f"  [{coin}] Yeterli veri yok, atlanıyor.")
            continue

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        if len(val_df) > 51:
            X_val, y_val, mask_val = build_features(val_df)
            X_val, y_val = X_val[mask_val], y_val[mask_val]
            if len(X_val) > 0:
                acc = accuracy_score(y_val, model.predict(X_val))
                print(f"  [{coin}] Parca {n_chunks_train}→{n_chunks_train+1} "
                      f"accuracy: {acc:.3f}")

        models[coin] = model
    return models


def plot_predictions(coin_data, all_models, coin):
    df = coin_data[coin]
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    fig.suptitle(f"{coin} — Walk-Forward Tahminler (her parca {CHUNK} gun)",
                 fontsize=13)

    for n in range(1, 5):
        ax = axes[n - 1]
        _, val_df = split_by_chunk(df, n)

        label = (f"Parca {n} → Parca {n+1} tahmini"
                 if n < 4 else
                 f"Parca {n} → TAHMIN DONEMI ({PREDICT_DAYS} gun)")

        if coin not in all_models[n] or len(val_df) < 5:
            ax.set_title(f"{label}: Yeterli veri yok")
            continue

        model  = all_models[n][coin]
        X_val, y_val, mask_val = build_features(val_df)
        closes = val_df["Close"].values[mask_val]
        preds  = model.predict(X_val[mask_val])

        ax.plot(closes, label="Gercek Fiyat", color="royalblue", linewidth=1.2)

        long_idx  = [i for i, p in enumerate(preds) if p == 1]
        short_idx = [i for i, p in enumerate(preds) if p == 0]

        ax.scatter(long_idx,  closes[long_idx],  color="green", s=10,
                   alpha=0.6, label="Long sinyali")
        ax.scatter(short_idx, closes[short_idx], color="red",   s=10,
                   alpha=0.4, label="Dur sinyali")

        ax.set_title(label)
        ax.set_ylabel("Fiyat (USD)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    safe = coin.replace("-", "_").replace("/", "_")
    path = f"outputs/{safe}_walkforward.png"
    plt.savefig(path, dpi=120)
    print(f"  Graf kaydedildi: {path}")
    plt.close()


class WalkForwardStrategy(BaseStrategy):

    def __init__(self, trained_models):
        super().__init__()
        self.modeller = trained_models[4]

    def _confidence(self, model, x):
        proba = model.predict_proba([x])[0]
        return proba[1]

    def _leverage_and_alloc(self, confidence):
        """Yüksek leverage → düşük pay, düşük leverage → yüksek pay."""
        if confidence >= 0.75:
            return 10, 0.10
        elif confidence >= 0.68:
            return 5,  0.20
        elif confidence >= 0.62:
            return 3,  0.30
        elif confidence >= 0.57:
            return 2,  0.40
        elif confidence >= 0.52:
            return 1,  0.55
        else:
            return 0,  0.0  # sinyal yok

    def predict(self, data: dict) -> list[dict]:
        decisions  = []
        candidates = []

        for coin in COINS:
            df = data[coin]

            if coin not in self.modeller or len(df) < 51:
                candidates.append((coin, 0, 0.0, 0.0))
                continue

            X, _, mask = build_features(df)
            if not mask[-1]:
                candidates.append((coin, 0, 0.0, 0.0))
                continue

            confidence      = self._confidence(self.modeller[coin], X[-1])
            leverage, alloc = self._leverage_and_alloc(confidence)
            candidates.append((coin, leverage, alloc, confidence))

        # Toplam allocation %90'ı geçmesin
        total_alloc = sum(a for _, l, a, _ in candidates if l > 0)
        scale       = min(1.0, 0.90 / total_alloc) if total_alloc > 0 else 1.0

        for coin, leverage, alloc, confidence in candidates:
            df            = data[coin]
            current_price = df["Close"].iloc[-1]

            if leverage > 0:
                final_alloc = round(alloc * scale, 2)
                decisions.append({
                    "coin":        coin,
                    "signal":      1,
                    "allocation":  final_alloc,
                    "leverage":    leverage,
                    "stop_loss":   current_price * STOP_LOSS_PCT,
                    "take_profit": current_price * TAKE_PROFIT_PCT,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })

        return decisions


if __name__ == "__main__":
    print("📦 Veri yukleniyor...")
    coin_data = get_full_data()

    for coin, df in coin_data.items():
        print(f"  {coin}: {len(df)} satir, "
              f"{df['Date'].min()} → {df['Date'].max()}")

    print(f"\n📐 Yapilandirma:")
    print(f"  Egitim verisi : {TRAIN_DAYS} gun (0 → {TRAIN_DAYS})")
    print(f"  Tahmin donemi : {PREDICT_DAYS} gun ({START_CANDLE} → {TOTAL_DAYS})")
    print(f"  Parca buyuklugu: {CHUNK} gun")

    print("\n🏋️  Walk-forward egitim basliyor...")
    os.makedirs("models", exist_ok=True)

    all_models = {}
    for n in range(1, 5):
        print(f"\n=== Parca {n} egitim → Parca {n+1} tahmin ===")
        models = train_models(coin_data, n_chunks_train=n)
        all_models[n] = models
        for coin, model in models.items():
            safe = coin.replace("-", "_").replace("/", "_")
            joblib.dump(model, f"models/{safe}_chunk{n}.pkl")

    print("\n📊 Grafikler olusturuluyor...")
    for coin in COINS:
        plot_predictions(coin_data, all_models, coin)

    print(f"\n🚀 Backtest basliyor (start_candle={START_CANDLE})...")
    strategy = WalkForwardStrategy(all_models)
    result   = backtest.run(
        strategy=strategy,
        initial_capital=3000.0,
        start_candle=START_CANDLE,
    )
    result.print_summary()