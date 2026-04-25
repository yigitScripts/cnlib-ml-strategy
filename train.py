import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
from features import build_features

COINS = [
    "kapcoin-usd_train",
    "metucoin-usd_train",
    "tamcoin-usd_train",
]

TOTAL_DAYS   = 1570
PREDICT_DAYS = 368
TRAIN_DAYS   = TOTAL_DAYS - PREDICT_DAYS  # 1202
START_CANDLE = TRAIN_DAYS                 # 1202
CHUNK        = TRAIN_DAYS // 4            # 300

STOP_LOSS_PCT   = 0.05
TAKE_PROFIT_PCT = 0.15


# ── Veri toplama ──────────────────────────────────────────────────────────────

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

    backtest.run(strategy=TempStrategy(), initial_capital=3000.0, silent=True)
    return collected


# ── Veri bölme ────────────────────────────────────────────────────────────────

def split_by_chunk(df, n):
    train_end = min(n * CHUNK, TRAIN_DAYS)
    val_end   = min(train_end + CHUNK, TRAIN_DAYS)
    return df.iloc[:train_end], df.iloc[train_end:val_end]


# ── Model eğitimi ─────────────────────────────────────────────────────────────

def make_model(seed=42):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=15,
        random_state=seed,
        n_jobs=-1,
    )

def make_model_gb(seed=42):
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=seed,
    )


def train_models(coin_data, n):
    """
    Her coin için iki model eğitir:
        long_model  → yarın fiyat yükselecek mi?
        short_model → yarın fiyat düşecek mi?
    Her biri için RF + GB ensemble kullanır.
    """
    models = {}
    for coin in COINS:
        df = coin_data[coin]
        train_df, val_df = split_by_chunk(df, n)

        print(f"  [{coin}] train={len(train_df)} gun, val={len(val_df)} gun")

        X_tr, y_long_tr, y_short_tr, mask_tr = build_features(train_df)
        X_tr      = X_tr[mask_tr]
        y_long_tr = y_long_tr[mask_tr]
        y_short_tr = y_short_tr[mask_tr]

        if len(X_tr) < 20:
            print(f"  [{coin}] Yetersiz veri.")
            continue

        # Long modelleri
        rf_long = make_model(42)
        gb_long = make_model_gb(42)
        rf_long.fit(X_tr, y_long_tr)
        gb_long.fit(X_tr, y_long_tr)

        # Short modelleri
        rf_short = make_model(99)
        gb_short = make_model_gb(99)
        rf_short.fit(X_tr, y_short_tr)
        gb_short.fit(X_tr, y_short_tr)

        if len(val_df) > 51:
            X_v, y_long_v, y_short_v, mask_v = build_features(val_df)
            X_v = X_v[mask_v]
            if len(X_v) > 0:
                # Ensemble tahmin: RF + GB ortalaması
                lp = (rf_long.predict_proba(X_v)[:, 1] +
                      gb_long.predict_proba(X_v)[:, 1]) / 2
                sp = (rf_short.predict_proba(X_v)[:, 1] +
                      gb_short.predict_proba(X_v)[:, 1]) / 2
                acc_l = accuracy_score(y_long_v[mask_v],  (lp > 0.5).astype(int))
                acc_s = accuracy_score(y_short_v[mask_v], (sp > 0.5).astype(int))
                print(f"  [{coin}] Parca {n}→{n+1}  "
                      f"long_acc={acc_l:.3f}  short_acc={acc_s:.3f}")

        models[coin] = {
            "rf_long":  rf_long,
            "gb_long":  gb_long,
            "rf_short": rf_short,
            "gb_short": gb_short,
        }

    return models


# ── Grafik ────────────────────────────────────────────────────────────────────

def plot_predictions(coin_data, all_models, coin):
    df = coin_data[coin]
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    fig.suptitle(f"{coin} — Walk-Forward (Long & Short Sinyaller)", fontsize=13)

    for n in range(1, 5):
        ax = axes[n - 1]
        _, val_df = split_by_chunk(df, n)

        label = (f"Parca {n}→{n+1}"
                 if n < 4 else f"Parca {n} → TAHMIN ({PREDICT_DAYS} gun)")

        if coin not in all_models[n] or len(val_df) < 5:
            ax.set_title(f"{label}: Yeterli veri yok")
            continue

        m = all_models[n][coin]
        X_v, _, _, mask_v = build_features(val_df)
        closes = val_df["Close"].values[mask_v]

        lp = (m["rf_long"].predict_proba(X_v[mask_v])[:, 1] +
              m["gb_long"].predict_proba(X_v[mask_v])[:, 1]) / 2
        sp = (m["rf_short"].predict_proba(X_v[mask_v])[:, 1] +
              m["gb_short"].predict_proba(X_v[mask_v])[:, 1]) / 2

        ax.plot(closes, color="royalblue", linewidth=1.2, label="Fiyat")

        long_idx  = [i for i, p in enumerate(lp) if p >= 0.55]
        short_idx = [i for i, p in enumerate(sp) if p >= 0.55]

        ax.scatter(long_idx,  closes[long_idx],
                   color="lime",   s=12, alpha=0.7, label="Long")
        ax.scatter(short_idx, closes[short_idx],
                   color="red",    s=12, alpha=0.7, label="Short")

        ax.set_title(label)
        ax.set_ylabel("Fiyat")
        ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    safe = coin.replace("-", "_").replace("/", "_")
    path = f"outputs/{safe}_advanced.png"
    plt.savefig(path, dpi=120)
    print(f"  Graf kaydedildi: {path}")
    plt.close()


# ── Strateji ──────────────────────────────────────────────────────────────────

class AdvancedStrategy(BaseStrategy):

    def __init__(self, trained_models):
        super().__init__()
        self.modeller = trained_models[4]

    # ── Regime tespiti ────────────────────────────────────────────────────────
    def _regime(self, df):
        """
        Piyasa rejimini tespit et.
        Döndürür: 'bull', 'bear', 'sideways'
        Çarpan:   bull=1.2, bear=1.2 (short için), sideways=0.6
        """
        closes = df["Close"].values
        if len(closes) < 200:
            return "sideways", 0.6

        ma50  = closes[-50:].mean()  if len(closes) >= 50  else closes.mean()
        ma200 = closes[-200:].mean() if len(closes) >= 200 else closes.mean()
        price = closes[-1]

        # Volatilite (sideways tespiti)
        std20 = closes[-20:].std() / price if len(closes) >= 20 else 0.02

        if price > ma50 > ma200:
            return "bull", 1.2        # güçlü boğa
        elif price < ma50 < ma200:
            return "bear", 1.2        # güçlü ayı (short için çarpan)
        elif std20 < 0.015:
            return "sideways", 0.6    # yatay → daha temkinli
        else:
            return "neutral", 0.9

    # ── Ensemble confidence ───────────────────────────────────────────────────
    def _signals(self, coin, X_last):
        m  = self.modeller[coin]
        lp = (m["rf_long"].predict_proba([X_last])[0][1] +
              m["gb_long"].predict_proba([X_last])[0][1]) / 2
        sp = (m["rf_short"].predict_proba([X_last])[0][1] +
              m["gb_short"].predict_proba([X_last])[0][1]) / 2
        return lp, sp

    # ── Risk kademesi ─────────────────────────────────────────────────────────
    def _risk_tier(self, confidence, regime_mult):
        """
        Yüksek leverage → düşük pay
        Düşük leverage  → yüksek pay
        Regime çarpanı ile ölçekle.
        """
        c = confidence * regime_mult
        if c >= 0.90:
            return 10, 0.08
        elif c >= 0.80:
            return 5,  0.18
        elif c >= 0.70:
            return 3,  0.28
        elif c >= 0.62:
            return 2,  0.38
        elif c >= 0.55:
            return 1,  0.50
        else:
            return 0,  0.0   # sinyal yok

    # ── Ana karar ─────────────────────────────────────────────────────────────
    def predict(self, data: dict) -> list[dict]:
        decisions  = []
        candidates = []

        for coin in COINS:
            df = data[coin]

            if coin not in self.modeller or len(df) < 51:
                candidates.append((coin, 0, 0.0, 0.0))
                continue

            X, _, _, mask = build_features(df)
            if not mask[-1]:
                candidates.append((coin, 0, 0.0, 0.0))
                continue

            regime, regime_mult = self._regime(df)
            long_p, short_p     = self._signals(coin, X[-1])

            # Regime'e göre hangi yönü tercih edelim
            if regime == "bull":
                # Boğada: long sinyali varsa aç, short sinyali çok güçlü değilse yoksay
                if long_p >= 0.52:
                    lev, alloc = self._risk_tier(long_p, regime_mult)
                    candidates.append((coin, 1, alloc, lev))
                elif short_p >= 0.70:   # çok güçlü short sinyali bile olsa dikkatli
                    lev, alloc = self._risk_tier(short_p * 0.8, regime_mult * 0.7)
                    candidates.append((coin, -1, alloc, lev))
                else:
                    candidates.append((coin, 0, 0.0, 0))

            elif regime == "bear":
                # Ayıda: short sinyali varsa aç, long çok güçlüyse al
                if short_p >= 0.52:
                    lev, alloc = self._risk_tier(short_p, regime_mult)
                    candidates.append((coin, -1, alloc, lev))
                elif long_p >= 0.70:
                    lev, alloc = self._risk_tier(long_p * 0.8, regime_mult * 0.7)
                    candidates.append((coin, 1, alloc, lev))
                else:
                    candidates.append((coin, 0, 0.0, 0))

            else:
                # Neutral/Sideways: her iki yön de mümkün ama daha küçük pozisyon
                if long_p > short_p and long_p >= 0.57:
                    lev, alloc = self._risk_tier(long_p, regime_mult)
                    candidates.append((coin, 1, alloc, lev))
                elif short_p > long_p and short_p >= 0.57:
                    lev, alloc = self._risk_tier(short_p, regime_mult)
                    candidates.append((coin, -1, alloc, lev))
                else:
                    candidates.append((coin, 0, 0.0, 0))

        # Toplam allocation %88'i geçmesin
        total_alloc = sum(a for _, sig, a, l in candidates if l > 0)
        scale       = min(1.0, 0.88 / total_alloc) if total_alloc > 0 else 1.0

        for coin, signal, alloc, leverage in candidates:
            df            = data[coin]
            current_price = df["Close"].iloc[-1]

            if leverage > 0 and signal != 0:
                final_alloc = round(alloc * scale, 2)
                decisions.append({
                    "coin":        coin,
                    "signal":      signal,
                    "allocation":  final_alloc,
                    "leverage":    leverage,
                    "stop_loss":   current_price * (1 - STOP_LOSS_PCT)
                                   if signal == 1
                                   else current_price * (1 + STOP_LOSS_PCT),
                    "take_profit": current_price * (1 + TAKE_PROFIT_PCT)
                                   if signal == 1
                                   else current_price * (1 - TAKE_PROFIT_PCT),
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })

        return decisions


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📦 Veri yukleniyor...")
    coin_data = get_full_data()

    for coin, df in coin_data.items():
        print(f"  {coin}: {len(df)} satir  "
              f"{df['Date'].min()} → {df['Date'].max()}")

    print(f"\n📐 Yapilandirma:")
    print(f"  Egitim : {TRAIN_DAYS} gun  (0 → {TRAIN_DAYS})")
    print(f"  Tahmin : {PREDICT_DAYS} gun  ({START_CANDLE} → {TOTAL_DAYS})")
    print(f"  Parca  : {CHUNK} gun x 4")

    print("\n🏋️  Walk-forward egitim (Long + Short + Ensemble)...")
    os.makedirs("models", exist_ok=True)

    all_models = {}
    for n in range(1, 5):
        print(f"\n=== Parca {n} → Parca {n + 1} ===")
        models = train_models(coin_data, n)
        all_models[n] = models
        for coin, m in models.items():
            safe = coin.replace("-", "_").replace("/", "_")
            for key, mdl in m.items():
                joblib.dump(mdl, f"models/{safe}_chunk{n}_{key}.pkl")

    print("\n📊 Grafikler olusturuluyor...")
    for coin in COINS:
        plot_predictions(coin_data, all_models, coin)

    print(f"\n🚀 Backtest basliyor (start_candle={START_CANDLE})...")
    strategy = AdvancedStrategy(all_models)
    result   = backtest.run(
        strategy=strategy,
        initial_capital=3000.0,
        start_candle=START_CANDLE,
    )
    result.print_summary()