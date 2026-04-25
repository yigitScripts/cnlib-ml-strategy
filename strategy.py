import joblib
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
from features import build_features

COINS = [
    "kapcoin-usd_train",
    "metucoin-usd_train",
    "tamcoin-usd_train",
]


class FinalStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.modeller = {}

    def yukle(self):
        for coin in COINS:
            safe = coin.replace("-", "_").replace("/", "_")
            path = f"models/{safe}_year4.pkl"
            self.modeller[coin] = joblib.load(path)
            print(f"Model yuklendi: {path}")

    def predict(self, data: dict) -> list[dict]:
        long_coins = []
        for coin in COINS:
            df = data[coin]
            if coin not in self.modeller or len(df) < 51:
                long_coins.append((coin, 0))
                continue
            X, _, mask = build_features(df)
            if not mask[-1]:
                long_coins.append((coin, 0))
                continue
            tahmin = self.modeller[coin].predict([X[-1]])[0]
            long_coins.append((coin, tahmin))

        actives = [c for c, s in long_coins if s == 1]
        alloc   = round(1.0 / len(actives), 2) if actives else 0.0

        STOP_LOSS_PCT   = 0.05   # %5 zarar → pozisyonu kapat
        TAKE_PROFIT_PCT = 0.15   # %15 kâr  → pozisyonu kapat

        decisions = []
        for coin, signal in long_coins:
            df            = data[coin]
            current_price = df["Close"].iloc[-1]
            if signal == 1:
                decisions.append({
                    "coin":        coin,
                    "signal":      1,
                    "allocation":  alloc,
                    "leverage":    2,
                    "stop_loss":   current_price * (1 - STOP_LOSS_PCT),
                    "take_profit": current_price * (1 + TAKE_PROFIT_PCT),
                })
            else:
                decisions.append({"coin": coin, "signal": 0,
                                   "allocation": 0.0, "leverage": 1})
        return decisions


if __name__ == "__main__":
    strategy = FinalStrategy()
    strategy.get_data()
    strategy.yukle()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()