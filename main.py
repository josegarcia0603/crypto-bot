import asyncio, os, aiohttp, numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─── CONFIG ───────────────────────
BASE = "https://api.binance.com"
SYMBOL = "BTCUSDT"

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

AUTO_LIVE = os.getenv("AUTO_LIVE", "false").lower() == "true"

STOP_LOSS_PCT = 0.008
TAKE_PROFIT_PCT = 0.025

BASE_RISK = 0.01
MAX_DAILY_LOSS = 0.04
MAX_LOSS_STREAK = 4

SLIPPAGE = 0.0005

state = {
    "model": None,
    "position": None,
    "entry": None,
    "balance": 1000,
    "daily_start": 1000,
    "loss_streak": 0,
    "daily_pnl": 0
}

# ─── TELEGRAM ─────────────────────
async def notify(msg):
    if not TELEGRAM_TOKEN:
        return
    async with aiohttp.ClientSession() as s:
        try:
            await s.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": msg}
            )
        except:
            pass

# ─── DATA ─────────────────────────
    
    async def klines(interval="1m"):

    async with aiohttp.ClientSession() as s:

        async with s.get(f"{BASE}/api/v3/klines", params={
            "symbol": SYMBOL,
            "interval": interval,
            "limit": 300
        }) as r:

            try:
                data = await r.json()
            except Exception:
                raise Exception("Resposta não é JSON válido")

    # 🔴 detectar erro da Binance
    if isinstance(data, dict):
        raise Exception(f"Erro da Binance: {data}")

    if not isinstance(data, list):
        raise Exception(f"Formato inesperado: {type(data)}")

    closes = []

    for x in data:
        if not isinstance(x, list) or len(x) < 5:
            continue

        try:
            closes.append(float(x[4]))
        except:
            continue

    if len(closes) < 20:
        raise Exception("Dados insuficientes recebidos da Binance")

    return np.array(closes)

# ─── FEATURES ─────────────────────
def features(c):
    returns = np.diff(c[-10:]) / c[-10:-1]
    trend = (np.mean(c[-20:]) - np.mean(c[-50:])) / np.mean(c[-50:])
    macro = (np.mean(c[-50:]) - np.mean(c[-200:])) / np.mean(c[-200:])
    vol = np.std(c[-20:])
    momentum = returns.mean()

    return np.array([trend, macro, vol, momentum])

# ─── REGIME ───────────────────────
def market_regime(c):
    trend = abs((np.mean(c[-50:]) - np.mean(c[-200:])) / np.mean(c[-200:]))
    vol = np.std(c[-50:])

    if trend > 0.012 and vol > 0.006:
        return "strong_trend"
    if trend > 0.006:
        return "weak_trend"
    return "range"

# ─── MODEL ────────────────────────
def train(c):
    X, y = [], []
    for i in range(100, len(c)-1):
        X.append(features(c[:i]))
        y.append(1 if c[i+1] > c[i] else 0)

    rf = RandomForestClassifier(n_estimators=200)
    gb = GradientBoostingClassifier(n_estimators=200)

    rf.fit(X, y)
    gb.fit(X, y)

    return rf, gb

def predict(models, c):
    rf, gb = models
    f = features(c).reshape(1, -1)
    return (rf.predict_proba(f)[0][1] + gb.predict_proba(f)[0][1]) / 2

# ─── RISCO ────────────────────────
def risk_multiplier():
    return max(0.3, 1 - (state["loss_streak"] * 0.2))

def position_size(price):
    return (state["balance"] * BASE_RISK * risk_multiplier()) / price

def risk_block():
    if (state["balance"] - state["daily_start"]) / state["daily_start"] <= -MAX_DAILY_LOSS:
        return True
    return False

# ─── TRAILING ─────────────────────
def trailing(entry, price):
    return max(entry, price * (1 - SLIPPAGE))

# ─── EXECUÇÃO ─────────────────────
async def execute_trade(side, price):
    if not AUTO_LIVE:
        return

    # aqui conectarias à Binance real
    await notify(f"⚡ EXEC {side} @ {price}")

# ─── LIVE ─────────────────────────
async def live():

    await notify("🚀 Bot iniciado")

    while True:
        try:

            if risk_block():
                await notify("⛔ Bloqueado por risco diário")
                await asyncio.sleep(300)
                continue

            if state["loss_streak"] >= MAX_LOSS_STREAK:
                await notify("⛔ Stop por perdas consecutivas")
                await asyncio.sleep(600)
                continue

            c1 = await klines("1m")
            c5 = await klines("5m")

            if state["model"] is None:
                state["model"] = train(c1)

            prob = predict(state["model"], c1)

            regime = market_regime(c1)

            score = 0
            if prob > 0.8: score += 2
            if regime == "strong_trend": score += 2

            price = c1[-1]

            # ─── ENTRADA ───
            if state["position"] is None:

                if score >= 3:

                    state["position"] = price
                    state["entry"] = price

                    await execute_trade("BUY", price)
                    await notify(f"📈 BUY {price}")

            # ─── SAÍDA ───
            else:

                entry = state["entry"]

                stop = entry * (1 - STOP_LOSS_PCT)
                take = entry * (1 + TAKE_PROFIT_PCT)

                state["entry"] = trailing(entry, price)

                if price <= stop:

                    pnl = (price - entry) / entry
                    state["balance"] *= (1 + pnl)

                    state["position"] = None
                    state["loss_streak"] += 1

                    await execute_trade("SELL", price)
                    await notify(f"🔴 STOP {price}")

                elif price >= take:

                    pnl = (price - entry) / entry
                    state["balance"] *= (1 + pnl)

                    state["position"] = None
                    state["loss_streak"] = 0

                    await execute_trade("SELL", price)
                    await notify(f"🟢 TP {price}")

        except Exception as e:
            await notify(f"⚠️ ERRO: {e}")

        await asyncio.sleep(30)

# ─── MAIN ─────────────────────────
async def main():
    await live()

if __name__ == "__main__":
    asyncio.run(main())
