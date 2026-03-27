import asyncio
import os
import time
import sqlite3
import aiohttp
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─── CONFIG ─────────────────────────────────────────────
ENDPOINTS = [
    "https://api.binance.me",
    "https://api.binance.com",
    "https://api.binance.com/api"
]

SYMBOL = "BTCUSDT"

BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("TELEGRAM_CHAT_ID")
AUTO_LIVE          = os.getenv("AUTO_LIVE", "false").lower() == "true"

STOP_LOSS_PCT   = 0.008
TAKE_PROFIT_PCT = 0.025
BASE_RISK       = 0.01
MAX_DAILY_LOSS  = 0.04
MAX_LOSS_STREAK = 4
SLIPPAGE        = 0.0005

state = {
    "model":       None,
    "position":    None,
    "entry":       None,
    "balance":     1000,
    "daily_start": 1000,
    "loss_streak": 0,
    "daily_pnl":   0,
    "BASE":        None,
    "paused":      False
}

# ─── BASE DE DADOS ──────────────────────────────────────
conn = sqlite3.connect("trades.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        side TEXT,
        price REAL,
        timestamp REAL
    )
""")
conn.commit()

def save_trade(side, price):
    cursor.execute(
        "INSERT INTO trades (side, price, timestamp) VALUES (?, ?, ?)",
        (side, price, time.time())
    )
    conn.commit()

def get_trades():
    cursor.execute("SELECT side, price, timestamp FROM trades ORDER BY id DESC LIMIT 10")
    return cursor.fetchall()

# ─── TELEGRAM ───────────────────────────────────────────
async def notify(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(f"[LOG] {msg}")
        return
    try:
        async with aiohttp.ClientSession() as s:
            resp = await s.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=10)
            )
            result = await resp.json()
            if not result.get("ok"):
                print(f"[TELEGRAM ERRO] {result}")
    except Exception as e:
        print(f"[TELEGRAM ERRO] {e}")

async def send_menu():
    keyboard = {
        "inline_keyboard": [
            [{"text": "📊 Dashboard", "callback_data": "dashboard"}],
            [{"text": "💹 Trades Abertos", "callback_data": "trades_abertos"}],
            [{"text": "🔍 Analisar Pares", "callback_data": "analisa_pares"}],
            [{"text": "💰 Saldo", "callback_data": "saldo"}],
            [
                {"text": "⏸ Pausar", "callback_data": "pausar"},
                {"text": "▶️ Retomar", "callback_data": "retomar"}
            ],
            [{"text": "❌ Fechar Trades", "callback_data": "fechar_trades"}]
        ]
    }
    await notify("🤖 Menu do Bot ativo!\nClique em qualquer botão.")
    async with aiohttp.ClientSession() as s:
        await s.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": "Escolha uma opção:", "reply_markup": keyboard}
        )

# ─── HANDLER DE BOTÕES ──────────────────────────────────
async def handle_callback(update):
    if "callback_query" not in update:
        return
    data = update["callback_query"]["data"]

    if data == "dashboard":
        msg = f"📊 Dashboard:\nSaldo: ${state['balance']:.2f}\nPosition: {state['position'] or 'Nenhuma'}\nLoss streak: {state['loss_streak']}"
        await notify(msg)
    elif data == "trades_abertos":
        trades = get_trades()
        if trades:
            msg = "💹 Últimos trades:\n"
            for t in trades:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t[2]))
                msg += f"{ts} | {t[0]} @ {t[1]:.2f}\n"
        else:
            msg = "💹 Nenhum trade aberto."
        await notify(msg)
    elif data == "analisa_pares":
        await notify("🔍 Analisando pares... (em breve funcional)")
    elif data == "saldo":
        await notify(f"💰 Saldo atual: ${state['balance']:.2f}")
    elif data == "pausar":
        state["paused"] = True
        await notify("⏸ Bot pausado.")
    elif data == "retomar":
        state["paused"] = False
        await notify("▶️ Bot retomado.")
    elif data == "fechar_trades":
        state["position"] = None
        await notify("❌ Todos trades fechados.")

# ─── HELPER BINANCE COM FAILOVER ───────────────────────
async def binance_request(path, params=None):
    last_error = None
    for base in ENDPOINTS:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{base}{path}", params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    data = await r.json()
                    if isinstance(data, dict) and data.get("code"):
                        raise Exception(f"Erro Binance {data}")
                    state["BASE"] = base
                    return data
        except Exception as e:
            last_error = e
            continue
    raise Exception(f"Todos endpoints falharam: {last_error}")

# ─── DATA ───────────────────────────────────────────────
async def klines(interval="1m"):
    data = await binance_request("/api/v3/klines", {"symbol": SYMBOL, "interval": interval, "limit": 300})
    closes = []
    for x in data:
        if not isinstance(x, list) or len(x) < 5:
            continue
        try: closes.append(float(x[4]))
        except: continue
    if len(closes) < 20:
        raise Exception("Dados insuficientes da Binance")
    return np.array(closes)

# ─── FEATURES / ML ─────────────────────────────────────
def features(c):
    returns = np.diff(c[-10:]) / c[-10:-1]
    trend = (np.mean(c[-20:]) - np.mean(c[-50:])) / np.mean(c[-50:])
    macro = (np.mean(c[-50:]) - np.mean(c[-200:])) / np.mean(c[-200:])
    vol = np.std(c[-20:])
    momentum = returns.mean()
    return np.array([trend, macro, vol, momentum])

def market_regime(c):
    trend = abs((np.mean(c[-50:]) - np.mean(c[-200:])) / np.mean(c[-200:]))
    vol = np.std(c[-50:])
    if trend > 0.012 and vol > 0.006: return "strong_trend"
    if trend > 0.006: return "weak_trend"
    return "range"

def train(c):
    X, y = [], []
    for i in range(100, len(c) - 1):
        X.append(features(c[:i]))
        y.append(1 if c[i + 1] > c[i] else 0)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    gb.fit(X, y)
    return rf, gb

def predict(models, c):
    rf, gb = models
    f = features(c).reshape(1, -1)
    return (rf.predict_proba(f)[0][1] + gb.predict_proba(f)[0][1]) / 2

# ─── RISCO / TRADING ───────────────────────────────────
def risk_multiplier(): return max(0.3, 1 - (state["loss_streak"]*0.2))
def position_size(price): return (state["balance"]*BASE_RISK*risk_multiplier())/price
def risk_block(): return (state["balance"] - state["daily_start"]) / state["daily_start"] <= -MAX_DAILY_LOSS
def trailing(entry, price): return max(entry, price*(1-SLIPPAGE))

async def execute_trade(side, price):
    save_trade(side, price)
    if AUTO_LIVE: await notify(f"⚡ EXEC {side} @ {price:.2f}")

# ─── LOOP PRINCIPAL ────────────────────────────────────
async def live():
    await send_menu()
    await notify(f"🚀 Bot iniciado! Símbolo: {SYMBOL} | Modo: {'LIVE' if AUTO_LIVE else 'SIMULAÇÃO'}")

    while True:
        try:
            if state["paused"]:
                await asyncio.sleep(30)
                continue
            if risk_block():
                await notify("⛔ Bloqueado — risco diário máximo atingido")
                await asyncio.sleep(300)
                continue
            if state["loss_streak"] >= MAX_LOSS_STREAK:
                await notify(f"⛔ Stop — {MAX_LOSS_STREAK} perdas consecutivas")
                await asyncio.sleep(600)
                continue

            c1 = await klines("1m")
            c5 = await klines("5m")

            if state["model"] is None:
                await notify("🧠 A treinar modelo ML... aguarda ~30s")
                state["model"] = train(c1)
                await notify("✅ Modelo treinado! A analisar mercado...")

            prob = predict(state["model"], c1)
            regime = market_regime(c1)
            price = c1[-1]
            score = 0
            if prob > 0.8: score += 2
            if regime == "strong_trend": score += 2

            if state["position"] is None:
                if score >= 3:
                    state["position"] = price
                    state["entry"] = price
                    await execute_trade("BUY", price)
                    await notify(f"📈 BUY @ {price:.2f} | Score: {score}/4 | Regime: {regime} | Prob: {prob:.1%}")
            else:
                entry = state["entry"]
                stop = entry*(1-STOP_LOSS_PCT)
                take = entry*(1+TAKE_PROFIT_PCT)
                state["entry"] = trailing(entry, price)

                if price <= stop:
                    pnl = (price-entry)/entry
                    state["balance"] *= (1+pnl)
                    state["position"] = None
                    state["loss_streak"] += 1
                    await execute_trade("SELL", price)
                    await notify(f"🔴 STOP LOSS @ {price:.2f} | PnL: {pnl*100:.2f}% | Saldo: {state['balance']:.2f} | Loss streak: {state['loss_streak']}")
                elif price >= take:
                    pnl = (price-entry)/entry
                    state["balance"] *= (1+pnl)
                    state["position"] = None
                    state["loss_streak"] = 0
                    await execute_trade("SELL", price)
                    await notify(f"🟢 TAKE PROFIT @ {price:.2f} | PnL: {pnl*100:.2f}% | Saldo: {state['balance']:.2f}")

        except Exception as e:
            await notify(f"⚠️ ERRO: {e}")
            print(f"[ERRO] {e}")

        await asyncio.sleep(30)

# ─── MAIN ──────────────────────────────────────────────
async def main():
    await live()

if __name__ == "__main__":
    asyncio.run(main())
    await live()

if __name__ == "__main__":
    asyncio.run(main())
