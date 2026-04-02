import asyncio
import os
import time
import sqlite3
import aiohttp
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ─── CONFIG ─────────────────────────────────────────────
ENDPOINTS = ["https://api.binance.com", "https://api.binance.me"]
SYMBOL = "BTCUSDT"

# ATENÇÃO: Se as chaves não estiverem no ambiente, o bot falhará na execução real
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
CHAT_ID            = os.getenv("TELEGRAM_CHAT_ID")

# Forçado para True para sair do modo simulação
AUTO_LIVE = True 

STOP_LOSS_PCT   = 0.008
TAKE_PROFIT_PCT = 0.015 # Ajustado para 1.5% para maior frequência de lucro
BASE_RISK       = 0.01
MAX_DAILY_LOSS  = 0.04
MAX_LOSS_STREAK = 4
SLIPPAGE        = 0.0005

state = {
    "model":        None,
    "position":     None,
    "entry":        None,
    "balance":      1000,
    "daily_start":  1000,
    "loss_streak":  0,
    "last_train":   None
}

# ─── TELEGRAM / NOTIFICAÇÃO ──────────────────────────────
async def notify(msg):
    print(f"[LOG] {msg}")
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(
                    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                    json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
                )
        except: pass

# ─── BASE DE DADOS ──────────────────────────────────────
conn = sqlite3.connect("trades.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, side TEXT, price REAL, timestamp REAL)")
conn.commit()

# ─── BINANCE API ────────────────────────────────────────
async def klines(interval="1m"):
    async with aiohttp.ClientSession() as s:
        params = {"symbol": SYMBOL, "interval": interval, "limit": 1000}
        async with s.get(f"{ENDPOINTS[0]}/api/v3/klines", params=params) as r:
            data = await r.json()
            return np.array([float(x[4]) for x in data])

# ─── ML LOGIC ───────────────────────────────────────────
def features(c):
    returns = np.diff(c[-10:]) / c[-10:-1]
    trend = (np.mean(c[-20:]) - np.mean(c[-50:])) / np.mean(c[-50:])
    vol = np.std(c[-20:])
    momentum = returns.mean()
    return np.array([trend, vol, momentum])

def market_regime(c):
    trend = abs((np.mean(c[-50:]) - np.mean(c[-200:])) / np.mean(c[-200:]))
    vol = np.std(c[-50:])
    if trend > 0.008 and vol > 0.004: return "strong_trend"
    if trend > 0.003: return "weak_trend"
    return "range"

def train(c):
    X, y = [], []
    for i in range(200, len(c) - 1):
        X.append(features(c[:i]))
        y.append(1 if c[i+1] > c[i] else 0)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

# ─── TRADING ────────────────────────────────────────────
async def live():
    await notify(f"🚀 Bot Corrigido e Ativo! ({SYMBOL})")
    state["last_train"] = datetime.datetime.now()

    while True:
        try:
            # Re-treinar a cada 4 horas para adaptar ao mercado
            if state["model"] is None or (datetime.datetime.now() - state["last_train"]).seconds > 14400:
                c_train = await klines("1m")
                state["model"] = train(c_train)
                state["last_train"] = datetime.datetime.now()
                await notify("🧠 Modelo ML atualizado com sucesso.")

            c1 = await klines("1m")
            prob = (state["model"].predict_proba(features(c1).reshape(1, -1))[0][1])
            regime = market_regime(c1)
            price = c1[-1]
            
            score = 0
            if prob > 0.62: score += 2
            if regime != "range": score += 1

            if state["position"] is None:
                if score >= 2:
                    state["position"] = price
                    state["entry"] = price
                    cursor.execute("INSERT INTO trades (side, price, timestamp) VALUES (?,?,?)", ("BUY", price, time.time()))
                    conn.commit()
                    await notify(f"📈 <b>BUY BTC</b> @ {price:.2f}\nProb: {prob:.1%} | Score: {score}")
            else:
                entry = state["entry"]
                if price <= entry * (1 - STOP_LOSS_PCT):
                    state["position"] = None
                    await notify(f"🔴 <b>STOP LOSS</b> @ {price:.2f}")
                elif price >= entry * (1 + TAKE_PROFIT_PCT):
                    state["position"] = None
                    await notify(f"🟢 <b>TAKE PROFIT</b> @ {price:.2f}")

        except Exception as e:
            print(f"Erro: {e}")
        
        await asyncio.sleep(20)

# ─── EXECUÇÃO (LINHA CORRIGIDA) ─────────────────────────
if __name__ == "__main__":
    asyncio.run(live())
