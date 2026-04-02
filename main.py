import asyncio
import os
import time
import hmac
import hashlib
import sqlite3
import aiohttp
import numpy as np
import datetime
from urllib.parse import urlencode
from sklearn.ensemble import RandomForestClassifier

# ─── CONFIGURAÇÃO PROFISSIONAL ──────────────────────────
ENDPOINT = "https://api.binance.com"
SYMBOL = "BTCUSDT"

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET  = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# GESTÃO DE RISCO PARA 11€
STOP_LOSS_PCT = 0.007    # 0.7% (Proteção do mínimo de 10 USDT)
INITIAL_TP_PCT = 0.012   # Alvo inicial de 1.2%
TRAILING_STOP_ACTIVATE = 0.008 # Começa a "perseguir" após 0.8% de lucro
MIN_NOTIONAL = 10.5      # Margem de segurança

state = {
    "model": None,
    "position": None,
    "entry_price": None,
    "highest_price": None, # Para o Trailing Stop
    "last_train": None,
    "qty": 0
}

# ─── UTILITÁRIOS ────────────────────────────────────────
async def notify(msg):
    print(f"[LOG] {msg}")
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                             json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
        except: pass

async def binance_signed_request(method, path, params={}):
    params['timestamp'] = int(time.time() * 1000)
    query = urlencode(params)
    signature = hmac.new(SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    url = f"{ENDPOINT}{path}?{query}&signature={signature}"
    headers = {'X-MBX-APIKEY': API_KEY}
    async with aiohttp.ClientSession() as s:
        async with s.request(method, url, headers=headers) as r:
            return await r.json()

# ─── DADOS MULTI-TIMEFRAME ──────────────────────────────
async def get_data(interval="1m", limit=500):
    async with aiohttp.ClientSession() as s:
        params = {"symbol": SYMBOL, "interval": interval, "limit": limit}
        async with s.get(f"{ENDPOINT}/api/v3/klines", params=params) as r:
            data = await r.json()
            return np.array([float(x[4]) for x in data])

def get_features(c):
    returns = np.diff(c[-20:]) / c[-20:-1]
    vol = np.std(c[-30:])
    trend = (np.mean(c[-10:]) - np.mean(c[-100:])) / np.mean(c[-100:])
    return np.array([trend, vol, returns.mean()])

def train_model(c):
    X, y = [], []
    for i in range(100, len(c) - 1):
        X.append(get_features(c[:i]))
        y.append(1 if c[i+1] > c[i] else 0)
    model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
    model.fit(X, y)
    return model

# ─── EXECUÇÃO DE ORDENS ─────────────────────────────────
async def place_order(side, quantity):
    quantity = round(float(quantity), 5)
    params = {
        "symbol": SYMBOL, "side": side, "type": "MARKET", "quantity": f"{quantity:.5f}"
    }
    res = await binance_signed_request("POST", "/api/v3/order", params)
    if "orderId" in res:
        await notify(f"✅ <b>{side} EXECUTADO</b>\nPreço: MARKET\nQtd: {quantity}")
        return res
    await notify(f"❌ <b>ERRO</b>: {res.get('msg')}")
    return None

# ─── LOOP PRINCIPAL OTIMIZADO ───────────────────────────
async def run_bot():
    await notify("🚀 <b>BOT ELITE INICIADO</b>\nBanca: 11€ | Estratégia: ML + 15m Filter")
    
    while True:
        try:
            # 1. Re-treino Inteligente (Cada 3 horas)
            if state["model"] is None or (datetime.datetime.now() - state["last_train"]).seconds > 10800:
                c_train = await get_data("1m", 1000)
                state["model"] = train_model(c_train)
                state["last_train"] = datetime.datetime.now()
                await notify("🧠 Modelo ML Re-treinado.")

            # 2. Obter dados de 1m e 15m
            c1 = await get_data("1m", 150)
            c15 = await get_data("15m", 50)
            
            price = c1[-1]
            prob = state["model"].predict_proba(get_features(c1).reshape(1, -1))[0][1]
            
            # Validação de Tendência Macro (EMA de 15m simplificada)
            trend_15m_up = np.mean(c15[-5:]) > np.mean(c15[-20:])

            # 3. Lógica de Entrada
            if state["position"] is None:
                # SÓ ENTRA SE: Probabilidade alta + Tendência de 15m a subir
                if prob > 0.72 and trend_15m_up:
                    acc = await binance_signed_request("GET", "/api/v3/account")
                    balance = next((float(b['free']) for b in acc['balances'] if b['asset'] == 'USDT'), 0)
                    
                    if balance >= MIN_NOTIONAL:
                        qty = (balance * 0.97) / price # 3% de folga para taxas
                        order = await place_order("BUY", qty)
                        if order:
                            state["position"] = "LONG"
                            state["entry_price"] = price
                            state["highest_price"] = price
                            state["qty"] = qty
                    else:
                        print(f"Aguardando saldo... ({balance} USDT)")

            # 4. Lógica de Saída (Trailing Stop + SL)
            elif state["position"] == "LONG":
                if price > state["highest_price"]:
                    state["highest_price"] = price
                
                entry = state["entry_price"]
                highest = state["highest_price"]
                
                # Cálculo de Trailing: Se subiu 0.8%, o stop sobe para garantir lucro
                profit_pct = (price - entry) / entry
                
                # Condição de Saída 1: Stop Loss Fixo
                stop_triggered = price <= entry * (1 - STOP_LOSS_PCT)
                
                # Condição de Saída 2: Trailing Stop (Caiu 0.5% desde o topo atingido)
                trailing_triggered = (profit_pct > TRAILING_STOP_ACTIVATE) and (price <= highest * 0.995)
                
                # Condição de Saída 3: Take Profit Final (1.5%)
                tp_triggered = price >= entry * (1 + INITIAL_TP_PCT)

                if stop_triggered or trailing_triggered or tp_triggered:
                    order = await place_order("SELL", state["qty"])
                    if order:
                        state["position"] = None
                        final_pnl = (price - entry) / entry * 100
                        msg = "🔴 STOP LOSS" if stop_triggered else "🟢 PROFIT"
                        await notify(f"💰 <b>{msg}</b>\nPnL: {final_pnl:.2f}%")

        except Exception as e:
            print(f"Erro: {e}")
        
        await asyncio.sleep(15) # Mais rápido para scalping

if __name__ == "__main__":
    asyncio.run(run_bot())
