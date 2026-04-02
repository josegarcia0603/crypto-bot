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

# ─── CONFIGURAÇÃO ───────────────────────────────────────
ENDPOINT = "https://api.binance.com"
SYMBOL = "BTCUSDT"

API_KEY = os.getenv("BINANCE_API_KEY")
SECRET  = os.getenv("BINANCE_SECRET_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

AUTO_LIVE = True 
STOP_LOSS_PCT = 0.008
TAKE_PROFIT_PCT = 0.018
BASE_RISK = 0.01  # Usa 1% do saldo por trade

state = {
    "model": None,
    "position": None,
    "entry_price": None,
    "last_train": None,
    "active_order_id": None
}

# ─── UTILITÁRIOS DE REDE ────────────────────────────────
async def notify(msg):
    print(f"[LOG] {msg}")
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                             json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
        except: pass

async def binance_signed_request(method, path, params={}):
    if not API_KEY or not SECRET:
        raise Exception("Chaves API ausentes! Configura as variáveis de ambiente.")
    
    params['timestamp'] = int(time.time() * 1000)
    query = urlencode(params)
    signature = hmac.new(SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    url = f"{ENDPOINT}{path}?{query}&signature={signature}"
    
    headers = {'X-MBX-APIKEY': API_KEY}
    async with aiohttp.ClientSession() as s:
        async with s.request(method, url, headers=headers) as r:
            return await r.json()

# ─── EXECUÇÃO DE ORDENS REAIS ───────────────────────────
async def place_order(side, quantity):
    params = {
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": quantity
    }
    res = await binance_signed_request("POST", "/api/v3/order", params)
    if "orderId" in res:
        await notify(f"✅ <b>ORDEM EXECUTADA</b>\nLado: {side}\nPreço: MARKET\nID: {res['orderId']}")
        return res
    else:
        await notify(f"❌ <b>ERRO NA BINANCE</b>\n{res.get('msg', 'Erro desconhecido')}")
        return None

# ─── LOGICA DE MERCADO / ML ─────────────────────────────
async def get_klines():
    async with aiohttp.ClientSession() as s:
        params = {"symbol": SYMBOL, "interval": "1m", "limit": 500}
        async with s.get(f"{ENDPOINT}/api/v3/klines", params=params) as r:
            data = await r.json()
            return np.array([float(x[4]) for x in data])

def get_features(c):
    returns = np.diff(c[-15:]) / c[-15:-1]
    vol = np.std(c[-20:])
    trend = (np.mean(c[-10:]) - np.mean(c[-50:])) / np.mean(c[-50:])
    return np.array([trend, vol, returns.mean()])

def train_model(c):
    X, y = [], []
    for i in range(100, len(c) - 1):
        X.append(get_features(c[:i]))
        y.append(1 if c[i+1] > c[i] else 0)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# ─── LOOP PRINCIPAL ─────────────────────────────────────
async def run_bot():
    await notify("🤖 Bot em execução no Railway. Modo LIVE Ativo.")
    
    while True:
        try:
            # Re-treino automático
            if state["model"] is None or (datetime.datetime.now() - state["last_train"]).seconds > 7200:
                candles = await get_klines()
                state["model"] = train_model(candles)
                state["last_train"] = datetime.datetime.now()
                await notify("🧠 Modelo re-treinado com dados recentes.")

            c = await get_klines()
            price = c[-1]
            feat = get_features(c).reshape(1, -1)
            prob = state["model"].predict_proba(feat)[0][1]

            # Lógica de Entrada (LONG)
            if state["position"] is None:
                if prob > 0.65: # Filtro de confiança
                    # Cálculo de quantidade (exemplo: comprar 0.001 BTC ou baseado em saldo)
                    # Para simplificar, definimos uma quantia mínima (ajusta conforme o teu saldo)
                    qty = 0.001 
                    order = await place_order("BUY", qty)
                    if order:
                        state["position"] = "LONG"
                        state["entry_price"] = price
                        state["qty"] = qty
            
            # Lógica de Saída (TP/SL)
            elif state["position"] == "LONG":
                entry = state["entry_price"]
                if price >= entry * (1 + TAKE_PROFIT_PCT) or price <= entry * (1 - STOP_LOSS_PCT):
                    order = await place_order("SELL", state["qty"])
                    if order:
                        state["position"] = None
                        pnl = (price - entry) / entry * 100
                        await notify(f"💰 Trade Fechada. PnL: {pnl:.2f}%")

        except Exception as e:
            print(f"Erro no loop: {e}")
        
        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_bot())
