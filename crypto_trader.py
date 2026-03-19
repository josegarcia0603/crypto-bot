"""
AITrader Crypto Bot — Binance Spot + Telegram
Bot de trading automático para BTC, ETH, BNB, SOL
Todas as estratégias profissionais + Dashboard Telegram
"""

import asyncio
import logging
import os
import time
import hmac
import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import urlencode
import aiohttp

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ─── Configuração ────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY",    "SUA_API_KEY_AQUI")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "SUA_SECRET_KEY_AQUI")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN",     "SEU_TOKEN_TELEGRAM_AQUI")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "SEU_CHAT_ID_AQUI")

BINANCE_BASE_URL   = "https://api.binance.com"
# BINANCE_BASE_URL = "https://testnet.binance.vision"  # Descomenta para Testnet

# ─── Pares e Parâmetros ──────────────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
QUOTE_ASSET      = "USDT"
RISK_PERCENT     = 1.0       # % do saldo por trade
SCORE_THRESHOLD  = 6.0       # Score mínimo para abrir trade
CHECK_INTERVAL   = 300       # Segundos entre análises
RSI_PERIOD       = 14
BB_PERIOD        = 20
EMA_FAST         = 8
EMA_SLOW         = 21
EMA_TREND        = 200
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
ATR_PERIOD       = 14

# ─── Estado Global ───────────────────────────────────────────────────────────
bot_state = {
    "running": True,
    "trades_today": 0,
    "open_trades": {},
    "last_check": None,
    "errors": [],
    "start_time": datetime.now(timezone.utc),
    "win_rate": {p: 0.5 for p in PAIRS},
    "total_trades": {p: 0 for p in PAIRS},
    "won_trades": {p: 0 for p in PAIRS},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE API
# ═══════════════════════════════════════════════════════════════════════════════

class BinanceAPI:
    def __init__(self):
        self.headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

    def _sign(self, params: dict) -> str:
        query = urlencode(params)
        return hmac.new(BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()

    async def get(self, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        if params is None:
            params = {}
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BINANCE_BASE_URL}{endpoint}", params=params,
                                   headers=self.headers if signed else {}) as r:
                return await r.json()

    async def post(self, endpoint: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{BINANCE_BASE_URL}{endpoint}", params=params,
                                    headers=self.headers) as r:
                return await r.json()

    async def delete(self, endpoint: str, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        async with aiohttp.ClientSession() as session:
            async with session.delete(f"{BINANCE_BASE_URL}{endpoint}", params=params,
                                      headers=self.headers) as r:
                return await r.json()

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 250) -> list:
        data = await self.get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        return data if isinstance(data, list) else []

    async def get_account(self) -> dict:
        return await self.get("/api/v3/account", signed=True)

    async def get_balance(self, asset: str) -> float:
        account = await self.get_account()
        for b in account.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    async def get_price(self, symbol: str) -> float:
        data = await self.get("/api/v3/ticker/price", {"symbol": symbol})
        return float(data.get("price", 0))

    async def get_open_orders(self, symbol: str = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self.get("/api/v3/openOrders", params, signed=True)

    async def place_order(self, symbol: str, side: str, quantity: float,
                          stop_loss: float = None, take_profit: float = None) -> dict:
        params = {
            "symbol":    symbol,
            "side":      side,
            "type":      "MARKET",
            "quantity":  f"{quantity:.6f}",
        }
        result = await self.post("/api/v3/order", params)
        return result

    async def get_symbol_info(self, symbol: str) -> dict:
        data = await self.get("/api/v3/exchangeInfo", {"symbol": symbol})
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return {}

    async def get_all_trades(self, symbol: str) -> list:
        return await self.get("/api/v3/myTrades", {"symbol": symbol, "limit": 50}, signed=True)


binance = BinanceAPI()


# ═══════════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_klines(klines: list) -> dict:
    """Converte klines da Binance em listas de OHLCV"""
    opens   = [float(k[1]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    return {"opens": opens, "highs": highs, "lows": lows, "closes": closes, "volumes": volumes}

def calc_ema(closes: list, period: int) -> list:
    if len(closes) < period:
        return [0.0] * len(closes)
    result = [0.0] * len(closes)
    k = 2 / (period + 1)
    result[period - 1] = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1 - k)
    return result

def calc_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-(i + 1)]
        (gains if diff > 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.0001
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_macd(closes: list) -> tuple:
    if len(closes) < 35:
        return 0.0, 0.0, 0.0
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal    = calc_ema(macd_line, 9)
    hist      = macd_line[-1] - signal[-1]
    return macd_line[-1], signal[-1], hist

def calc_bollinger(closes: list, period: int = 20, dev: float = 2.0) -> tuple:
    if len(closes) < period:
        return 0.0, closes[-1], 0.0
    recent = closes[-period:]
    mid = sum(recent) / period
    std = (sum((x - mid) ** 2 for x in recent) / period) ** 0.5
    return mid + dev * std, mid, mid - dev * std

def calc_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 0.001
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / period

def calc_stochastic(highs: list, lows: list, closes: list, k: int = 14) -> float:
    if len(closes) < k:
        return 50.0
    h, l = max(highs[-k:]), min(lows[-k:])
    return (closes[-1] - l) / (h - l) * 100 if h != l else 50.0

def calc_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """ADX simplificado"""
    if len(closes) < period * 2:
        return 20.0
    # True Range
    trs, plus_dm, minus_dm = [], [], []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        pdm = max(highs[i] - highs[i-1], 0) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0
        mdm = max(lows[i-1] - lows[i], 0) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0
        trs.append(tr); plus_dm.append(pdm); minus_dm.append(mdm)
    atr_val = sum(trs[-period:]) / period
    if atr_val == 0:
        return 20.0
    plus_di  = (sum(plus_dm[-period:])  / period) / atr_val * 100
    minus_di = (sum(minus_dm[-period:]) / period) / atr_val * 100
    dx = abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001) * 100
    return dx


# ═══════════════════════════════════════════════════════════════════════════════
# DETECÇÃO DE CONTEXTO DE MERCADO
# ═══════════════════════════════════════════════════════════════════════════════

def detect_context(data: dict) -> int:
    """1=Tendência 2=Lateral 3=Reversão 4=Momentum Forte"""
    closes = data["closes"]
    highs  = data["highs"]
    lows   = data["lows"]

    adx   = calc_adx(highs, lows, closes)
    rsi   = calc_rsi(closes)
    atr   = calc_atr(highs, lows, closes)
    atr_avg = calc_atr(highs[:-14], lows[:-14], closes[:-14]) if len(closes) > 28 else atr

    strong_trend = adx > 30
    weak_trend   = adx < 20
    high_vol     = atr > atr_avg * 1.3
    extreme_rsi  = rsi > 70 or rsi < 30

    # Momentum Forte: ADX alto + RSI extremo + alta volatilidade
    if strong_trend and high_vol and extreme_rsi:
        return 4

    # Tendência: ADX forte
    if strong_trend:
        return 1

    # Reversão: extremos RSI + volatilidade
    if extreme_rsi and high_vol:
        return 3

    # Lateral: ADX fraco
    if weak_trend:
        return 2

    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# SISTEMA DE PONTUAÇÃO POR CONTEXTO
# ═══════════════════════════════════════════════════════════════════════════════

async def calculate_scores(symbol: str) -> tuple:
    """Retorna (buy_score, sell_score, context_name, reason)"""
    try:
        # Buscar dados
        klines_1h  = await binance.get_klines(symbol, "1h",  250)
        klines_15m = await binance.get_klines(symbol, "15m",  60)
        klines_4h  = await binance.get_klines(symbol, "4h",   50)

        if len(klines_1h) < 200:
            return 0.0, 0.0, "?", "Dados insuficientes"

        data    = parse_klines(klines_1h)
        data_15 = parse_klines(klines_15m)
        data_4h = parse_klines(klines_4h)

        closes  = data["closes"]
        highs   = data["highs"]
        lows    = data["lows"]
        volumes = data["volumes"]
        price   = closes[-1]

        context = detect_context(data)
        ctx_names = {1: "Tendência", 2: "Lateral", 3: "Reversão", 4: "Momentum"}
        ctx_name  = ctx_names.get(context, "?")

        buy_score = sell_score = 0.0
        reasons = [f"📊 Contexto: {ctx_name}"]

        # Adaptive Learning — multiplicador baseado em win rate histórica
        wr = bot_state["win_rate"].get(symbol, 0.5)
        tt = bot_state["total_trades"].get(symbol, 0)
        al_mult = (0.5 + wr) if tt >= 5 else 1.0
        al_mult = max(0.3, min(1.5, al_mult))

        # ──────────────────────────────────────────────
        # CONTEXTO 1 — TENDÊNCIA
        # ──────────────────────────────────────────────
        if context == 1:
            # EMA 200 (peso 2.5)
            ema200 = calc_ema(closes, 200)[-1]
            if price > ema200:
                buy_score += 2.5; reasons.append("✅ Preço > EMA200")
            else:
                sell_score += 2.5; reasons.append("🔴 Preço < EMA200")

            # EMA Alignment 8/21 (peso 2)
            e8  = calc_ema(closes, EMA_FAST)[-1]
            e21 = calc_ema(closes, EMA_SLOW)[-1]
            if e8 > e21 and price > e8:
                buy_score += 2.0; reasons.append("✅ EMAs alinhadas alta")
            elif e8 < e21 and price < e8:
                sell_score += 2.0; reasons.append("🔴 EMAs alinhadas baixa")

            # ADX (peso 2)
            adx = calc_adx(highs, lows, closes)
            if adx > 25:
                if e8 > e21: buy_score += 2.0
                else:        sell_score += 2.0
                reasons.append(f"✅ ADX forte ({adx:.0f})")

            # MACD (peso 1.5)
            mm, ms, mh = calc_macd(closes)
            mm2, ms2, _ = calc_macd(closes[:-1])
            if mm > ms and mm2 <= ms2:
                buy_score += 1.5; reasons.append("✅ MACD cruzou alta")
            elif mm < ms and mm2 >= ms2:
                sell_score += 1.5; reasons.append("🔴 MACD cruzou baixa")
            elif mm > ms:
                buy_score += 1.0; reasons.append("✅ MACD bullish")
            elif mm < ms:
                sell_score += 1.0; reasons.append("🔴 MACD bearish")

            # Confirmação 4H (peso 2)
            if len(data_4h["closes"]) >= 50:
                ema200_4h = calc_ema(data_4h["closes"], 50)[-1]
                if data_4h["closes"][-1] > ema200_4h:
                    buy_score += 2.0; reasons.append("✅ 4H confirma alta")
                else:
                    sell_score += 2.0; reasons.append("🔴 4H confirma baixa")

            # Volume (peso 1.5)
            avg_vol = sum(volumes[-20:]) / 20
            if volumes[-1] > avg_vol * 1.5:
                if price > closes[-2]: buy_score += 1.5; reasons.append("✅ Volume bullish")
                else: sell_score += 1.5; reasons.append("🔴 Volume bearish")

        # ──────────────────────────────────────────────
        # CONTEXTO 2 — LATERAL (Mean Reversion)
        # ──────────────────────────────────────────────
        elif context == 2:
            # Bollinger Bands extremos (peso 3)
            bb_u, bb_m, bb_l = calc_bollinger(closes, BB_PERIOD, 2.5)
            if price < bb_l:
                buy_score += 3.0; reasons.append("✅ Abaixo BB inferior")
            elif price > bb_u:
                sell_score += 3.0; reasons.append("🔴 Acima BB superior")
            elif price < bb_m:
                sell_score += 1.0
            else:
                buy_score += 1.0

            # RSI extremos (peso 2.5)
            rsi = calc_rsi(closes)
            if rsi < 30:
                buy_score += 2.5; reasons.append(f"✅ RSI sobrevendido ({rsi:.0f})")
            elif rsi > 70:
                sell_score += 2.5; reasons.append(f"🔴 RSI sobrecomprado ({rsi:.0f})")

            # Stochastic (peso 2)
            stoch = calc_stochastic(highs, lows, closes)
            if stoch < 20:
                buy_score += 2.0; reasons.append(f"✅ Stoch sobrevendido ({stoch:.0f})")
            elif stoch > 80:
                sell_score += 2.0; reasons.append(f"🔴 Stoch sobrecomprado ({stoch:.0f})")

            # Fibonacci (peso 1.5)
            fib = detect_fibonacci(closes, highs, lows)
            if fib == 1:   buy_score  += 1.5; reasons.append("✅ Nível Fibonacci")
            elif fib == -1: sell_score += 1.5; reasons.append("🔴 Nível Fibonacci")

        # ──────────────────────────────────────────────
        # CONTEXTO 3 — REVERSÃO
        # ──────────────────────────────────────────────
        elif context == 3:
            # Divergência RSI (peso 3)
            div = detect_divergence(closes, highs, lows)
            if div == 1:   buy_score  += 3.0; reasons.append("✅ Divergência bullish")
            elif div == -1: sell_score += 3.0; reasons.append("🔴 Divergência bearish")

            # Padrões de velas (peso 2.5)
            candle = detect_candle_patterns(data)
            if candle == 1:   buy_score  += 2.5; reasons.append("✅ Padrão vela bullish")
            elif candle == -1: sell_score += 2.5; reasons.append("🔴 Padrão vela bearish")

            # CHoCH (peso 2)
            choch = detect_choch(closes, highs, lows)
            if choch == 1:   buy_score  += 2.0; reasons.append("✅ CHoCH bullish")
            elif choch == -1: sell_score += 2.0; reasons.append("🔴 CHoCH bearish")

            # RSI cruzamento 50 (peso 1.5)
            rsi      = calc_rsi(closes)
            rsi_prev = calc_rsi(closes[:-1])
            if rsi > 50 and rsi_prev <= 50:
                buy_score += 1.5; reasons.append("✅ RSI cruzou 50 alta")
            elif rsi < 50 and rsi_prev >= 50:
                sell_score += 1.5; reasons.append("🔴 RSI cruzou 50 baixa")

        # ──────────────────────────────────────────────
        # CONTEXTO 4 — MOMENTUM FORTE
        # ──────────────────────────────────────────────
        elif context == 4:
            # BOS (peso 3)
            bos = detect_bos(closes, highs, lows)
            if bos == 1:   buy_score  += 3.0; reasons.append("✅ BOS bullish")
            elif bos == -1: sell_score += 3.0; reasons.append("🔴 BOS bearish")

            # EMA 200 (peso 2)
            ema200 = calc_ema(closes, 200)[-1]
            if price > ema200: buy_score  += 2.0
            else:              sell_score += 2.0

            # Volume extremo (peso 2)
            avg_vol = sum(volumes[-20:]) / 20
            if volumes[-1] > avg_vol * 2.0:
                if price > closes[-2]: buy_score  += 2.0; reasons.append("✅ Volume extremo bullish")
                else:                  sell_score += 2.0; reasons.append("🔴 Volume extremo bearish")

            # MACD (peso 1.5)
            mm, ms, _ = calc_macd(closes)
            if mm > ms: buy_score  += 1.5
            else:       sell_score += 1.5

        # ─── Filtros comuns a todos os contextos ───
        # Confirmação M15 (peso 1)
        if len(data_15["closes"]) >= 21:
            e8_15  = calc_ema(data_15["closes"], EMA_FAST)[-1]
            e21_15 = calc_ema(data_15["closes"], EMA_SLOW)[-1]
            if e8_15 > e21_15: buy_score  += 1.0; reasons.append("✅ M15 confirma")
            else:              sell_score += 1.0; reasons.append("🔴 M15 confirma")

        # Filtro ATR (volatilidade mínima)
        atr     = calc_atr(highs, lows, closes)
        atr_avg = calc_atr(highs[:-14], lows[:-14], closes[:-14]) if len(closes) > 28 else atr
        if atr < atr_avg * 0.4:
            buy_score  *= 0.6
            sell_score *= 0.6
            reasons.append("⚠️ Volatilidade baixa")

        # Aplicar Adaptive Learning
        buy_score  *= al_mult
        sell_score *= al_mult

        return round(buy_score, 2), round(sell_score, 2), ctx_name, "\n".join(reasons)

    except Exception as e:
        log.error(f"Erro scores {symbol}: {e}")
        return 0.0, 0.0, "?", f"Erro: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES DE ESTRATÉGIA
# ═══════════════════════════════════════════════════════════════════════════════

def detect_bos(closes: list, highs: list, lows: list, lookback: int = 20) -> int:
    price = closes[-1]
    h = max(highs[-lookback-1:-1])
    l = min(lows[-lookback-1:-1])
    if price > h: return 1
    if price < l: return -1
    return 0

def detect_choch(closes: list, highs: list, lows: list) -> int:
    if len(closes) < 12: return 0
    hh1, hh2, hh3 = highs[-2], highs[-6], highs[-10]
    ll1, ll2, ll3 = lows[-2],  lows[-6],  lows[-10]
    ch, cl = highs[-1], lows[-1]
    if hh1 < hh2 < hh3 and ll1 < ll2 < ll3 and ch > hh1: return 1
    if hh1 > hh2 > hh3 and ll1 > ll2 > ll3 and cl < ll1: return -1
    return 0

def detect_fibonacci(closes: list, highs: list, lows: list, lookback: int = 50) -> int:
    if len(closes) < lookback: return 0
    sh = max(highs[-lookback:])
    sl = min(lows[-lookback:])
    price = closes[-1]
    rng = sh - sl
    if rng <= 0: return 0
    tol = rng * 0.02  # 2% de tolerância
    sh_idx = highs[-lookback:].index(sh)
    sl_idx = lows[-lookback:].index(sl)
    if sl_idx > sh_idx:  # Tendência de baixa
        for lvl in [0.382, 0.5, 0.618]:
            if abs(price - (sh - rng * lvl)) <= tol: return -1
    else:  # Tendência de alta
        for lvl in [0.382, 0.5, 0.618]:
            if abs(price - (sl + rng * lvl)) <= tol: return 1
    return 0

def detect_divergence(closes: list, highs: list, lows: list, lookback: int = 20) -> int:
    if len(closes) < lookback + 5: return 0
    # Pivot highs e lows simples
    ph_prices, pl_prices = [], []
    ph_rsi, pl_rsi = [], []
    for i in range(2, min(lookback, len(closes) - 2)):
        if highs[-i] > highs[-(i-1)] and highs[-i] > highs[-(i+1)]:
            ph_prices.append(highs[-i])
            ph_rsi.append(calc_rsi(closes[:-(i-1)]))
        if lows[-i] < lows[-(i-1)] and lows[-i] < lows[-(i+1)]:
            pl_prices.append(lows[-i])
            pl_rsi.append(calc_rsi(closes[:-(i-1)]))
    if len(ph_prices) >= 2 and ph_prices[0] > ph_prices[1] and ph_rsi[0] < ph_rsi[1]:
        return -1  # Divergência bearish
    if len(pl_prices) >= 2 and pl_prices[0] < pl_prices[1] and pl_rsi[0] > pl_rsi[1]:
        return 1   # Divergência bullish
    return 0

def detect_candle_patterns(data: dict) -> int:
    opens  = data["opens"]
    closes = data["closes"]
    highs  = data["highs"]
    lows   = data["lows"]
    if len(closes) < 4: return 0
    o1, c1, h1, l1 = opens[-2], closes[-2], highs[-2], lows[-2]
    o2, c2 = opens[-3], closes[-3]
    o3, c3 = opens[-4], closes[-4]
    r1 = h1 - l1
    b1 = abs(c1 - o1)
    b2 = abs(c2 - o2)
    b3 = abs(c3 - o3)
    if r1 <= 0: return 0
    lw = min(o1, c1) - l1
    uw = h1 - max(o1, c1)
    # Pin Bar Bullish
    if b1/r1 < 0.3 and lw > r1 * 0.6 and uw < r1 * 0.2: return 1
    # Pin Bar Bearish
    if b1/r1 < 0.3 and uw > r1 * 0.6 and lw < r1 * 0.2: return -1
    # Bullish Engulfing
    if c1 > o1 and c2 < o2 and c1 > o2 and o1 < c2 and b1 > b2 * 1.1: return 1
    # Bearish Engulfing
    if c1 < o1 and c2 > o2 and c1 < o2 and o1 > c2 and b1 > b2 * 1.1: return -1
    # Morning Star
    if c3 < o3 and b2 < (highs[-3]-lows[-3]) * 0.3 and c1 > o1 and b1 > b3 * 0.5: return 1
    # Evening Star
    if c3 > o3 and b2 < (highs[-3]-lows[-3]) * 0.3 and c1 < o1 and b1 > b3 * 0.5: return -1
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# GESTÃO DE TRADES
# ═══════════════════════════════════════════════════════════════════════════════

async def calculate_quantity(symbol: str, price: float) -> float:
    """Calcula a quantidade a comprar baseado no risco"""
    balance = await binance.get_balance(QUOTE_ASSET)
    risk_money = balance * RISK_PERCENT / 100
    quantity = risk_money / price
    # Arredondar para 6 casas decimais
    return round(quantity, 6)

async def execute_trade(symbol: str, direction: str, score: float, ctx: str, bot_app):
    try:
        price = await binance.get_price(symbol)
        if price <= 0:
            return

        quantity = await calculate_quantity(symbol, price)
        if quantity <= 0:
            log.warning(f"Saldo insuficiente para {symbol}")
            return

        side = "BUY" if direction == "BUY" else "SELL"
        result = await binance.place_order(symbol, side, quantity)

        if result.get("status") == "FILLED" or result.get("orderId"):
            order_id = result.get("orderId", "?")
            filled_price = float(result.get("fills", [{}])[0].get("price", price)) if result.get("fills") else price

            bot_state["trades_today"] += 1
            bot_state["open_trades"][str(order_id)] = {
                "symbol": symbol, "direction": direction,
                "price": filled_price, "quantity": quantity,
                "score": score, "context": ctx,
                "time": datetime.now(timezone.utc).strftime("%H:%M UTC")
            }

            emoji = "🟢" if direction == "BUY" else "🔴"
            msg = (
                f"{emoji} *Trade Aberto — Binance*\n"
                f"Par: `{symbol}`\n"
                f"Direção: *{direction}*\n"
                f"Preço: `${filled_price:,.4f}`\n"
                f"Quantidade: `{quantity}`\n"
                f"Contexto: `{ctx}`\n"
                f"Score: `{score}/12`"
            )
            await bot_app.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
            log.info(f"✅ Trade: {symbol} {direction} @ {filled_price} | Score: {score}")

            # Atualizar Adaptive Learning
            bot_state["total_trades"][symbol] = bot_state["total_trades"].get(symbol, 0) + 1

        else:
            err = result.get("msg", str(result))
            log.error(f"Erro ordem {symbol}: {err}")
            bot_state["errors"].append(f"{symbol}: {err}")

    except Exception as e:
        log.error(f"Erro execute_trade {symbol}: {e}")
        bot_state["errors"].append(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

async def trading_loop(bot_app):
    log.info("🚀 Bot Crypto Binance iniciado!")
    await bot_app.bot.send_message(
        TELEGRAM_CHAT_ID,
        "🚀 *AITrader Crypto Bot iniciado!*\n"
        "Pares: BTC, ETH, BNB, SOL\n"
        "Digita /start para ver o menu.",
        parse_mode="Markdown"
    )

    while True:
        try:
            if not bot_state["running"]:
                await asyncio.sleep(30)
                continue

            bot_state["last_check"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

            for symbol in PAIRS:
                # Verificar se já tem trade aberto neste par
                open_orders = await binance.get_open_orders(symbol)
                if open_orders:
                    continue

                buy_score, sell_score, ctx, reasons = await calculate_scores(symbol)

                if buy_score >= SCORE_THRESHOLD and buy_score > sell_score:
                    await execute_trade(symbol, "BUY", buy_score, ctx, bot_app)
                    await asyncio.sleep(2)
                elif sell_score >= SCORE_THRESHOLD and sell_score > buy_score:
                    await execute_trade(symbol, "SELL", sell_score, ctx, bot_app)
                    await asyncio.sleep(2)

                await asyncio.sleep(1)

        except Exception as e:
            log.error(f"Erro no loop: {e}")
            bot_state["errors"].append(str(e)[-100:])

        await asyncio.sleep(CHECK_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM — Dashboard Completo
# ═══════════════════════════════════════════════════════════════════════════════

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Dashboard",      callback_data="dashboard")],
        [InlineKeyboardButton("📈 Trades Abertos", callback_data="trades")],
        [InlineKeyboardButton("🔍 Analisar Pares", callback_data="analyze")],
        [InlineKeyboardButton("💰 Saldos",         callback_data="balances")],
        [InlineKeyboardButton("⏸ Pausar",          callback_data="pause"),
         InlineKeyboardButton("▶️ Retomar",         callback_data="resume")],
        [InlineKeyboardButton("❌ Fechar Trades",   callback_data="close_all")],
    ])

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *AITrader Crypto Bot — Binance*\nEscolhe uma opção:",
        reply_markup=main_keyboard(),
        parse_mode="Markdown"
    )

async def send_dashboard(chat_id, bot):
    try:
        usdt_balance = await binance.get_balance(QUOTE_ASSET)
        uptime  = str(datetime.now(timezone.utc) - bot_state["start_time"]).split(".")[0]
        status  = "🟢 A correr" if bot_state["running"] else "⏸ Pausado"
        last    = bot_state["last_check"] or "—"
        errors  = len(bot_state["errors"])

        msg = (
            f"📊 *Dashboard AITrader Crypto*\n"
            f"{'─'*30}\n"
            f"💰 Saldo USDT: `${usdt_balance:,.2f}`\n"
            f"{'─'*30}\n"
            f"📈 Trades abertos: `{len(bot_state['open_trades'])}`\n"
            f"📅 Trades hoje: `{bot_state['trades_today']}`\n"
            f"⏰ Última análise: `{last}`\n"
            f"⏱ Uptime: `{uptime}`\n"
            f"⚠️ Erros: `{errors}`\n"
            f"{'─'*30}\n"
            f"Status: {status}\n"
            f"Risco/trade: `{RISK_PERCENT}%`\n"
            f"Score mínimo: `{SCORE_THRESHOLD}/12`"
        )

        # Win rates por par
        wr_text = ""
        for p in PAIRS:
            tt = bot_state["total_trades"].get(p, 0)
            wr = bot_state["win_rate"].get(p, 0.5)
            if tt > 0:
                wr_text += f"\n`{p}`: {wr*100:.0f}% win ({tt} trades)"
        if wr_text:
            msg += f"\n{'─'*30}\n📊 *Win Rate por Par*{wr_text}"

        kb = [[InlineKeyboardButton("🔄 Atualizar", callback_data="dashboard"),
               InlineKeyboardButton("📈 Trades", callback_data="trades")]]
        await bot.send_message(chat_id, msg, parse_mode="Markdown",
                               reply_markup=InlineKeyboardMarkup(kb))
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Erro dashboard: {e}")

async def send_trades(chat_id, bot):
    if not bot_state["open_trades"]:
        await bot.send_message(chat_id, "📭 Nenhum trade aberto.")
        return
    msg = f"📈 *Trades Abertos ({len(bot_state['open_trades'])})*\n{'─'*30}\n"
    for oid, t in bot_state["open_trades"].items():
        try:
            current = await binance.get_price(t["symbol"])
            pnl_pct = ((current - t["price"]) / t["price"] * 100)
            if t["direction"] == "SELL":
                pnl_pct = -pnl_pct
            emoji = "✅" if pnl_pct >= 0 else "❌"
            msg += (
                f"{'🟢' if t['direction']=='BUY' else '🔴'} `{t['symbol']}`\n"
                f"Entrada: `${t['price']:,.4f}` → Actual: `${current:,.4f}`\n"
                f"P&L: {emoji} `{pnl_pct:+.2f}%` | Score: `{t['score']}`\n\n"
            )
        except:
            msg += f"`{t['symbol']}` {t['direction']} — erro ao obter preço\n\n"
    await bot.send_message(chat_id, msg, parse_mode="Markdown",
                           reply_markup=InlineKeyboardMarkup([
                               [InlineKeyboardButton("🔄 Atualizar", callback_data="trades")]
                           ]))

async def send_analysis(chat_id, bot):
    await bot.send_message(chat_id, "🔍 A analisar todos os pares... aguarda ~30s")
    msg = f"📊 *Análise dos Pares*\n{'─'*30}\n"
    for symbol in PAIRS:
        try:
            buy, sell, ctx, _ = await calculate_scores(symbol)
            price = await binance.get_price(symbol)
            if buy >= SCORE_THRESHOLD:
                signal = f"🟢 BUY ({buy:.1f}/12)"
            elif sell >= SCORE_THRESHOLD:
                signal = f"🔴 SELL ({sell:.1f}/12)"
            else:
                signal = f"⚪ Neutro (B:{buy:.1f} S:{sell:.1f})"
            msg += f"`{symbol}` ${price:,.2f}\n{signal} | {ctx}\n\n"
        except:
            msg += f"`{symbol}`: ⚠️ Erro\n\n"
    await bot.send_message(chat_id, msg, parse_mode="Markdown")

async def send_balances(chat_id, bot):
    try:
        assets = ["BTC", "ETH", "BNB", "SOL", "USDT"]
        msg = f"💰 *Saldos Binance*\n{'─'*30}\n"
        for asset in assets:
            bal = await binance.get_balance(asset)
            if bal > 0:
                if asset != "USDT":
                    try:
                        price = await binance.get_price(f"{asset}USDT")
                        value = bal * price
                        msg += f"`{asset}`: {bal:.6f} (~${value:,.2f})\n"
                    except:
                        msg += f"`{asset}`: {bal:.6f}\n"
                else:
                    msg += f"`{asset}`: ${bal:,.2f}\n"
        await bot.send_message(chat_id, msg, parse_mode="Markdown")
    except Exception as e:
        await bot.send_message(chat_id, f"❌ Erro saldos: {e}")

async def button_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    data    = query.data

    if data == "dashboard":    await send_dashboard(chat_id, ctx.bot)
    elif data == "trades":     await send_trades(chat_id, ctx.bot)
    elif data == "analyze":    await send_analysis(chat_id, ctx.bot)
    elif data == "balances":   await send_balances(chat_id, ctx.bot)
    elif data == "pause":
        bot_state["running"] = False
        await ctx.bot.send_message(chat_id, "⏸ *Bot pausado.*", parse_mode="Markdown")
    elif data == "resume":
        bot_state["running"] = True
        await ctx.bot.send_message(chat_id, "▶️ *Bot retomado!*", parse_mode="Markdown")
    elif data == "close_all":
        await ctx.bot.send_message(chat_id,
            "⚠️ Para fechar trades no Spot da Binance vai à app da Binance e vende manualmente.\n"
            "O bot não fecha posições Spot automaticamente por segurança.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(button_handler))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    await trading_loop(app)

if __name__ == "__main__":
    asyncio.run(main())
