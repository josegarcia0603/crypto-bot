"""
AITrader V5 — Multi-Bot Institutional System
"""

import asyncio, os, time, aiohttp, hmac, hashlib, logging
from urllib.parse import urlencode

BINANCE_API_KEY=os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY=os.getenv("BINANCE_SECRET_KEY")

BASE="https://api.binance.com"
PAIRS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]

logging.basicConfig(level=logging.INFO)
log=logging.getLogger()

# ─── STATE ─────────────────────────
state={
 "balance":0,
 "initial":0,
 "peak":0,
 "drawdown":0,
 "kill":False,
 "bots":{},
 "open":[]
}

# ─── API ───────────────────────────
class API:
 def sign(self,p):
  q=urlencode(p)
  return hmac.new(BINANCE_SECRET_KEY.encode(),q.encode(),hashlib.sha256).hexdigest()

 async def get(self,e,p={},s=False):
  if s:
   p["timestamp"]=int(time.time()*1000)
   p["signature"]=self.sign(p)
  async with aiohttp.ClientSession() as sess:
   async with sess.get(BASE+e,params=p,headers={"X-MBX-APIKEY":BINANCE_API_KEY} if s else {}) as r:
    return await r.json()

 async def post(self,e,p):
  p["timestamp"]=int(time.time()*1000)
  p["signature"]=self.sign(p)
  async with aiohttp.ClientSession() as sess:
   async with sess.post(BASE+e,params=p,headers={"X-MBX-APIKEY":BINANCE_API_KEY}) as r:
    return await r.json()

api=API()

# ─── INDICATORS ───────────────────
def ema(d,p):
 k=2/(p+1);e=d[0]
 for x in d: e=x*k+e*(1-k)
 return e

def rsi(c):
 g=[max(c[i]-c[i-1],0) for i in range(1,len(c))]
 l=[max(c[i-1]-c[i],0) for i in range(1,len(c))]
 return 100-(100/(1+(sum(g[-14:])/14)/(sum(l[-14:])/14+1e-6)))

def atr(h,l,c):
 return sum([abs(h[i]-l[i]) for i in range(-14,0)])/14

# ─── AI ───────────────────────────
class AI:
 def __init__(self):
  self.w=[0.2]*5; self.lr=0.02
 def p(self,f): return 1/(1+2.7**(-sum(self.w[i]*f[i] for i in range(len(f)))))
 def u(self,f,r):
  e=r-self.p(f)
  for i in range(len(self.w)): self.w[i]+=self.lr*e*f[i]

ai=AI()

# ─── FEATURES ─────────────────────
def feats(c,h,l,v):
 return [
  rsi(c)/100,
  ema(c,50)/c[-1],
  atr(h,l,c),
  v[-1]/(sum(v[-20:])/20),
  1 if c[-1]>ema(c,50) else 0
 ]

# ─── BOTS ─────────────────────────
async def bot_trend(sym,data):
 c=data["c"]; e50=ema(c,50)
 return 1 if c[-1]>e50 else 0

async def bot_mean(sym,data):
 val=rsi(data["c"])
 return 1 if val<30 else 0

async def bot_breakout(sym,data):
 return 1 if data["c"][-1]>max(data["c"][-20:-1]) else 0

BOTS=[bot_trend,bot_mean,bot_breakout]

# ─── DATA ─────────────────────────
async def klines(sym):
 return await api.get("/api/v3/klines",{"symbol":sym,"interval":"1h","limit":200})

def parse(k):
 return {
  "c":[float(x[4]) for x in k],
  "h":[float(x[2]) for x in k],
  "l":[float(x[3]) for x in k],
  "v":[float(x[5]) for x in k],
 }

# ─── BALANCE ──────────────────────
async def balance():
 acc=await api.get("/api/v3/account",{},True)
 for b in acc["balances"]:
  if b["asset"]=="USDT":
   return float(b["free"])
 return 0

# ─── POSITION SIZE ────────────────
def size(price):
 base=state["balance"]*0.01
 if state["drawdown"]>0.1: base*=0.5
 return base/price

# ─── TRADE ────────────────────────
async def buy(sym,qty):
 return await api.post("/api/v3/order",{"symbol":sym,"side":"BUY","type":"MARKET","quantity":f"{qty:.6f}"})

async def sell(sym,qty):
 return await api.post("/api/v3/order",{"symbol":sym,"side":"SELL","type":"MARKET","quantity":f"{qty:.6f}"})

# ─── CORE ─────────────────────────
async def run():
 while True:
  try:
   state["balance"]=await balance()

   if state["initial"]==0: state["initial"]=state["balance"]
   if state["balance"]>state["peak"]: state["peak"]=state["balance"]

   state["drawdown"]=(state["peak"]-state["balance"])/state["peak"] if state["peak"] else 0

   if state["drawdown"]>0.25:
    state["kill"]=True

   if state["kill"]:
    await asyncio.sleep(60)
    continue

   for sym in PAIRS:
    if len(state["open"])>=3: break

    k=await klines(sym)
    d=parse(k)

    f=feats(d["c"],d["h"],d["l"],d["v"])

    votes=sum([await b(sym,d) for b in BOTS])

    prob=ai.p(f)

    if votes>=2 and prob>0.6:
     price=d["c"][-1]
     qty=size(price)

     a=atr(d["h"],d["l"],d["c"])
     sl=price-2*a
     tp=price+3*a

     r=await buy(sym,qty)
     if r.get("orderId"):
      state["open"].append({"sym":sym,"qty":qty,"sl":sl,"tp":tp,"f":f,"entry":price})

  except Exception as e:
   log.error(e)

  await asyncio.sleep(60)

# ─── MONITOR ──────────────────────
async def monitor():
 while True:
  for t in state["open"][:]:
   k=await klines(t["sym"])
   price=float(k[-1][4])

   if price<=t["sl"]:
    await sell(t["sym"],t["qty"])
    ai.u(t["f"],0)
    state["open"].remove(t)

   elif price>=t["tp"]:
    await sell(t["sym"],t["qty"])
    ai.u(t["f"],1)
    state["open"].remove(t)

  await asyncio.sleep(10)

# ─── MAIN ─────────────────────────
async def main():
 await asyncio.gather(run(),monitor())

if __name__=="__main__":
 asyncio.run(main())
