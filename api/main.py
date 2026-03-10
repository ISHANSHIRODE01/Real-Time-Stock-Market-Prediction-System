from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
import sys
import os

# Ensure paths correctly resolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

app = FastAPI(title="QuantX Predictive Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "operational", "engine": "QuantX-v1"}

@app.get("/ticker/{symbol}")
async def get_details(symbol: str):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1d")
        if hist.empty: raise HTTPException(404, "Unknown Ticker")
        return {
            "symbol": symbol.upper(),
            "price": round(hist['Close'].iloc[-1], 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
