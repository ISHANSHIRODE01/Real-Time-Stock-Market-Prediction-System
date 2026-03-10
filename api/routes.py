from fastapi import APIRouter, HTTPException
import yfinance as yf
import random

router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "ok"}

@router.post("/predict/{ticker}")
async def run_forecast(ticker: str):
    # Simulated high-performance inference bridge
    return {
        "ticker": ticker.upper(),
        "predictions": [round(150 + random.uniform(0, 10), 2) for _ in range(7)]
    }
