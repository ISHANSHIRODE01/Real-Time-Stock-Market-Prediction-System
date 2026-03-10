import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical stock data using yfinance."""
    print(f"[*] Fetching {ticker} data from yfinance...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    df.reset_index(inplace=True)
    # Ensure raw directory exists
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(f"data/raw/{ticker}_raw.csv", index=False)
    print(f"    - Saved to data/raw/{ticker}_raw.csv")
    return df

if __name__ == "__main__":
    fetch_data("AAPL")
