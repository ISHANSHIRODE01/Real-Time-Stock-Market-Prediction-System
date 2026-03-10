import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for technical analysis."""
    print("[*] Calculating technical features...")
    data = df.copy()
    
    # 1. SMAs
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # 2. RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Drop rows with NaN (due to rolling windows)
    data.dropna(inplace=True)
    return data

def save_processed_data(df: pd.DataFrame, ticker: str):
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(f"data/processed/{ticker}_cleaned.csv", index=False)
    print(f"[*] Processed data saved for {ticker}.")

if __name__ == "__main__":
    ticker = "AAPL"
    try:
        raw = pd.read_csv(f"data/raw/{ticker}_raw.csv")
        clean = calculate_technical_indicators(raw)
        save_processed_data(clean, ticker)
    except FileNotFoundError:
        print("[!] Raw data not found. Run fetch_stock_data.py first.")
