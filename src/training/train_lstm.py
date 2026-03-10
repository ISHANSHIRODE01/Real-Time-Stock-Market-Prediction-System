import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Adjust path to import from src/models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.lstm_model import StockLSTM

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_model(ticker: str, epochs: int = 10):
    """Execution script for training the LSTM."""
    print(f"[*] Training core predictive engine for {ticker}...")
    
    # Load data
    df = pd.read_csv(f"data/processed/{ticker}_cleaned.csv")
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volume']
    
    scaler = StandardScaler()
    data = scaler.fit_transform(df[features])
    
    # Windowing
    seq_len = 60
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
        
    X, y = np.array(X), np.array(y)
    
    dataset = TimeSeriesDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = StockLSTM(input_dim=len(features))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for e in range(epochs):
        epoch_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred.squeeze(), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    - Epoch {e+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")

    # Persistence
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/lstm_model.pt")
    print(f"[*] Optimization complete. Model artifact saved.")

if __name__ == "__main__":
    train_model("AAPL")
