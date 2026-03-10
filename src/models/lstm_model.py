import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """Deep LSTM architecture for sequence-to-value prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1) # Yields the next Close price
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Context extraction from the last hidden state
        out = hn[-1]
        out = self.dropout(out)
        return self.fc(out)

if __name__ == "__main__":
    model = StockLSTM(6, 128)
    print(f"[*] Initialized model:\n{model}")
