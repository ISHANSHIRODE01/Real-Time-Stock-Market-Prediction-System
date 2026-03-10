import torch
import numpy as np

def forecast_next_day(model, input_sequence):
    """Core inference function for LSTM."""
    model.eval()
    with torch.no_grad():
        # Ensure sequence is (1, seq_len, input_dim)
        if input_sequence.ndim == 2:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        
        tensor_in = torch.tensor(input_sequence, dtype=torch.float32)
        prediction = model(tensor_in)
        return prediction.item()

if __name__ == "__main__":
    print("[*] Inference module loaded.")
