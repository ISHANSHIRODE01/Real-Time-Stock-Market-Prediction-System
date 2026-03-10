from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os

def calculate_metrics(actual, predicted, model_name="LSTM"):
    """Evaluates accuracy metrics for lead dev report."""
    metrics = {
        "model": model_name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted, squared=False)
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[*] Evaluation report generated at reports/metrics.json")
    return metrics
