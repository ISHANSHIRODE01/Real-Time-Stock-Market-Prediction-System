from prophet import Prophet
import pandas as pd

def generate_prophet_forecast(df: pd.DataFrame, days: int = 30):
    """
    Statistical baseline using Facebook Prophet.
    Expected DF must have 'Date' and 'Close' columns.
    """
    print(f"[*] Generating {days}-day Prophet forecast...")
    # Prepare DS/Y columns
    pdf = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # ds needs to be dt without timezone
    pdf['ds'] = pd.to_datetime(pdf['ds']).dt.tz_localize(None)
    
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(pdf)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return model, forecast

if __name__ == "__main__":
    # Test stub
    pass
