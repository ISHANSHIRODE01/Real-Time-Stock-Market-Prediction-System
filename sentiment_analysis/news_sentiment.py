import random

def fetch_sentiment_score(ticker: str) -> float:
    """
    Mock sentiment scores representing news aggregation.
    Logic: Aggregate NLP signal from headlines.
    """
    # Production: Replace with real NewsAPI + Transformer pass
    return random.uniform(-0.6, 0.9)

if __name__ == "__main__":
    print(f"Sentiment for AAPL: {fetch_sentiment_score('AAPL'):.4f}")
