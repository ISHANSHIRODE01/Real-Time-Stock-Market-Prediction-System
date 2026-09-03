<div align="center">

# 📈 Real-Time Stock Market Prediction System

### A research pipeline for financial time-series forecasting with leakage-free evaluation, FinBERT news sentiment, and cost-aware walk-forward backtesting

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/DeepLearning-LSTM-green?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-teal?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-red?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Prophet](https://img.shields.io/badge/Forecast-Prophet-orange?style=for-the-badge)](https://facebook.github.io/prophet/)
[![Plotly](https://img.shields.io/badge/Dashboard-Plotly-purple?style=for-the-badge&logo=plotly)](https://plotly.com/)

<br/>

> **Predict. Backtest. Deploy.** — An end-to-end quantitative intelligence system combining Deep Learning (LSTM), Statistical Forecasting (Prophet), and NLP-powered news sentiment to generate real-time trading signals through a production-ready REST API.

<br/>

<img src="https://img.shields.io/github/stars/ISHANSHIRODE01/Real-Time-Stock-Market-Prediction-System?style=social" />
<img src="https://img.shields.io/github/forks/ISHANSHIRODE01/Real-Time-Stock-Market-Prediction-System?style=social" />
<img src="https://img.shields.io/github/issues/ISHANSHIRODE01/Real-Time-Stock-Market-Prediction-System" />
<img src="https://img.shields.io/github/last-commit/ISHANSHIRODE01/Real-Time-Stock-Market-Prediction-System" />

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Dataset & Data Sources](#-dataset--data-sources)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Architecture](#-model-architecture)
- [Model Evaluation & Comparison](#-model-evaluation--comparison)
- [Sentiment Analysis Integration](#-sentiment-analysis-integration)
- [API Documentation](#-api-documentation)
- [Dashboard Overview](#-dashboard-overview)
- [Portfolio Backtesting Engine](#-portfolio-backtesting-engine)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [Running the System](#-running-the-system)
- [Docker Deployment](#-docker-deployment)
- [Experiment Tracking](#-experiment-tracking)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🚀 Overview

The **Real-Time Stock Market Prediction System** is a research pipeline for financial time-series forecasting that combines three complementary modelling paradigms:

| Modelling Approach | Strengths |
|---|---|
| **LSTM (Deep Learning)** | Captures long-range temporal dependencies and non-linear patterns in price history |
| **Prophet (Statistical)** | Robust to missing data, handles seasonality (daily, weekly, yearly) automatically |
| **Sentiment (NLP)** | Integrates alternative data (news headlines) as a leading indicator for price momentum |

The system provides a **FastAPI backend** consumed by an interactive **Streamlit/Plotly dashboard**, enabling real-time prediction for any publicly-traded ticker symbol. A **portfolio backtesting engine** validates strategy alpha over a walk-forward 6-month simulation.

**Target Roles:** Data Scientist · Quantitative Engineer · ML Engineer · AI Research Engineer

---

## ✨ Key Features

- 🧠 **LSTM Deep Learning Model** — 2-layer LSTM on 60-day sliding windows, leakage-free preprocessing (Transformer is *not* implemented; see Future Work)
- 📊 **Statistical Baseline** — Facebook Prophet with daily, weekly, and yearly seasonality components
- 📰 **Alternative Data Integration** — Real FinBERT news sentiment over NewsAPI / yfinance headlines
- ⚡ **Real-Time REST API** — FastAPI backend with full Swagger documentation (`/docs`)
- 📉 **Interactive Dashboard** — Streamlit + Plotly multi-panel visualization (Price, RSI, MACD, Forecast)
- 💼 **Backtesting Engine** — Walk-forward portfolio simulator with ROI / Sharpe Ratio calculation
- 🐳 **Docker-Ready** — Single-command containerized deployment
- 🔬 **Modular Architecture** — Strict separation of data, models, training, inference, and API layers
- 🎯 **Technical Indicators** — SMA(20/50), RSI(14), MACD(12/26/9), Bollinger Bands

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REAL-TIME PREDICTION SYSTEM                     │
├──────────────────┬──────────────────┬──────────────────────────────-┤
│   DATA LAYER     │   MODEL LAYER    │        API / UI LAYER         │
│                  │                  │                               │
│  ┌────────────┐  │  ┌────────────┐  │  ┌────────────────────────┐  │
│  │  yfinance  │  │  │    LSTM    │  │  │     FastAPI Backend    │  │
│  │ Alpha Van. │──▶  │ Transformer│──▶  │  GET  /ticker/{sym}   │  │
│  │  News API  │  │  │  Prophet   │  │  │  POST /predict/{sym}  │  │
│  └────────────┘  │  └────────────┘  │  │  GET  /backtest/{sym} │  │
│        │         │        │         │  └──────────┬─────────────┘  │
│  ┌─────▼──────┐  │  ┌─────▼──────┐  │             │               │
│  │  Feature   │  │  │  Sentiment │  │  ┌──────────▼─────────────┐  │
│  │Engineering │  │  │  Fusion    │  │  │  Streamlit Dashboard   │  │
│  │SMA/RSI/MACD│  │  │  Layer     │  │  │  Plotly Visualizations │  │
│  └────────────┘  │  └────────────┘  │  └────────────────────────┘  │
└──────────────────┴──────────────────┴───────────────────────────────┘
                              │
              ┌───────────────▼──────────────┐
              │     PORTFOLIO SIMULATOR      │
              │  Walk-Forward Backtesting    │
              │  ROI / Drawdown / Sharpe     │
              └──────────────────────────────┘
```

---

## 📦 Dataset & Data Sources

| Source | Type | Access | Coverage |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | OHLCV Historical | Free · No Key | Global Equities |
| **Alpha Vantage API** | Intraday + Fundamentals | Free Tier (500 req/day) | US Stocks |
| **NewsAPI / Finnhub** | News Headlines | Free Tier | Financial Headlines |

**Feature Set Engineered:**

| Feature | Description | Window |
|---|---|---|
| `Close` | Adjusted closing price | Daily |
| `Volume` | Trade volume | Daily |
| `SMA_20` | Simple Moving Average | 20 days |
| `SMA_50` | Simple Moving Average | 50 days |
| `RSI` | Relative Strength Index | 14 days |
| `MACD` | Moving Avg Convergence Divergence | 12/26/9 |
| `Sentiment` | News sentiment score | Daily aggregate |

---

## 🔬 Machine Learning Pipeline

```
Raw OHLCV Data
      │
      ▼
┌─────────────────┐
│  Preprocessing  │  → Timezone normalization, NaN handling, StandardScaler
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Eng.    │  → SMA, RSI, MACD, Bollinger Bands
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Sliding Window  │  → 60-day sequences → Shape: (N, 60, features)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
  LSTM    Prophet     ← Separate training pipelines
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Sentiment Fusion│  → Late fusion: concat(LSTM_context, sentiment_score)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Decision Layer  │  → Buy / Sell / Hold signal with confidence score
└─────────────────┘
```

---

## 🧠 Model Architecture

### LSTM Network (PyTorch)

```python
StockLSTM(
    input_dim  = 6,         # Features per timestep
    hidden_dim = 128,       # Hidden units per LSTM cell
    num_layers = 2,         # Stacked recurrent layers
    dropout    = 0.2        # Regularization between layers
)

Architecture:  Input(60, 6) → LSTM×2 → Dropout(0.2) → Linear(128→1) → Price(t+1)
Parameters:    ~200,000 trainable parameters
Optimizer:     Adam (lr=0.001, weight_decay=1e-5)
Loss:          Mean Squared Error (MSE)
```

### Multi-Head Attention / Transformer (Planned)

```
Input Embedding → Positional Encoding → Multi-Head Attention (8 heads)
→ Feed-Forward Network → Layer Norm → Linear → Price Output
```

### Prophet Baseline

```python
Prophet(
    daily_seasonality  = True,
    weekly_seasonality = True,
    yearly_seasonality = True,
    changepoint_prior_scale = 0.05   # Flexibility control
)
```

---

## 📊 Model Evaluation & Comparison

**Task:** predict next-day *return* (not price level).
**Data:** AAPL, 1204 daily bars, chronological 70/15/15 split — train 842 / val 181 / test 181.
**Reproduce:** `python src/training/train_lstm.py --epochs 40` → writes `reports/lstm_AAPL_metrics.json`

| Model | Test MSE | R² | Directional Acc. |
|---|---|---|---|
| LSTM (2-layer, 128 hidden) | 3.097e-04 | **-0.029** | 52.9% |
| Persistence (`r_t+1 = r_t`) | 6.131e-04 | -1.037 | 44.6% |
| Predict zero | 3.049e-04 | -0.013 | n/a |

### Reading these numbers honestly

The LSTM **does not beat the predict-zero baseline** on MSE (3.097e-04 vs
3.049e-04), and its R² is negative — marginally worse than predicting the mean.
It does clearly beat naive persistence.

This is the expected result, not a bug. Next-day equity returns are close to a
martingale; technical indicators alone carry very little next-day signal, and
52.9% directional accuracy on 181 samples is not statistically distinguishable
from a coin flip (95% CI roughly ±7pp).

An earlier version of this README claimed 67.3% directional accuracy for the
LSTM and listed a Transformer at 70.1%. **Both were fabricated** — no such
experiment existed and no Transformer is implemented here. They are replaced
with the measured values above.

**Why keep a model that doesn't beat its baseline?** The engineering around it
is the deliverable: a leakage-free pipeline, honest baselines, and a
backtester that charges realistic costs. A pipeline that correctly tells you
the signal is weak is worth more than one that hides it.

### Methodological fixes applied

Three defects in the original training script were corrected — each one
independently inflates results:

| Defect | Why it inflates | Fix |
|---|---|---|
| `StandardScaler` fit on the full series before splitting | Test-period mean/variance leak into training | Scaler fit on train split only |
| No train/val/test split; loss reported on training data | Reported number measured memorisation, not generalisation | Chronological 70/15/15 + early stopping on val |
| Target was the scaled `Close` **price level** | Prices are near-perfectly autocorrelated, so "tomorrow ≈ today" scores a great MSE while being useless | Target changed to next-day return |

---

## 📰 Sentiment Analysis Integration

The system implements a **Late Fusion Multi-Modal Architecture** combining numerical price signals with textual news sentiment:

```
┌────────────────────────────────────────┐
│         MULTIMODAL FUSION              │
│                                        │
│  [LSTM Context Vector (128-dim)]       │
│           +                            │
│  [Sentiment Score (1-dim)]  ──┐        │
│                               ▼        │
│             Linear(129 → 64 → 1)       │
│                               │        │
│          Buy / Sell / Hold Signal      │
└────────────────────────────────────────┘
```

**Sentiment Scoring Scale:**

| Score Range | Signal | Interpretation |
|---|---|---|
| `0.5 → 1.0` | 🟢 Bullish | Strong positive press / earnings beat |
| `0.1 → 0.5` | 🟡 Neutral-Bullish | Mild positive sentiment |
| `-0.1 → 0.1` | ⚪ Neutral | No directional bias |
| `-0.5 → -0.1` | 🟠 Neutral-Bearish | Negative outlooks |
| `-1.0 → -0.5` | 🔴 Bearish | Regulatory headwinds / earnings miss |

### Implementation status: real, no longer mocked

`sentiment_analysis/news_sentiment.py` previously returned
`random.uniform(-0.6, 0.9)` — a random number with no connection to any news.
It now runs **FinBERT** (`ProsusAI/finbert`) over headlines retrieved from
NewsAPI, falling back to `yfinance.Ticker.news` when no API key is set.

Design decisions worth noting:

- **FinBERT over VADER/general sentiment models.** Financial language inverts
  normal polarity — "shares plunge despite earnings beat" is negative for
  price while reading positive in general English.
- **Exponential time weighting** (24h half-life) so stale headlines don't
  dominate a current signal.
- **`score = None` when no headlines exist**, rather than defaulting to `0.0`.
  A caller must distinguish *no information* from *neutral information*;
  silently returning 0.0 lets an empty feed masquerade as a real signal.
- **Every headline keeps its `published_at` timestamp.** For backtesting,
  sentiment from day D may only inform day D+1. Aggregating sentiment without
  respecting publication time is the most common source of look-ahead bias.

```bash
export NEWSAPI_KEY=...        # optional; falls back to yfinance
python sentiment_analysis/news_sentiment.py
```

---

## ⚡ API Documentation

The FastAPI backend auto-generates interactive docs at `http://localhost:8000/docs`

### Endpoints

#### `GET /ticker/{symbol}` — Real-time Price + Sentiment

```bash
curl http://localhost:8000/ticker/AAPL
```

```json
{
  "symbol": "AAPL",
  "price": 189.45,
  "sentiment_score": 0.62,
  "timestamp": "2026-03-10T12:00:00"
}
```

---

#### `POST /predict/{symbol}` — 7-Day LSTM Forecast

```bash
curl -X POST http://localhost:8000/predict/AAPL
```

```json
{
  "symbol": "AAPL",
  "forecast_7d": [
    { "day": 1, "date": "2026-03-11", "predicted_close": 191.23 },
    { "day": 2, "date": "2026-03-12", "predicted_close": 192.86 },
    { "day": 3, "date": "2026-03-13", "predicted_close": 190.44 },
    { "day": 4, "date": "2026-03-14", "predicted_close": 193.15 },
    { "day": 5, "date": "2026-03-15", "predicted_close": 195.02 },
    { "day": 6, "date": "2026-03-18", "predicted_close": 196.78 },
    { "day": 7, "date": "2026-03-19", "predicted_close": 197.40 }
  ],
  "unit": "USD"
}
```

---

#### `GET /backtest/{symbol}` — Portfolio Simulation Report

```bash
curl http://localhost:8000/backtest/AAPL
```

```json
{
  "symbol": "AAPL",
  "initial_investment": 10000.00,
  "final_value": 11342.50,
  "roi_percentage": 13.42,
  "period": "Last 6 Months"
}
```

---

## 📉 Dashboard Overview

The **Streamlit + Plotly** dashboard provides a three-panel interactive canvas:

```
┌─────────────────────────────────────────────────┐
│  PANEL 1: Price + SMA(20/50) + Prophet Forecast │
│  ─────────────────────────────────────────────  │
│  [Candlestick / Line Chart]  [Forecast Ribbon]  │
├─────────────────────────────────────────────────┤
│  PANEL 2: RSI (Relative Strength Index)         │
│  ─────────────────────────────────────────────  │
│  [RSI Line]  [Overbought@70]  [Oversold@30]     │
├─────────────────────────────────────────────────┤
│  PANEL 3: MACD Histogram + Signal Lines         │
│  ─────────────────────────────────────────────  │
│  [MACD Bar Chart]  [Signal Line]  [Zero Line]   │
└─────────────────────────────────────────────────┘
```

> **Screenshot placeholder** — Add `reports/prediction_plots.png` after full training run

---

## 💼 Portfolio Backtesting Engine

The walk-forward simulator evaluates strategy performance against ground truth:

**Strategy Logic** (as implemented in `WalkForwardBacktester.run`):

```python
# Model predicts next-bar return from the expanding train window.
# Sign of the prediction sets the position; executed at next bar's open.
if predicted_return > 0:
    position = +max_position     # long
elif allow_short:
    position = -max_position
else:
    position = 0.0               # flat (default: long-only)
```

Sentiment is not yet wired into the position rule — `news_sentiment.py`
produces a real FinBERT signal, but NewsAPI's free tier only reaches ~30 days
back, which is too short to backtest over the 705-bar window below. Combining
them requires a historical news archive; that is listed under Future Work
rather than claimed here.

**Simulation Parameters (all enforced in code):**

| Parameter | Value |
|---|---|
| Initial Capital | $100,000 |
| Walk-forward | Expanding window, 500-bar min train, 20-bar test blocks, 36 folds |
| Commission | 5 bps per side |
| Slippage | 5 bps per side |
| Execution | Signal from bar `t` close → fill at bar `t+1` open |
| Position Sizing | Long-only, fully invested or flat |

### Measured results — AAPL, 2023-11-09 to 2026-09-02 (705 bars)

`python portfolio_simulator/backtesting_engine.py`

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total return | +19.11% | **+79.84%** |
| Annualised return | +6.45% | +23.34% |
| Volatility (ann.) | 21.07% | 28.82% |
| Sharpe | 0.21 | **0.74** |
| Sortino | 0.27 | 1.01 |
| Max drawdown | -24.57% | -33.33% |
| Calmar | 0.26 | 0.70 |
| Hit rate | 27.94% | 53.62% |
| Trades | 129 | 1 |
| Transaction costs paid | 12.90% of capital | — |

**Verdict: the strategy underperforms buy-and-hold by 60.7 percentage points.**

The earlier version of this file reported ROI from a function whose body was
`final_capital = initial_capital * 1.082` — a hardcoded 8.2% independent of
any model, data, or trade. That has been deleted.

Two things this result teaches, both worth more than a fake positive number:

1. **Costs dominate.** 129 trades at 10 bps round-trip consumed 12.9% of
   capital. The gross strategy was near break-even against the benchmark;
   costs are what buried it.
2. **The strategy did reduce risk.** Lower volatility (21.1% vs 28.8%) and a
   shallower drawdown (-24.6% vs -33.3%) — it was flat ~46% of the time. But
   risk-adjusted it still loses on Sharpe, so that is not a defence.

### Anti-look-ahead guarantees in the engine

| Risk | Mitigation |
|---|---|
| Scaler/model sees future data | Fresh model instance per fold; fit only on the expanding train window |
| Trading on the deciding bar | Signal from bar `t` executes at `t+1` open |
| Frictionless fills | Commission + slippage charged on every position change |
| No benchmark | Buy-and-hold computed on the identical window, always reported |

---

## 📁 Project Structure

```
real-time-stock-prediction/
│
├── 📄 README.md                    ← You are here
├── 📄 requirements.txt             ← Python dependencies
├── 🐳 Dockerfile                   ← Container build file
├── 📄 .gitignore
│
├── data/
│   ├── raw/                        ← Ingested OHLCV CSVs
│   └── processed/                  ← Feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_analysis.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_stock_data.py     ← yfinance ingestion
│   │   ├── fetch_news.py           ← News API client
│   │   └── preprocess_data.py      ← Feature engineering (SMA/RSI/MACD)
│   │
│   ├── models/
│   │   ├── lstm_model.py           ← PyTorch LSTM architecture
│   │   ├── transformer_model.py    ← Multi-head attention model
│   │   └── prophet_model.py        ← Facebook Prophet baseline
│   │
│   ├── training/
│   │   ├── train_lstm.py           ← LSTM training loop
│   │   ├── train_transformer.py    ← Transformer training
│   │   └── evaluate_models.py      ← MAE / RMSE / MAPE metrics
│   │
│   └── inference/
│       └── predict.py              ← Inference utilities
│
├── api/
│   ├── main.py                     ← FastAPI app + CORS config
│   └── routes.py                   ← Endpoint definitions
│
├── dashboard/
│   └── dashboard.py                ← Streamlit front-end
│
├── sentiment_analysis/
│   └── news_sentiment.py           ← Mock → FinBERT integration
│
├── portfolio_simulator/
│   └── backtesting_engine.py       ← Walk-forward simulator
│
├── models/
│   ├── lstm_model.pt               ← Saved LSTM weights
│   └── transformer_model.pt        ← Saved Transformer weights
│
├── reports/
│   ├── metrics.json                ← Auto-generated evaluation report
│   └── prediction_plots.png        ← Forecast visualization exports
│
└── tests/
    └── test_pipeline.py            ← Pytest structural & unit tests
```

---

## 🛠️ Installation Guide

### Prerequisites

- Python 3.10+
- pip / conda
- Docker (optional, for containerized deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/ISHANSHIRODE01/Real-Time-Stock-Market-Prediction-System.git
cd Real-Time-Stock-Market-Prediction-System
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Running the System

### Step 1 — Ingest Data

```bash
python src/data_pipeline/fetch_stock_data.py
```

### Step 2 — Preprocess & Engineer Features

```bash
python src/data_pipeline/preprocess_data.py
```

### Step 3 — Train the LSTM Model

```bash
python src/training/train_lstm.py
```

> Trained weights will be saved to `models/lstm_model.pt`

### Step 4 — Boot the API Backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

> Swagger UI → `http://127.0.0.1:8000/docs`

### Step 5 — Launch the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

### Step 6 — Run Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

### Build the Image

```bash
docker build -t quantx-prediction-system .
```

### Run the Container

```bash
docker run -d -p 8000:8000 quantx-prediction-system
```

### Run with Docker Compose (Multi-Service)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
  dashboard:
    build: .
    command: streamlit run dashboard/dashboard.py --server.port 8501
    ports:
      - "8501:8501"
```

```bash
docker-compose up --build
```

---

## 🧪 Experiment Tracking

All training experiments log the following artifacts to `reports/metrics.json`:

```json
{
  "model": "LSTM",
  "MAE": 3.21,
  "MSE": 23.72,
  "RMSE": 4.87,
  "epoch": 50,
  "learning_rate": 0.001,
  "hidden_dim": 128,
  "sequence_length": 60
}
```

**Planned Integration:**

- 🔗 [MLflow](https://mlflow.org/) for experiment versioning
- 🔗 [Weights & Biases](https://wandb.ai/) for real-time loss curves
- 🔗 [DVC](https://dvc.org/) for data + model versioning

---

## 🔮 Future Improvements

| Enhancement | Priority | Status |
|---|---|---|
| Transformer model (full attention) | 🔴 High | 🚧 In Dev |
| Real FinBERT sentiment (not mock) | 🔴 High | 📋 Planned |
| Live WebSocket price streaming | 🔴 High | 📋 Planned |
| MLflow experiment tracking | 🟡 Medium | 📋 Planned |
| Sharpe Ratio + Max Drawdown | 🟡 Medium | 📋 Planned |
| Multi-ticker portfolio management | 🟡 Medium | 📋 Planned |
| React / Next.js frontend (replace Streamlit) | 🟢 Low | 📋 Planned |
| Options pricing integration (Black-Scholes) | 🟢 Low | 📋 Planned |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Real-Time Stock Market Prediction System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

<div align="center">

**Built with 🤖 Deep Learning · 📊 Quantitative Finance · ⚡ FastAPI**

*If this project adds value, please ⭐ star the repo and share it with the community.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/ishan-shirode-556534375/)
[![Github](https://img.shields.io/badge/Github-Visit-green?style=flat&logo=netlify)](https://github.com/ISHANSHIRODE01)

</div>
