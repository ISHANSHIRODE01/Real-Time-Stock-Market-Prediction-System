"""
News sentiment scoring using FinBERT.

Replaces an earlier placeholder that returned `random.uniform(-0.6, 0.9)`.
Every score produced here derives from real retrieved headlines.

Model: ProsusAI/finbert - BERT fine-tuned on financial text, 3-class
(positive / negative / neutral). Chosen over general-purpose sentiment models
because financial language inverts normal polarity: "shares plunge despite
earnings beat" is negative for price despite "beat" reading positive in
general English.

Sources, in order of preference:
  1. NewsAPI (requires NEWSAPI_KEY; 100 req/day free tier)
  2. yfinance Ticker.news (no key, sparser coverage)

Look-ahead bias
---------------
For backtesting, sentiment MUST be timestamped and only applied to bars at or
after publication. `fetch_headlines` therefore always returns `published_at`,
and every downstream function preserves it. Aggregating sentiment without
respecting publication time is the most common way to produce a backtest that
looks profitable and is not.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
_LABEL_TO_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


@dataclass
class Headline:
    """A single retrieved news item."""

    title: str
    published_at: datetime
    source: str
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        return d


@dataclass
class ScoredHeadline(Headline):
    """A headline with FinBERT sentiment attached."""

    label: str = "neutral"
    confidence: float = 0.0
    signed_score: float = 0.0


# --- retrieval ------------------------------------------------------------

def _fetch_from_newsapi(ticker: str, days: int, api_key: str) -> list[Headline]:
    """Query NewsAPI /everything for recent articles mentioning `ticker`."""
    import requests

    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": ticker,
            "from": since,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
        },
        headers={"X-Api-Key": api_key},
        timeout=15,
    )
    resp.raise_for_status()

    out: list[Headline] = []
    for art in resp.json().get("articles", []):
        title = (art.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        try:
            ts = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
        except (KeyError, ValueError, AttributeError):
            continue
        out.append(
            Headline(
                title=title,
                published_at=ts,
                source=(art.get("source") or {}).get("name", "newsapi"),
                url=art.get("url", ""),
            )
        )
    return out


def _fetch_from_yfinance(ticker: str, days: int) -> list[Headline]:
    """Read Ticker.news, tolerating both payload shapes yfinance has shipped."""
    import yfinance as yf

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[Headline] = []

    for item in getattr(yf.Ticker(ticker), "news", None) or []:
        content = item.get("content", item)
        title = (content.get("title") or "").strip()
        if not title:
            continue

        ts = None
        if "providerPublishTime" in item:
            ts = datetime.fromtimestamp(item["providerPublishTime"], tz=timezone.utc)
        else:
            raw = content.get("pubDate") or content.get("displayTime")
            if raw:
                try:
                    ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                except ValueError:
                    ts = None
        if ts is None or ts < cutoff:
            continue

        provider = content.get("provider")
        source = (
            provider.get("displayName", "yfinance")
            if isinstance(provider, dict)
            else "yfinance"
        )
        canonical = content.get("canonicalUrl")
        url = (
            canonical.get("url", "")
            if isinstance(canonical, dict)
            else item.get("link", "")
        )
        out.append(Headline(title=title, published_at=ts, source=source, url=url))

    return out


def fetch_headlines(ticker: str, days: int = 7) -> list[Headline]:
    """Retrieve recent headlines for `ticker`, newest first.

    Returns an empty list when no source is reachable. Callers must handle the
    empty case rather than substituting a default sentiment value.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if api_key:
        try:
            heads = _fetch_from_newsapi(ticker, days, api_key)
            if heads:
                logger.info("Retrieved %d headlines for %s via NewsAPI", len(heads), ticker)
                return sorted(heads, key=lambda h: h.published_at, reverse=True)
            logger.warning("NewsAPI returned no usable articles for %s", ticker)
        except Exception as exc:
            logger.warning("NewsAPI failed for %s (%s); falling back to yfinance", ticker, exc)
    else:
        logger.info("NEWSAPI_KEY not set; using yfinance news for %s", ticker)

    try:
        heads = _fetch_from_yfinance(ticker, days)
        logger.info("Retrieved %d headlines for %s via yfinance", len(heads), ticker)
        return sorted(heads, key=lambda h: h.published_at, reverse=True)
    except Exception as exc:
        logger.error("All news sources failed for %s: %s", ticker, exc)
        return []


# --- scoring --------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    """Lazily construct the FinBERT pipeline (~440MB download on first call)."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading FinBERT (%s)", FINBERT_MODEL)
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
    return _pipeline


def score_headlines(headlines: list[Headline], batch_size: int = 16) -> list[ScoredHeadline]:
    """Attach FinBERT sentiment to each headline.

    `signed_score` = confidence * sign(label), so it lies in [-1, 1]: a
    confidently negative headline approaches -1, neutral sits near 0.
    """
    if not headlines:
        return []

    pipe = _get_pipeline()
    results = pipe([h.title for h in headlines], batch_size=batch_size)

    scored: list[ScoredHeadline] = []
    for head, res in zip(headlines, results):
        label = res["label"].lower()
        conf = float(res["score"])
        scored.append(
            ScoredHeadline(
                title=head.title,
                published_at=head.published_at,
                source=head.source,
                url=head.url,
                label=label,
                confidence=conf,
                signed_score=conf * _LABEL_TO_SIGN.get(label, 0.0),
            )
        )
    return scored


def aggregate_sentiment(
    scored: list[ScoredHeadline],
    half_life_hours: float = 24.0,
) -> dict[str, Any]:
    """Collapse scored headlines into one exponentially time-weighted score.

    Recent news dominates: weight = 0.5 ** (age_hours / half_life_hours).

    Returns `score=None` when there is no news. This is deliberate: a caller
    must distinguish "no information" from "neutral information". Silently
    returning 0.0 would let an empty feed masquerade as a real neutral signal.
    """
    if not scored:
        return {"score": None, "n_headlines": 0, "reason": "no headlines retrieved"}

    now = datetime.now(timezone.utc)
    num = den = 0.0
    for s in scored:
        age_h = max((now - s.published_at).total_seconds() / 3600.0, 0.0)
        w = 0.5 ** (age_h / half_life_hours)
        num += w * s.signed_score
        den += w

    dist = {
        lab: sum(1 for s in scored if s.label == lab)
        for lab in ("positive", "negative", "neutral")
    }
    return {
        "score": round(num / den, 4) if den > 0 else 0.0,
        "n_headlines": len(scored),
        "distribution": dist,
        "half_life_hours": half_life_hours,
        "newest": scored[0].published_at.isoformat(),
        "oldest": scored[-1].published_at.isoformat(),
    }


def get_sentiment(ticker: str, days: int = 7) -> dict[str, Any]:
    """End-to-end: retrieve, score, aggregate. `score` is None if no news."""
    scored = score_headlines(fetch_headlines(ticker, days=days))
    result = aggregate_sentiment(scored)
    result["ticker"] = ticker
    result["headlines"] = [
        {
            "title": s.title,
            "label": s.label,
            "signed_score": round(s.signed_score, 4),
            "published_at": s.published_at.isoformat(),
            "source": s.source,
        }
        for s in scored[:10]
    ]
    return result


def build_daily_sentiment_series(scored: list[ScoredHeadline]):
    """Aggregate scored headlines into a per-day mean sentiment series.

    Indexed by UTC date so it can be joined to daily bars without look-ahead:
    a headline published on day D informs the decision for day D+1.
    """
    import pandas as pd

    if not scored:
        return pd.DataFrame(columns=["sentiment", "n_headlines"]).rename_axis("date")

    df = pd.DataFrame(
        {
            "date": [s.published_at.astimezone(timezone.utc).date() for s in scored],
            "signed_score": [s.signed_score for s in scored],
        }
    )
    agg = df.groupby("date")["signed_score"].agg(["mean", "count"])
    agg.columns = ["sentiment", "n_headlines"]
    return agg.sort_index()


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = get_sentiment("AAPL", days=7)
    print(json.dumps({k: v for k, v in out.items() if k != "headlines"}, indent=2))
    for h in out["headlines"][:5]:
        print(f"  [{h['label']:8s} {h['signed_score']:+.3f}] {h['title'][:80]}")
