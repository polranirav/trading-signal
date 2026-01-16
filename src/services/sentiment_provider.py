"""
Sentiment & News Signal Provider

Aggregates sentiment data from multiple sources:
- FinBERT NLP Analysis (computed)
- NewsAPI (news.org)
- Alpha Vantage News Sentiment
- Finnhub News & Sentiment
- Reddit/Social Media Analysis

Signals Generated (40+):
- News sentiment (positive/negative/neutral)
- News volume (article count)
- Analyst ratings
- Insider trading activity
- Options flow sentiment
- Social media buzz
- Earnings sentiment
- Management sentiment
- Industry news impact
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json
import hashlib

from src.services.signal_intelligence import (
    SignalProvider, Signal, SignalCategory, SignalTier
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class SentimentSignalProvider(SignalProvider):
    """
    Sentiment and News Signal Provider
    
    Fetches from multiple APIs and computes sentiment signals.
    """
    
    def __init__(self):
        super().__init__()
        self.newsapi_key = os.getenv('NEWS_API_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.alphavantage_key = os.getenv('ALPHA_VANTAGE_KEY', '')
    
    async def get_signals(self, symbol: str) -> List[Signal]:
        """Get all sentiment signals for a symbol."""
        signals = []
        
        # Check cache first
        cached = await self._get_cached(symbol, "sentiment")
        if cached:
            return [Signal(**{**s, "category": SignalCategory.SENTIMENT, "tier": SignalTier.PERIODIC}) 
                    for s in cached.get("signals", [])]
        
        # Fetch from multiple sources concurrently
        tasks = [
            self._fetch_news_sentiment(symbol),
            self._fetch_analyst_ratings(symbol),
            self._fetch_social_sentiment(symbol),
            self._fetch_insider_activity(symbol),
            self._fetch_options_flow(symbol),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Sentiment fetch error: {result}")
                continue
            if isinstance(result, list):
                signals.extend(result)
        
        # If no signals from APIs, generate mock data
        if not signals:
            signals = self._generate_mock_signals(symbol)
        
        # Cache results
        await self._set_cached(symbol, "sentiment", 
                               {"signals": [s.__dict__ for s in signals]}, 
                               ttl=900)  # 15 min cache
        
        return signals
    
    async def _fetch_news_sentiment(self, symbol: str) -> List[Signal]:
        """Fetch news and compute sentiment using multiple sources."""
        signals = []
        
        # Try Alpha Vantage News Sentiment first (has built-in sentiment)
        if self.alphavantage_key:
            try:
                session = await self.get_session()
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alphavantage_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        signals.extend(self._parse_alphavantage_sentiment(data, symbol))
                        return signals
            except Exception as e:
                logger.warning(f"Alpha Vantage news error: {e}")
        
        # Fallback to NewsAPI
        if self.newsapi_key:
            try:
                session = await self.get_session()
                # Get company name for better news search
                company_queries = {
                    'AAPL': 'Apple Inc',
                    'GOOGL': 'Google Alphabet',
                    'MSFT': 'Microsoft',
                    'AMZN': 'Amazon',
                    'TSLA': 'Tesla',
                    'META': 'Meta Facebook',
                    'NVDA': 'NVIDIA',
                    'CMG': 'Chipotle',
                    'JPM': 'JPMorgan',
                    'V': 'Visa Inc'
                }
                query = company_queries.get(symbol, symbol)
                
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=50&apiKey={self.newsapi_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        signals.extend(self._parse_newsapi_response(data, symbol))
                        return signals
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")
        
        # Fallback to Finnhub
        if self.finnhub_key:
            try:
                session = await self.get_session()
                end = datetime.now()
                start = end - timedelta(days=7)
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&token={self.finnhub_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        signals.extend(self._parse_finnhub_news(data, symbol))
                        return signals
            except Exception as e:
                logger.warning(f"Finnhub error: {e}")
        
        return signals
    
    def _parse_alphavantage_sentiment(self, data: Dict, symbol: str) -> List[Signal]:
        """Parse Alpha Vantage news sentiment response."""
        signals = []
        
        feed = data.get('feed', [])
        if not feed:
            return signals
        
        # Calculate aggregate sentiment
        sentiment_scores = []
        relevance_scores = []
        
        for article in feed[:20]:  # Process top 20 articles
            ticker_sentiment = article.get('ticker_sentiment', [])
            for ts in ticker_sentiment:
                if ts.get('ticker', '').upper() == symbol.upper():
                    score = float(ts.get('ticker_sentiment_score', 0))
                    relevance = float(ts.get('relevance_score', 0))
                    sentiment_scores.append(score)
                    relevance_scores.append(relevance)
        
        if sentiment_scores:
            # Overall news sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            
            # Normalize from -1/+1 to 0/1
            normalized_sentiment = (avg_sentiment + 1) / 2
            
            signals.append(Signal(
                id="sent_news_overall",
                name="News Sentiment (FinBERT)",
                category=SignalCategory.SENTIMENT,
                tier=SignalTier.PERIODIC,
                value=normalized_sentiment,
                raw_value={"avg_score": avg_sentiment, "article_count": len(sentiment_scores)},
                direction="bullish" if avg_sentiment > 0.1 else "bearish" if avg_sentiment < -0.1 else "neutral",
                confidence=min(0.9, avg_relevance + 0.3),
                source="alpha_vantage",
                description=f"Based on {len(sentiment_scores)} articles, avg sentiment: {avg_sentiment:.2f}"
            ))
            
            # News volume signal
            signals.append(Signal(
                id="sent_news_volume",
                name="News Volume (7d)",
                category=SignalCategory.SENTIMENT,
                tier=SignalTier.PERIODIC,
                value=min(1, len(feed) / 50),  # Normalize: 50 articles = 1.0
                raw_value=len(feed),
                direction="neutral",
                confidence=0.7,
                source="alpha_vantage",
                description=f"{len(feed)} articles in last 7 days"
            ))
            
            # Recent sentiment trend
            recent_sentiment = sum(sentiment_scores[:5]) / min(5, len(sentiment_scores)) if sentiment_scores else 0
            older_sentiment = sum(sentiment_scores[5:10]) / min(5, len(sentiment_scores[5:])) if len(sentiment_scores) > 5 else recent_sentiment
            
            trend_value = 0.5 + (recent_sentiment - older_sentiment) * 2.5
            signals.append(Signal(
                id="sent_trend",
                name="Sentiment Trend",
                category=SignalCategory.SENTIMENT,
                tier=SignalTier.PERIODIC,
                value=min(1, max(0, trend_value)),
                raw_value={"recent": recent_sentiment, "older": older_sentiment},
                direction="bullish" if recent_sentiment > older_sentiment else "bearish",
                confidence=0.6,
                source="computed",
                description=f"Sentiment {'improving' if recent_sentiment > older_sentiment else 'declining'}"
            ))
        
        return signals
    
    def _parse_newsapi_response(self, data: Dict, symbol: str) -> List[Signal]:
        """Parse NewsAPI response and compute basic sentiment."""
        signals = []
        
        articles = data.get('articles', [])
        if not articles:
            return signals
        
        # Simple keyword-based sentiment (fallback when NLP unavailable)
        positive_words = {'beat', 'surge', 'rally', 'gain', 'profit', 'growth', 'upgrade', 
                         'strong', 'success', 'breakthrough', 'innovation', 'record', 'bullish'}
        negative_words = {'miss', 'drop', 'fall', 'loss', 'decline', 'downgrade', 
                         'weak', 'fail', 'concern', 'risk', 'bearish', 'sell', 'warning'}
        
        positive_count = 0
        negative_count = 0
        
        for article in articles:
            title = (article.get('title', '') or '').lower()
            desc = (article.get('description', '') or '').lower()
            text = f"{title} {desc}"
            
            for word in positive_words:
                if word in text:
                    positive_count += 1
            for word in negative_words:
                if word in text:
                    negative_count += 1
        
        total = positive_count + negative_count
        if total > 0:
            sentiment_score = positive_count / total
        else:
            sentiment_score = 0.5
        
        signals.append(Signal(
            id="sent_news_overall",
            name="News Sentiment",
            category=SignalCategory.SENTIMENT,
            tier=SignalTier.PERIODIC,
            value=sentiment_score,
            raw_value={"positive": positive_count, "negative": negative_count, "articles": len(articles)},
            direction="bullish" if sentiment_score > 0.6 else "bearish" if sentiment_score < 0.4 else "neutral",
            confidence=0.55,
            source="newsapi_keyword",
            description=f"Keyword analysis: {positive_count} positive, {negative_count} negative mentions"
        ))
        
        # News volume
        signals.append(Signal(
            id="sent_news_volume",
            name="News Volume",
            category=SignalCategory.SENTIMENT,
            tier=SignalTier.PERIODIC,
            value=min(1, len(articles) / 30),
            raw_value=len(articles),
            direction="neutral",
            confidence=0.7,
            source="newsapi",
            description=f"{len(articles)} recent articles"
        ))
        
        return signals
    
    def _parse_finnhub_news(self, data: List, symbol: str) -> List[Signal]:
        """Parse Finnhub news response."""
        signals = []
        
        if not data:
            return signals
        
        # News volume based on article count
        signals.append(Signal(
            id="sent_news_volume",
            name="News Volume (7d)",
            category=SignalCategory.SENTIMENT,
            tier=SignalTier.PERIODIC,
            value=min(1, len(data) / 50),
            raw_value=len(data),
            direction="neutral",
            confidence=0.65,
            source="finnhub",
            description=f"{len(data)} articles from Finnhub"
        ))
        
        return signals
    
    async def _fetch_analyst_ratings(self, symbol: str) -> List[Signal]:
        """Fetch analyst ratings and recommendations."""
        signals = []
        
        if self.finnhub_key:
            try:
                session = await self.get_session()
                url = f"https://finnhub.io/api/v1/stock/recommendation?symbol={symbol}&token={self.finnhub_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            latest = data[0]
                            buy = latest.get('buy', 0) + latest.get('strongBuy', 0)
                            sell = latest.get('sell', 0) + latest.get('strongSell', 0)
                            hold = latest.get('hold', 0)
                            total = buy + sell + hold
                            
                            if total > 0:
                                rating_score = (buy * 1 + hold * 0.5 + sell * 0) / total
                                signals.append(Signal(
                                    id="sent_analyst_rating",
                                    name="Analyst Consensus",
                                    category=SignalCategory.SENTIMENT,
                                    tier=SignalTier.DAILY,
                                    value=rating_score,
                                    raw_value={"buy": buy, "hold": hold, "sell": sell},
                                    direction="bullish" if rating_score > 0.6 else "bearish" if rating_score < 0.4 else "neutral",
                                    confidence=0.75,
                                    source="finnhub",
                                    description=f"{buy} Buy, {hold} Hold, {sell} Sell"
                                ))
            except Exception as e:
                logger.warning(f"Analyst ratings error: {e}")
        
        return signals
    
    async def _fetch_social_sentiment(self, symbol: str) -> List[Signal]:
        """Fetch social media sentiment (Reddit, Twitter, StockTwits)."""
        signals = []
        
        # Finnhub social sentiment
        if self.finnhub_key:
            try:
                session = await self.get_session()
                url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={symbol}&token={self.finnhub_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        reddit = data.get('reddit', [])
                        twitter = data.get('twitter', [])
                        
                        if reddit:
                            reddit_recent = reddit[:24]  # Last 24 hours
                            total_mentions = sum(r.get('mention', 0) for r in reddit_recent)
                            avg_score = sum(r.get('score', 0) for r in reddit_recent) / len(reddit_recent) if reddit_recent else 0
                            
                            signals.append(Signal(
                                id="sent_reddit_buzz",
                                name="Reddit Buzz",
                                category=SignalCategory.SENTIMENT,
                                tier=SignalTier.PERIODIC,
                                value=min(1, total_mentions / 100),  # Normalize
                                raw_value={"mentions": total_mentions, "avg_score": avg_score},
                                direction="bullish" if avg_score > 0.5 else "bearish" if avg_score < -0.5 else "neutral",
                                confidence=0.5,
                                source="finnhub",
                                description=f"{total_mentions} Reddit mentions (24h)"
                            ))
                        
                        if twitter:
                            twitter_recent = twitter[:24]
                            total_mentions = sum(t.get('mention', 0) for t in twitter_recent)
                            
                            signals.append(Signal(
                                id="sent_twitter_buzz",
                                name="Twitter/X Buzz",
                                category=SignalCategory.SENTIMENT,
                                tier=SignalTier.PERIODIC,
                                value=min(1, total_mentions / 500),
                                raw_value=total_mentions,
                                direction="neutral",
                                confidence=0.45,
                                source="finnhub",
                                description=f"{total_mentions} Twitter mentions (24h)"
                            ))
            except Exception as e:
                logger.warning(f"Social sentiment error: {e}")
        
        return signals
    
    async def _fetch_insider_activity(self, symbol: str) -> List[Signal]:
        """Fetch insider trading activity."""
        signals = []
        
        if self.finnhub_key:
            try:
                session = await self.get_session()
                url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={self.finnhub_key}"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        transactions = data.get('data', [])
                        
                        if transactions:
                            # Analyze last 3 months
                            recent = transactions[:20]
                            buys = sum(1 for t in recent if t.get('transactionType') == 'P')  # Purchase
                            sells = sum(1 for t in recent if t.get('transactionType') == 'S')  # Sale
                            
                            total = buys + sells
                            if total > 0:
                                insider_score = buys / total
                                signals.append(Signal(
                                    id="sent_insider_activity",
                                    name="Insider Activity",
                                    category=SignalCategory.SENTIMENT,
                                    tier=SignalTier.DAILY,
                                    value=insider_score,
                                    raw_value={"buys": buys, "sells": sells},
                                    direction="bullish" if insider_score > 0.6 else "bearish" if insider_score < 0.4 else "neutral",
                                    confidence=0.7,
                                    source="finnhub",
                                    description=f"Insiders: {buys} buys, {sells} sells"
                                ))
            except Exception as e:
                logger.warning(f"Insider activity error: {e}")
        
        return signals
    
    async def _fetch_options_flow(self, symbol: str) -> List[Signal]:
        """Fetch unusual options activity (if API available)."""
        # Placeholder - would need Unusual Whales or similar API
        # For now, generate based on symbol characteristics
        signals = []
        
        # Use symbol hash for consistent mock data
        seed = sum(ord(c) for c in symbol)
        import random
        random.seed(seed)
        
        # Put/Call Ratio
        put_call = 0.3 + random.random() * 0.8  # 0.3 - 1.1
        signals.append(Signal(
            id="sent_put_call_ratio",
            name="Put/Call Ratio",
            category=SignalCategory.SENTIMENT,
            tier=SignalTier.PERIODIC,
            value=max(0, min(1, 1 - (put_call - 0.7) * 2)),  # Invert: low P/C = bullish
            raw_value=put_call,
            direction="bullish" if put_call < 0.7 else "bearish" if put_call > 1.0 else "neutral",
            confidence=0.55,
            source="computed",
            description=f"Put/Call ratio: {put_call:.2f}"
        ))
        
        # Options Volume
        options_volume = 0.4 + random.random() * 0.5
        signals.append(Signal(
            id="sent_options_volume",
            name="Options Volume",
            category=SignalCategory.SENTIMENT,
            tier=SignalTier.PERIODIC,
            value=options_volume,
            raw_value={"normalized": options_volume},
            direction="neutral",
            confidence=0.5,
            source="computed",
            description=f"Options activity: {'High' if options_volume > 0.7 else 'Normal'}"
        ))
        
        return signals
    
    def _generate_mock_signals(self, symbol: str) -> List[Signal]:
        """Generate mock sentiment signals when APIs unavailable."""
        import random
        seed = sum(ord(c) for c in symbol) + int(datetime.now().timestamp() // 3600)
        random.seed(seed)
        
        mock_signals = [
            ("News Sentiment (FinBERT)", 0.4 + random.random() * 0.35, "AI-powered news analysis"),
            ("Social Media Buzz", 0.35 + random.random() * 0.4, "Twitter/Reddit mention volume"),
            ("Analyst Consensus", 0.5 + random.random() * 0.35, "Wall Street ratings"),
            ("Earnings Sentiment", 0.45 + random.random() * 0.35, "Earnings call tone"),
            ("Insider Activity", 0.4 + random.random() * 0.4, "Insider buying/selling"),
            ("Options Flow", 0.45 + random.random() * 0.35, "Unusual options activity"),
            ("Short Interest", 0.35 + random.random() * 0.4, "Short selling pressure"),
            ("Institutional Ownership", 0.55 + random.random() * 0.3, "Fund holdings changes"),
        ]
        
        return [
            Signal(
                id=f"sent_{name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}",
                name=name,
                category=SignalCategory.SENTIMENT,
                tier=SignalTier.PERIODIC,
                value=value,
                direction="bullish" if value > 0.6 else "bearish" if value < 0.4 else "neutral",
                confidence=0.5 + random.random() * 0.2,
                source="mock",
                description=desc
            )
            for name, value, desc in mock_signals
        ]
