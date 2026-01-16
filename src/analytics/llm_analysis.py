"""
LLM-Powered Analysis with RAG (Retrieval-Augmented Generation).

Uses GPT-4 with Pinecone vector database for:
1. Earnings call analysis
2. Research report synthesis
3. News context augmentation
4. Investment thesis generation

The RAG approach allows us to:
- Store and retrieve relevant financial documents
- Provide context to LLM for more accurate analysis
- Generate professional-grade research reports

Architecture:
    Documents → Chunk → Embed → Pinecone Store
    Query → Retrieve Top-K → Augment Prompt → GPT-4 → Response

Usage:
    engine = RAGAnalysisEngine()
    report = engine.synthesize_research_report(symbol, data)
"""

from openai import OpenAI
from typing import List, Dict, Optional
from datetime import datetime
import json
import hashlib

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# Optional Pinecone import
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not installed. RAG features will be limited.")


class RAGAnalysisEngine:
    """
    Retrieval-Augmented Generation engine for financial analysis.
    
    Combines:
    - Pinecone vector store for document retrieval
    - OpenAI embeddings for semantic search
    - GPT-4 for analysis synthesis
    """
    
    INDEX_NAME = "trading-signals"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIM = 1536
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 5
    
    def __init__(self):
        self.client = None
        self.pinecone = None
        self.index = None
        self._initialized = False
        
        # Check if API keys are configured
        self.openai_enabled = bool(settings.OPENAI_API_KEY)
        self.pinecone_enabled = PINECONE_AVAILABLE and bool(settings.PINECONE_API_KEY)
    
    def _ensure_initialized(self):
        """Lazy initialization of API clients."""
        if self._initialized:
            return
        
        # Initialize OpenAI
        if self.openai_enabled:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.openai_enabled = False
        
        # Initialize Pinecone
        if self.pinecone_enabled:
            try:
                self.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
                
                # Check if index exists
                existing_indexes = [idx.name for idx in self.pinecone.list_indexes()]
                
                if self.INDEX_NAME not in existing_indexes:
                    self.pinecone.create_index(
                        name=self.INDEX_NAME,
                        dimension=self.EMBEDDING_DIM,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    logger.info(f"Created Pinecone index: {self.INDEX_NAME}")
                
                self.index = self.pinecone.Index(self.INDEX_NAME)
                logger.info("Pinecone client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                self.pinecone_enabled = False
        
        self._initialized = True
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI."""
        if not self.openai_enabled:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]
            
            # Find natural break point (sentence end)
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.CHUNK_SIZE // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.CHUNK_OVERLAP
        
        return [c for c in chunks if c]
    
    def store_document(
        self,
        symbol: str,
        doc_type: str,  # 'earnings', 'news', 'filing', 'research'
        content: str,
        metadata: Dict = None
    ) -> int:
        """
        Store document in vector database.
        
        Args:
            symbol: Stock ticker
            doc_type: Type of document
            content: Full document text
            metadata: Additional metadata
        
        Returns:
            Number of chunks stored
        """
        self._ensure_initialized()
        
        if not self.pinecone_enabled or not self.openai_enabled:
            logger.warning("RAG storage disabled - missing API keys")
            return 0
        
        # Chunk the document
        chunks = self._chunk_text(content)
        
        if not chunks:
            return 0
        
        # Prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            if not embedding:
                continue
            
            # Create unique ID
            doc_hash = hashlib.md5(content[:500].encode()).hexdigest()[:8]
            vector_id = f"{symbol}_{doc_type}_{doc_hash}_{i}"
            
            # Prepare metadata
            chunk_metadata = {
                "symbol": symbol,
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk[:500],  # Store preview
                "created_at": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": chunk_metadata
            })
        
        # Upsert to Pinecone
        if vectors:
            self.index.upsert(vectors=vectors)
            logger.info(f"Stored {len(vectors)} chunks for {symbol} ({doc_type})")
        
        return len(vectors)
    
    def retrieve_context(
        self,
        query: str,
        symbol: str = None,
        doc_types: List[str] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            symbol: Filter by symbol (optional)
            doc_types: Filter by document types (optional)
            top_k: Number of results
        
        Returns:
            List of relevant document chunks with scores
        """
        self._ensure_initialized()
        
        if not self.pinecone_enabled or not self.openai_enabled:
            return []
        
        top_k = top_k or self.TOP_K
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        # Build filter
        filter_dict = {}
        if symbol:
            filter_dict["symbol"] = symbol
        if doc_types:
            filter_dict["doc_type"] = {"$in": doc_types}
        
        # Query Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "symbol": match.metadata.get("symbol"),
                    "doc_type": match.metadata.get("doc_type")
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def analyze_with_context(
        self,
        query: str,
        symbol: str,
        context: List[Dict] = None
    ) -> str:
        """
        Answer a query using retrieved context (RAG).
        
        Args:
            query: User question
            symbol: Stock symbol
            context: Pre-retrieved context (optional)
        
        Returns:
            LLM response grounded in context
        """
        self._ensure_initialized()
        
        if not self.openai_enabled:
            return "OpenAI API not configured"
        
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve_context(query, symbol=symbol)
        
        # Build context string
        context_str = "\n\n".join([
            f"[Source: {c.get('doc_type', 'unknown')}]\n{c.get('text', '')}"
            for c in context
        ]) if context else "No additional context available."
        
        prompt = f"""You are a senior financial analyst analyzing {symbol}.

RELEVANT CONTEXT:
{context_str}

QUESTION: {query}

Provide a detailed, professional analysis based on the context above. 
If the context doesn't contain relevant information, say so and provide general analysis.
Be specific with numbers and cite sources when possible."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior equity research analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def synthesize_research_report(
        self,
        symbol: str,
        market_data: Dict,
        technical_data: Dict,
        sentiment_data: Dict
    ) -> str:
        """
        Generate comprehensive research report using GPT-4.
        
        Combines:
        - Market data (price, volume, market cap)
        - Technical analysis (indicators, signals)
        - Sentiment analysis (FinBERT scores)
        
        Args:
            symbol: Stock ticker
            market_data: Dict with price, volume, market_cap, etc.
            technical_data: Dict with indicators and signals
            sentiment_data: Dict with sentiment scores
        
        Returns:
            Markdown-formatted research report
        """
        self._ensure_initialized()
        
        if not self.openai_enabled:
            return self._generate_fallback_report(symbol, market_data, technical_data, sentiment_data)
        
        # Build structured context
        context = f"""
MARKET DATA FOR {symbol}:
- Current Price: ${market_data.get('close', 0):.2f}
- Previous Close: ${market_data.get('prev_close', 0):.2f}
- Daily Change: {market_data.get('change_pct', 0):+.2f}%
- Volume: {market_data.get('volume', 0):,.0f}
- Avg Volume (20d): {market_data.get('volume_sma_20', 0):,.0f}
- Volume Ratio: {market_data.get('volume_ratio', 1):.2f}x

TECHNICAL ANALYSIS:
- Technical Score: {technical_data.get('technical_score', 0.5):.3f} (0-1 scale)
- Signal: {technical_data.get('signal_type', 'HOLD')}
- RSI (14): {technical_data.get('rsi_14', 50):.1f}
- MACD: {technical_data.get('macd', 0):.4f}
- MACD Signal: {technical_data.get('macd_signal', 0):.4f}
- Trend: {technical_data.get('trend_signal', 'SIDEWAYS')}
- SMA 50: ${technical_data.get('sma_50', 0):.2f}
- SMA 200: ${technical_data.get('sma_200', 0):.2f}
- ATR: ${technical_data.get('atr', 0):.2f}
- Bollinger %B: {technical_data.get('bb_percent_b', 0.5):.2f}

SENTIMENT ANALYSIS:
- Weighted Sentiment: {sentiment_data.get('weighted_score', 0):+.3f} (-1 to +1)
- Overall Label: {sentiment_data.get('overall_label', 'neutral')}
- Article Count: {sentiment_data.get('article_count', 0)}
- Signal Quality: {sentiment_data.get('signal_quality', 'NO_DATA')}
- Peak Window Signal (7-30d): {sentiment_data.get('peak_window_signal', 0):+.3f}
- Sentiment Momentum: {sentiment_data.get('sentiment_momentum', 0):+.3f}
- Positive/Negative/Neutral: {sentiment_data.get('label_counts', {}).get('positive', 0)}/{sentiment_data.get('label_counts', {}).get('negative', 0)}/{sentiment_data.get('label_counts', {}).get('neutral', 0)}
"""

        prompt = f"""Generate a professional investment research report for {symbol}.

DATA PROVIDED:
{context}

REPORT STRUCTURE (use markdown formatting):

# {symbol} Investment Analysis

## Executive Summary
(2-3 sentences with clear recommendation and rationale)

## Technical Analysis
- Price action and momentum
- Key support/resistance levels
- Trend assessment

## Sentiment Analysis
- News sentiment interpretation
- Information flow assessment (are we in the peak predictive window?)
- Sentiment momentum signal

## Signal Confluence
- How do technical and sentiment signals align?
- Overall conviction level
- Risk assessment

## Trading Recommendation
- Signal: [STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL]
- Conviction: [HIGH/MEDIUM/LOW]
- Time Horizon: [Days/Weeks/Months]
- Position Size Suggestion: [% of portfolio]

## Risk Factors
- Key risks to monitor (bullet list)

## Catalysts
- Upcoming events that could move the stock (bullet list)

Be specific with numbers. Use professional analyst language.
If data suggests conflicting signals, acknowledge the uncertainty.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior equity research analyst at a top-tier investment bank. You provide clear, actionable analysis with specific price levels and time horizons."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2500
            )
            
            report = response.choices[0].message.content
            logger.info(f"Generated research report for {symbol}, length: {len(report)}")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return self._generate_fallback_report(symbol, market_data, technical_data, sentiment_data)
    
    def _generate_fallback_report(
        self,
        symbol: str,
        market_data: Dict,
        technical_data: Dict,
        sentiment_data: Dict
    ) -> str:
        """Generate basic report without GPT-4."""
        
        signal = technical_data.get('signal_type', 'HOLD')
        score = technical_data.get('technical_score', 0.5)
        sentiment = sentiment_data.get('weighted_score', 0)
        
        return f"""# {symbol} Analysis Report

## Summary
- Technical Signal: {signal}
- Technical Score: {score:.3f}
- Sentiment Score: {sentiment:+.3f}

## Technical Indicators
- RSI (14): {technical_data.get('rsi_14', 'N/A')}
- Trend: {technical_data.get('trend_signal', 'N/A')}
- MACD: {technical_data.get('macd', 'N/A')}

## Sentiment
- Label: {sentiment_data.get('overall_label', 'N/A')}
- Articles: {sentiment_data.get('article_count', 0)}
- Quality: {sentiment_data.get('signal_quality', 'N/A')}

*Note: Full analysis requires OpenAI API configuration.*
"""


    def analyze_trading_history(
        self,
        transactions: List[Dict],
        summary_by_symbol: Dict,
        total_realized_pnl: float
    ) -> str:
        """
        Analyze user's trading history and generate personalized lessons learned.
        
        This is the core of the "Lessons from My Trades" feature.
        
        Args:
            transactions: List of transactions with pnl data from timeline endpoint
            summary_by_symbol: Dict with symbol -> {current_shares, realized_pnl, avg_cost}
            total_realized_pnl: Total realized P&L across all trades
        
        Returns:
            Markdown-formatted lessons and insights
        """
        self._ensure_initialized()
        
        if not self.openai_enabled:
            return self._generate_fallback_lessons(transactions, summary_by_symbol, total_realized_pnl)
        
        # Prepare transaction summary
        symbols_traded = list(summary_by_symbol.keys())
        winners = {s: d for s, d in summary_by_symbol.items() if d.get('realized_pnl', 0) > 0}
        losers = {s: d for s, d in summary_by_symbol.items() if d.get('realized_pnl', 0) < 0}
        
        # Analyze trades
        buy_count = len([t for t in transactions if t.get('transaction_type') == 'buy'])
        sell_count = len([t for t in transactions if t.get('transaction_type') == 'sell'])
        
        # Calculate timing patterns
        profitable_sells = [t for t in transactions if t.get('transaction_type') == 'sell' and (t.get('pnl') or 0) > 0]
        losing_sells = [t for t in transactions if t.get('transaction_type') == 'sell' and (t.get('pnl') or 0) < 0]
        
        # Build context
        context = f"""
TRADING HISTORY SUMMARY:
- Total Symbols Traded: {len(symbols_traded)}
- Total Transactions: {len(transactions)}
- Buy Transactions: {buy_count}
- Sell Transactions: {sell_count}
- Total Realized P&L: ${total_realized_pnl:,.2f}

WINNERS (Profitable Trades):
{self._format_positions(winners) if winners else "No profitable closed positions yet"}

LOSERS (Unprofitable Trades):
{self._format_positions(losers) if losers else "No losing closed positions yet"}

WIN/LOSS STATISTICS:
- Profitable Sells: {len(profitable_sells)}
- Losing Sells: {len(losing_sells)}
- Win Rate: {len(profitable_sells) / max(len(profitable_sells) + len(losing_sells), 1) * 100:.1f}%

SAMPLE TRANSACTIONS (most recent):
{self._format_transactions(transactions[:20])}
"""

        prompt = f"""You are a professional trading coach analyzing a personal trading history.
Your goal is to provide honest, actionable lessons learned from this trading history.

{context}

Generate a comprehensive "Lessons Learned" report with the following sections:

# 📚 Lessons From My Trades

## 🎯 Executive Summary
(2-3 sentences summarizing overall trading performance and the key takeaway)

## ✅ What Went Right
(Bullet points of positive patterns observed - timing, position sizing, sector selection, etc.)

## ❌ What Went Wrong
(Bullet points of mistakes identified - selling too early/late, chasing losses, etc.)

## 📊 Pattern Analysis
- Trading frequency assessment
- Position sizing patterns
- Timing patterns (holding period analysis)
- Sector/Symbol concentration

## 💡 Actionable Recommendations
(Specific, numbered recommendations for improvement)
1. ...
2. ...
3. ...

## 🎓 Key Lesson
(Single most important lesson from this trading history - make it personal and memorable)

Be specific. Reference actual symbols and trades when possible.
Be honest but constructive - the goal is to help the trader improve.
If there's limited data, acknowledge it and provide general guidance.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an experienced trading coach and behavioral finance expert. You analyze trading patterns and provide constructive feedback to help traders improve. Be specific, honest, and actionable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            report = response.choices[0].message.content
            logger.info(f"Generated trading lessons, length: {len(report)}")
            return report
            
        except Exception as e:
            logger.error(f"Trading history analysis failed: {e}")
            return self._generate_fallback_lessons(transactions, summary_by_symbol, total_realized_pnl)
    
    def _format_positions(self, positions: Dict) -> str:
        """Format positions for prompt."""
        lines = []
        for symbol, data in positions.items():
            pnl = data.get('realized_pnl', 0)
            shares = data.get('current_shares', 0)
            lines.append(f"- {symbol}: P&L ${pnl:,.2f}, Current Shares: {shares}")
        return "\n".join(lines) if lines else "None"
    
    def _format_transactions(self, transactions: List[Dict]) -> str:
        """Format recent transactions for prompt."""
        lines = []
        for t in transactions:
            date = t.get('transaction_date', '')[:10] if t.get('transaction_date') else 'N/A'
            pnl = t.get('pnl')
            pnl_str = f", P&L: ${pnl:,.2f}" if pnl is not None else ""
            lines.append(
                f"- {date}: {t.get('transaction_type', '?').upper()} {t.get('symbol', '?')} "
                f"{t.get('shares', 0)} shares @ ${t.get('price', 0):.2f}{pnl_str}"
            )
        return "\n".join(lines) if lines else "No transactions"
    
    def _generate_fallback_lessons(
        self,
        transactions: List[Dict],
        summary_by_symbol: Dict,
        total_realized_pnl: float
    ) -> str:
        """Generate basic lessons without GPT-4."""
        
        symbols = list(summary_by_symbol.keys())
        winners = len([s for s, d in summary_by_symbol.items() if d.get('realized_pnl', 0) > 0])
        losers = len([s for s, d in summary_by_symbol.items() if d.get('realized_pnl', 0) < 0])
        
        return f"""# 📚 Lessons From My Trades

## Summary
- Symbols Traded: {len(symbols)}
- Total Transactions: {len(transactions)}
- Realized P&L: ${total_realized_pnl:,.2f}

## Performance
- Winning Positions: {winners}
- Losing Positions: {losers}
- Win Rate: {winners / max(winners + losers, 1) * 100:.1f}%

## Symbols Traded
{', '.join(symbols) if symbols else 'None yet'}

*Note: Full AI-powered analysis requires OpenAI API configuration.*

To enable personalized trading lessons:
1. Set your OPENAI_API_KEY in the .env file
2. Restart the backend server
3. Return to this page for detailed insights
"""


# Convenience function
def get_rag_engine() -> RAGAnalysisEngine:
    """Get a RAGAnalysisEngine instance."""
    return RAGAnalysisEngine()
