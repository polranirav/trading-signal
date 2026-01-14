"""
Unit tests for sentiment analysis module.

Tests:
- FinBERT text analysis
- Time-weighted aggregation
- Signal quality assessment
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestFinBERTAnalyzer:
    """Test suite for FinBERTAnalyzer class."""
    
    def test_time_weight_day_0_discounted(self):
        """Test that day 0-1 news is heavily discounted."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        # Day 0 should be heavily discounted (30% weight)
        weight_day_0 = analyzer._calculate_time_weight(0)
        assert weight_day_0 == 0.3, "Day 0 should be 30% weight (discounted)"
        
        # Day 1 should also be discounted
        weight_day_1 = analyzer._calculate_time_weight(1)
        assert weight_day_1 == 0.3, "Day 1 should be 30% weight (discounted)"
    
    def test_time_weight_peak_window(self):
        """Test that days 6-30 have full weight (peak signal)."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        # Days 6-30 should have full weight (research peak window)
        for day in [6, 10, 15, 20, 25, 30]:
            weight = analyzer._calculate_time_weight(day)
            assert weight == 1.0, f"Day {day} should be 100% weight (peak window)"
    
    def test_time_weight_decays_after_30(self):
        """Test that weight decays after day 30."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        weight_35 = analyzer._calculate_time_weight(35)
        weight_60 = analyzer._calculate_time_weight(60)
        weight_90 = analyzer._calculate_time_weight(90)
        weight_120 = analyzer._calculate_time_weight(120)
        
        assert weight_35 == 0.75, "Day 35 should be 75% weight"
        assert weight_60 == 0.75, "Day 60 should still be 75% weight"
        assert weight_90 == 0.5, "Day 90 should be 50% weight"
        assert weight_120 < 0.5, "Day 120+ should be below 50% weight"
    
    def test_signal_quality_strong(self):
        """Test strong signal quality detection."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        # Strong signal: consistent positive with high peak window score
        period_scores = {'0-7d': 0.4, '7-30d': 0.5, '30-90d': 0.3, '90d+': 0.2}
        label_counts = {'positive': 8, 'negative': 1, 'neutral': 1}
        
        quality = analyzer._assess_signal_quality(period_scores, label_counts)
        assert quality == "STRONG"
    
    def test_signal_quality_conflicting(self):
        """Test conflicting signal detection."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        # Conflicting: recent negative, peak window positive
        period_scores = {'0-7d': -0.3, '7-30d': 0.4, '30-90d': 0.2, '90d+': 0.1}
        label_counts = {'positive': 4, 'negative': 4, 'neutral': 2}
        
        quality = analyzer._assess_signal_quality(period_scores, label_counts)
        assert quality == "CONFLICTING"
    
    def test_empty_sentiment(self):
        """Test empty sentiment structure."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        result = analyzer._empty_sentiment("TEST")
        
        assert result['symbol'] == "TEST"
        assert result['weighted_score'] == 0.0
        assert result['normalized_score'] == 0.5
        assert result['signal_quality'] == 'NO_DATA'
    
    def test_aggregate_sentiment_structure(self):
        """Test aggregated sentiment has correct structure."""
        from src.analytics.sentiment import FinBERTAnalyzer
        
        analyzer = FinBERTAnalyzer()
        
        # Mock news items
        now = datetime.utcnow()
        news_items = [
            {
                'headline': 'Company reports strong earnings',
                'published_at': (now - timedelta(days=10)).isoformat(),
                'source': 'Reuters'
            },
            {
                'headline': 'Positive growth outlook for next quarter',
                'published_at': (now - timedelta(days=5)).isoformat(),
                'source': 'Bloomberg'
            }
        ]
        
        # Mock the analyze_batch to avoid loading model
        with patch.object(analyzer, 'analyze_batch') as mock_analyze:
            from src.analytics.sentiment import SentimentResult, SentimentLabel
            mock_analyze.return_value = [
                SentimentResult(
                    label=SentimentLabel.POSITIVE,
                    score=0.9,
                    normalized_score=0.8,
                    raw_scores={'positive': 0.9, 'negative': 0.05, 'neutral': 0.05}
                ),
                SentimentResult(
                    label=SentimentLabel.POSITIVE,
                    score=0.85,
                    normalized_score=0.7,
                    raw_scores={'positive': 0.85, 'negative': 0.1, 'neutral': 0.05}
                ),
            ]
            
            result = analyzer.aggregate_sentiment("TEST", news_items)
        
        # Check structure
        assert 'symbol' in result
        assert 'weighted_score' in result
        assert 'normalized_score' in result
        assert 'overall_label' in result
        assert 'period_scores' in result
        assert 'signal_quality' in result
        assert 'peak_window_signal' in result


class TestRAGAnalysisEngine:
    """Test suite for RAGAnalysisEngine class."""
    
    def test_chunk_text(self):
        """Test text chunking."""
        from src.analytics.llm_analysis import RAGAnalysisEngine
        
        engine = RAGAnalysisEngine()
        
        # Generate long text
        long_text = "This is a sentence. " * 100
        
        chunks = engine._chunk_text(long_text)
        
        assert len(chunks) > 1, "Long text should be split into multiple chunks"
        assert all(len(c) <= engine.CHUNK_SIZE for c in chunks), "Chunks should not exceed max size"
    
    def test_fallback_report_structure(self):
        """Test fallback report generation."""
        from src.analytics.llm_analysis import RAGAnalysisEngine
        
        engine = RAGAnalysisEngine()
        
        market_data = {'close': 150.0, 'volume': 1000000}
        technical_data = {
            'technical_score': 0.65,
            'signal_type': 'BUY',
            'rsi_14': 55.0,
            'trend_signal': 'UPTREND'
        }
        sentiment_data = {
            'weighted_score': 0.3,
            'overall_label': 'positive',
            'article_count': 5,
            'signal_quality': 'MODERATE'
        }
        
        report = engine._generate_fallback_report(
            "AAPL", market_data, technical_data, sentiment_data
        )
        
        assert "AAPL" in report
        assert "BUY" in report
        assert "UPTREND" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
