"""
Email Templates for Signal Alerts.

Provides HTML and plain text email templates for trading signals.
"""

from typing import Dict, Optional
from datetime import datetime
from jinja2 import Template


class EmailTemplates:
    """Email templates for signal alerts."""
    
    @staticmethod
    def signal_alert_html(signal_data: Dict) -> str:
        """
        Generate HTML email template for signal alert.
        
        Args:
            signal_data: Dictionary with signal information
        
        Returns:
            HTML email content
        """
        signal_type = signal_data.get('signal_type', 'HOLD')
        symbol = signal_data.get('symbol', 'N/A')
        confluence_score = signal_data.get('confluence_score', 0.5)
        
        # Determine badge color
        if 'STRONG_BUY' in signal_type or 'BUY' in signal_type:
            badge_color = '#10b981'  # Green
            badge_text = signal_type
        elif 'STRONG_SELL' in signal_type or 'SELL' in signal_type:
            badge_color = '#ef4444'  # Red
            badge_text = signal_type
        else:
            badge_color = '#f59e0b'  # Amber
            badge_text = 'HOLD'
        
        # Risk-reward ratio
        rr_ratio = signal_data.get('risk_reward_ratio', 0.0)
        rr_display = f"{rr_ratio:.2f}:1" if rr_ratio > 0 else "N/A"
        rr_highlight = "font-weight: bold; color: #10b981;" if rr_ratio >= 2.0 else ""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Trading Signal: {symbol}</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #1a1a1a; background-color: #f5f5f5; margin: 0; padding: 20px;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #e5e5e5;">
            <h1 style="margin: 0; color: #1a1a1a; font-size: 28px;">🚀 New Trading Signal</h1>
            <p style="margin: 10px 0 0; color: #666; font-size: 14px;">{datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}</p>
        </div>
        
        <!-- Signal Badge -->
        <div style="text-align: center; margin-bottom: 30px;">
            <span style="display: inline-block; background-color: {badge_color}; color: white; padding: 12px 24px; border-radius: 6px; font-size: 18px; font-weight: bold; text-transform: uppercase;">
                {badge_text}
            </span>
        </div>
        
        <!-- Symbol & Price -->
        <div style="background-color: #f9f9f9; border-radius: 6px; padding: 20px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 32px; font-weight: bold; color: #1a1a1a; margin-bottom: 5px;">{symbol}</div>
                    <div style="font-size: 14px; color: #666;">Confluence Score: <strong>{confluence_score:.2f}</strong></div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; color: #666; margin-bottom: 5px;">Price</div>
                    <div style="font-size: 24px; font-weight: bold; color: #1a1a1a;">
                        ${signal_data.get('price_at_signal', 0):.2f}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Entry/Exit Points -->
        <div style="margin-bottom: 25px;">
            <h2 style="font-size: 18px; margin-bottom: 15px; color: #1a1a1a; border-bottom: 1px solid #e5e5e5; padding-bottom: 10px;">Entry & Exit Points</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 10px; color: #666;">Entry Price:</td>
                    <td style="padding: 10px; text-align: right; font-weight: bold; color: #1a1a1a;">
                        ${signal_data.get('price_at_signal', 0):.2f}
                    </td>
                </tr>
                <tr style="background-color: #fef2f2;">
                    <td style="padding: 10px; color: #666;">Stop Loss:</td>
                    <td style="padding: 10px; text-align: right; font-weight: bold; color: #ef4444;">
                        {signal_data.get('stop_loss_pct', 0) * 100:.1f}%
                    </td>
                </tr>
                <tr style="background-color: #f0fdf4;">
                    <td style="padding: 10px; color: #666;">Take Profit:</td>
                    <td style="padding: 10px; text-align: right; font-weight: bold; color: #10b981;">
                        {signal_data.get('take_profit_pct', 0) * 100:.1f}%
                    </td>
                </tr>
                <tr style="background-color: #f9fafb; border-top: 2px solid #e5e5e5;">
                    <td style="padding: 10px; color: #666; font-weight: bold;">Risk-Reward Ratio:</td>
                    <td style="padding: 10px; text-align: right; font-weight: bold; font-size: 18px; {rr_highlight}">
                        {rr_display}
                    </td>
                </tr>
            </table>
        </div>
        
        <!-- Scores Breakdown -->
        <div style="margin-bottom: 25px;">
            <h2 style="font-size: 18px; margin-bottom: 15px; color: #1a1a1a; border-bottom: 1px solid #e5e5e5; padding-bottom: 10px;">Score Breakdown</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px; color: #666;">Technical Score:</td>
                    <td style="padding: 8px; text-align: right; font-weight: bold;">{signal_data.get('technical_score', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #666;">Sentiment Score:</td>
                    <td style="padding: 8px; text-align: right; font-weight: bold;">{signal_data.get('sentiment_score', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #666;">ML Score:</td>
                    <td style="padding: 8px; text-align: right; font-weight: bold;">{signal_data.get('ml_score', 0):.2f}</td>
                </tr>
            </table>
        </div>
        
        <!-- Rationale -->
        <div style="margin-bottom: 25px;">
            <h2 style="font-size: 18px; margin-bottom: 15px; color: #1a1a1a; border-bottom: 1px solid #e5e5e5; padding-bottom: 10px;">Analysis Rationale</h2>
            <p style="color: #333; line-height: 1.8; margin: 0;">
                {signal_data.get('rationale', signal_data.get('overall_rationale', 'No rationale available.'))}
            </p>
        </div>
        
        <!-- CTA Button -->
        <div style="text-align: center; margin-top: 30px; padding-top: 25px; border-top: 1px solid #e5e5e5;">
            <a href="{signal_data.get('dashboard_url', 'https://yourdomain.com/dashboard')}" 
               style="display: inline-block; background-color: #3b82f6; color: white; padding: 14px 28px; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 16px;">
                View Full Analysis in Dashboard
            </a>
        </div>
        
        <!-- Footer -->
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e5e5; text-align: center; color: #999; font-size: 12px;">
            <p style="margin: 5px 0;">Trading Signals Pro - AI-Powered Trading Signals</p>
            <p style="margin: 5px 0;">
                <a href="{signal_data.get('unsubscribe_url', '#')}" style="color: #3b82f6; text-decoration: none;">Unsubscribe</a> | 
                <a href="{signal_data.get('preferences_url', '#')}" style="color: #3b82f6; text-decoration: none;">Email Preferences</a>
            </p>
            <p style="margin: 10px 0 0; font-size: 11px; color: #bbb;">
                This is not financial advice. Trading involves risk. Past performance does not guarantee future results.
            </p>
        </div>
        
    </div>
</body>
</html>
"""
        return html.strip()
    
    @staticmethod
    def signal_alert_text(signal_data: Dict) -> str:
        """
        Generate plain text email template for signal alert.
        
        Args:
            signal_data: Dictionary with signal information
        
        Returns:
            Plain text email content
        """
        signal_type = signal_data.get('signal_type', 'HOLD')
        symbol = signal_data.get('symbol', 'N/A')
        confluence_score = signal_data.get('confluence_score', 0.5)
        rr_ratio = signal_data.get('risk_reward_ratio', 0.0)
        
        text = f"""
NEW TRADING SIGNAL
==================

Signal: {signal_type}
Symbol: {symbol}
Confluence Score: {confluence_score:.2f}
Date: {datetime.utcnow().strftime('%B %d, %Y at %H:%M UTC')}

ENTRY & EXIT POINTS
-------------------
Entry Price: ${signal_data.get('price_at_signal', 0):.2f}
Stop Loss: {signal_data.get('stop_loss_pct', 0) * 100:.1f}%
Take Profit: {signal_data.get('take_profit_pct', 0) * 100:.1f}%
Risk-Reward Ratio: {rr_ratio:.2f}:1

SCORE BREAKDOWN
---------------
Technical Score: {signal_data.get('technical_score', 0):.2f}
Sentiment Score: {signal_data.get('sentiment_score', 0):.2f}
ML Score: {signal_data.get('ml_score', 0):.2f}

ANALYSIS RATIONALE
------------------
{signal_data.get('rationale', signal_data.get('overall_rationale', 'No rationale available.'))}

View full analysis: {signal_data.get('dashboard_url', 'https://yourdomain.com/dashboard')}

---
Trading Signals Pro
This is not financial advice. Trading involves risk.
Unsubscribe: {signal_data.get('unsubscribe_url', '#')}
"""
        return text.strip()
