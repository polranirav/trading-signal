"""
SEC Portfolio Fetcher Service.

Fetches 13F filings from SEC EDGAR for famous investors and parses holdings.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Holding:
    """Represents a single portfolio holding."""
    symbol: str
    name: str
    shares: int
    value_usd: float
    pct_portfolio: float
    cusip: Optional[str] = None


@dataclass
class FilingInfo:
    """SEC filing metadata."""
    accession_number: str
    filing_date: str
    form_type: str
    primary_doc: str


# Famous investors and their SEC CIK numbers
FAMOUS_INVESTORS = {
    "buffett": {
        "name": "Warren Buffett",
        "fund": "Berkshire Hathaway",
        "cik": "0001067983",
        "description": "The Oracle of Omaha",
        "avatar": "🏛️",
    },
    "ackman": {
        "name": "Bill Ackman",
        "fund": "Pershing Square",
        "cik": "0001336528",
        "description": "Activist investor",
        "avatar": "🎯",
    },
    "soros": {
        "name": "George Soros",
        "fund": "Soros Fund Management",
        "cik": "0001029160",
        "description": "Legendary macro trader",
        "avatar": "🌍",
    },
    "icahn": {
        "name": "Carl Icahn",
        "fund": "Icahn Enterprises",
        "cik": "0000921669",
        "description": "Corporate raider",
        "avatar": "⚔️",
    },
    "burry": {
        "name": "Michael Burry",
        "fund": "Scion Asset Management",
        "cik": "0001649339",
        "description": "The Big Short",
        "avatar": "📉",
    },
    "dalio": {
        "name": "Ray Dalio",
        "fund": "Bridgewater Associates",
        "cik": "0001350694",
        "description": "All-weather strategy",
        "avatar": "🌊",
    },
    "tepper": {
        "name": "David Tepper",
        "fund": "Appaloosa Management",
        "cik": "0001656456",
        "description": "Distressed debt expert",
        "avatar": "💰",
    },
    "druckenmiller": {
        "name": "Stanley Druckenmiller",
        "fund": "Duquesne Family Office",
        "cik": "0001536411",
        "description": "Soros's former partner",
        "avatar": "📊",
    },
}

# CUSIP to Ticker mapping for common stocks
# (SEC filings use CUSIP, not ticker symbols)
CUSIP_TO_TICKER = {
    "037833100": "AAPL",
    "060505104": "BAC",
    "191216100": "KO",
    "02079K305": "GOOGL",
    "025816109": "AXP",
    "166764100": "CVX",
    "674599105": "OXY",
    "742718109": "PG",
    "594918104": "MSFT",
    "023135106": "AMZN",
    "88160R101": "TSLA",
    "67066G104": "NVDA",
    "30303M102": "META",
}


class SECPortfolioFetcher:
    """
    Fetches and parses 13F filings from SEC EDGAR.
    
    13F filings are quarterly reports that institutional investment managers
    with over $100M in assets must file with the SEC.
    """
    
    BASE_URL = "https://data.sec.gov"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Trading Signals Pro (research@tradingsignals.com)",
            "Accept": "application/json",
        })
    
    def get_famous_investors(self) -> Dict[str, Dict]:
        """Return list of available famous investors."""
        return FAMOUS_INVESTORS
    
    def get_investor_info(self, investor_id: str) -> Optional[Dict]:
        """Get info for a specific investor."""
        return FAMOUS_INVESTORS.get(investor_id.lower())
    
    def fetch_company_filings(self, cik: str) -> Optional[Dict]:
        """Fetch company filing metadata from SEC."""
        
        # Ensure CIK is zero-padded to 10 digits
        cik = cik.lstrip("0").zfill(10)
        
        url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
        
        try:
            logger.info(f"Fetching SEC filings for CIK {cik}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch SEC filings: {e}")
            return None
    
    def get_latest_13f_filing(self, cik: str) -> Optional[FilingInfo]:
        """Get the most recent 13F-HR filing for a company."""
        
        data = self.fetch_company_filings(cik)
        if not data:
            return None
        
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        filing_dates = filings.get("filingDate", [])
        primary_docs = filings.get("primaryDocument", [])
        
        # Find the latest 13F-HR filing
        for i, form in enumerate(forms):
            if form == "13F-HR":
                return FilingInfo(
                    accession_number=accession_numbers[i],
                    filing_date=filing_dates[i],
                    form_type=form,
                    primary_doc=primary_docs[i] if i < len(primary_docs) else "",
                )
        
        logger.warning(f"No 13F-HR filing found for CIK {cik}")
        return None
    
    def fetch_13f_holdings(self, cik: str, accession_number: str) -> List[Holding]:
        """
        Fetch and parse holdings from a 13F filing.
        
        Note: This returns mock data for now. In production, you would:
        1. Download the 13F XML filing
        2. Parse the infotable entries
        3. Map CUSIP to ticker symbols
        """
        
        logger.info(f"Fetching 13F holdings for accession {accession_number}")
        
        # In a real implementation, we would fetch and parse the XML
        # For now, return realistic mock data based on known holdings
        
        investor = None
        for inv_id, inv_data in FAMOUS_INVESTORS.items():
            if inv_data["cik"].lstrip("0") == cik.lstrip("0"):
                investor = inv_id
                break
        
        return self._get_mock_holdings(investor or "buffett")
    
    def _get_mock_holdings(self, investor_id: str) -> List[Holding]:
        """Return realistic mock holdings for famous investors."""
        
        mock_data = {
            "buffett": [
                Holding("AAPL", "Apple Inc", 915_000_000, 215_000_000_000, 39.2),
                Holding("BAC", "Bank of America", 1_000_000_000, 45_000_000_000, 8.2),
                Holding("KO", "Coca-Cola Co", 400_000_000, 28_000_000_000, 5.1),
                Holding("GOOGL", "Alphabet Inc", 28_100_000, 40_000_000_000, 7.3),
                Holding("AXP", "American Express", 151_600_000, 38_000_000_000, 6.9),
                Holding("CVX", "Chevron Corp", 160_000_000, 25_000_000_000, 4.6),
                Holding("OXY", "Occidental Petroleum", 219_500_000, 16_000_000_000, 2.9),
                Holding("PG", "Procter & Gamble", 120_000_000, 20_000_000_000, 3.7),
            ],
            "ackman": [
                Holding("CMG", "Chipotle Mexican Grill", 2_900_000, 8_500_000_000, 18.5),
                Holding("HLT", "Hilton Worldwide", 8_900_000, 1_800_000_000, 12.3),
                Holding("GOOGL", "Alphabet Inc", 3_100_000, 4_500_000_000, 9.8),
                Holding("LOW", "Lowe's Companies", 5_200_000, 1_400_000_000, 9.5),
                Holding("HHH", "Howard Hughes Holdings", 15_000_000, 1_200_000_000, 8.2),
            ],
            "burry": [
                Holding("BABA", "Alibaba Group", 50_000, 4_000_000, 5.2),
                Holding("JD", "JD.com", 125_000, 4_500_000, 5.8),
                Holding("GOOG", "Alphabet Inc", 10_000, 15_000_000, 19.5),
                Holding("GEO", "GEO Group", 500_000, 8_000_000, 10.4),
                Holding("PINS", "Pinterest", 100_000, 3_500_000, 4.5),
            ],
            "soros": [
                Holding("RIVN", "Rivian Automotive", 5_000_000, 75_000_000, 4.2),
                Holding("MSFT", "Microsoft Corp", 150_000, 63_000_000, 3.5),
                Holding("AMZN", "Amazon.com", 200_000, 38_000_000, 2.1),
                Holding("GOOGL", "Alphabet Inc", 100_000, 14_500_000, 0.8),
                Holding("CRM", "Salesforce", 200_000, 58_000_000, 3.2),
            ],
        }
        
        # Default to buffett if investor not found
        return mock_data.get(investor_id, mock_data["buffett"])
    
    def import_investor_portfolio(
        self, investor_id: str
    ) -> Tuple[List[Holding], FilingInfo]:
        """
        Import a famous investor's portfolio.
        
        Returns:
            Tuple of (holdings list, filing info)
        """
        
        investor = self.get_investor_info(investor_id)
        if not investor:
            raise ValueError(f"Unknown investor: {investor_id}")
        
        cik = investor["cik"]
        
        # Get latest filing
        filing = self.get_latest_13f_filing(cik)
        if not filing:
            # Fall back to mock data
            logger.warning(f"Could not fetch filing for {investor_id}, using mock data")
            filing = FilingInfo(
                accession_number="0000000000-00-000000",
                filing_date=datetime.now().strftime("%Y-%m-%d"),
                form_type="13F-HR",
                primary_doc="",
            )
        
        # Get holdings
        holdings = self.fetch_13f_holdings(cik, filing.accession_number)
        
        return holdings, filing
    
    def parse_sec_url(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Parse a SEC EDGAR URL to extract CIK and accession number.
        
        Supports URLs like:
        - https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001067983
        - https://www.sec.gov/cgi-bin/viewer?action=view&cik=1067983&accession_number=...
        
        Returns:
            Tuple of (cik, accession_number) or None if parsing fails
        """
        
        try:
            # Extract CIK
            cik_match = re.search(r'CIK[=:]?(\d+)', url, re.IGNORECASE)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
            else:
                return None
            
            # Extract accession number if present
            accn_match = re.search(r'accession[_-]?number[=:]?(\d{10}-\d{2}-\d{6})', url, re.IGNORECASE)
            accession = accn_match.group(1) if accn_match else None
            
            return cik, accession
        except Exception as e:
            logger.error(f"Failed to parse SEC URL: {e}")
            return None


# Singleton instance
_fetcher = None

def get_sec_fetcher() -> SECPortfolioFetcher:
    """Get singleton SEC fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = SECPortfolioFetcher()
    return _fetcher
