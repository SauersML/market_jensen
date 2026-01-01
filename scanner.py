
import time
import logging
import sqlite3
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dateutil import parser
import polars as pl
from kalshi_client import KalshiClient
from config import Config

logger = logging.getLogger(__name__)

class SettlementResolver:
    """
    Resolves the final outcome of markets (0 or 1).
    """
    def __init__(self, client: KalshiClient):
        self.client = client
        self.db_path = Config.DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settled_outcomes (
                    market_ticker TEXT PRIMARY KEY,
                    outcome INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def resolve_market(self, ticker: str, m_info: Optional[Dict] = None) -> Optional[int]:
        # Check DB first
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT outcome FROM settled_outcomes WHERE market_ticker = ?", (ticker,))
            row = cursor.fetchone()
            if row:
                return row[0]

        # If info not provided, we might need it.
        # Logic to resolve using m_info or API would go here.
        # For MVP, assume we have logic or fail soft.
        return None

    def set_outcome(self, ticker: str, outcome: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO settled_outcomes (market_ticker, outcome) VALUES (?, ?)",
                (ticker, outcome)
            )
            
    def _determine_outcome(self, last_candle) -> int:
        """
        Heuristic: If last price near 100 -> 1, near 0 -> 0.
        """
        close_price = last_candle['c']
        return 1 if close_price > 90 else 0

class Scanner:
    """
    Phase 1: Series Discovery & Data Ingestion
    Dynamically identifies liquid, recurring markets and fetches data.
    """
    def __init__(self, client: KalshiClient):
        self.client = client
        self.resolver = SettlementResolver(client)

    def find_liquid_series(self) -> str:
        """
        Scans for series with highest volume/interest.
        """
        logger.info("Scanning for liquid series...")
        # Placeholder for complex scanning.
        # For now return a hardcoded/config liquid ticker or find one.
        # e.g. "KX-HIGH"
        # We can search markets and group by series.
        try:
             # Example: Search for daily markets?
             markets_resp = self.client.get_markets(limit=10)
             if markets_resp.get("markets"):
                 return markets_resp["markets"][0].get("series_ticker", "DEMO")
        except Exception:
            pass
            
        return "DEMO_SERIES"

    def fetch_history(self, series_ticker: str) -> pl.DataFrame:
        """
        Fetches historical data for the series.
        Includes Time-Alignment (time_remaining_hours).
        """
        logger.info(f"Fetching history for {series_ticker}...")
        
        # 1. Get all markets in series (past and present)
        # We might need pagination here?
        all_markets = []
        for m in self.client.get_paginated("/markets", params={"series_ticker": series_ticker}, data_key="markets"):
            all_markets.append(m)
            
        if not all_markets:
            logger.warning("No markets found.")
            return pl.DataFrame()
            
        # 2. Fetch candles for each
        data = []
        now = datetime.now(timezone.utc)
        
        for m in all_markets:
            ticker = m['ticker']
            expiry_str = m.get('expiration_time') 
            if not expiry_str: continue
            
            # Robust Parsing
            try:
                expiry = parser.isoparse(expiry_str)
            except Exception:
                continue

            # Skip if expiry too far in future? No, we want history.
            
            # Get candles
            # 1 hour candles
            start_ts = int((expiry - timedelta(days=7)).timestamp()) # 1 Week history per market
            end_ts = int(expiry.timestamp())
            
            resp = self.client.batch_get_market_candlesticks([ticker], start_ts, end_ts, 60)
            candles = resp.get("candlesticks", {}).get(ticker, [])
            
            outcome = self.resolver.resolve_market(ticker, m)
            if outcome is None and candles and expiry < now:
                # Try to determine outcome from last candle
                last = candles[-1]
                outcome = self.resolver._determine_outcome(last)
                if outcome is not None:
                    self.resolver.set_outcome(ticker, outcome)

            for c in candles:
                ts = datetime.fromtimestamp(c['end_period_ts'], tz=timezone.utc)
                hours_left = (expiry - ts).total_seconds() / 3600.0
                
                # Normalize price
                price = c['c'] / 100.0
                
                row = {
                    "timestamp": ts,
                    "market_ticker": ticker,
                    "price_normalized": price,
                    "time_remaining_hours": hours_left,
                    "outcome": outcome,
                    "volume": c['v']
                }
                data.append(row)
                
        return pl.DataFrame(data)
