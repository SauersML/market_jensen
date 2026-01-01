import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import polars as pl
from kalshi_client import KalshiClient
from config import Config

logger = logging.getLogger(__name__)

class Scanner:
    """
    Phase 1: Series Discovery & Data Ingestion
    Dynamically identifies liquid, recurring markets and fetches data.
    """
    def __init__(self, client: KalshiClient):
        self.client = client

    def find_liquid_series(self) -> str:
        """
        Scans for a viable series based on frequency, category, and liquidity.
        Returns the ticker of the best series found.
        Raises Exception if no series found.
        """
        logger.info("Scanning for liquid series...")

        # 1. Fetch all series
        series_resp = self.client.get_series_list()
        series_list = series_resp.get("series", [])

        candidates = []

        for s in series_list:
            ticker = s.get("ticker")
            frequency = s.get("frequency", "").lower()
            category = s.get("category", "").lower()

            # Filter by frequency
            if frequency not in ["daily", "weekly", "monthly"]:
                continue

            # Filter by category (preference)
            # if category not in ["economics", "weather", "politics"]:
            #     continue

            candidates.append(ticker)

        logger.info(f"Found {len(candidates)} candidate series based on metadata.")

        best_series = None
        best_volume = 0

        # 2. Check Liquidity (Volume)
        # We need to check recent markets for these series.
        # This can be slow if we check all. We might want to prioritize.

        for series_ticker in candidates:
            try:
                # Fetch recent settled markets
                markets_resp = self.client.get_markets(series_ticker=series_ticker, status="settled", limit=30)
                markets = markets_resp.get("markets", [])

                if len(markets) < Config.MIN_HISTORY_EVENTS:
                    continue

                total_volume = sum(m.get("volume", 0) for m in markets)
                avg_volume = total_volume / len(markets)

                # Assume volume is in contracts? Or USD? Kalshi volume is contracts usually.
                # Let's assume contracts. If 1000 contracts * $0.50 ~ $500.
                # User config says MIN_DAILY_VOLUME_USD.
                # Rough approx: avg_price ~ 50c.
                avg_volume_usd = avg_volume * 0.50 * 100 # 100 cents? No, volume is number of contracts. Each contract is max $1.
                # Actually, volume is often reported in number of contracts traded.
                # Let's just use raw contract volume.

                if avg_volume > Config.MIN_DAILY_VOLUME_USD: # Treating this config as contracts count for simplicity or assume dollar volume
                     logger.info(f"Series {series_ticker} Avg Volume: {avg_volume}")
                     if avg_volume > best_volume:
                         best_volume = avg_volume
                         best_series = series_ticker

            except Exception as e:
                logger.warning(f"Error checking liquidity for {series_ticker}: {e}")
                continue

        if not best_series:
            raise RuntimeError("No viable liquid series found.")

        logger.info(f"Selected Best Series: {best_series} (Avg Volume: {best_volume})")
        return best_series

    def fetch_history(self, series_ticker: str) -> pl.DataFrame:
        """
        Fetches historical data for the series.
        Returns a Polars DataFrame with columns: timestamp, market_ticker, price, etc.
        """
        logger.info(f"Fetching history for {series_ticker}...")

        # 1. Get past markets
        markets_resp = self.client.get_markets(series_ticker=series_ticker, status="settled", limit=100)
        markets = markets_resp.get("markets", [])

        if len(markets) < Config.MIN_HISTORY_EVENTS:
            raise RuntimeError(f"Insufficient history for {series_ticker}. Found {len(markets)} events.")

        market_tickers = [m['ticker'] for m in markets]

        # 2. Batch Get Candlesticks
        # Chunk into 100s
        chunk_size = 100
        all_candles = []

        end_ts = int(time.time())
        # Look back enough to cover these markets.
        # Since we don't know exact dates easily without parsing events, let's just go back 60 days.
        start_ts = end_ts - (60 * 24 * 60 * 60)

        for i in range(0, len(market_tickers), chunk_size):
            chunk = market_tickers[i:i + chunk_size]
            try:
                logger.info(f"Fetching batch candlesticks for {len(chunk)} markets...")
                resp = self.client.batch_get_market_candlesticks(
                    market_tickers=chunk,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=60 # Hourly data
                )

                batch_markets = resp.get("markets", [])
                for m_data in batch_markets:
                    m_ticker = m_data.get("market_ticker")
                    candles = m_data.get("candlesticks", [])
                    for c in candles:
                        # Extract data
                        # c structure: {price: {open, high, low, close}, volume, created_time, etc}
                        price = c.get("price", {}).get("close")
                        ts = c.get("end_period_ts") # or created_time

                        if price is not None and ts is not None:
                            all_candles.append({
                                "timestamp": ts,
                                "market_ticker": m_ticker,
                                "price": price, # Cents
                                "volume": c.get("volume", 0)
                            })

            except Exception as e:
                logger.error(f"Error fetching batch candlesticks: {e}")
                # Don't fail completely, try next chunk

        if not all_candles:
            raise RuntimeError("No candlestick data retrieved.")

        df = pl.DataFrame(all_candles)
        # Normalize price to 0-1
        df = df.with_columns((pl.col("price") / 100.0).alias("price_normalized"))

        # Save for caching?
        # df.write_parquet(Config.DATA_DIR / f"{series_ticker}_history.parquet")

        return df

if __name__ == "__main__":
    # Test Scanner
    logging.basicConfig(level=logging.INFO)
    try:
        client = KalshiClient()
        scanner = Scanner(client)
        series = scanner.find_liquid_series()
        df = scanner.fetch_history(series)
        print(df.head())
    except Exception as e:
        print(e)
