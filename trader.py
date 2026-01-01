import logging
import asyncio
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import websockets
from kalshi_client import KalshiClient
from scanner import Scanner
from analysis_engine import AnalysisEngine
from config import Config

logger = logging.getLogger(__name__)

class Trader:
    """
    Phase 4: Live Execution Strategy (The Trader)
    Uses polling for market discovery and REST API for orderbook snapshots.
    (WebSocket implementation for orderbook delta is possible but complex to maintain
    without a full orderbook reconstruction engine. Using frequent polling or
    basic websocket ticker subscription is more robust for this MVP).
    """
    def __init__(self):
        self.client = KalshiClient()
        self.scanner = Scanner(self.client)
        self.engine = AnalysisEngine()
        self.active_markets = {} # {ticker: details}
        self.running = False

    def initialize(self):
        logger.info("Initializing Trader...")

        # 1. Find Series
        try:
            self.series_ticker = self.scanner.find_liquid_series()
        except Exception as e:
            logger.critical(f"Initialization failed: {e}")
            raise

        # 2. Build Model
        try:
            history_df = self.scanner.fetch_history(self.series_ticker)
            self.engine.calculate_empirical_volatility(history_df)
            # Calibration would go here if we had outcomes
        except Exception as e:
            logger.critical(f"Model building failed: {e}")
            raise

    def update_active_markets(self):
        """Fetches currently open markets for the target series."""
        logger.info(f"Updating active markets for {self.series_ticker}...")
        markets_resp = self.client.get_markets(series_ticker=self.series_ticker, status="open")
        markets = markets_resp.get("markets", [])

        current_tickers = set()
        for m in markets:
            ticker = m['ticker']
            current_tickers.add(ticker)
            if ticker not in self.active_markets:
                logger.info(f"Tracking new market: {ticker}")
                self.active_markets[ticker] = m

        # Remove closed markets
        remove = [t for t in self.active_markets if t not in current_tickers]
        for t in remove:
            del self.active_markets[t]

    def get_time_to_expiry_hours(self, market_info: Dict) -> Optional[float]:
        """Calculates hours remaining until market expiration."""
        # API returns 'close_time' or 'expiration_time' usually in ISO format or similar.
        # Example: "2024-11-05T00:00:00Z"
        # Or sometimes it's a timestamp.
        # Let's inspect what we get typically. Kalshi V2 often uses ISO 8601 strings.

        try:
            # Prefer close_time (trading stops) over expiration_time (settlement)
            ts_str = market_info.get("close_time") or market_info.get("expiration_time")
            if not ts_str:
                return None

            # Parse ISO format
            # Handle 'Z' for UTC
            if ts_str.endswith('Z'):
                ts_str = ts_str[:-1]

            # Try parsing with microseconds or without
            try:
                expiry_dt = datetime.fromisoformat(ts_str)
            except ValueError:
                 # Fallback for different formats if needed
                 # Maybe it's just a date?
                 return None

            now = datetime.utcnow()
            diff = expiry_dt - now
            hours = diff.total_seconds() / 3600.0
            return max(0.0, hours)

        except Exception as e:
            logger.warning(f"Failed to parse expiry time: {e}")
            return None

    def evaluate_market(self, ticker: str):
        """
        Main logic loop for a single market.
        """
        try:
            # Get Orderbook
            ob_resp = self.client.get_market_orderbook(ticker)
            ob = ob_resp.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            no_bids = ob.get("no", [])

            if not yes_bids or not no_bids:
                return

            best_yes_bid = yes_bids[0][0]
            best_no_bid = no_bids[0][0] # "Sell Yes" ~ "Buy No" logic
            best_yes_ask = 100 - best_no_bid

            # Mid price
            mid_price_cents = (best_yes_bid + best_yes_ask) / 2
            current_price = mid_price_cents / 100.0

            # Time to expiry
            market_info = self.active_markets[ticker]
            time_to_expiry = self.get_time_to_expiry_hours(market_info)

            if time_to_expiry is None:
                logger.warning(f"Skipping {ticker}: Unknown expiry time")
                return

            if time_to_expiry < 1.0:
                # Don't trade if less than 1 hour left (volatility model might be unstable for very short term)
                return

            # Run Simulation
            # Fair Value
            fair_value = self.engine.run_mcmc_simulation(current_price, time_to_expiry_hours=time_to_expiry)
            fair_value_cents = fair_value * 100

            logger.info(f"{ticker}: Price={current_price:.3f}, Fair={fair_value:.3f}, HoursLeft={time_to_expiry:.1f}")

            # Decision
            threshold = Config.JENSEN_GAP_THRESHOLD_CENTS

            if fair_value_cents > best_yes_ask + threshold:
                logger.info(f"SIGNAL BUY: Fair {fair_value_cents:.1f} > Ask {best_yes_ask}")
                self.execute_trade(ticker, "yes", "buy", best_yes_ask)

            elif fair_value_cents < best_yes_bid - threshold:
                logger.info(f"SIGNAL SELL: Fair {fair_value_cents:.1f} < Bid {best_yes_bid}")
                # Sell Yes = Buy No
                self.execute_trade(ticker, "no", "buy", best_no_bid)

        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")

    def execute_trade(self, ticker: str, side: str, action: str, price: int):
        """
        Executes order.
        """
        # Check current position/orders first to avoid over-trading (not implemented fully here)

        qty = Config.ORDER_SIZE_CONTRACTS
        logger.info(f"Placing Order: {action} {qty} {side} @ {price} on {ticker}")

        try:
            if side == "yes":
                resp = self.client.create_order(ticker, side, action, qty, yes_price=price)
            else:
                resp = self.client.create_order(ticker, side, action, qty, no_price=price)

            order_id = resp.get("order_id")
            logger.info(f"Order Placed: {order_id}")
        except Exception as e:
            logger.error(f"Order execution failed: {e}")

    def run_loop(self):
        """
        Main Event Loop.
        """
        self.initialize()
        self.running = True

        while self.running:
            try:
                self.update_active_markets()

                for ticker in list(self.active_markets.keys()):
                    self.evaluate_market(ticker)

                time.sleep(10) # Poll interval

            except KeyboardInterrupt:
                logger.info("Stopping...")
                self.running = False
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    # Test
    pass
