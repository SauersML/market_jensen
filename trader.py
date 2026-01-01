
import logging
import asyncio
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from dateutil import parser
from kalshi_client import AsyncKalshiClient, KalshiWebSocket
from analysis_engine import AnalysisEngine
from config import Config
import sortedcontainers

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages active orders, inventory, and sizing (Kelly Criterion).
    """
    def __init__(self, client: AsyncKalshiClient):
        self.client = client
        self.open_orders = {} 
        self.positions = {}   
        self.balance = 100000 
        
    async def refresh_account(self):
        try:
            bal_resp = await self.client.get_portfolio_balance()
            self.balance = bal_resp.get("balance", 0) 
            # pos_resp = await self.client.get_positions()
            # Update self.positions...
        except Exception as e:
            logger.error(f"Failed to refresh account: {e}")

    def calculate_kelly_size(self, ticker: str, edge: float, odds: float, bankroll: float) -> int:
        """
        Config-based sizing for MVP, cap at Max %
        """
        # Fraction of bankroll
        max_amt = bankroll * Config.MAX_POSITION_SIZE_PERCENT
        # Determine size...
        return Config.ORDER_SIZE_CONTRACTS

class Trader:
    """
    Phase 3: Async Trader
    Events -> Analysis -> Decision -> Execution
    """
    def __init__(self, key_id: str, key_file: str):
        self.client = AsyncKalshiClient(key_id, key_file)
        self.engine = AnalysisEngine()
        self.scanner = None # Init async
        self.order_manager = OrderManager(self.client)
        self.active_tickers = [] 
        self.running = False
        
        # Split Loop State
        self.posterior_cache = {} # {ticker: {'mus': [], 'sigmas': [], 'last_updated': ts}}
        self.posterior_lock = asyncio.Lock() # Protect cache
        self.active_markets_meta = {} # {ticker: market_info_dict} for expiry etc.
        self.orderbooks = {} # {ticker: {'bids': SortedList, 'asks': SortedList}}
        
        self.ws = None

    async def initialize_strategy(self):
        logger.info("Initializing Strategy...")
        # Sync Setup
        from kalshi_client import AsyncKalshiClient
        from scanner import Scanner
        
        # Use our async client logic
        # Scanner needs a client.
        # We can reuse self.client? Yes.
        scanner = Scanner(self.client)

    async def update_market_structure(self):
        """
        Fetch active markets, update metadata cache.
        """
        # Example: Get markets for KXINX
        resp = await self.client.get_markets(series_ticker="KXINX", status="open")
        markets = resp.get("markets", [])
        
        self.active_tickers = []
        for m in markets:
            t = m['ticker']
            self.active_tickers.append(t)
            self.active_markets_meta[t] = m
            
        # Update WS Subscriptions
        if self.ws:
            await self.ws.subscribe(self.active_tickers)

    async def on_market_update(self, msg: Dict):
        """
        WebSocket Callback.
        """
        t = msg.get("type")
        if t == "orderbook_delta":
            ticker = msg.get("market_ticker")
            # Update local book (skipped detail for brevity)
            # Trigger Eval
            await self.evaluate_ticker(ticker)

    async def evaluate_ticker(self, ticker: str):
        try:
            # 1. Get Price (from local book or fetch)
            # For robustness, we fetch snapshot if book empty
            ob_resp = await self.client.get_market_orderbook(ticker)
            ob = ob_resp.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            if not yes_bids: return
            
            best_bid = yes_bids[0][0]
            # Synth ask
            best_ask = 100 - ob.get("no", [[0,0]])[0][0]
            mid_price = (best_bid + best_ask) / 2.0 / 100.0
            
            # 2. Time to expiry (Principaled)
            meta = self.active_markets_meta.get(ticker)
            if not meta: return 
            
            close_time = parser.isoparse(meta['close_time'])
            now = datetime.now(timezone.utc)
            hours_left = (close_time - now).total_seconds() / 3600.0
            
            if hours_left <= 0: return

            # 3. Fast Prediction (Using Cached Posteriors)
            # The Analyst Loop updates the cache in background.
            # Here we just read.
            posterior = self.posterior_cache.get(ticker)
            if not posterior:
                # No inference yet, skip or use priors?
                # User plan implies we need inference first.
                # Let's wait for Analyst.
                return

            # Predict with Full Bayes (returns dict with gap metrics)
            prediction = self.engine.predict_fast(mid_price, hours_left, posterior)
            fair_value = prediction["fair_value"]
            gap_cents = prediction["gap_cents"]
            
            # 4. Decision Based on Jensen's Gap
            # Gap > 0: Market underpriced (Fair Value > Naive Price)
            # Gap < 0: Market overpriced (Fair Value < Naive Price)
            
            edge_threshold = Config.JENSEN_GAP_THRESHOLD_CENTS
            
            # BUY signal: Gap is positive and significant
            # Fair value exceeds market ask
            if gap_cents > edge_threshold and fair_value > (best_ask/100.0):
                 # Aggressive take
                logger.info(f"BUY {ticker}: Gap={gap_cents:.2f}¢ Fair={fair_value:.3f} Ask={best_ask}¢")
                qty = self.order_manager.calculate_kelly_size(ticker, fair_value - (best_ask/100.0), 1.0, self.order_manager.balance)
                if qty > 0:
                    await self.client.create_order(ticker, "buy", "yes", qty, best_ask)
            
            # SELL signal: Gap is negative and significant
            # Fair value below market bid
            elif gap_cents < -edge_threshold and fair_value < (best_bid/100.0):
                logger.info(f"SELL {ticker}: Gap={gap_cents:.2f}¢ Fair={fair_value:.3f} Bid={best_bid}¢")
                no_price = 100 - best_bid
                qty = self.order_manager.calculate_kelly_size(ticker, (best_bid/100.0) - fair_value, 1.0, self.order_manager.balance)
                if qty > 0:
                    resp = await self.client.create_order(ticker, "buy", "no", qty, no_price)

        except Exception as e:
            logger.error(f"Eval Error {ticker}: {e}")

    async def _analyst_loop(self):
        """
        The Analyst (Slow Loop).
        Periodically infers latent parameters for all active markets.
        """
        logger.info("Analyst Loop Started")
        while self.running:
            try:
                for ticker in self.active_tickers:
                    # 1. Fetch History (Last 24h)
                    now = datetime.now(timezone.utc)
                    end_ts = int(now.timestamp())
                    start_ts = int((now - timedelta(hours=Config.INFERENCE_WINDOW_HOURS)).timestamp())
                    
                    try:
                        # Fetch meta to know close_time for TTE Calc
                        # Optimization: Use self.active_markets_meta if populated, or fetch candles meta
                        # 3b. Run Inference (CPU Bound)
                        
                        # We need TTE.
                        # We can get it from the last candle? No, candle is history.
                        # We need 'close_time' of the market.
                        # `self.active_markets_meta` should have it?
                        # If not, we fetched candles.
                        # Let's assume we have it or can get it.
                        # For robustness, let's fetch market details if missing.
                        
                        # Assumption: active_markets_meta is populated in update_market_structure
                        # But loop runs async.
                        market_meta = self.active_markets_meta.get(ticker)
                        if not market_meta:
                            # Fetch
                            m_resp = await self.client.get_markets(ticker=ticker) # Singular?
                            if m_resp and 'markets' in m_resp and len(m_resp['markets']) > 0:
                                market_meta = m_resp['markets'][0]
                                self.active_markets_meta[ticker] = market_meta
                        
                        if not market_meta: continue
                        
                        close_time_str = market_meta.get('close_time') or market_meta.get('expiration_time')
                        close_time = parser.isoparse(close_time_str)
                        if close_time.tzinfo is None: close_time = close_time.replace(tzinfo=timezone.utc)
                        
                        tte_hours = (close_time - now).total_seconds() / 3600.0
                        if tte_hours <= 0: continue

                        c_resp = await self.client.get_candlesticks(ticker, start_ts, end_ts, 60)
                        candles = c_resp.get("candlesticks", [])
                        recent_prices = [c['c'] / 100.0 for c in candles]
                        
                        # 2. Infer (Heavy CPU)
                        posterior = await asyncio.get_running_loop().run_in_executor(
                            None, 
                            self.engine.infer_posterior, 
                            recent_prices,
                            tte_hours
                        )
                        
                        # 3. Update Cache
                        async with self.posterior_lock:
                            self.posterior_cache[ticker] = posterior
                            
                        logger.info(f"Analyst: Updated Posterior for {ticker}")
                        
                    except Exception as e:
                        logger.error(f"Analyst Error {ticker}: {e}")
                        
            except Exception as e:
                logger.error(f"Analyst Loop Crash: {e}")
                
            await asyncio.sleep(60 * 5) # Run every 5 mins

    async def run(self):
        async with self.client:
            await self.order_manager.refresh_account()
            
            # Init Markets
            await self.update_market_structure()
            
            # Init Engine (Train)
            # Need to fetch history using Async Client now
            # ... (logic to fetch history via self.client and train self.engine)
            
            # WS
            self.ws = KalshiWebSocket(self.client, self.on_market_update)
            asyncio.create_task(self.ws.connect())
            
            logger.info("Trader Running...")
            self.running = True
            # Start Analyst Loop
            asyncio.create_task(self._analyst_loop())
            
            # Keep Process Alive
            while self.running:
                await asyncio.sleep(60)
                await self.update_market_structure()
                await self.order_manager.refresh_account()

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(Trader().run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
