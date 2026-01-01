
import logging
import asyncio
import json
import time
import numpy as np
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
        
        # Split Loop State with Trajectory Validity
        # {ticker: {'trace': InferenceData, 'timestamp': float, 'valid_until': float}}
        self.posterior_cache = {}
        self.posterior_lock = asyncio.Lock() # Protect cache
        self.TRAJECTORY_VALIDITY_MINUTES = 10  # Cache lifetime (volatility regime assumption)
        
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
        """
        Fast Loop: Evaluate arbitrage opportunity using cached posterior.
        """
        try:
            # ═══════════════════════════════════════════════════════════
            # 1. GET ORDERBOOK (Sub-Penny Precision)
            # ═══════════════════════════════════════════════════════════
            ob_resp = await self.client.get_market_orderbook(ticker)
            ob = ob_resp.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            yes_asks = ob.get("no", [])  # No asks = complement of yes bids
            
            if not yes_bids or not yes_asks:
                return  # Illiquid market
            
            # Parse prices with sub-penny support
            if isinstance(yes_bids[0], dict):
                # V2 format with sub-penny
                bid_price = self.client.parse_price(yes_bids[0], "yes")
            else:
                # Legacy format [price_cents, quantity]
                bid_price = yes_bids[0][0] / 100.0
            
            # Ask = 1 - no_bid (complement)
            if isinstance(yes_asks[0], dict):
                no_bid_price = self.client.parse_price(yes_asks[0], "no")
            else:
                no_bid_price = yes_asks[0][0] / 100.0
            
            ask_price = 1.0 - no_bid_price
            
            # Frozen orderbook detection (wide spread = illiquid)
            spread = ask_price - bid_price
            if spread > 0.20:  # 20 cents
                logger.debug(f"{ticker}: Spread too wide ({spread:.2f}), skipping")
                return
            
            mid_price = (bid_price + ask_price) / 2.0
            
            # ═══════════════════════════════════════════════════════════
            # 2. CHECK CACHED POSTERIOR VALIDITY
            # ═══════════════════════════════════════════════════════════
            posterior_data = self.posterior_cache.get(ticker)
            if not posterior_data:
                # No inference yet, wait for analyst
                return
            
            # Staleness check
            current_time = time.time()
            if current_time > posterior_data.get('valid_until', 0):
                logger.warning(f"{ticker}: Cached posterior expired, awaiting fresh inference...")
                return
            
            posterior = posterior_data['trace']
            
            # ═══════════════════════════════════════════════════════════
            # 3. TIME TO EXPIRY
            # ═══════════════════════════════════════════════════════════
            meta = self.active_markets_meta.get(ticker)
            if not meta:
                return
            
            close_time = parser.isoparse(meta['close_time'])
            now = datetime.now(timezone.utc)
            hours_left = (close_time - now).total_seconds() / 3600.0
            
            if hours_left <= 0:
                return  # Market closed
            
            # ═══════════════════════════════════════════════════════════
            # 4. CALCULATE JENSEN'S GAP
            # ═══════════════════════════════════════════════════════════
            prediction = self.engine.predict_fast(mid_price, hours_left, posterior)
            fair_value = prediction["fair_value"]
            
            # ═══════════════════════════════════════════════════════════
            # 5. DECISION LOGIC - LIQUIDITY-AWARE EXECUTION
            # ═══════════════════════════════════════════════════════════
            # Edge must beat: Spread + Fees + Confidence Buffer
            #
            # The critic's insight: Jensen's Gap is often small (cents).
            # We must ensure the edge exceeds ALL transaction costs.
            
            buy_gap = fair_value - ask_price
            sell_gap = bid_price - fair_value
            
            # Calculate effective costs for BUY side
            buy_fees = ask_price * Config.TAKER_FEE_RATE
            buy_spread_cost = spread * Config.CONFIDENCE_MULTIPLIER
            buy_effective_threshold = (
                Config.JENSEN_GAP_THRESHOLD_CENTS / 100.0 +
                buy_fees +
                buy_spread_cost
            )
            
            # Calculate effective costs for SELL side
            sell_fees = bid_price * Config.TAKER_FEE_RATE
            sell_spread_cost = spread * Config.CONFIDENCE_MULTIPLIER
            sell_effective_threshold = (
                Config.JENSEN_GAP_THRESHOLD_CENTS / 100.0 +
                sell_fees +
                sell_spread_cost
            )
            
            # BUY signal: Fair value exceeds ask PLUS all costs
            if buy_gap > buy_effective_threshold:
                net_edge = buy_gap - buy_effective_threshold
                logger.info(
                    f"BUY {ticker}: "
                    f"Gross Gap={buy_gap*100:.2f}¢ "
                    f"Costs={buy_effective_threshold*100:.2f}¢ (fees={buy_fees*100:.2f}¢, spread={buy_spread_cost*100:.2f}¢) "
                    f"NET EDGE={net_edge*100:.2f}¢ | "
                    f"Fair={fair_value:.4f} Ask={ask_price:.4f} TTE={hours_left:.1f}h"
                )
                qty = self.order_manager.calculate_kelly_size(
                    ticker, net_edge, 1.0, self.order_manager.balance
                )
                if qty > 0:
                    ask_cents = int(round(ask_price * 100))
                    await self.client.create_order(ticker, "buy", "yes", qty, ask_cents)
            
            # SELL signal: Bid exceeds fair value PLUS all costs
            elif sell_gap > sell_effective_threshold:
                net_edge = sell_gap - sell_effective_threshold
                logger.info(
                    f"SELL {ticker}: "
                    f"Gross Gap={sell_gap*100:.2f}¢ "
                    f"Costs={sell_effective_threshold*100:.2f}¢ (fees={sell_fees*100:.2f}¢, spread={sell_spread_cost*100:.2f}¢) "
                    f"NET EDGE={net_edge*100:.2f}¢ | "
                    f"Fair={fair_value:.4f} Bid={bid_price:.4f} TTE={hours_left:.1f}h"
                )
                qty = self.order_manager.calculate_kelly_size(
                    ticker, net_edge, 1.0, self.order_manager.balance
                )
                if qty > 0:
                    no_price_cents = int(round((1.0 - bid_price) * 100))
                    await self.client.create_order(ticker, "buy", "no", qty, no_price_cents)

        except Exception as e:
            logger.error(f"Eval Error {ticker}: {e}")

    async def _analyst_loop(self):
        """
        The Analyst (Slow Loop).
        Periodically infers latent parameters for all active markets.
        Creates "Trajectories" valid for ~10 minutes.
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
                        # Get market metadata for TTE calculation
                        market_meta = self.active_markets_meta.get(ticker)
                        if not market_meta:
                            # Fetch if missing
                            m_resp = await self.client.get_markets(ticker=ticker)
                            if m_resp and 'markets' in m_resp and len(m_resp['markets']) > 0:
                                market_meta = m_resp['markets'][0]
                                self.active_markets_meta[ticker] = market_meta
                        
                        if not market_meta:
                            continue
                        
                        close_time_str = market_meta.get('close_time') or market_meta.get('expiration_time')
                        close_time = parser.isoparse(close_time_str)
                        if close_time.tzinfo is None:
                            close_time = close_time.replace(tzinfo=timezone.utc)
                        
                        tte_hours = (close_time - now).total_seconds() / 3600.0
                        if tte_hours <= 0:
                            continue  # Market closed

                        # 2. Fetch candlestick data
                        c_resp = await self.client.get_candlesticks(ticker, start_ts, end_ts, 60)
                        candles = c_resp.get("candlesticks", [])
                        
                        if not candles or len(candles) < Config.MIN_OBSERVATIONS_FOR_INFERENCE:
                            logger.debug(f"{ticker}: Insufficient candles ({len(candles)}), skipping")
                            continue
                        
                        recent_prices = [c['c'] / 100.0 for c in candles]
                        
                        # 3. Infer (Heavy CPU - run in executor to avoid blocking)
                        posterior = await asyncio.get_running_loop().run_in_executor(
                            None, 
                            lambda: asyncio.run(self.engine.infer_posterior(recent_prices, tte_hours))
                        )
                        
                        # 4. Update Cache with Validity Timestamp
                        current_time = time.time()
                        valid_until = current_time + (self.TRAJECTORY_VALIDITY_MINUTES * 60)
                        
                        async with self.posterior_lock:
                            self.posterior_cache[ticker] = {
                                'trace': posterior,
                                'timestamp': current_time,
                                'valid_until': valid_until
                            }
                            
                        logger.info(f"Analyst: Updated Posterior for {ticker} (valid for {self.TRAJECTORY_VALIDITY_MINUTES}min)")
                        
                    except Exception as e:
                        logger.error(f"Analyst Error {ticker}: {e}")
                        
            except Exception as e:
                logger.error(f"Analyst Loop Crash: {e}")
                
            # Run every 1 minute (was 5 minutes - too slow for volatile markets)
            await asyncio.sleep(60)

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
