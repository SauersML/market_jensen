
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

    def calculate_kelly_size(self, fair_value: float, market_price: float, 
                               bankroll: float, posterior_samples: np.ndarray) -> int:
        """
        Exact Kelly Criterion via Numerical Optimization (Phase 2D).
        
        Maximizes E[log(1 + f * payoff)] over full posterior distribution.
        No approximation - uses scipy.optimize for exact f*.
        
        Args:
            fair_value: Posterior mean (not used in exact version)
            market_price: Current ask (for buy) or bid (for sell)
            bankroll: Total available capital
            posterior_samples: Array of posterior terminal probabilities
        """
        from scipy.optimize import minimize_scalar
        
        def expected_log_growth(f):
            """
            Expected log wealth change over full posterior.
            
            For each posterior sample of win probability p:
            - Win: wealth multiplied by (1 + f * payoff_win)
            - Lose: wealth multiplied by (1 + f * payoff_lose)
            """
            if f <= 0:
                return 1e10  # Large penalty for non-positive bets
            
            log_growths = []
            for p_win in posterior_samples:
                # Binary contract payoffs
                payoff_win = (1.0 - market_price) / market_price  # Winning return
                payoff_lose = -1.0  # Lose entire bet
                
                # Expected log growth for this posterior sample
                if 1 + f * payoff_lose <= 0:  # Avoid catastrophic loss
                    return 1e10
                
                log_g = (
                    p_win * np.log(1 + f * payoff_win) +
                    (1 - p_win) * np.log(1 + f * payoff_lose)
                )
                log_growths.append(log_g)
            
            # Return negative (for minimization)
            return -np.mean(log_growths)
        
        # Optimize f in [0, 0.5] (max 50% of bankroll)
        result = minimize_scalar(
            expected_log_growth,
            bounds=(0.0, 0.5),
            method='bounded'
        )
        
        f_optimal = result.x
        
        # Apply fractional Kelly dampener
        from config import Config
        f_dampened = f_optimal * Config.KELLY_FRACTION
        
        # Convert to contract quantity
        position_value = bankroll * f_dampened
        contracts = int(position_value / market_price)
        
        logger.info(f"Kelly optimization: f*={f_optimal:.4f}, dampened={f_dampened:.4f}, contracts={contracts}")
        
        return max(0, min(contracts, int(bankroll / market_price)))

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
        
        
        # Cache for posteriors with learned validity
        self.posterior_cache = {}
        self.posterior_lock = asyncio.Lock()
        # NO HARDCODED VALIDITY: Learn from posterior vol_of_vol
        # For now, use conservative 5 minutes (can be made dynamic later)
        self.TRAJECTORY_VALIDITY_SECONDS = 300
        
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
        # Get markets for active series (series_ticker should be dynamically determined)
        # This is a placeholder - actual series should come from scanner
        resp = await self.client.get_markets(status="open")
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
            
            mid_price = (bid_price + ask_price) / 2.0
            spread = ask_price - bid_price
            
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
            
            prediction = self.engine.predict_fast(mid_price, hours_left, posterior)
            fair_value = prediction["fair_value"]
            gap_cents = prediction["gap_cents"]
            
            # Extract posterior samples for Kelly sizing
            # We need the full distribution, not just the mean
            posterior_samples = self.engine._extract_posterior_fair_values(posterior, mid_price, hours_left)
            
            # ═══════════════════════════════════════════════════════════
            # 5. PURE NET EV DECISION (No Arbitrary Thresholds)
            # ═══════════════════════════════════════════════════════════
            # Calculate Net EV = Expected Payoff - All Costs
            # No magic numbers like "gap > 2 cents" or "spread < 20 cents"
            
            # BUY opportunity
            buy_gap = fair_value - ask_price
            buy_fees = ask_price * Config.TAKER_FEE_RATE
            buy_net_ev = buy_gap - buy_fees  # No spread multiplier, no gap threshold
            
            # SELL opportunity  
            sell_gap = bid_price - fair_value
            sell_fees = bid_price * Config.TAKER_FEE_RATE
            sell_net_ev = sell_gap - sell_fees
            
            # Execute if ANY positive EV exists (the fundamental economic condition)
            if buy_net_ev > 0:
                # Calculate Kelly size using full posterior
                qty = self.order_manager.calculate_kelly_size(
                    fair_value, ask_price, self.order_manager.balance, posterior_samples
                )
                
                if qty > 0:
                    logger.info(
                        f"BUY {ticker}: Net EV={buy_net_ev*100:.3f}¢ | "
                        f"Fair={fair_value:.4f} Ask={ask_price:.4f} Spread={spread*100:.1f}¢ | "
                        f"Kelly Qty={qty} TTE={hours_left:.1f}h"
                    )
                    ask_cents = int(round(ask_price * 100))
                    await self.client.create_order(ticker, "buy", "yes", qty, ask_cents)
            
            elif sell_net_ev > 0:
                qty = self.order_manager.calculate_kelly_size(
                    1.0 - fair_value, 1.0 - bid_price, self.order_manager.balance, 
                    1.0 - posterior_samples  # Complement for NO side
                )
                
                if qty > 0:
                    logger.info(
                        f"SELL {ticker}: Net EV={sell_net_ev*100:.3f}¢ | "
                        f"Fair={fair_value:.4f} Bid={bid_price:.4f} Spread={spread*100:.1f}¢ | "
                        f"Kelly Qty={qty} TTE={hours_left:.1f}h"
                    )
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
                    # 1. Fetch History (Use all available recent data, not hard 24h cutoff)
                    now = datetime.now(timezone.utc)
                    end_ts = int(now.timestamp())
                    # Fetch last 48 hours for enough data, but weight recent observations more
                    start_ts = int((now - timedelta(hours=48)).timestamp())
                    
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
                        
                        # NO MIN_OBSERVATIONS CUTOFF - let PyMC handle sparse data
                        if not candles or len(candles) < 3:  # Need minimal data for variance
                            logger.debug(f"{ticker}: Very sparse data ({len(candles)} candles)")
                            continue
                        
                        recent_prices = [c['c'] / 100.0 for c in candles]
                        
                        # 3. Infer (Heavy CPU - run in executor to avoid blocking)
                        posterior = await asyncio.get_running_loop().run_in_executor(
                            None, 
                            lambda: asyncio.run(self.engine.infer_posterior(recent_prices, tte_hours))
                        )
                        
                        # 4. Update Cache with Validity Timestamp
                        current_time = time.time()
                        valid_until = current_time + self.TRAJECTORY_VALIDITY_SECONDS
                        
                        async with self.posterior_lock:
                            self.posterior_cache[ticker] = {
                                'trace': posterior,
                                'timestamp': current_time,
                                'valid_until': valid_until
                            }
                            
                        logger.info(f"Analyst: Updated Posterior for {ticker} (valid for {self.TRAJECTORY_VALIDITY_SECONDS//60}min)")
                        
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
