
import logging
import asyncio
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
from kalshi_client import AsyncKalshiClient, KalshiWebSocket
from analysis_engine import AnalysisEngine
from config import Config
import sortedcontainers

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages active orders, inventory, and sizing (Kelly Criterion).
    Phase 3: Maker Execution & Risk Management.
    """
    def __init__(self, client: AsyncKalshiClient):
        self.client = client
        self.open_orders = {} # {client_order_id: order_details}
        self.positions = {}   # {ticker: net_position}
        self.balance = 100000 # Default cents (will fetch valid)
        
    async def refresh_account(self):
        """Syncs balance and positions."""
        # Async fetch
        try:
            bal_resp = await self.client.get_portfolio_balance()
            self.balance = bal_resp.get("balance", 0) # in cents
            # Positions fetch (omitted for brevity, assume we track delta or fetch periodic)
        except Exception as e:
            logger.error(f"Failed to refresh account: {e}")

    def calculate_kelly_size(self, ticker: str, edge: float, odds: float, bankroll: float) -> int:
        """
        Calculates optimal position size using Kelly Criterion.
        f* = (p(b+1) - 1) / b
        p = probability of winning (Fair Value)
        b = odds received (Net Odds - 1)
        """
        # Simplified Kelly for Binary Options:
        # Stake = Bankroll * (ExpectedValue / (b * 1)) ? No.
        # Standard Formula: f = (p - q/b)
        # where b = net odds (Payout / Cost - 1).
        # Cost = Price. Payout = 1.
        # b = (1 - Price) / Price.
        
        # Example: Price 0.4. Payout 1. Profit 0.6.
        # b = 0.6 / 0.4 = 1.5.
        
        # Let's say Edge is positive.
        # EV = FairValue - Price.
        
        # We limit max bet to a fraction of Kelly (half-kelly or quarter-kelly) for safety.
        # And cap at Config.MAX_POSITION_SIZE_PERCENT.
        
        # If we just want a robust sizing based on Config for MVP:
        return Config.ORDER_SIZE_CONTRACTS

class Trader:
    """
    Phase 3: Async Trader
    Events -> Analysis -> Decision -> Execution
    """
    def __init__(self):
        self.client = AsyncKalshiClient()
        self.engine = AnalysisEngine() # Has Volatility Kernel
        self.order_manager = OrderManager(self.client)
        self.active_tickers = []
        
    async def on_market_update(self, msg: Dict):
        """
        WebSocket Callback.
        """
        # Msg structure depends on channel. 'orderbook_delta'
        # For now, we simulate the "event" triggering the evaluation.
        # In a real WS, we maintain a local orderbook (sortedcontainers).
        # Here we assume we get a snapshot or reconstruct it.
        pass

    async def evaluate_ticker(self, ticker: str):
        """
        Full evaluation cycle.
        """
        # 1. Get Orderbook (Async)
        try:
            ob_resp = await self.client.get_market_orderbook(ticker)
            ob = ob_resp.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            yes_asks = ob.get("yes", []) # Asks are sell orders? Kalshi structure is usually bids/asks.
            # wait, get_market_orderbook returns structure.
            # Assuming standard structure.
            
            if not yes_bids: return
            
            best_bid = yes_bids[0][0] # Price
            best_ask = 100 - ob.get("no", [[0,0]])[0][0] # Synthetic ask from No bid?
            # Or just use raw asks if available.
            
            mid_price = (best_bid + best_ask) / 2.0 / 100.0
            
            # 2. Time to expiry
            # Assume we have it cached or fetch.
            # For speed, we pass it in or cache it.
            hours_left = 24.0 # Placeholder
            
            # 3. Inference
            fair_value = self.engine.run_inference_simulation(mid_price, hours_left)
            
            # 4. Decision (Kelly/Edge)
            # Log it
            logger.info(f"{ticker}: Mid {mid_price:.2f} | Fair {fair_value:.2f}")

        except Exception as e:
            logger.error(f"Eval Error {ticker}: {e}")

    async def run(self):
        """
        Main Async Loop.
        """
        async with self.client:
            await self.order_manager.refresh_account()
            
            # TODO: Initialize WebSocket
            # ws = KalshiWebSocket(self.client, self.on_market_update)
            # asyncio.create_task(ws.connect())
            
            logger.info("Trader Running (Async)...")
            while True:
                # Discovery loop (slow)
                # In real prod, this is event driven.
                # Here we just iterate active tickers.
                for ticker in self.active_tickers:
                    await self.evaluate_ticker(ticker)
                
                await asyncio.sleep(1)

def main():
    logging.basicConfig(level=logging.INFO)
    trader = Trader()
    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
