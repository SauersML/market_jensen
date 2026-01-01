import logging
import polars as pl
import numpy as np
from typing import Dict, List
from analysis_engine import AnalysisEngine
from config import Config

logger = logging.getLogger(__name__)

class Backtester:
    """
    Phase 3: Backtesting Framework (The Validator)
    Walk-forward validation without look-ahead bias.
    """
    def __init__(self, engine: AnalysisEngine):
        self.engine = engine
        self.results = []

    def run_walk_forward(self, series_ticker: str, full_history_df: pl.DataFrame, outcomes_map: Dict[str, int]):
        """
        Runs a walk-forward backtest.
        """
        logger.info(f"Starting walk-forward backtest for {series_ticker}...")

        # Sort markets by time (using their earliest timestamp in history)
        # We need to know when each market *started* or *ended*.
        # Let's group by market and get the start time.

        market_times = full_history_df.group_by("market_ticker").agg([
            pl.col("timestamp").min().alias("start_ts"),
            pl.col("timestamp").max().alias("end_ts")
        ]).sort("end_ts") # Sort by settlement time roughly

        sorted_markets = market_times["market_ticker"].to_list()

        # Need minimum history to start
        min_history = Config.MIN_HISTORY_EVENTS
        if len(sorted_markets) <= min_history:
            logger.warning("Not enough markets for walk-forward backtest.")
            return

        # Walk forward
        for i in range(min_history, len(sorted_markets)):
            train_markets = sorted_markets[:i]
            test_market = sorted_markets[i]

            # 1. Train
            # Filter history for train markets
            train_df = full_history_df.filter(pl.col("market_ticker").is_in(train_markets))

            # Re-calibrate volatility
            try:
                self.engine.calculate_empirical_volatility(train_df)
                # self.engine.calibrate_link_function(train_df, outcomes_map) # Optional: if we had outcome data accessible here easily
            except Exception as e:
                logger.warning(f"Skipping {test_market} due to training failure: {e}")
                continue

            # 2. Test (Intraday Simulation)
            test_df = full_history_df.filter(pl.col("market_ticker") == test_market).sort("timestamp")

            # We simulate trading through the life of the test market
            market_pnl = 0.0
            position = 0 # 1 for Yes, -1 for No (Short Yes)
            entry_price = 0.0

            # Iterate through minutes/hours of the test market
            # For speed, maybe just check every hour?

            rows = test_df.to_dicts()
            if not rows:
                continue

            final_outcome = outcomes_map.get(test_market)
            if final_outcome is None:
                # Can't backtest if we don't know the result
                continue

            expiry_ts = rows[-1]["timestamp"]

            for row in rows:
                current_ts = row["timestamp"]
                price = row["price_normalized"]

                # Time to expiry in hours
                hours_left = (expiry_ts - current_ts) / 3600
                if hours_left < 1:
                    continue # Don't trade last hour

                # Run Simulation
                try:
                    fair_value = self.engine.run_mcmc_simulation(price, int(hours_left))
                except:
                    continue

                # Signal
                # Price is normalized 0-1.
                # Threshold is in cents, convert to 0-1
                threshold = Config.JENSEN_GAP_THRESHOLD_CENTS / 100.0

                signal = "HOLD"
                if fair_value > price + threshold:
                    signal = "BUY"
                elif fair_value < price - threshold:
                    signal = "SELL"

                # Execute (Simplified)
                # We assume we can trade at the current 'price' (Close of candle)
                # In reality, we'd pay spread. Let's add slippage/fee penalty.
                fee = 0.02 # 2 cents round trip + spread estimate

                # Logic: If we don't have a position, enter. If we have one, check exit?
                # For this backtest, let's assume we hold until expiration or signal reversal.

                if position == 0:
                    if signal == "BUY":
                        position = 1
                        entry_price = price
                        # logger.debug(f"Buy at {price:.2f}")
                    elif signal == "SELL":
                        position = -1
                        entry_price = price
                        # logger.debug(f"Sell at {price:.2f}")

                elif position == 1: # Long
                    if signal == "SELL":
                        # Close and Flip
                        pnl = (price - entry_price) - fee
                        market_pnl += pnl
                        position = -1
                        entry_price = price

                elif position == -1: # Short
                    if signal == "BUY":
                        # Close and Flip
                        pnl = (entry_price - price) - fee
                        market_pnl += pnl
                        position = 1
                        entry_price = price

            # Settle final position
            if position == 1:
                pnl = (final_outcome - entry_price) - (fee/2) # Exit fee only? Kalshi settlement is free.
                market_pnl += pnl
            elif position == -1:
                pnl = (entry_price - final_outcome) - (fee/2)
                market_pnl += pnl

            self.results.append({
                "market": test_market,
                "pnl": market_pnl,
                "volume": len(rows) # Proxy for duration/activity
            })

        logger.info(f"Backtest Complete. Markets Traded: {len(self.results)}")
        total_pnl = sum(r['pnl'] for r in self.results)
        logger.info(f"Total PnL (Units): {total_pnl:.4f}")

if __name__ == "__main__":
    # Test
    pass
