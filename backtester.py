
import logging
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from analysis_engine import AnalysisEngine
from config import Config

logger = logging.getLogger(__name__)

class BacktestResult:
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.metrics: Dict = {}

class Backtester:
    def __init__(self, history_df: pl.DataFrame):
        self.history_df = history_df
        self.engine = AnalysisEngine()
        self.capital = 100000.0 # Cents
        self.results = BacktestResult()
        
    def run(self):
        logger.info("Starting Backtest...")
        df = self.history_df.sort("timestamp")
        
        # Train on first chunk
        min_train = Config.MIN_VOLATILITY_DATA_POINTS
        
        for i in range(min_train, len(df)):
            row = df.row(i, named=True)
            
            # Periodic Re-fit (e.g. daily)
            if i % 24 == 0:
                train_data = df.slice(0, i)
                self.engine.calculate_empirical_volatility(train_data)
                
            # Simulate
            price = row['price_normalized']
            hours = row['time_remaining_hours']
            
            if hours < 1: continue
            
            # Prepare Recent History for Inference (e.g. last 24h window)
            # We have 'timestamp' and 'price_normalized' in df up to i.
            # We need to slice effective recent window.
            # Ideally fetch via polars efficient slice.
            current_ts = row['timestamp']
            start_window = current_ts - timedelta(hours=Config.INFERENCE_WINDOW_HOURS)
            
            # Polars filter is fast
            # We already have train_data slice (0..i)
            # Just take last N rows? Assuming 1h candles?
            # Or strict time filter.
            
            # Split Loop Simulation in Backtest
            # We don't want to run ADVI every hour (too slow). Le's run it every 4 hours.
            # If current_ts - last_inference_ts > 4h:
            #    posterior = infer(...)
            
            # State needed for backtest loop
            if 'last_inference_ts' not in self.__dict__: self.last_inference_ts = datetime.min.replace(tzinfo=timezone.utc)
            if 'current_posterior' not in self.__dict__: self.current_posterior = None
            
            hours_since_last = (current_ts - self.last_inference_ts).total_seconds() / 3600.0
            
            if hours_since_last >= 4 or self.current_posterior is None:
                # Re-infer
                # Filter recent window
                start_window = current_ts - timedelta(hours=Config.INFERENCE_WINDOW_HOURS)
                recent_rows = df.filter(
                    (pl.col("timestamp") >= start_window) & 
                    (pl.col("timestamp") <= current_ts) &
                    (pl.col("market_ticker") == row['market_ticker'])
                )
                recent_prices = recent_rows["price_normalized"].to_list()
                
                # We need to run this sync for backtest
                # Engine.infer_posterior is async but we can just call the body if we refactored or just use asyncio.run
                # Wait, infer_posterior is defined as `async def`.
                # We should probably run it synchronously in backtester.
                # Hack: `asyncio.run(self.engine.infer_posterior(recent_prices))`
                # But PyMC inside async function is fine.
                
                try:
                    self.current_posterior = asyncio.run(self.engine.infer_posterior(recent_prices, hours))
                    self.last_inference_ts = current_ts
                except Exception as e:
                    print(f"Inference error: {e}")
                    continue
            
            if not self.current_posterior: continue
            
            fair = self.engine.predict_fast(price, hours, self.current_posterior)
            
            # Decision
            edge = fair - price
            if abs(edge * 100) > Config.JENSEN_GAP_THRESHOLD_CENTS:
                # Trade
                self.record_trade(row, fair, edge)
                
        self.calculate_metrics()

    def record_trade(self, row, fair, edge):
        # Simplistic outcome lookahead for PnL
        outcome = row['outcome']
        if outcome is None: return 
        
        # If BUY YES
        if edge > 0:
            price = row['price_normalized']
            pnl = (outcome - price) * 100 # cents per contract
            # Fee: Taker 
            fee = 0.0 # Placeholder for calc
            net = pnl - fee
            
            self.capital += net
            self.results.trades.append({
                "ts": row['timestamp'],
                "type": "BUY_YES",
                "pnl": net
            })
            self.results.equity_curve.append(self.capital)

    def calculate_metrics(self):
        # Calc Sharpe etc
        pass
