
import logging
import numpy as np
import polars as pl
from analysis_engine import AnalysisEngine

def test_engine():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Test")
    
    engine = AnalysisEngine()
    
    logger.info("Generating mock data...")
    # Mock some data: A market drifting up then down
    data = []
    # Market 1: High Volatility
    price = 0.5
    for i in range(100):
        # Human readable time? Scanner expects timestamp
        market_data = {
            "timestamp": i * 3600,
            "market_ticker": "DEMO-1",
            "price_normalized": price
        }
        data.append(market_data)
        # Random walk step
        price += np.random.normal(0, 0.05)
        price = max(0.01, min(0.99, price))
        
    df = pl.DataFrame(data)
    
    logger.info("Fitting Volatility Kernel (Empirical)...")
    engine.calculate_empirical_volatility(df)
    
    if engine.empirical_residuals is None:
        logger.error("Residuals failed to initialize!")
        return

    logger.info("Running Simulation (Bootstrap)...")
    current_price = 0.60
    hours_left = 24
    
    fair_value = engine.run_inference_simulation(current_price, hours_left)
    logger.info(f"Result: Current {current_price} -> Fair {fair_value}")

if __name__ == "__main__":
    test_engine()
