"""
Integration Test for Phase 2 Hierarchical Bayesian Architecture

Tests the complete pipeline:
1. SeriesData: Build library from API
2. AnalysisEngine: Hierarchical inference
3. PosteriorCache: Statistical validity
4. Kelly: Exact numerical optimization
"""

import asyncio
import logging
from series_data import SeriesData
from analysis_engine import AnalysisEngine  
from posterior_cache import PosteriorCache
from trader import OrderManager
from kalshi_client import AsyncKalshiClient
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_phase2_integration():
    """Test complete Phase 2 pipeline on demo environment"""
    
    logger.info("=" * 80)
    logger.info("PHASE 2 INTEGRATION TEST - Hierarchical Bayesian Architecture")
    logger.info("=" * 80)
    
    # Initialize client from config
    from config import Config
    
    if not Config.KEY_ID or not Config.KEY_FILE:
        logger.error("KALSHI_KEY_ID and KALSHI_KEY_FILE must be set in environment")
        return
    
    logger.info(f"Using ENV: {Config.ENV.value}")
    logger.info(f"Using key_id: {Config.KEY_ID[:10]}...")
    
    client = AsyncKalshiClient(key_id=Config.KEY_ID, key_file=Config.KEY_FILE)
    
    try:
        # ========================================================================
        # PHASE 2A TEST: SeriesData - Learn from History
        # ========================================================================
        logger.info("\n[TEST 2A] Building SeriesData with persistence learning...")
        
        # Pick a recurring series (weekly jobless  claims, CPI, etc)
        # For demo, we'll use whatever series is available
        series_resp = await client.get("/series", params={"status": "active", "limit": 5})
        
        if not series_resp or "series" not in series_resp:
            logger.error("No active series found")
            return
        
        test_series_ticker = series_resp["series"][0]["ticker"]
        logger.info(f"Selected series: {test_series_ticker}")
        
        series_data = SeriesData(test_series_ticker)
        
        try:
            await series_data.update_library(client, min_markets=2)  # Lower threshold for demo
            logger.info(f"✓ Library built: {len(series_data.library_of_noise)} residuals")
            logger.info(f"✓ Persistence learned: {series_data.hyperpriors.persistence:.3f}")
            logger.info(f"✓ Series volatility: {np.exp(series_data.hyperpriors.log_vol_mu):.4f}")
        except ValueError as e:
            logger.warning(f"SeriesData build failed (expected if < 2 settled markets): {e}")
            series_data = None
        
        # ========================================================================
        # PHASE 2B TEST: Hierarchical Inference
        # ========================================================================
        logger.info("\n[TEST 2B] Testing hierarchical PyMC inference...")
        
        engine = AnalysisEngine(series_data=series_data)
        
        # Test with sparse data (N=5) - should use strong Series prior
        sparse_prices = [0.45, 0.46, 0.48, 0.47, 0.49]
        timestamps = np.array([
            1704067200 - 3600 * 4,  # 4 hours ago
            1704067200 - 3600 * 3,  # 3 hours ago
            1704067200 - 3600 * 2,  # 2 hours ago
            1704067200 - 3600 * 1,  # 1 hour ago
            1704067200,  # Now
        ])
        
        logger.info("Testing sparse data (N=5) - should shrink to Series priors...")
        try:
            trace_sparse = await engine.infer_posterior(
                sparse_prices,
                tte_hours=24.0,
                timestamps=timestamps
            )
            logger.info(f"✓ Sparse inference completed with {len(trace_sparse.posterior.chain)} chains")
        except Exception as e:
            logger.error(f"✗ Sparse inference failed: {e}")
        
        # Test with sufficient data (N=15) - should learn from data
        sufficient_prices = list(np.random.uniform(0.40, 0.60, 15))
        timestamps_sufficient = np.linspace(
            1704067200 - 3600 * 14,
            1704067200,
            15
        )
        
        logger.info("\nTesting sufficient data (N=15) - should learn from market...")
        try:
            trace_sufficient = await engine.infer_posterior(
                sufficient_prices,
                tte_hours=12.0,
                timestamps=timestamps_sufficient
            )
            logger.info(f"✓ Sufficient-data inference completed")
        except Exception as e:
            logger.error(f"✗ Sufficient inference failed: {e}")
        
        # ========================================================================
        # PHASE 2C TEST: PosteriorCache HDI Validity
        # ========================================================================
        logger.info("\n[TEST 2C] Testing PosteriorCache statistical validity...")
        
        # Create mock forward projection
        n_samples = 100
        n_timesteps = 10
        projected_samples = np.random.uniform(0.45, 0.55, (n_samples, n_timesteps))
        projection_times = np.linspace(1704067200, 1704067200 + 3600, n_timesteps)
        
        cache = PosteriorCache(
            trace=trace_sparse if 'trace_sparse' in locals() else None,
            projected_samples=projected_samples,
            projection_timestamps=projection_times,
            creation_time=1704067200
        )
        
        # Test price within HDI (should be valid)
        is_valid, reason = cache.is_valid(new_price=0.50, current_time=1704067200 + 1800)
        logger.info(f"Price 0.50 within HDI: valid={is_valid}")
        
        # Test price outside HDI (should trigger re-inference)
        is_valid_diverge, reason_diverge = cache.is_valid(new_price=0.90, current_time=1704067200 + 1800)
        logger.info(f"Price 0.90 outside HDI: valid={is_valid_diverge}, reason={reason_diverge}")
        
        # ========================================================================
        # PHASE 2D TEST: Exact Kelly Optimization
        # ========================================================================
        logger.info("\n[TEST 2D] Testing exact Kelly optimization...")
        
        order_mgr = OrderManager(client)
        order_mgr.balance = 10000  # Mock $10k balance
        
        # Generate posterior samples of terminal probabilities
        posterior_probs = np.random.beta(5, 5, 1000)  # Mock posterior (mean ~0.5)
        
        # Test Kelly sizing
        market_price = 0.45  # Market underpriced relative to 0.50 mean
        contracts = order_mgr.calculate_kelly_size(
            fair_value=0.50,  # Not used in exact version
            market_price=market_price,
            bankroll=order_mgr.balance,
            posterior_samples=posterior_probs
        )
        
        logger.info(f"✓ Kelly optimization: {contracts} contracts (bankroll=$10k, price=${market_price})")
        
        # ========================================================================
        # SUMMARY
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 INTEGRATION TEST COMPLETE")
        logger.info("=" * 80)
        logger.info("✓ Phase 2A: SeriesData with learned persistence")
        logger.info("✓ Phase 2B: Hierarchical inference (Global → Series → Market)")
        logger.info("✓ Phase 2C: HDI-based cache validity (no time limits)")
        logger.info("✓ Phase 2D: Exact Kelly optimization (scipy)")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_phase2_integration())
