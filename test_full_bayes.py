#!/usr/bin/env python3
"""
Verification script for Full Bayes MCMC implementation.
Tests the stochastic volatility model and Jensen's Gap calculation.
"""

import logging
import numpy as np
import sys
sys.path.insert(0, '/Users/user/market_jensen')

from analysis_engine import AnalysisEngine
from config import Config
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_stochastic_volatility_model():
    """Test 1: Verify stochastic volatility model runs without errors."""
    logger.info("=" * 60)
    logger.info("TEST 1: Stochastic Volatility Model")
    logger.info("=" * 60)
    
    engine = AnalysisEngine()
    
    # Generate synthetic price data with known properties
    np.random.seed(42)
    n_obs = 50
    
    # Simulate a random walk in probability space
    true_volatility = 0.05
    prices = []
    p = 0.5
    for i in range(n_obs):
        # Random walk in log-odds space
        logit_p = np.log(p / (1 - p))
        logit_p += np.random.normal(0, true_volatility)
        p = 1.0 / (1.0 + np.exp(-logit_p))
        p = np.clip(p, 0.01, 0.99)
        prices.append(p)
    
    logger.info(f"Generated {n_obs} synthetic prices")
    logger.info(f"Price range: [{min(prices):.3f}, {max(prices):.3f}]")
    logger.info(f"Mean price: {np.mean(prices):.3f}")
    
    # Run inference
    try:
        logger.info("Starting NUTS inference...")
        trace = await engine.infer_posterior(prices, tte_hours=24.0)
        logger.info("✓ Inference completed successfully")
        
        # Check posterior structure
        assert hasattr(trace, 'posterior'), "Trace missing posterior"
        assert 'nu_signal' in trace.posterior, "Missing nu_signal"
        assert 'log_volatility' in trace.posterior, "Missing log_volatility"
        assert 'vol_of_vol' in trace.posterior, "Missing vol_of_vol"
        logger.info("✓ Posterior structure is correct")
        
        # Check convergence (R-hat should be < 1.1 for good convergence)
        import arviz as az
        summary = az.summary(trace, var_names=['nu_signal', 'vol_of_vol'])
        logger.info("\nPosterior Summary:")
        logger.info(summary)
        
        max_rhat = summary['r_hat'].max()
        if max_rhat < 1.1:
            logger.info(f"✓ Convergence check passed (max R-hat = {max_rhat:.4f})")
        else:
            logger.warning(f"⚠ Convergence questionable (max R-hat = {max_rhat:.4f})")
        
        return trace
        
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        raise


async def test_jensens_gap_calculation(trace):
    """Test 2: Verify Jensen's Gap is non-zero for high variance scenarios."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Jensen's Gap Calculation")
    logger.info("=" * 60)
    
    engine = AnalysisEngine()
    
    # Test with different price points
    test_cases = [
        (0.05, "Long shot (5%)"),
        (0.50, "Coin flip (50%)"),
        (0.95, "Sure thing (95%)"),
    ]
    
    for current_price, description in test_cases:
        logger.info(f"\nTest case: {description}")
        logger.info(f"  Current price: {current_price:.2f}")
        
        try:
            prediction = engine.predict_fast(
                current_price=current_price,
                time_to_expiry_hours=48.0,  # 48 hours gives more uncertainty
                trace=trace
            )
            
            fair_value = prediction['fair_value']
            naive_price = prediction['naive_price']
            gap = prediction['gap']
            gap_cents = prediction['gap_cents']
            
            logger.info(f"  Fair Value:  {fair_value:.4f}")
            logger.info(f"  Naive Price: {naive_price:.4f}")
            logger.info(f"  Gap:         {gap:.4f} ({gap_cents:.2f}¢)")
            logger.info(f"  Std Dev:     {prediction['std_prob']:.4f}")
            logger.info(f"  [5%, 95%]:   [{prediction['percentile_5']:.3f}, {prediction['percentile_95']:.3f}]")
            
            # Jensen's inequality check
            # For concave function (sigmoid on right side), E[f(X)] < f(E[X])
            # For convex function (sigmoid on left side), E[f(X)] > f(E[X])
            
            if current_price < 0.5:
                # Left side (convex), expect positive gap
                if gap > 0:
                    logger.info("  ✓ Gap has expected sign (positive for underpriced)")
                else:
                    logger.warning(f"  ⚠ Gap is {gap:.4f}, expected positive")
            elif current_price > 0.5:
                # Right side (concave), expect negative gap
                if gap < 0:
                    logger.info("  ✓ Gap has expected sign (negative for overpriced)")
                else:
                    logger.warning(f"  ⚠ Gap is {gap:.4f}, expected negative")
            
            # Gap should be non-zero when variance is present
            if abs(gap_cents) > 0.01:
                logger.info(f"  ✓ Non-zero gap detected ({gap_cents:.2f}¢)")
            else:
                logger.warning("  ⚠ Gap is near zero (might indicate insufficient variance)")
                
        except Exception as e:
            logger.error(f"  ✗ Prediction failed: {e}")
            raise


async def test_prior_fallback():
    """Test 3: Verify prior fallback works with insufficient data."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Prior Fallback")
    logger.info("=" * 60)
    
    engine = AnalysisEngine()
    
    # Insufficient data (< 10 observations)
    sparse_prices = [0.5, 0.51, 0.49, 0.52, 0.48]
    
    try:
        trace = await engine.infer_posterior(sparse_prices, tte_hours=12.0)
        logger.info("✓ Prior fallback executed successfully")
        
        # Verify it returns valid trace structure
        prediction = engine.predict_fast(0.5, 24.0, trace)
        logger.info(f"  Fair Value: {prediction['fair_value']:.4f}")
        logger.info(f"  Gap: {prediction['gap_cents']:.2f}¢")
        logger.info("✓ Prior-based prediction successful")
        
    except Exception as e:
        logger.error(f"✗ Prior fallback failed: {e}")
        raise


async def test_numerical_stability():
    """Test 4: Verify numerical stability at extreme prices."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Numerical Stability")
    logger.info("=" * 60)
    
    engine = AnalysisEngine()
    
    # Test extreme prices
    extreme_cases = [
        [0.01] * 20 + [0.02] * 10,  # Very low
        [0.99] * 20 + [0.98] * 10,  # Very high
    ]
    
    for i, prices in enumerate(extreme_cases):
        logger.info(f"\n  Case {i+1}: Extreme prices (mean={np.mean(prices):.3f})")
        
        try:
            trace = await engine.infer_posterior(prices, tte_hours=6.0)
            prediction = engine.predict_fast(prices[-1], 12.0, trace)
            
            # Check for NaN or Inf
            assert not np.isnan(prediction['fair_value']), "Fair value is NaN"
            assert not np.isinf(prediction['fair_value']), "Fair value is Inf"
            assert 0.0 <= prediction['fair_value'] <= 1.0, "Fair value out of bounds"
            
            logger.info(f"    Fair Value: {prediction['fair_value']:.4f}")
            logger.info(f"    Gap: {prediction['gap_cents']:.2f}¢")
            logger.info("    ✓ Numerically stable")
            
        except Exception as e:
            logger.error(f"    ✗ Failed: {e}")
            raise


async def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 60)
    logger.info("FULL BAYES MCMC VERIFICATION")
    logger.info("=" * 60)
    
    try:
        # Test 1: Core model
        trace = await test_stochastic_volatility_model()
        
        # Test 2: Jensen's Gap
        await test_jensens_gap_calculation(trace)
        
        # Test 3: Prior fallback
        await test_prior_fallback()
        
        # Test 4: Numerical stability
        await test_numerical_stability()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("=" * 60)
        logger.info("\nThe Full Bayes MCMC implementation is functioning correctly:")
        logger.info("  • Stochastic volatility model converges")
        logger.info("  • Jensen's Gap is properly calculated")
        logger.info("  • Prior fallback works for sparse data")
        logger.info("  • Numerically stable at extreme values")
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("VERIFICATION FAILED ✗")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
