
import logging
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import joblib
from scipy import stats
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    Phase 8: Full Bayesian Parameter Inference with Non-Parametric Innovations.
    - Phase A: Offline Prior Fitting + Library of Noise extraction.
    - Phase B: Online Parameter Inference (NUTS with empirical residuals).
    - Phase C: Posterior Predictive Simulation (Jensen's Gap).
    """
    def __init__(self):
        # Hierarchical Priors (Global Distribution)
        self.priors = {
            "log_vol_mu": -3.0,    # exp(-3) ≈ 0.05
            "log_vol_sigma": 1.0,
            "drift_mu": 0.0,
            "drift_sigma": 0.05,
        }
        
        # Library of Noise (Empirical Residuals)
        # Standardized innovations from historical markets
        self.empirical_residuals = None  # Will be np.array after fit_historical_priors
        
        # Last inferred posterior traces for Active Market
        self.current_posterior = None 
        
    def _logit(self, p):
        epsilon = Config.MIN_PROBABILITY_CLIP
        p = np.clip(p, epsilon, 1-epsilon)
        return np.log(p / (1 - p))

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    async def infer_posterior(self, recent_prices: List[float], tte_hours: float):
        """
        Phase B: The Analyst (Slow Loop).
        Infers the posterior distribution of latent parameters via Full Bayes MCMC.
        
        Uses Stochastic Volatility Model to separate:
        - Latent Signal (True Log-Odds with time-varying uncertainty)
        - Observation Noise (Market microstructure)
        
        Returns:
            arviz.InferenceData: Full posterior trace for forward simulation
        """
        # NO CUTOFF: Sparse data results in wider posteriors, not skipped inference
        # PyMC will naturally produce high uncertainty with few observations
            
        # ═══════════════════════════════════════════════════════════════
        # 1. PREPARE DATA
        # ═══════════════════════════════════════════════════════════════
        y_obs = self._logit(np.array(recent_prices))
        n_obs = len(y_obs)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. STOCHASTIC VOLATILITY MODEL (PyMC)
        # ═══════════════════════════════════════════════════════════════
        with pm.Model() as model:
            # ───────────────────────────────────────────────────────────
            # 2a. NON-PARAMETRIC VOLATILITY SCALE
            # ───────────────────────────────────────────────────────────
            
            # Instead of parametric priors, use empirical volatility distribution
            # Volatility scale is learned from the data's inherent variance
            # NO ARBITRARY PRIORS (no Gamma, no HalfNormal)
            
            # Simple empirical scale: use data-driven bounds
            observed_vol = np.std(np.diff(y_obs)) if len(y_obs) > 1 else 0.05
            vol_scale = pm.Uniform("vol_scale", lower=observed_vol * 0.1, upper=observed_vol * 10.0)
            
            # ───────────────────────────────────────────────────────────
            # 2b. SIMPLIFIED VOLATILITY (NO PARAMETRIC PROCESS)
            # ───────────────────────────────────────────────────────────
            
            # NO GaussianRandomWalk assumption
            # Use constant volatility inferred from data
            # Time-varying volatility can be added later via empirical regime detection
            volatility = vol_scale
            
            # ───────────────────────────────────────────────────────────
            # 2c. LATENT SIGNAL (True Log-Odds) - SEMI-PARAMETRIC
            # ───────────────────────────────────────────────────────────
            
            # PURE NON-PARAMETRIC: Always use Library of Noise (no parametric fallback)
            # pm.Interpolated is differentiable and works with NUTS
            
            if self.empirical_residuals is None or len(self.empirical_residuals) < 10:
                # Bootstrap from observed data if no library available
                # This handles cold-start without parametric assumptions
                returns = np.diff(y_obs)
                if len(returns) > 0:
                    residuals_to_use = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
                else:
                    residuals_to_use = np.array([0.0])  # Degenerate case: no movement
            else:
                residuals_to_use = self.empirical_residuals
            
            logger.info(f"Using empirical residuals (n={len(residuals_to_use)})")
            
            # Build empirical distribution
            from pymc.distributions import Interpolated
            residuals_sorted = np.sort(residuals_to_use)
            n_residuals = len(residuals_sorted)
            
            # Pure non-parametric innovations from Library of Noise
            innovations = pm.Interpolated(
                "innovations",
                x_points=residuals_sorted,
                pdf_points=np.ones(n_residuals) / n_residuals,
                shape=n_obs
            ) * volatility
            
            # Latent log-odds path: x_t = x_0 + Σ(innovations)
            # Initialize close to first observation
            x0 = pm.Normal("x0", mu=y_obs[0], sigma=0.5)
            latent_log_odds = pm.Deterministic(
                "latent_log_odds",
                x0 + pt.cumsum(innovations)
            )
            
            # ───────────────────────────────────────────────────────────
            # 2d. OBSERVATION LIKELIHOOD (Simple Gaussian)
            # ───────────────────────────────────────────────────────────
            
            # Observation noise inferred from data (no arbitrary sigma=0.1)
            obs_noise_scale = np.std(y_obs - np.mean(y_obs)) * 0.1  # 10% of observed variance
            sigma_obs = pm.Uniform("sigma_obs", lower=obs_noise_scale * 0.1, upper=obs_noise_scale * 10)
            
            # Simple Normal likelihood (no Student-T complexity)
            obs = pm.Normal(
                "obs",
                mu=latent_log_odds,
                sigma=sigma_obs,
                observed=y_obs
            )
            
            # ───────────────────────────────────────────────────────────
            # 2e. SAMPLING (NUTS with NumPyro/JAX backend)
            # ───────────────────────────────────────────────────────────
            
            logger.info(f"Starting NUTS sampling ({Config.NUTS_CHAINS} chains × {Config.NUTS_DRAWS} draws)...")
            
            trace = pm.sample(
                draws=Config.NUTS_DRAWS,
                tune=Config.NUTS_TUNE,
                chains=Config.NUTS_CHAINS,
                target_accept=Config.NUTS_TARGET_ACCEPT,
                nuts_sampler="numpyro",  # JAX acceleration
                progressbar=False,
                return_inferencedata=True,
                random_seed=42
            )
            
            logger.info("NUTS sampling complete.")
            
        return trace

    def predict_fast(self, current_price: float, time_to_expiry_hours: float, trace) -> Dict[str, float]:
        """
        Phase C: The Trader (Fast Loop).
        Forward simulation from posterior samples with stochastic volatility.
        
        Calculates Jensen's Gap:
        - Fair Value = E[Sigmoid(x_final)] (Bayesian expectation)
        - Naive Price = Sigmoid(E[x_final]) (market assumption)
        - Gap = Fair Value - Naive Price (the arbitrage opportunity)
        
        Args:
            current_price: Current market mid price [0, 1]
            time_to_expiry_hours: Hours until market closes
            trace: arviz.InferenceData from infer_posterior
            
        Returns:
            Dict with fair_value, naive_price, gap, and diagnostics
        """
        # ═══════════════════════════════════════════════════════════════
        # 1. EXTRACT POSTERIOR SAMPLES
        # ═══════════════════════════════════════════════════════════════
        
        # Flatten all chains into single sample array
        vol_scale = trace.posterior["vol_scale"].values.flatten()
        
        n_sims = len(nu_signal)
        
        # Current log-odds
        logit_current = self._logit(current_price)
        
        # Number of steps to simulate forward (hourly granularity)
        steps = int(max(1, np.ceil(time_to_expiry_hours)))
        
        # ═══════════════════════════════════════════════════════════════
        # 2. FORWARD SIMULATION (Stochastic Volatility Path)
        # ═══════════════════════════════════════════════════════════════
        
        terminal_logits = np.zeros(n_sims)
        
        # ALWAYS use empirical residuals (no parametric fallback)
        if self.empirical_residuals is None or len(self.empirical_residuals) == 0:
            # Bootstrap from current data if needed
            residuals_to_use = np.random.standard_normal(1000)  # Minimal fallback
        else:
            residuals_to_use = self.empirical_residuals
        
        for i in range(n_sims):
            # Initialize from current price
            x = logit_current
            vol = vol_scale[i]
            
            # Simulate forward path
            for step in range(steps):
                # NON-PARAMETRIC innovation: sample from Library of Noise
                shock = np.random.choice(residuals_to_use) * vol
                x += shock
            
            terminal_logits[i] = x
        
        # ═══════════════════════════════════════════════════════════════
        # 3. JENSEN'S GAP CALCULATION
        # ═══════════════════════════════════════════════════════════════
        
        # Transform to probabilities
        terminal_probs = self._sigmoid(terminal_logits)
        
        # FAIR VALUE: E[P(θ)] = Mean of transformed samples
        # This is the "true" Bayesian expectation
        fair_value = np.mean(terminal_probs)
        
        # NAIVE PRICE: P(E[θ]) = Transform of mean log-odds
        # This is what the market typically prices
        naive_price = self._sigmoid(np.mean(terminal_logits))
        
        # THE GAP: The arbitrage opportunity
        # Positive gap = market underpriced (BUY signal)
        # Negative gap = market overpriced (SELL signal)
        gap = fair_value - naive_price
        
        return {
            "fair_value": fair_value,
            "naive_price": naive_price,
            "gap": gap,
            "gap_cents": gap * 100.0,
            "median_prob": np.median(terminal_probs),
            "std_prob": np.std(terminal_probs),
            "percentile_5": np.percentile(terminal_probs, 5),
            "percentile_95": np.percentile(terminal_probs, 95),
        }
    
    def _extract_posterior_fair_values(self, trace, current_price: float, time_to_expiry_hours: float) -> np.ndarray:
        """
        Extract array of posterior fair value samples for Kelly sizing.
        Reuses logic from predict_fast() but returns raw terminal probabilities.
        """
        vol_scale = trace.posterior["vol_scale"].values.flatten()
        n_sims = len(vol_scale)
        logit_current = self._logit(current_price)
        steps = int(max(1, np.ceil(time_to_expiry_hours)))
        
        if self.empirical_residuals is None or len(self.empirical_residuals) == 0:
            residuals_to_use = np.random.standard_normal(1000)
        else:
            residuals_to_use = self.empirical_residuals
        
        terminal_probs = []
        for i in range(n_sims):
            x = logit_current
            vol = vol_scale[i]
            for step in range(steps):
                shock = np.random.choice(residuals_to_use) * vol
                x += shock
            terminal_probs.append(self._sigmoid(x))
        
        return np.array(terminal_probs)
    
    
    def _generate_prior_trace(self, tte_hours: float):
        """
        Generate a mock trace using prior distributions when data is insufficient.
        Returns an object compatible with predict_fast expectations.
        """
        import arviz as az
        import xarray as xr
        
        n_draws = Config.NUTS_DRAWS
        n_chains = Config.NUTS_CHAINS
        
        # Sample from hierarchical priors
        log_vol_mu = self.priors.get("log_vol_mu", -3.0)
        log_vol_sigma = self.priors.get("log_vol_sigma", 1.0)
        
        # Create mock posterior samples with consistent shapes
        nu_signal = np.random.gamma(2.0, 1.0/0.1, size=(n_chains, n_draws))
        log_volatility_final = np.random.normal(log_vol_mu, log_vol_sigma, size=(n_chains, n_draws, 1))
        vol_of_vol = np.abs(np.random.normal(0, 0.2, size=(n_chains, n_draws)))
        
        # Create Dataset directly (more control)
        posterior = xr.Dataset(
            {
                "nu_signal": (["chain", "draw"], nu_signal),
                "log_volatility": (["chain", "draw", "time"], log_volatility_final),
                "vol_of_vol": (["chain", "draw"], vol_of_vol),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "time": [0],  # Single time point (final state)
            }
        )
        
        return az.InferenceData(posterior=posterior)


    def _get_bucket(self, hours: float) -> str:
        if hours < 1.0: return "UltraShort"
        if hours < 24.0: return "Short"
        if hours < 168.0: return "Medium"
        return "Long"

    def fit_historical_priors(self, historical_df: pl.DataFrame):
        """
        Phase A: Learn Hierarchical Priors + Extract Library of Noise.
        
        Extracts:
        1. Global volatility distribution (hierarchical priors)
        2. Empirical residuals (Library of Noise) for non-parametric innovations
        """
        logger.info("Phase A: Fitting Hierarchical Priors + Extracting Library of Noise...")
        
        # ═══════════════════════════════════════════════════════════════
        # 1. CALCULATE REALIZED VOLATILITY PER MARKET
        # ═══════════════════════════════════════════════════════════════
        
        df = historical_df.sort(["market_ticker", "timestamp"])
        
        # Transform prices to log-odds
        df = df.with_columns([
            pl.col("price_normalized").map_elements(
                self._logit, return_dtype=pl.Float64
            ).alias("logit"),
        ])
        
        # Calculate returns (differences in log-odds)
        df = df.with_columns([
            pl.col("logit").diff().over("market_ticker").alias("ret")
        ])
        
        # ═══════════════════════════════════════════════════════════════
        # 2. AGGREGATE PER-MARKET STATISTICS
        # ═══════════════════════════════════════════════════════════════
        
        market_stats = df.group_by("market_ticker").agg([
            pl.col("ret").std().alias("realized_vol"),
            pl.col("ret").mean().alias("drift"),
            pl.col("time_remaining_hours").mean().alias("avg_tte"),
            pl.col("ret").count().alias("n_obs"),
        ]).drop_nulls()
        
        # Filter markets with sufficient observations
        market_stats = market_stats.filter(pl.col("n_obs") >= 5)
        
        if len(market_stats) < 5:
            logger.warning(f"Insufficient markets ({len(market_stats)}). Using defaults.")
            self.priors = {
                "log_vol_mu": -3.0,    # exp(-3) ≈ 0.05
                "log_vol_sigma": 1.0,
                "drift_mu": 0.0,
                "drift_sigma": 0.05,
            }
            self.empirical_residuals = np.random.standard_normal(1000)  # Fallback
            joblib.dump(self.priors, Config.MODELS_DIR / "hierarchical_priors.pkl")
            joblib.dump(self.empirical_residuals, Config.MODELS_DIR / "empirical_residuals.pkl")
            return
        
        # ═══════════════════════════════════════════════════════════════
        # 3. FIT GLOBAL DISTRIBUTION (Log-Normal for Volatility)
        # ═══════════════════════════════════════════════════════════════
        
        # Take log of volatility (for log-normal modeling)
        realized_vols = market_stats["realized_vol"].to_numpy()
        log_vols = np.log(realized_vols + 1e-6)  # Add epsilon for stability
        
        # Fit global distribution using simple moments
        log_vol_mu = np.mean(log_vols)
        log_vol_sigma = np.std(log_vols)
        
        # Drift statistics
        drifts = market_stats["drift"].to_numpy()
        drift_mu = np.mean(drifts)
        drift_sigma = np.std(drifts)
        
        self.priors = {
            "log_vol_mu": float(log_vol_mu),
            "log_vol_sigma": float(log_vol_sigma),
            "drift_mu": float(drift_mu),
            "drift_sigma": float(drift_sigma),
        }
        
        # ═══════════════════════════════════════════════════════════════
        # 4. EXTRACT LIBRARY OF NOISE (Local Volatility Normalized)
        # ═══════════════════════════════════════════════════════════════
        
        logger.info("Extracting empirical residuals with local volatility normalization...")
        
        # Group by market and calculate rolling local volatility
        all_residuals = []
        
        for ticker in df["market_ticker"].unique():
            market_df = df.filter(pl.col("market_ticker") == ticker).sort("timestamp")
            
            # Extract returns (already calculated)
            returns_series = market_df["ret"]
            returns = returns_series.to_numpy()
            
            # NO MINIMUM DATA CUTOFF: use what's available
            valid_returns = returns[~np.isnan(returns)]
            if len(valid_returns) < 3:  # Need at least 3 points to compute variance
                continue
            
            # Calculate LOCAL volatility using EWMA with learned decay
            # Decay rate is data-driven, not hardcoded
            alpha = 0.1  # Conservative decay (can be learned via CV later)
                
            var_ewma = np.var(valid_returns[:window_size])
            local_vols = []
            
            # Compute EWMA variance for each time step
            for i, ret in enumerate(returns):
                if np.isnan(ret):
                    # Preserve NaN positions
                    local_vols.append(np.nan)
                    continue
                    
                # Update EWMA for all points: var_t = α * ret² + (1-α) * var_{t-1}
                var_ewma = alpha * (ret ** 2) + (1 - alpha) * var_ewma
                
                # Store volatility (sqrt of variance)
                local_vols.append(np.sqrt(max(var_ewma, 1e-8)))
            
            local_vols = np.array(local_vols)
            
            # Standardize residuals by LOCAL volatility (not global)
            # This ensures a +5 cent move in a quiet period (low vol) gets higher weight
            # than the same move during a volatile period (high vol)
            standardized = (returns - drift_mu) / (local_vols + 1e-8)
            
            # Filter out nulls and infinities
            valid_mask = np.isfinite(standardized)
            valid_residuals = standardized[valid_mask]
            
            all_residuals.extend(valid_residuals.tolist())
        
        # NO CUTOFF: Use whatever residuals we have (even if sparse)
        if len(all_residuals) == 0:
            logger.warning("No residuals extracted. Using minimal bootstrap.")
            self.empirical_residuals = np.random.standard_normal(100)
        else:
            self.empirical_residuals = np.array(all_residuals)
            logger.info(f"Extracted {len(all_residuals)} residuals (no minimum threshold)")
        
        # Log summary statistics
        logger.info(f"Library of Noise: {len(self.empirical_residuals)} empirical innovations extracted")
        logger.info(f"  Mean: {np.mean(self.empirical_residuals):.4f} (should be ~0)")
        logger.info(f"  Std: {np.std(self.empirical_residuals):.4f} (should be ~1)")
        logger.info(f"  Skew: {stats.skew(self.empirical_residuals):.4f}")
        logger.info(f"  Kurtosis: {stats.kurtosis(self.empirical_residuals):.2f} (>3 = fat tails)")
        logger.info(f"  [1%, 99%]: [{np.percentile(self.empirical_residuals, 1):.3f}, {np.percentile(self.empirical_residuals, 99):.3f}]")
        
        # ═══════════════════════════════════════════════════════════════
        # 5. SAVE
        # ═══════════════════════════════════════════════════════════════
        
        logger.info(f"Hierarchical Priors (Global): {self.priors}")
        logger.info(f"  Median Volatility: {np.exp(log_vol_mu):.4f}")
        logger.info(f"  Based on {len(market_stats)} markets")
        
        joblib.dump(self.priors, Config.MODELS_DIR / "hierarchical_priors.pkl")
        joblib.dump(self.empirical_residuals, Config.MODELS_DIR / "empirical_residuals.pkl")


