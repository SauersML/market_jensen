
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
    Phase 8: Full Bayesian Parameter Inference.
    - Phase A: Offline Prior Fitting (Hierarchical).
    - Phase B: Online Parameter Inference (ADVI).
    - Phase C: Posterior Predictive Simulation (Jensen's Gap).
    """
    def __init__(self):
        # Hyperparameters (Priors for Mu and Sigma)
        # Default Weakly Informative if no history fit yet
        self.priors = {
            "drift_mu": 0.0, "drift_sigma": 0.05, "drift_nu": 5,
            "vol_mu": 0.05, "vol_sigma": 0.05, "vol_nu": 5
        }
        # Last inferred posterior traces for the Active Market
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
        if len(recent_prices) < Config.MIN_OBSERVATIONS_FOR_INFERENCE:
            # Fallback: Return prior samples in compatible format
            logger.warning(f"Insufficient data ({len(recent_prices)} obs). Using prior samples.")
            return self._generate_prior_trace(tte_hours)
            
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
            # 2a. HIERARCHICAL PRIORS
            # ───────────────────────────────────────────────────────────
            
            # Degrees of freedom for Student-T (controls tail fatness)
            # Lower nu = fatter tails (more robust to jumps/news events)
            nu_signal = pm.Gamma(
                "nu_signal",
                alpha=2.0,
                beta=0.1,
                initval=5.0
            )
            nu_obs = pm.Gamma(
                "nu_obs", 
                alpha=2.0,
                beta=0.1,
                initval=4.0
            )
            
            # Initial log-volatility (from learned hierarchical priors)
            log_vol_init = pm.Normal(
                "log_vol_init",
                mu=self.priors.get("log_vol_mu", -3.0),  # exp(-3) ≈ 0.05
                sigma=self.priors.get("log_vol_sigma", 1.0)
            )
            
            # Volatility of volatility (how much uncertainty changes)
            vol_of_vol = pm.HalfNormal("vol_of_vol", sigma=0.2)
            
            # ───────────────────────────────────────────────────────────
            # 2b. STOCHASTIC VOLATILITY PROCESS
            # ───────────────────────────────────────────────────────────
            
            # Log-volatility follows random walk (ensures vol > 0)
            # log(s_t) = log(s_{t-1}) + vol_of_vol * ξ_t
            log_volatility = pm.GaussianRandomWalk(
                "log_volatility",
                mu=0.0,  # No drift in log-volatility
                sigma=vol_of_vol,
                shape=n_obs,
                init_dist=pm.Normal.dist(mu=log_vol_init, sigma=0.1)
            )
            
            # Transform to volatility: s_t = exp(log(s_t))
            volatility = pm.Deterministic("volatility", pt.exp(log_volatility))
            
            # ───────────────────────────────────────────────────────────
            # 2c. LATENT SIGNAL (True Log-Odds)
            # ───────────────────────────────────────────────────────────
            
            # Student-T innovations with TIME-VARYING volatility
            # η_t ~ StudentT(nu_signal, 0, s_t)
            # This is the KEY innovation: volatility changes over time!
            innovations = pm.StudentT(
                "innovations",
                nu=nu_signal,
                mu=0.0,
                sigma=volatility,  # Time-varying!
                shape=n_obs
            )
            
            # Latent log-odds path: x_t = x_0 + Σ(innovations)
            # Initialize close to first observation
            x0 = pm.Normal("x0", mu=y_obs[0], sigma=0.5)
            latent_log_odds = pm.Deterministic(
                "latent_log_odds",
                x0 + pt.cumsum(innovations)
            )
            
            # ───────────────────────────────────────────────────────────
            # 2d. OBSERVATION LIKELIHOOD (Microstructure Noise)
            # ───────────────────────────────────────────────────────────
            
            # Observation noise (bid-ask bounce, liquidity gaps)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)
            
            # Observed prices ~ StudentT(latent, sigma_obs, nu_obs)
            # Using Student-T for robustness to outlier prices
            obs = pm.StudentT(
                "obs",
                nu=nu_obs,
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
        nu_signal = trace.posterior["nu_signal"].values.flatten()
        log_vol_final = trace.posterior["log_volatility"].values[:, :, -1].flatten()
        vol_of_vol = trace.posterior["vol_of_vol"].values.flatten()
        
        n_sims = len(nu_signal)
        
        # Current log-odds
        logit_current = self._logit(current_price)
        
        # Number of steps to simulate forward (hourly granularity)
        steps = int(max(1, np.ceil(time_to_expiry_hours)))
        
        # ═══════════════════════════════════════════════════════════════
        # 2. FORWARD SIMULATION (Stochastic Volatility Path)
        # ═══════════════════════════════════════════════════════════════
        
        terminal_logits = np.zeros(n_sims)
        
        for i in range(n_sims):
            # Initialize from final state of inference
            log_vol = log_vol_final[i]
            x = logit_current
            nu = nu_signal[i]
            vov = vol_of_vol[i]
            
            # Simulate forward path
            for step in range(steps):
                # Evolve volatility (log-space random walk)
                log_vol += np.random.normal(0, vov)
                vol = np.exp(log_vol)
                
                # Signal innovation (Student-T with time-varying vol)
                shock = np.random.standard_t(nu) * vol
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
    
    def _generate_prior_trace(self, tte_hours: float):
        """
        Generate a mock trace using prior distributions when data is insufficient.
        Returns an object compatible with predict_fast expectations.
        """
        import arviz as az
        
        n_draws = Config.NUTS_DRAWS
        n_chains = Config.NUTS_CHAINS
        
        # Sample from hierarchical priors
        log_vol_mu = self.priors.get("log_vol_mu", -3.0)
        log_vol_sigma = self.priors.get("log_vol_sigma", 1.0)
        
        # Create mock posterior samples
        nu_signal = np.random.gamma(2.0, 1.0/0.1, size=(n_chains, n_draws))
        log_volatility_final = np.random.normal(log_vol_mu, log_vol_sigma, size=(n_chains, n_draws))
        vol_of_vol = np.abs(np.random.normal(0, 0.2, size=(n_chains, n_draws)))
        
        # Create InferenceData structure
        posterior_dict = {
            "nu_signal": (["chain", "draw"], nu_signal),
            "log_volatility": (["chain", "draw", "time"], log_volatility_final[:, :, np.newaxis]),
            "vol_of_vol": (["chain", "draw"], vol_of_vol),
        }
        
        return az.from_dict(posterior=posterior_dict)


    def _get_bucket(self, hours: float) -> str:
        if hours < 1.0: return "UltraShort"
        if hours < 24.0: return "Short"
        if hours < 168.0: return "Medium"
        return "Long"

    def fit_historical_priors(self, historical_df: pl.DataFrame):
        """
        Phase A: Learn Hierarchical Priors (No Bucketing).
        Fits a global distribution of volatility across all settled markets.
        """
        logger.info("Phase A: Fitting Hierarchical Priors (Global Model)...")
        
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
            joblib.dump(self.priors, Config.MODELS_DIR / "hierarchical_priors.pkl")
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
        
        logger.info(f"Hierarchical Priors (Global): {self.priors}")
        logger.info(f"  Median Volatility: {np.exp(log_vol_mu):.4f}")
        logger.info(f"  Based on {len(market_stats)} markets")
        
        joblib.dump(self.priors, Config.MODELS_DIR / "hierarchical_priors.pkl")

