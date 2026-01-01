
import logging
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
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

    async def infer_posterior(self, recent_prices: List[float], tte_hours: float) -> Dict[str, np.ndarray]:
        """
        Phase B: The Analyst (Slow Loop).
        Infers the posterior distribution of latent parameters (Mu, Sigma) given recent history.
        Uses State Space Model to separate Signal (Latent) from Noise (Microstructure).
        """
        if len(recent_prices) < 10:
            # Fallback to Priors
            bucket = self._get_bucket(tte_hours)
            prior = self.priors.get(bucket, self.priors.get("Short")) # Fallback
            
            n_sims = Config.MCMC_N_SIMULATIONS
            return {
                "mus": np.random.standard_t(prior["drift_nu"], n_sims) * prior["drift_sigma"] + prior["drift_mu"],
                "sigmas": np.abs(np.random.standard_t(prior["vol_nu"], n_sims) * prior["vol_sigma"] + prior["vol_mu"])
            }
            
        # 1. Prepare Data
        # Logits of observed prices
        y_obs = self._logit(np.array(recent_prices))
        n_obs = len(y_obs)
        
        # Select Prior Bucket
        bucket = self._get_bucket(tte_hours)
        prior = self.priors.get(bucket, self.priors.get("Short"))
        
        # 2. PyMC State Space Model
        # Latent Random Walk + Observation Noise
        with pm.Model() as model:
            # 2a. Priors (Hierarchical, Stratified)
            # Drift (Trend)
            mu = pm.StudentT("mu", 
                             nu=prior["drift_nu"], 
                             mu=prior["drift_mu"], 
                             sigma=prior["drift_sigma"])
            
            # Process Volatility (True underlying uncertainty)
            sigma_process = pm.Truncated("sigma_process", 
                                 pm.StudentT.dist(nu=prior["vol_nu"], mu=prior["vol_mu"], sigma=prior["vol_sigma"]), 
                                 lower=0.0001)
                                 
            # Microstructure Noise (Observation Noise)
            # We assume this is small but exists. HalfNormal prior.
            sigma_noise = pm.HalfNormal("sigma_noise", sigma=0.1)
            
            # 2b. Latent Process (Gaussian Random Walk)
            # x_t = x_{t-1} + mu + sigma_process * eps
            # We use GaussianRandomWalk distribution.
            # Shape is n_obs.
            # Initial state x_0 roughly matches y_obs[0]
            
            x_latent = pm.GaussianRandomWalk("x_latent", 
                                             mu=mu, 
                                             sigma=sigma_process, 
                                             shape=n_obs,
                                             init_dist=pm.Normal.dist(mu=y_obs[0], sigma=0.1))
            
            # 2c. Likelihood
            # Observed Logits ~ Normal(Latent Logits, Noise)
            obs = pm.Normal("obs", mu=x_latent, sigma=sigma_noise, observed=y_obs)
            
            # Inference: ADVI
            # This is higher dimensional (x_latent has n_obs params).
            # ADVI is crucial here.
            approx = pm.fit(n=30000, method='advi', progressbar=False)
            trace = approx.sample(Config.MCMC_N_SIMULATIONS)
            
        return {
            "mus": trace.posterior["mu"].values.flatten(),
            "sigmas": trace.posterior["sigma_process"].values.flatten()
        }

    def predict_fast(self, current_price: float, time_to_expiry_hours: float, posterior: Dict[str, np.ndarray]) -> float:
        """
        Phase C: The Trader (Fast Loop).
        Vectorized forward simulation of the LATENT PROCESS (Signal).
        We do NOT add observation noise here because we want the True Fair Value.
        """
        mus = posterior["mus"]
        sigmas = posterior["sigmas"]
        
        n_sims = min(len(mus), len(sigmas), Config.MCMC_N_SIMULATIONS)
        mus = mus[:n_sims]
        sigmas = sigmas[:n_sims]
        
        return self._simulate_paths(current_price, time_to_expiry_hours, mus, sigmas)

    def _simulate_paths(self, current_price, hours, mus, sigmas) -> float:
        """
        Vectorized Random Walk (Latent Signal).
        """
        steps = int(max(1, np.ceil(hours)))
        n_sims = len(mus)
        logit_current = self._logit(current_price)
        
        # Latent Process Shocks
        # Assuming Process is Gaussian (or StudentT if we matched fit).
        # Model used GaussianRandomWalk. Let's use Normal.
        # User requested Fat Tails possible, but GRW is Normal.
        # Let's stick to Normal for the Process to match the PyMC model.
        raw_shocks = np.random.normal(0, 1, size=(n_sims, steps))
        
        # Path
        step_moves = mus[:, np.newaxis] + sigmas[:, np.newaxis] * raw_shocks
        cumulative = np.cumsum(step_moves, axis=1)
        terminal_logits = logit_current + cumulative[:, -1]
        
        # Transform & Mean
        terminal_probs = self._sigmoid(terminal_logits)
        return np.mean(terminal_probs)

    def _get_bucket(self, hours: float) -> str:
        if hours < 1.0: return "UltraShort"
        if hours < 24.0: return "Short"
        if hours < 168.0: return "Medium"
        return "Long"

    def fit_historical_priors(self, historical_df: pl.DataFrame):
        """
        Phase A: Learn TTE-Stratified Priors.
        """
        logger.info("Phase A: Fitting Stratified Priors...")
        
        # 1. Calc Parameter Estimates per Market
        df = historical_df.sort(["market_ticker", "timestamp"])
        df = df.with_columns([
            pl.col("price_normalized").map_elements(self._logit, return_dtype=pl.Float64).alias("logit"),
            pl.col("time_remaining_hours")
        ])
        
        # We need to assign each MARKET to a bucket?
        # A market evolves through buckets.
        # But we want the volatility characteristic of that market *when it was in that bucket*?
        # Or do we treat a market as a single entity?
        # Volatility assumes constant parameter fitting.
        # Let's fit (Drift, Vol) for each market *overall*, but classify the market by its *Mean TTE* or *Start TTE*?
        # No, better: segment the time series of each market into TTE chunks and calc vol for each chunk.
        # Complexity: 8.
        # Simpler: Each market has a dominant TTE characteristic in the data we fetched?
        # Let's iterate markets and compute (mu, sigma) *per TTE window*.
        
        # For simplicity/robustness:
        # We calculate (mu, sigma) for *segments* of history.
        # But `agg` works on groups.
        # Let's bin the *rows* by TTE first.
        
        buckets = {
            "UltraShort": df.filter(pl.col("time_remaining_hours") < 1.0),
            "Short": df.filter((pl.col("time_remaining_hours") >= 1.0) & (pl.col("time_remaining_hours") < 24.0)),
            "Medium": df.filter((pl.col("time_remaining_hours") >= 24.0) & (pl.col("time_remaining_hours") < 168.0)),
            "Long": df.filter(pl.col("time_remaining_hours") >= 168.0)
        }
        
        self.priors = {}
        
        for name, sub_df in buckets.items():
            if sub_df.is_empty():
                logger.warning(f"No data for bucket {name}. Using default.")
                self.priors[name] = {"drift_nu": 5, "drift_mu": 0, "drift_sigma": 0.05, "vol_nu": 5, "vol_mu": 0.05, "vol_sigma": 0.05}
                continue

            # Calculate observed return stats per market within this bucket
            sub_df = sub_df.with_columns([
                pl.col("logit").diff().over("market_ticker").alias("ret")
            ])
            
            aggs = sub_df.group_by("market_ticker").agg([
                pl.col("ret").mean().alias("mu"),
                pl.col("ret").std().alias("sigma")
            ]).drop_nulls()
            
            mus = aggs["mu"].to_numpy()
            sigmas = aggs["sigma"].to_numpy()
            
            if len(mus) < 5:
                logger.warning(f"Insufficient markets for bucket {name}. Using default.")
                self.priors[name] = {"drift_nu": 5, "drift_mu": 0, "drift_sigma": 0.05, "vol_nu": 5, "vol_mu": 0.05, "vol_sigma": 0.05}
                continue

            # Fit StudentT
            try:
                d_nu, d_mu, d_sigma = stats.t.fit(mus)
                v_nu, v_mu, v_sigma = stats.t.fit(sigmas)
                
                self.priors[name] = {
                    "drift_nu": d_nu, "drift_mu": d_mu, "drift_sigma": d_sigma,
                    "vol_nu": v_nu, "vol_mu": v_mu, "vol_sigma": v_sigma
                }
                logger.info(f"Bucket {name} Priors: {self.priors[name]}")
            except Exception as e:
                logger.error(f"Fit failed for {name}: {e}")
                self.priors[name] = {"drift_nu": 5, "drift_mu": 0, "drift_sigma": 0.05, "vol_nu": 5, "vol_mu": 0.05, "vol_sigma": 0.05}

        joblib.dump(self.priors, Config.MODELS_DIR / "hierarchical_priors.pkl")
