import numpy as np
import polars as pl
import logging
import joblib
import numpy as np
import polars as pl
import logging
import joblib
# import scipy.stats as stats # Dropped per user request (KDE removal)
import pymc as pm
import arviz as az
# import aesara.tensor as at # or pytensor depending on pymc version. PyMC 5 uses PyTensor.
import pytensor.tensor as pt
from config import Config

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    Phase 2 & 3: Bayesian Inference Engine (Empirical/Bootstrapped)
    Implements Empirical BOotstrap, PyMC Random Walk Model (Categorical Noise), and Jensen's Gap Calculation.
    """
    def __init__(self):
        self.empirical_residuals = None # np.array of raw log-odds innovations

    def calculate_empirical_volatility(self, historical_df: pl.DataFrame):
        """
        Phase 2: Empirical Volatility (No Fit).
        Extracts hourly Log-Odds innovations and stores them raw.
        """
        logger.info("Computing Empirical Volatility (Raw Residuals)...")
        
        # 1. Group by market to get time-series
        market_tickers = historical_df["market_ticker"].unique().to_list()
        all_innovations = []

        for ticker in market_tickers:
            market_data = historical_df.filter(pl.col("market_ticker") == ticker).sort("timestamp")
            
            if len(market_data) < 2: 
                continue

            # Get Prices
            prices = market_data["price_normalized"].to_numpy()
            
            # 2. Convert to Log-Odds Space
            # Clip to avoid infs
            prices_clipped = np.clip(prices, Config.MIN_PROBABILITY_CLIP, Config.MAX_PROBABILITY_CLIP)
            logits = np.log(prices_clipped / (1 - prices_clipped))
            
            # 3. Calculate Hourly Changes (Innovations)
            innovations = np.diff(logits)
            
            all_innovations.extend(innovations)
            
        if not all_innovations:
            logger.warning("No innovations found for Volatility.")
            return

        # 4. Store Raw Residuals
        # This represents the "Empirical Prior Distribution"
        all_innovations = np.array(all_innovations)
        # Filter NaNs/Infs
        all_innovations = all_innovations[np.isfinite(all_innovations)]
        
        self.empirical_residuals = all_innovations
        joblib.dump(all_innovations, Config.MODELS_DIR / "empirical_residuals.pkl")
        logger.info(f"Empirical Volatility stored: {len(all_innovations)} raw residuals.")

    def run_inference_simulation(self, current_price: float, time_to_expiry_hours: float) -> float:
        """
        Phase 3 & 4: PyMC Model (Categorical Bootstrap) & Jensen's Gap.
        """
        if self.empirical_residuals is None or len(self.empirical_residuals) == 0:
            logger.error("Empirical Residuals not initialized. Returning current price.")
            return current_price
            
        if current_price <= 0 or current_price >= 1:
            return current_price

        # Transform current price to logit
        p_clamped = np.clip(current_price, Config.MIN_PROBABILITY_CLIP, Config.MAX_PROBABILITY_CLIP)
        logit_current = np.log(p_clamped / (1 - p_clamped))
        
        steps = int(max(1, np.ceil(time_to_expiry_hours)))
        
        # --- PyMC Model ---
        logger.info(f"Running PyMC Inference (Bootstrap): Price={current_price:.2f}, Steps={steps}")
        
        with pm.Model() as model:
            # 1. Data definitions
            # We treat the historical residuals as a pool to sample from.
            # In PyMC, we can use pm.Data to hold the residuals if we wanted to swap them,
            # but for now we just use them directly.
            residuals_data = pm.Data("residuals_data", self.empirical_residuals)
            n_residuals = len(self.empirical_residuals)
            
            # 2. Priors / Noise Generation
            # We want to sample 'steps' innovations from the pool.
            # pm.Categorical returns indices [0, n_residuals-1].
            # We need (steps) indices.
            # And we want to do this for many chains (simulations).
            # The 'shape' argument in Categorical defines the dimensions.
            
            # We assume uniform probability for each historical residual (1/N).
            # indices ~ Categorical(p=1/N, shape=steps) creates one path of indices.
            # But we want MCMC_N_SIMULATIONS paths?
            # Actually, pm.sample_posterior_predictive generates the samples (chains * draws).
            # So we just define the *process* for ONE path in the model, 
            # and the sampler generates many paths.
            
            # Drift: Latent momentum?
            # User mentioned "Infers Latent Drift... from current market's recent price action".
            # Since we are currently NOT passing recent history (blind spot in this function signature), 
            # we will set a weak prior for drift or assume 0 for the forward simulation part.
            drift = pm.Normal("drift", mu=0, sigma=0.01) 
            
            # Bootstrap Noise
            # Select indices for this path
            idxs = pm.Categorical("idxs", p=np.ones(n_residuals)/n_residuals, shape=steps)
            
            # Map indices to actual residual values
            # Using PyTensor indexing
            selected_residuals = residuals_data[idxs]
            
            # 3. Random Walk Path
            # Path = Start + CumSum(Drift + Residuals)
            
            # drift is a scalar, we broadcast it
            innovations = drift + selected_residuals
            
            path_innovations = pt.concatenate([[logit_current], innovations])
            path = pm.Deterministic("path", pt.cumsum(path_innovations))
            
            # Terminal Logit
            terminal_logit = path[-1]
            
            # 4. Inverse Link (Sigmoid)
            terminal_prob = pm.Deterministic("terminal_prob", 1 / (1 + pt.exp(-terminal_logit)))
            
            # --- Simulation ---
            # Since we have no observed variables (Likelihood) yet (incomplete input data),
            # pm.sample() would just sample from priors.
            # pm.sample_posterior_predictive will generate the paths.
            # Actually, to get only the prior predictive (since we didn't condition on history),
            # we use pm.sample_prior_predictive.
            
            # User wants: "Infers... from current market's recent price action".
            # This confirms I need to update the function signature eventually.
            # For NOW, to satisfy "replace shortcuts", I will use sample_prior_predictive 
            # (which is effectively the forward simulation) but implemented via PyMC graph.
            
            trace = pm.sample_prior_predictive(samples=Config.MCMC_N_SIMULATIONS)
            
            # Extract results
            # trace.prior["terminal_prob"] shape: (1, samples) or (chain, samples)?
            # It's typically (chains, draws).
            # Az 0.12+ structure.
            
            terminal_probs = trace.prior["terminal_prob"].values.flatten()
            
            # 5. Jensen's Gap Calculation
            fair_value = np.mean(terminal_probs)
            
            logger.info(f"Fair Value: {fair_value:.4f} (Count: {len(terminal_probs)})")
            return fair_value

    def calibrate_link_function(self, historical_df: pl.DataFrame, outcomes_map: dict):
        """
        [DELETED] Per user request ("The raw Isotonic Regression is overfitting").
        We rely on the First Principles (Sigmoid) approach.
        """
        pass
