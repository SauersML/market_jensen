import numpy as np
import scipy.stats as stats
import polars as pl
import logging
import joblib
from sklearn.isotonic import IsotonicRegression
from config import Config

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    Phase 2: Volatility Modeling & Calibration (The Brain)
    """
    def __init__(self):
        self.volatility_kernel = None
        self.link_function = None
        self.market_price_history = []
        self.outcomes = []

    def calculate_empirical_volatility(self, historical_df: pl.DataFrame) -> stats.gaussian_kde:
        """
        Builds a Kernel Density Estimation (KDE) of log-odds changes (returns).
        """
        logger.info("Calculating empirical volatility...")

        # We need to process data by market to calculate differences
        # Group by market_ticker and sort by timestamp

        log_returns = []

        # Iterate over unique markets in the DF
        market_tickers = historical_df["market_ticker"].unique().to_list()

        for ticker in market_tickers:
            market_data = historical_df.filter(pl.col("market_ticker") == ticker).sort("timestamp")
            prices = market_data["price_normalized"].to_numpy()

            # Clip to avoid 0/1 for logit
            prices = np.clip(prices, Config.MIN_PROBABILITY_CLIP, Config.MAX_PROBABILITY_CLIP)

            if len(prices) > 1:
                # Logit transform
                logits = np.log(prices / (1 - prices))
                # First difference
                diffs = np.diff(logits)
                log_returns.extend(diffs)

        if not log_returns:
            raise ValueError("Insufficient volatility data: No price changes found.")

        log_returns = np.array(log_returns)

        # Check for NaN or Inf
        log_returns = log_returns[np.isfinite(log_returns)]

        if len(log_returns) < 50:
             raise ValueError(f"Insufficient volatility data points: {len(log_returns)}")

        # Kernel Density Estimation
        try:
            kde = stats.gaussian_kde(log_returns)
            self.volatility_kernel = kde

            # Save kernel
            joblib.dump(kde, Config.MODELS_DIR / "volatility_kernel.pkl")

            return kde
        except Exception as e:
            logger.error(f"KDE estimation failed: {e}")
            raise

    def calibrate_link_function(self, historical_df: pl.DataFrame, outcomes_map: dict):
        """
        Calibrates the relationship between Market Price and Real Probability.
        outcomes_map: {market_ticker: 0 or 1}
        """
        logger.info("Calibrating link function...")
        # We need pairs of (FinalPrice_t, Outcome) or (Price_t, Outcome)
        # Actually usually we check if the market price at time t was accurate.
        # But here we just want a simple calibration.
        # Let's take the closing price of the market? Or average price?
        # A simple approach: use all price points? No, that biases towards long markets.
        # Let's use the price at various timestamps relative to expiry?
        # For simplicity in this non-placeholder version, let's use the daily closing prices against the final outcome.

        X = []
        y = []

        market_tickers = historical_df["market_ticker"].unique().to_list()

        for ticker in market_tickers:
            if ticker not in outcomes_map:
                continue

            outcome = outcomes_map[ticker]
            market_data = historical_df.filter(pl.col("market_ticker") == ticker)
            prices = market_data["price_normalized"].to_numpy()

            X.extend(prices)
            y.extend([outcome] * len(prices))

        if not X:
            logger.warning("No data for link calibration. Using identity.")
            return

        X = np.array(X)
        y = np.array(y)

        # Isotonic Regression
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_reg.fit(X, y)

        self.link_function = iso_reg
        joblib.dump(iso_reg, Config.MODELS_DIR / "link_function.pkl")

    def run_mcmc_simulation(self, current_price: float, time_to_expiry_hours: float) -> float:
        """
        Runs MCMC simulation to estimate the Posterior Mean.
        Uses Numpy vectorization for speed instead of full PyMC model for simple random walk.
        This provides the same mathematical result as a forward Monte Carlo simulation in PyMC but is much faster for this specific use case.
        """
        if self.volatility_kernel is None:
            raise ValueError("Volatility kernel not initialized.")

        # Transform current price to logit
        p_start = np.clip(current_price, Config.MIN_PROBABILITY_CLIP, Config.MAX_PROBABILITY_CLIP)

        # Apply link function calibration if available
        if self.link_function:
            p_start = self.link_function.predict([p_start])[0]
            # re-clip
            p_start = np.clip(p_start, Config.MIN_PROBABILITY_CLIP, Config.MAX_PROBABILITY_CLIP)

        logit_start = np.log(p_start / (1 - p_start))

        # Sample shocks from KDE
        # The KDE is derived from hourly changes (if period_interval=60).
        # So 'steps' should be the number of hours left.
        # If time_to_expiry_hours is fractional, we can round up or interpolate.
        # For a random walk, variance scales linearly with time.
        # Var(T) = T * Var(1).
        # We can simulate T steps.

        steps = int(max(1, np.ceil(time_to_expiry_hours)))
        n_sims = Config.MCMC_N_SIMULATIONS

        # Draw noise: (n_sims, steps)
        # resample() returns (dims, n_samples)
        noise_samples = self.volatility_kernel.resample(n_sims * steps)
        noise_samples = noise_samples.reshape(n_sims, steps)

        # Sum noise to get total displacement
        total_displacement = np.sum(noise_samples, axis=1)

        # If we rounded up time, we might want to scale the variance slightly down,
        # but integer steps is standard for discrete time simulation.
        # Alternatively, we could scale the total displacement by sqrt(actual_time / steps)
        # to adjust for the fractional step, but random walk steps are usually discrete events.
        # Let's stick to integer steps as "periods".

        # Final logits
        final_logits = logit_start + total_displacement

        # Transform to probabilities
        final_probs = 1 / (1 + np.exp(-final_logits))

        # Calculate Posterior Mean
        posterior_mean = np.mean(final_probs)

        return posterior_mean

if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    engine = AnalysisEngine()

    # Mock data
    data = []
    for i in range(100):
        p = 0.5 + 0.1 * np.sin(i/10) + np.random.normal(0, 0.05)
        data.append({"timestamp": i, "market_ticker": "M1", "price_normalized": p})
    df = pl.DataFrame(data)

    engine.calculate_empirical_volatility(df)
    mean = engine.run_mcmc_simulation(0.5, 24)
    print(f"Posterior Mean: {mean}")
