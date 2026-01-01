"""
Hierarchical Data Layer for Series-Specific Bayesian Inference

This module implements a 3-level hierarchy:
- Level 1 (Global): Cross-series baseline volatility
- Level 2 (Series): Event-specific noise profile (e.g., Jobless Claims)
- Level 3 (Market): Current contract observations

No time windows - all historical data ingested with learned decay weighting.
"""

import numpy as np
import polars as pl
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SeriesHyperpriors:
    """Learned hyperparameters for a series"""
    log_vol_mu: float  # Mean log-volatility for this series
    log_vol_sigma: float  # Std of log-volatility
    drift_mu: float  # Mean drift
    drift_sigma: float  # Std of drift
    persistence: float  # Information decay rate (learned from autocorrelation)
    n_markets: int  # Number of markets used to learn these priors


class SeriesData:
    """
    Hierarchical data container for a recurring event series.
    
    Maintains Library of Noise (empirical residuals) and learned hyperpriors.
    No time windows - uses ALL available historical data with exponential decay.
    """
    
    def __init__(self, series_ticker: str, global_priors: Optional[Dict] = None):
        self.series_ticker = series_ticker
        self.library_of_noise = None  # np.array of standardized residuals
        self.hyperpriors = None  # SeriesHyperpriors object
        self.market_data = {}  # {market_ticker: DataFrame of observations}
        
        # Global priors (cross-series baseline)
        self.global_priors = global_priors or {
            "log_vol_mu": -3.0,
            "log_vol_sigma": 1.5,
            "drift_mu": 0.0,
            "drift_sigma": 0.1,
        }
    
    async def update_library(self, client, min_markets: int = 3):
        """
        Build Library of Noise from ALL settled markets in series.
        Learn persistence parameter from autocorrelation of returns.
        
        Args:
            client: AsyncKalshiClient for API calls
            min_markets: Minimum markets required (raises ValueError if insufficient)
        """
        logger.info(f"Building Library of Noise for {self.series_ticker}...")
        
        # Fetch ALL settled markets (no time window)
        all_markets = []
        async for market in client.get_paginated(
            "/markets",
            params={"series_ticker": self.series_ticker, "status": "settled"},
            data_key="markets"
        ):
            all_markets.append(market)
        
        if len(all_markets) < min_markets:
            raise ValueError(
                f"Insufficient historical data for {self.series_ticker}: "
                f"found {len(all_markets)} markets, need at least {min_markets}"
            )
        
        logger.info(f"Found {len(all_markets)} settled markets for {self.series_ticker}")
        
        # Extract standardized residuals from each market
        all_residuals = []
        all_log_vols = []
        all_drifts = []
        returns_for_persistence = []  # For learning decay rate
        
        for market in all_markets:
            try:
                residuals, log_vol, drift, returns = await self._extract_market_residuals(
                    client, market
                )
                
                if residuals is not None and len(residuals) > 0:
                    all_residuals.extend(residuals)
                    all_log_vols.append(log_vol)
                    all_drifts.append(drift)
                    returns_for_persistence.extend(returns)
                    
            except Exception as e:
                logger.warning(f"Failed to process {market['ticker']}: {e}")
                continue
        
        if len(all_residuals) == 0:
            raise ValueError(
                f"Failed to extract any residuals from {self.series_ticker} history"
            )
        
        # Build Library of Noise (ECDF of standardized residuals)
        self.library_of_noise = np.array(all_residuals)
        
        # Learn persistence from autocorrelation of pooled returns
        persistence = self._learn_persistence(returns_for_persistence)
        
        # Compute hyperpriors
        self.hyperpriors = SeriesHyperpriors(
            log_vol_mu=float(np.mean(all_log_vols)),
            log_vol_sigma=float(np.std(all_log_vols)),
            drift_mu=float(np.mean(all_drifts)),
            drift_sigma=float(np.std(all_drifts)),
            persistence=persistence,
            n_markets=len([x for x in [all_log_vols] if x])
        )
        
        logger.info(
            f"Library of Noise built: {len(all_residuals)} residuals, "
            f"persistence={persistence:.3f}, "
            f"volatility_mu={np.exp(self.hyperpriors.log_vol_mu):.4f}"
        )
        
        # Log distribution characteristics
        logger.info(f"  Skewness: {stats.skew(self.library_of_noise):.3f}")
        logger.info(f"  Kurtosis: {stats.kurtosis(self.library_of_noise):.2f} (>3 = fat tails)")
        logger.info(f"  [1%, 99%]: [{np.percentile(self.library_of_noise, 1):.3f}, "
                   f"{np.percentile(self.library_of_noise, 99):.3f}]")
    
    async def _extract_market_residuals(
        self, client, market_info: Dict
    ) -> Tuple[Optional[List[float]], Optional[float], Optional[float], List[float]]:
        """
        Extract standardized residuals from a single market.
        
        Returns:
            (residuals, log_volatility, drift, raw_returns)
        """
        from dateutil import parser
        from datetime import timezone
        
        ticker = market_info['ticker']
        
        # Get market timeframe
        try:
            open_time = parser.isoparse(market_info['open_time'])
            close_time = parser.isoparse(market_info.get('close_time') or market_info.get('expiration_time'))
            
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
                
            start_ts = int(open_time.timestamp())
            end_ts = int(close_time.timestamp())
            
        except Exception as e:
            logger.warning(f"Failed to parse times for {ticker}: {e}")
            return None, None, None, []
        
        # Fetch candlesticks
        try:
            c_resp = await client.get_candlesticks(ticker, start_ts, end_ts, period_interval=60)
            candles = c_resp.get("candlesticks", [])
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {ticker}: {e}")
            return None, None, None, []
        
        if len(candles) < 3:
            return None, None, None, []
        
        # Convert to log-odds and compute returns
        prices = np.array([c['c'] / 100.0 for c in candles])
        epsilon = 0.001
        prices = np.clip(prices, epsilon, 1 - epsilon)
        log_odds = np.log(prices / (1 - prices))
        
        returns = np.diff(log_odds)
        
        if len(returns) < 2:
            return None, None, None, []
        
        # Calculate local volatility using EWMA
        alpha = 0.1  # Conservative decay
        var_ewma = np.var(returns[:min(10, len(returns))])
        
        local_vols = []
        for i, ret in enumerate(returns):
            if i > 0:
                var_ewma = alpha * (ret ** 2) + (1 - alpha) * var_ewma
            local_vols.append(np.sqrt(max(var_ewma, 1e-8)))
        
        local_vols = np.array(local_vols)
        
        # Standardize residuals by local volatility
        drift = np.mean(returns)
        standardized = (returns - drift) / (local_vols + 1e-8)
        
        # Filter infinities
        valid = np.isfinite(standardized)
        standardized = standardized[valid]
        
        # Market-level stats
        log_vol = np.log(np.mean(local_vols) + 1e-8)
        
        return standardized.tolist(), log_vol, drift, returns.tolist()
    
    def _learn_persistence(self, returns: List[float], max_lag: int = 20) -> float:
        """
        Learn information persistence from autocorrelation of returns.
        
        persistence = 1 / half-life where half-life is the lag at which ACF = 0.5
        Higher persistence → information decays slower
        
        Args:
            returns: Pooled returns from all markets
            max_lag: Maximum lag to check
            
        Returns:
            persistence parameter (0 to 1)
        """
        if len(returns) < 50:
            # Default to moderate persistence if insufficient data
            logger.warning("Insufficient data for persistence learning, using default")
            return 0.5
        
        returns_arr = np.array(returns)
        returns_arr = returns_arr[np.isfinite(returns_arr)]
        
        # Calculate autocorrelation function
        mean = np.mean(returns_arr)
        var = np.var(returns_arr)
        
        if var < 1e-10:
            return 0.5
        
        acf = []
        for lag in range(1, min(max_lag + 1, len(returns_arr) // 4)):
            cov = np.mean((returns_arr[:-lag] - mean) * (returns_arr[lag:] - mean))
            acf.append(cov / var)
        
        # Find half-life (lag where ACF ~ 0.5)
        acf_arr = np.array(acf)
        
        # If ACF doesn't cross 0.5, use exponential fit
        try:
            # Fit exponential decay: ACF(lag) = exp(-lag / tau)
            # persistence = 1 / tau
            lags = np.arange(1, len(acf) + 1)
            
            # Robust fit: only use positive ACF values
            pos_mask = acf_arr > 0
            if np.sum(pos_mask) < 3:
                return 0.5
            
            log_acf = np.log(acf_arr[pos_mask] + 1e-10)
            valid_lags = lags[pos_mask]
            
            # Linear fit in log space
            slope, _ = np.polyfit(valid_lags, log_acf, 1)
            tau = -1.0 / slope if slope < 0 else 10.0
            
            # Convert to persistence (bounded to [0.1, 0.9])
            persistence = 1.0 / (tau + 1.0)
            persistence = np.clip(persistence, 0.1, 0.9)
            
            logger.info(f"Learned persistence={persistence:.3f} (tau={tau:.1f} lags)")
            return float(persistence)
            
        except Exception as e:
            logger.warning(f"Persistence learning failed: {e}, using default")
            return 0.5
    
    def get_weights_for_observations(self, timestamps: np.ndarray, current_time: float) -> np.ndarray:
        """
        Calculate exponential decay weights for observations based on age.
        Uses learned persistence parameter instead of arbitrary time windows.
        
        Args:
            timestamps: Unix timestamps of observations
            current_time: Current unix timestamp
            
        Returns:
            Array of weights (same length as timestamps)
        """
        if self.hyperpriors is None:
            # No learned persistence yet, use moderate decay
            persistence = 0.5
        else:
            persistence = self.hyperpriors.persistence
        
        # Age in hours
        ages = (current_time - timestamps) / 3600.0
        
        # Exponential decay: weight = exp(-age / half_life)
        # half_life related to persistence
        half_life = 24.0 * (1.0 / persistence)  # Higher persistence → longer half-life
        
        weights = np.exp(-ages / half_life)
        
        # Normalize to sum to 1
        return weights / np.sum(weights)
