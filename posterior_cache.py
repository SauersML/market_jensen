"""
Dynamic Posterior Cache with Statistical Validity (Phase 2C)

Replaces time-based cache expiration with statistical divergence detection.
Re-inference triggered when new prices fall outside 95% HDI of projection.
"""

import numpy as np
import logging
from typing import Optional, Tuple
import time

logger = logging.getLogger(__name__)


class PosteriorCache:
    """
    Intelligent cache that invalidates based on statistical divergence,
    not arbitrary time limits.
    """
    
    def __init__(self, trace, projected_samples: np.ndarray, 
                 projection_timestamps: np.ndarray, creation_time: float):
        """
        Args:
            trace: arviz.InferenceData from inference
            projected_samples: Forward simulation samples [n_samples, n_timesteps]
            projection_timestamps: Unix timestamps for each projection step
            creation_time: When this cache was created
        """
        self.trace = trace
        self.projected_samples = projected_samples
        self.projection_timestamps = projection_timestamps
        self.creation_time = creation_time
        
    def is_valid(self, new_price: float, current_time: float, 
                 hdi_alpha: float = 0.05) -> Tuple[bool, Optional[str]]:
        """
        Check if new price is within HDI of forward projection.
        
        Args:
            new_price: Newly observed market price
            current_time: Current unix timestamp
            hdi_alpha: HDI level (0.05 = 95% interval)
            
        Returns:
            (is_valid, reason_if_invalid)
        """
        # Find closest projection timestamp
        time_idx = self._get_time_index(current_time)
        
        if time_idx is None:
            return False, "Projection expired (beyond forecast horizon)"
        
        # Get projected distribution at this time
        projected_prices_at_t = self.projected_samples[:, time_idx]
        
        # Calculate HDI
        hdi_lower = np.percentile(projected_prices_at_t, hdi_alpha / 2 * 100)
        hdi_upper = np.percentile(projected_prices_at_t, (1 - hdi_alpha / 2) * 100)
        
        # Check if new price is within HDI
        if new_price < hdi_lower or new_price > hdi_upper:
            distance_lower = max(0, hdi_lower - new_price)
            distance_upper = max(0, new_price - hdi_upper)
            total_distance = distance_lower + distance_upper
            
            logger.warning(
                f"Statistical divergence detected: "
                f"price={new_price:.4f} outside HDI=[{hdi_lower:.4f}, {hdi_upper:.4f}], "
                f"distance={total_distance:.4f}"
            )
            
            return False, f"Price outside 95% HDI by {total_distance:.4f}"
        
        return True, None
    
    def _get_time_index(self, current_time: float) -> Optional[int]:
        """Find the projection timestep closest to current_time"""
        if len(self.projection_timestamps) == 0:
            return None
        
        # Find closest timestamp
        time_diffs = np.abs(self.projection_timestamps - current_time)
        closest_idx = np.argmin(time_diffs)
        
        # If more than 10 minutes away from any projection point, invalid
        if time_diffs[closest_idx] > 600:
            return None
        
        return int(closest_idx)
    
    def get_median_projection(self, current_time: float) -> Optional[float]:
        """Get median projected price at current time"""
        time_idx = self._get_time_index(current_time)
        if time_idx is None:
            return None
        
        return float(np.median(self.projected_samples[:, time_idx]))
