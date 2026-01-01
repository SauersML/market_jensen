import os
from enum import Enum
from pathlib import Path

class Environment(Enum):
    DEMO = "demo"
    PROD = "prod"

class Config:
    # Environment
    ENV = Environment.DEMO if os.getenv("KALSHI_ENV", "demo").lower() == "demo" else Environment.PROD
    KEY_ID = os.getenv("KALSHI_KEY_ID")
    KEY_FILE = os.getenv("KALSHI_KEY_FILE")

    # API URLs (V2)
    DEMO_API_URL = "https://demo-api.kalshi.co/trade-api/v2"
    PROD_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
    API_URL = DEMO_API_URL if ENV == Environment.DEMO else PROD_API_URL

    # Scanner Settings
    MIN_DAILY_VOLUME_USD = 1000  # Minimum average daily volume to consider a series
    MIN_HISTORY_EVENTS = 10      # Minimum number of past settled events to calculate volatility

    # MCMC & Inference Settings
    MCMC_N_SIMULATIONS = 10000
    INFERENCE_WINDOW_HOURS = 24 # Lookback for online inference
    
    # NUTS Sampler Configuration
    NUTS_CHAINS = 4              # Parallel chains for convergence diagnostics
    NUTS_DRAWS = 2000            # Post-warmup samples per chain
    NUTS_TUNE = 1000             # Warmup/tuning samples
    NUTS_TARGET_ACCEPT = 0.90    # Higher = more accurate, slower (0.8-0.95 typical)
    
    # Student-T Distribution Configuration
    STUDENT_T_NU_MIN = 2.0       # Minimum degrees of freedom (fatter tails)
    STUDENT_T_NU_MAX = 30.0      # Maximum degrees of freedom (closer to Gaussian)
    
    # Minimum observations required for full Bayesian inference
    MIN_OBSERVATIONS_FOR_INFERENCE = 10
    
    # Clipping for logit transform to avoid -inf/inf at boundaries
    MIN_PROBABILITY_CLIP = 0.001
    MAX_PROBABILITY_CLIP = 0.999

    # Trading Thresholds
    # Jensen's Gap: Difference between Fair Value and Market Price
    JENSEN_GAP_THRESHOLD_CENTS = 2.0
    
    # Trading Costs & Risk Controls
    TAKER_FEE_RATE = 0.007  # 70 basis points (Kalshi taker fee)
    CONFIDENCE_MULTIPLIER = 1.5  # Conservative safety margin for spread

    # Data & Volatility
    MIN_VOLATILITY_DATA_POINTS = 50
    MIN_DAILY_VOLUME_USD = 1000 # Contracts

    # API & Persistence
    # DEPRECATED: Use API_URL instead (auto-selects based on ENV)
    BASE_URL = API_URL
    WS_BASE_URL = "wss://demo-api.kalshi.co/trade-api/v2/ws" if ENV == Environment.DEMO else "wss://api.elections.kalshi.com/trade-api/v2/ws"
    
    # Batch API Configuration
    BATCH_CANDLESTICK_CHUNK_SIZE = 100  # API limit per request

    # Trading Settings
    MAX_POSITION_SIZE_PERCENT = 0.05
    ORDER_SIZE_CONTRACTS = 1      # Default order size

    # Paths
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models")
    LOGS_DIR = Path("logs")
    DB_PATH = DATA_DIR / "market_data.db"

    @classmethod
    def validate(cls):
        if not cls.KEY_ID or not cls.KEY_FILE:
            raise ValueError("Environment variables KALSHI_KEY_ID and KALSHI_KEY_FILE must be set.")
        if not os.path.exists(cls.KEY_FILE):
             raise FileNotFoundError(f"Key file not found at {cls.KEY_FILE}")

# Ensure directories exist
Config.DATA_DIR.mkdir(exist_ok=True)
Config.MODELS_DIR.mkdir(exist_ok=True)
Config.LOGS_DIR.mkdir(exist_ok=True)
