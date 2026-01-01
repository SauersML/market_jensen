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

    # API
    DEMO_API_URL = "https://demo-api.kalshi.co"
    PROD_API_URL = "https://api.elections.kalshi.com"
    API_URL = DEMO_API_URL if ENV == Environment.DEMO else PROD_API_URL

    # Scanner Settings
    MIN_DAILY_VOLUME_USD = 1000  # Minimum average daily volume to consider a series
    MIN_HISTORY_EVENTS = 10      # Minimum number of past settled events to calculate volatility

    # Analysis Settings
    MCMC_N_SIMULATIONS = 2000    # Number of paths
    MCMC_TUNE_STEPS = 500
    # Clipping removed per user request (logic handles logits naturally, though we clip to avoid -inf/inf at boundaries of 0/1)
    # We keep a tiny epsilon just for safe logit transform if needed locally.
    MIN_PROBABILITY_CLIP = 0.001 
    MAX_PROBABILITY_CLIP = 0.999
    JENSEN_GAP_THRESHOLD_CENTS = 0.0 # Logic removed in trader, kept for reference or removed.

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
