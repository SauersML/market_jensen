import logging
import sys
import os
from trader import Trader
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOGS_DIR / "system.log")
    ]
)

logger = logging.getLogger("Main")

def main():
    logger.info("Starting Jensen's Inequality Arbitrage System")

    try:
        Config.validate()
        trader = Trader()
        trader.run_loop()
    except Exception as e:
        logger.critical(f"System crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
