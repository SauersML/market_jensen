
import asyncio
import logging
from config import Config
from trader import Trader

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_jensen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Market Jensen Bot (Async)...")
    
    # Initialize Trader
    trader = Trader()
    
    # Run
    await trader.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot Stopped by User.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}", exc_info=True)
