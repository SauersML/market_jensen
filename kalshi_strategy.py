import os
import sys
import numpy as np
import scipy.stats as stats
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from cryptography.hazmat.primitives import serialization
from kalshi_client import KalshiHttpClient, Environment

# Configuration
# Assuming environment variables are set or we will mock them for this implementation logic
# For the purpose of "Fully implement the idea", I will assume keys are present or I will
# provide instructions on how to set them.

class KalshiStrategy:
    def __init__(self, key_id: str, private_key_path: str, environment: str = "demo"):
        try:
            with open(private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )
            self.client = KalshiHttpClient(
                key_id=key_id,
                private_key=private_key,
                environment=Environment.DEMO if environment.lower() == "demo" else Environment.PROD
            )
            print("Client initialized.")
        except Exception as e:
            print(f"Failed to initialize client: {e}")
            sys.exit(1)

    def get_recurring_series(self) -> str:
        """
        Identify a recurring series.
        Finds a series with daily frequency to ensure enough data points.
        Defaults to "KXHIGHNY" if no suitable series is found or on error.
        """
        try:
            print("Fetching series list to identify recurring events...")
            series_resp = self.client.get_series_list()
            series_list = series_resp.get("series", [])

            # Look for a daily series
            for s in series_list:
                # Check for daily frequency in metadata or title/frequency field
                freq = s.get("frequency", "").lower()
                if "daily" in freq or "day" in freq:
                    print(f"Found recurring series: {s.get('ticker')} ({freq})")
                    return s.get("ticker")

        except Exception as e:
            print(f"Error fetching series list: {e}")

        # Fallback to a known high-frequency series
        target_series = "KXHIGHNY"
        print(f"Targeting default series: {target_series}")
        return target_series

    def get_historical_volatility(self, series_ticker: str) -> float:
        """
        Retrieve Historical Volatility (The Input).
        Get past events for the series, get candlesticks, calculate variance/std dev.
        """
        print("Retrieving historical data...")

        # 1. Get past events
        # We want "settled" events to see final outcomes?
        # Actually the prompt says: "By pulling historical candlestick data... we can calculate the standard deviation or variance... noise or uncertainty that surrounds any given forecast."
        # We want to measure the volatility of the *price* during the trading period, not just the outcome.
        # So we need settled events or closed events.

        events_response = self.client.get_events(series_ticker=series_ticker, status="settled", limit=10) # Get last 10 events
        events = events_response.get("events", [])

        if not events:
            print("No historical events found. Using default volatility.")
            return 0.5 # Default high volatility if no history

        market_tickers = []
        for event in events:
             # Assuming single market per event or we pick the first one
             # The prompt says "batch request candlestick data for the past instances"
             # We need market tickers.
             # get_events returns events, but we need markets.
             # events endpoint has "with_nested_markets" param but we didn't use it in client wrapper default.
             # Let's use the markets endpoint filtered by event ticker? Or just request events with nested markets.
             # I'll use get_markets for each event or just assume I can find markets.

             # Actually, let's fetch events with nested markets if possible.
             # My client `get_events` passes kwargs to params? No.
             # I'll just iterate and fetch markets for these events.
             markets_resp = self.client.get_markets(event_ticker=event['event_ticker'])
             markets = markets_resp.get("markets", [])
             if markets:
                 market_tickers.append(markets[0]['ticker'])

        if not market_tickers:
            print("No historical markets found.")
            return 0.5

        # Batch get candlesticks
        # We need a time range. Let's say we want the last 5 days of trading for each market?
        # But wait, batch request takes a single start/end_ts.
        # If the markets are from different times (which they are), we can't easily batch them in one request
        # unless we ask for a huge range covering all of them.
        # But "candlesticks" endpoint returns data by market_id.
        # If we ask for a wide range, we might get it.
        # Let's find the min start_ts and max end_ts.

        # For simplicity in this implementation, I will iterate and fetch candlesticks for each market
        # (ignoring the "critical for efficiency" batch advice for a moment if the time ranges are disjoint and wide,
        #  or I will use the batch endpoint with a range covering all events if they are recent enough).
        # Assuming these are daily events, last 10 days is fine.

        end_ts = int(time.time())
        start_ts = end_ts - (30 * 24 * 60 * 60) # Last 30 days

        # Batch limit is 100 tickers. We have max 10.
        # Note: The API doc says "Returns up to 10,000 candlesticks total".
        # 10 markets * 24 hours * 60 minutes = 14,400 if we do minute data.
        # Let's do hourly data (period_interval=60) to be safe.

        print(f"Fetching candlesticks for {len(market_tickers)} markets...")
        try:
            candles_resp = self.client.batch_get_market_candlesticks(
                market_tickers=market_tickers,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=60
            )
        except Exception as e:
            print(f"Batch request failed: {e}. Falling back to default.")
            return 0.5

        all_markets_data = candles_resp.get("markets", [])

        # Calculate volatility
        # We look at the log-odds changes or just price changes.
        # Prompt: "standard deviation or variance... noise... assuming price fluctuations tell us about latent uncertainty."
        # We will calculate the std dev of the changes in Logit(Price) over time.

        log_returns = []

        for m_data in all_markets_data:
            candles = m_data.get("candlesticks", [])
            prices = []
            for c in candles:
                # price.close is in cents.
                close_price = c.get("price", {}).get("close")
                if close_price:
                    p = close_price / 100.0
                    # Clip to avoid infinity
                    p = max(0.01, min(0.99, p))
                    prices.append(p)

            if len(prices) > 1:
                # Convert to log odds: log(p / (1-p))
                log_odds = np.log(np.array(prices) / (1 - np.array(prices)))
                # Calculate differences (daily/hourly fluctuations)
                diffs = np.diff(log_odds)
                log_returns.extend(diffs)

        if not log_returns:
            print("No sufficient price data to calc volatility.")
            return 0.5

        volatility = np.std(log_returns)
        print(f"Calculated Historical Volatility (std dev of log-odds changes): {volatility:.4f}")
        return volatility

    def simulate_fair_value(self, current_price: float, volatility: float, n_simulations: int = 5000) -> float:
        """
        MCMC/NUTS simulation (simplified to Monte Carlo for this context).
        "We simulate the event thousands of times, adding the calculated random noise to the current forecast in every iteration."
        "By averaging the results... we arrive at the 'Posterior Mean'."
        """
        # Current forecast (market consensus)
        p_market = current_price

        # Convert to log-odds space (Linear Predictor)
        # Avoid p=0 or p=1
        p_market = max(0.01, min(0.99, p_market))
        logit_market = np.log(p_market / (1 - p_market))

        # Simulate: Add noise to the logit
        # "adding the calculated random noise to the current forecast"
        # We assume the noise is normally distributed with sigma = calculated volatility
        # We are simulating the "true" probability distribution.

        # Generate samples from the distribution of latent true log-odds
        # We assume the current price is the mode or mean of the latent process,
        # but there is uncertainty around it.
        noise = np.random.normal(loc=0.0, scale=volatility, size=n_simulations)
        simulated_logits = logit_market + noise

        # Convert back to probability space (Sigmoid)
        simulated_probs = 1 / (1 + np.exp(-simulated_logits))

        # "By averaging the results of these thousands of scenarios, we arrive at the 'Posterior Mean'."
        posterior_mean = np.mean(simulated_probs)

        return posterior_mean

    def run(self):
        series_ticker = self.get_recurring_series()

        # 1. Get Volatility Input
        volatility = self.get_historical_volatility(series_ticker)

        # 2. Get Active Target Market
        print("Fetching active markets...")
        markets_resp = self.client.get_markets(series_ticker=series_ticker, status="open")
        active_markets = markets_resp.get("markets", [])

        if not active_markets:
            print("No active markets found for this series.")
            return

        print(f"Found {len(active_markets)} active markets.")

        for market in active_markets:
            ticker = market['ticker']
            print(f"\nAnalyzing Market: {ticker}")

            # Get Orderbook for live price
            ob_resp = self.client.get_market_orderbook(ticker)
            ob = ob_resp.get("orderbook", {})
            yes_bids = ob.get("yes", [])
            no_bids = ob.get("no", [])

            # Determine current market price (midpoint or best bid)
            # If we want to sell, we hit the bid. If we want to buy, we lift the ask (which is 100 - no_bid).
            # Let's take the "Mark Price" as roughly the midpoint for fair value comparison.

            best_yes_bid = yes_bids[0][0] if yes_bids else 0
            best_no_bid = no_bids[0][0] if no_bids else 0

            if best_yes_bid == 0 and best_no_bid == 0:
                print("Market has no liquidity.")
                continue

            # Best Ask for Yes is (100 - Best No Bid)
            best_yes_ask = 100 - best_no_bid

            current_mid_price_cents = (best_yes_bid + best_yes_ask) / 2
            current_price = current_mid_price_cents / 100.0

            print(f"Current Market Price: {current_mid_price_cents:.1f} cents ({current_price:.3f})")
            print(f"  Bid: {best_yes_bid}, Ask: {best_yes_ask}")

            # 3. Simulate
            fair_value = self.simulate_fair_value(current_price, volatility)
            fair_value_cents = fair_value * 100

            print(f"Simulated Fair Value (Posterior Mean): {fair_value_cents:.2f} cents")

            # 4. Trading Logic
            # "If our simulation calculates a fair value of 82 cents but the market is trading at 90 cents... we sell."
            # "If our simulation says the fair value is 12 cents but the market is trading at 5 cents... we buy."

            # Define a margin of safety/threshold
            threshold = 2.0 # cents

            if fair_value_cents > best_yes_ask + threshold:
                print(f"SIGNAL: BUY YES (Fair: {fair_value_cents:.2f} > Ask: {best_yes_ask})")
                # self.client.create_order(ticker, "yes", "buy", count=1, yes_price=int(best_yes_ask))
            elif fair_value_cents < best_yes_bid - threshold:
                 print(f"SIGNAL: SELL YES (Fair: {fair_value_cents:.2f} < Bid: {best_yes_bid})")
                 # Kalshi "Sell Yes" is technically "Buy No" if you don't have a position,
                 # or you can just "Buy No" which is equivalent to shorting Yes.
                 # The prompt says "we sell".
                 # If we don't have inventory, we "Buy No".
                 print(f"       Action: Buy NO at {best_no_bid} cents (equivalent to selling YES at {100-best_no_bid})")
                 # self.client.create_order(ticker, "no", "buy", count=1, no_price=best_no_bid)
            else:
                print("Signal: HOLD (Price within fair value range)")


if __name__ == "__main__":
    # In a real scenario, the user would provide these via environment variables.
    KEY_ID = os.getenv("KALSHI_KEY_ID")
    KEY_FILE = os.getenv("KALSHI_KEY_FILE")

    if not KEY_ID or not KEY_FILE:
        print("Please set KALSHI_KEY_ID and KALSHI_KEY_FILE environment variables.")
        sys.exit(1)

    strategy = KalshiStrategy(KEY_ID, KEY_FILE, environment="demo")

    # Run the strategy
    try:
        strategy.run()
    except Exception as e:
        print(f"Execution finished with error: {e}")
