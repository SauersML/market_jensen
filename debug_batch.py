#!/usr/bin/env python3
"""Debug batch candlestick API response."""
import asyncio
import os
import sys
import json
from pathlib import Path

os.environ["KALSHI_ENV"] = "demo"
os.environ["KALSHI_KEY_ID"] = "d0891b60-c4d8-4527-82eb-06bec8730e33"
os.environ["KALSHI_KEY_FILE"] = str(Path(__file__).parent / "demo_key.pem")

from kalshi_client import AsyncKalshiClient

async def debug_batch_api():
    async with AsyncKalshiClient() as client:
        # Get a few active markets
        markets_resp = await client.get_markets(status="open", limit=3)
        markets = markets_resp.get("markets", [])
        
        if not markets:
            print("No markets found")
            return
        
        tickers = [m['ticker'] for m in markets]
        print(f"Testing with tickers: {tickers}")
        
        import time
        end_ts = int(time.time())
        start_ts = end_ts - (7 * 24 * 3600)  # 7 days ago
        
        # Test the actual batch endpoint
        print(f"\nCalling batch API with:")
        print(f"  Tickers: {tickers}")
        print(f"  Start: {start_ts}")
        print(f"  End: {end_ts}")
        
        try:
            # Raw request to see actual response
            params = {
                "market_tickers": ",".join(tickers),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": 60
            }
            resp = await client._request("GET", "/markets/candlesticks", params=params)
            
            print(f"\nRaw API Response:")
            print(json.dumps(resp, indent=2)[:500])  # First 500 chars
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_batch_api())
