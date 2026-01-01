#!/usr/bin/env python3
"""
Quick test script to validate API fixes work with demo environment.
"""
import asyncio
import os
import sys
from pathlib import Path

# Set demo environment variables
os.environ["KALSHI_ENV"] = "demo"
os.environ["KALSHI_KEY_ID"] = "d0891b60-c4d8-4527-82eb-06bec8730e33"
os.environ["KALSHI_KEY_FILE"] = str(Path(__file__).parent / "demo_key.pem")

from kalshi_client import AsyncKalshiClient

async def test_api_connection():
    """Test that the new API URLs and headers work."""
    print("=" * 60)
    print("Testing Kalshi V2 API Connection with Demo Credentials")
    print("=" * 60)
    
    async with AsyncKalshiClient() as client:
        try:
            # Test 1: Get series list
            print("\n[Test 1] Fetching series list...")
            series_resp = await client.get_series_list()
            series = series_resp.get("series", [])
            print(f"✓ Connected! Found {len(series)} series")
            
            if series:
                print(f"  Sample: {series[0].get('ticker', 'N/A')}")
            
            # Test 2: Get active markets
            print("\n[Test 2] Fetching active markets...")
            markets_resp = await client.get_markets(status="open", limit=5)
            markets = markets_resp.get("markets", [])
            print(f"✓ Found {len(markets)} open markets")
            
            if markets:
                sample = markets[0]
                print(f"  Sample: {sample.get('ticker', 'N/A')}")
                
                # Test 3: Get orderbook (with sub-penny check)
                ticker = sample.get('ticker')
                if ticker:
                    print(f"\n[Test 3] Fetching orderbook for {ticker}...")
                    ob_resp = await client.get_market_orderbook(ticker)
                    ob = ob_resp.get("orderbook", {})
                    
                    yes_bids = ob.get("yes", [])
                    if yes_bids:
                        print(f"✓ Orderbook retrieved")
                        # Check if sub-penny data exists
                        if isinstance(yes_bids[0], dict) and "yes_price_dollars" in yes_bids[0]:
                            print("  ✓ Sub-penny pricing available!")
                            print(f"    Best bid: ${yes_bids[0]['yes_price_dollars']}")
                        else:
                            print(f"  Best bid: {yes_bids[0][0] / 100.0:.4f} (cent precision)")
                
                # Test 4: Batch candlesticks
                if len(markets) >= 2:
                    print(f"\n[Test 4] Testing batch candlestick API...")
                    tickers = [m['ticker'] for m in markets[:2]]
                    
                    import time
                    end_ts = int(time.time())
                    start_ts = end_ts - (24 * 3600)  # 24 hours ago
                    
                    batch_resp = await client.batch_get_market_candlesticks(
                        tickers, start_ts, end_ts, 60
                    )
                    candles_dict = batch_resp.get("candlesticks", {})
                    print(f"✓ Batch API returned data for {len(candles_dict)} markets")
                    
                    for t, candles in candles_dict.items():
                        print(f"  {t}: {len(candles)} candles")
            
            print("\n" + "=" * 60)
            print("All API tests passed! ✓")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n✗ API Test Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = asyncio.run(test_api_connection())
    sys.exit(0 if success else 1)
