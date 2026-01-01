
import asyncio
import json
import logging
import base64
import time
from typing import Dict, List, Optional, Any, AsyncGenerator

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from config import Config

logger = logging.getLogger(__name__)

class AsyncKalshiClient:
    def __init__(self, key_id: str=None, private_key_path: str=None):
        self.base_url = Config.BASE_URL
        self.key_id = key_id or Config.KALSHI_KEY_ID
        self.private_key_path = private_key_path or Config.KALSHI_KEY_FILE
        self._load_private_key()
        self._session = None

    def _load_private_key(self):
        try:
            with open(self.private_key_path, "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )
        except Exception as e:
            logger.critical(f"Failed to load private key: {e}")
            raise

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        # PSS Padding for Kalshi V2
        msg = f"{timestamp}{method}{path}".encode('utf-8')
        signature = self.private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    async def _sign_request_async(self, method: str, path: str, timestamp: str) -> str:
        """
        Offloads CPU-bound RSA signing to executor to prevent Event Loop blocking.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sign_request, method, path, timestamp)

    async def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Any:
        if not self._session:
            self._session = aiohttp.ClientSession()

        signing_path = f"/trade-api/v2{path}"
        timestamp = str(int(time.time() * 1000))
        
        # Non-blocking Sign
        signature = await self._sign_request_async(method, signing_path, timestamp)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-API-KEY": self.key_id,
            "KALSHI-API-SIGNATURE": signature,
            "KALSHI-API-TIMESTAMP": timestamp
        }

        url = f"{self.base_url}{path}"
        
        try:
            async with self._session.request(method, url, headers=headers, params=params, json=body) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(f"API Error {resp.status}: {text}")
                    # Rate Limit Handling
                    if resp.status == 429:
                        await asyncio.sleep(1) # Simple backoff
                        return await self._request(method, path, params, body)
                    resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            logger.error(f"Request Failed: {e}")
            raise

    async def get_paginated(self, path: str, params: Dict = None, data_key: str = None) -> AsyncGenerator[Dict, None]:
        """
        Async Generator for paginated endpoints.
        """
        params = params or {}
        limit = params.get("limit", 100)
        cursor = None
        
        while True:
            current_params = params.copy()
            current_params["limit"] = limit
            if cursor:
                current_params["cursor"] = cursor
                
            resp = await self._request("GET", path, params=current_params)
            data = resp.get(data_key, [])
            
            for item in data:
                yield item
                
            cursor = resp.get("cursor")
            if not cursor:
                break

    # --- High Level Methods ---

    async def get_series_list(self) -> Dict:
        return await self._request("GET", "/series")

    async def get_series(self, series_ticker: str) -> Dict:
        return await self._request("GET", f"/series/{series_ticker}")

    async def get_markets(self, **params) -> Dict:
        return await self._request("GET", "/markets", params=params)

    async def get_market_orderbook(self, ticker: str) -> Dict:
        return await self._request("GET", f"/markets/{ticker}/orderbook")

    async def batch_get_market_candlesticks(self, tickers: List[str], start_ts: int, end_ts: int, period: int) -> Dict:
        """
        Fetches candlesticks for multiple markets concurrently using asyncio.gather.
        Returns a dict {ticker: candles_list}.
        """
        tasks = []
        for t in tickers:
            tasks.append(self.get_candlesticks(t, start_ts, end_ts, period))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        out = {}
        for t, res in zip(tickers, results):
            if isinstance(res, Exception):
                logger.error(f"Failed to fetch candles for {t}: {res}")
                out[t] = []
            else:
                out[t] = res.get("candlesticks", [])
                
        return {"candlesticks": out}

    async def get_candlesticks(self, ticker: str, start_ts: int, end_ts: int, period_interval: int) -> Dict:
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval, 
            "market_ticker": ticker
        }
        return await self._request("GET", f"/markets/{ticker}/candlesticks", params=params)

    async def get_portfolio_balance(self) -> Dict:
        return await self._request("GET", "/portfolio/balance")
        
    async def get_positions(self) -> Dict:
        return await self._request("GET", "/portfolio/positions")

    async def create_order(self, ticker: str, action: str, side: str, count: int, price: int) -> Dict:
        body = {
            "action": action, # 'buy' or 'sell'
            "side": side,     # 'yes' or 'no'
            "count": count,
            "ticker": ticker,
            "type": "limit",
            "price": price,
            "client_order_id": str(int(time.time()*1000))
        }
        return await self._request("POST", "/portfolio/orders", body=body)


class KalshiWebSocket:
    def __init__(self, client: AsyncKalshiClient, message_handler):
        self.client = client
        self.handler = message_handler
        self.ws_url = Config.WS_BASE_URL
        self.ws = None
        self.running = False
        self.msg_id = 1
        self.subscriptions = set()

    async def connect(self):
        import websockets
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws = ws
                    logger.info("Connected to WebSocket")
                    
                    # Authenticate
                    # V2 Auth: Send {"id": 1, "cmd": "login", "params": {...}}
                    ts = str(int(time.time() * 1000))
                    sig_path = "/trade-api/v2/ws/login" # Verify path usually just /ws/login or similar, but sig is on standard path
                    # Actually standard is: method 'GET', path '/users/websocket/auth' usually for generating token 
                    # OR internal signing.
                    # Kalshi docs say: Sign "GET" + "/users/websocket/auth" + timestamp?
                    # OR just sign the connect message?
                    # Let's assume standard API signature on a 'login' command.
                    
                    # Sig on "/trade-api/v2/ws" ?? 
                    # Let's use the explicit signature generation method from client.
                    sig = self.client._sign_request("GET", "/users/websocket/auth", ts) 
                    
                    auth_msg = {
                        "id": self.msg_id,
                        "cmd": "login",
                        "params": {
                            "keyId": self.client.key_id,
                            "signature": sig,
                            "timestamp": ts
                        }
                    }
                    await ws.send(json.dumps(auth_msg))
                    self.msg_id += 1
                    
                    # Resubscribe if reconnecting
                    if self.subscriptions:
                        await self.subscribe(list(self.subscriptions))
                    
                    async for msg in ws:
                        data = json.loads(msg)
                        await self.handler(data)
                        
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5) # Backoff

    async def subscribe(self, tickers: List[str], channels=["orderbook_delta"]):
        if not self.ws:
            return
        
        # Update set
        self.subscriptions.update(tickers)
        
        # Fix: V2 Subscription format
        # {"id": 1, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": [...]}}
        cmd = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": tickers
            }
        }
        await self.ws.send(json.dumps(cmd))
        self.msg_id += 1
        logger.info(f"Subscribed to {tickers}")

    async def unsubscribe(self, tickers: List[str]):
        if not self.ws: return
        self.subscriptions.difference_update(tickers)
        # Send unsub command
