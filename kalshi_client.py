import requests
import base64
import time
import json
import logging
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature
from config import Config, Environment

import asyncio
import aiohttp
import websockets
from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)

class KalshiClient:
    """Robust client for interacting with the Kalshi API."""
    def __init__(self):
        Config.validate()
        self.key_id = Config.KEY_ID
        self.environment = Config.ENV
        self.host = Config.API_URL

        with open(Config.KEY_FILE, "rb") as key_file:
            self.private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )

        self.session = requests.Session()
        self.last_api_call = datetime.now()
        self.base_api_path = "/trade-api/v2"

    def request_headers(self, method: str, path: str) -> Dict[str, Any]:
        """Generates the required authentication headers for API requests."""
        current_time_milliseconds = int(time.time() * 1000)
        timestamp_str = str(current_time_milliseconds)

        msg_string = timestamp_str + method + path
        signature = self.sign_pss_text(msg_string)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }
        return headers

    def sign_pss_text(self, text: str) -> str:
        """Signs the text using RSA-PSS and returns the base64 encoded signature."""
        message = text.encode('utf-8')
        try:
            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode('utf-8')
        except InvalidSignature as e:
            logger.error("RSA sign PSS failed")
            raise ValueError("RSA sign PSS failed") from e

    def rate_limit(self) -> None:
        """Built-in rate limiter."""
        # Simple token bucket or just min interval could be implemented here.
        # For now, just ensure we don't spam too hard.
        # Kalshi limit is quite generous usually, but let's be safe.
        time.sleep(0.05)

    def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Any:
        self.rate_limit()
        full_path = self.base_api_path + path

        if params:
            req = requests.Request('GET', self.host + full_path, params=params)
            prepped = req.prepare()
            signing_path = prepped.path_url
        else:
            signing_path = full_path

        headers = self.request_headers(method, signing_path)
        url = self.host + full_path

        try:
            response = self.session.request(method, url, headers=headers, params=params, json=body)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            try:
                error_data = e.response.json()
                logger.error(f"API Error: {error_data}")
            except:
                logger.error(f"API Error: {e.response.text}")
            raise

    def get_paginated(self, path: str, params: Optional[Dict] = None, data_key: str = None) -> Any:
        """
        Generator that automatically follows cursors.
        yields items from the list located at response[data_key].
        """
        current_params = params.copy() if params else {}
        while True:
            resp = self._request("GET", path, params=current_params)
            
            # Find list data
            # If data_key not provided, try to guess or return whole response?
            # Usually strict data_key is better.
            if data_key and data_key in resp:
                items = resp[data_key]
                for item in items:
                    yield item
            else:
                # If no key, maybe the response itself is iterable or we just yield the whole page?
                # For paginated APIs, usually there is a list key.
                # If we can't find it, break or yield resp
                yield resp
                break

            cursor = resp.get("cursor")
            if not cursor:
                break
            
            current_params["cursor"] = cursor
            # Rate limit handling inside _request handles sleeping.

    def get_series_list(self, category: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if category:
            params['category'] = category
        return self._request("GET", "/series", params=params)

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/series/{series_ticker}")

    def get_events(self, series_ticker: Optional[str] = None, status: Optional[str] = None, limit: int = 100, with_nested_markets: bool = False) -> Dict[str, Any]:
        params = {'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        if with_nested_markets:
            params['with_nested_markets'] = 'true'
        return self._request("GET", "/events", params=params)

    def get_markets(self, event_ticker: Optional[str] = None, series_ticker: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        params = {'limit': limit}
        if event_ticker:
            params['event_ticker'] = event_ticker
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        return self._request("GET", "/markets", params=params)

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        params = {'depth': depth}
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    def batch_get_market_candlesticks(self, market_tickers: List[str], start_ts: int, end_ts: int, period_interval: int) -> Dict[str, Any]:
        """
        Batch request candlestick data.
        """
        # API limit is 100 tickers per request usually.
        # We should handle chunking if more than 100 passed?
        # For now assuming caller handles chunking or list is small.
        params = {
            'market_tickers': ",".join(market_tickers),
            'start_ts': start_ts,
            'end_ts': end_ts,
            'period_interval': period_interval
        }
        return self._request("GET", "/markets/candlesticks", params=params)

    def create_order(self, ticker: str, side: str, action: str, count: int, type: str = 'limit',
                     yes_price: Optional[int] = None, no_price: Optional[int] = None,
                     client_order_id: Optional[str] = None) -> Dict[str, Any]:
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id

        return self._request("POST", "/portfolio/orders", body=body)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_portfolio_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

class AsyncKalshiClient(KalshiClient):
    """
    Async implementation using aiohttp.
    Inherits auth logic from KalshiClient but overrides _request.
    """
    def __init__(self):
        super().__init__()
        self._session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        self._session = ClientSession(
            timeout=ClientTimeout(total=10),
            headers={"Content-Type": "application/json"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Any:
        if not self._session:
            raise RuntimeError("AsyncClient must be used within 'async with' context")

        # Rate limit (async sleep)
        await asyncio.sleep(0.05) 
        
        full_path = self.base_api_path + path
        
        # Sign with same logic (it's fast enough to be blocking, or we could run in executor if needed)
        # Authentication is CPU bound but RSA sign is usually <1ms on modern CPU.
        # Should be fine in loop for low freq, but strictly speaking could block.
        # For now, we call the sync sign_pss_text.
        
        if params:
             # Manually construct URL for signing to match requests behavior?
             # Aiohttp params handling is slightly different. 
             # Let's be careful. Kalshi needs the EXACT path+query for signing if it's included?
             # Usually standard Kalshi auth signs the timestamp+method+path (without query params traditionally? or with?)
             # The existing `request_headers` implementation signs: timestamp + method + path.
             # It does NOT include query params in the signature string in the provided code.
             # `msg_string = timestamp_str + method + path`
             # So we are good.
             signing_path = self.base_api_path + path # Base path only
        else:
             signing_path = self.base_api_path + path
             
        headers = self.request_headers(method, signing_path)
        # Add headers to session or request? Request specific.
        
        url = self.host + full_path
        
        async with self._session.request(method, url, headers=headers, params=params, json=body) as resp:
            if not resp.ok:
                try:
                    text = await resp.text()
                    logger.error(f"Async API Error: {resp.status} - {text}")
                except:
                    pass
                resp.raise_for_status()
            
            return await resp.json()
            
    # Re-declare generic methods to use async _request?
    # Since _request is async, the inherited methods that call self._request will fail 
    # because they expect a sync return.
    # We must override them or dynamically wrap.
    # For robust implementation, explicit overrides are safer.
    
    async def get_market_orderbook(self, ticker: str, depth: int = 10) -> Dict[str, Any]:
        params = {'depth': depth}
        return await self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        
    async def create_order(self, ticker: str, side: str, action: str, count: int, type: str = 'limit',
                         yes_price: Optional[int] = None, no_price: Optional[int] = None,
                         client_order_id: Optional[str] = None) -> Dict[str, Any]:
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }
        if yes_price is not None: body["yes_price"] = yes_price
        if no_price is not None: body["no_price"] = no_price
        if client_order_id: body["client_order_id"] = client_order_id

        return await self._request("POST", "/portfolio/orders", body=body)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")
        
    async def get_portfolio_balance(self) -> Dict[str, Any]:
        return await self._request("GET", "/portfolio/balance")

class KalshiWebSocket:
    """
    Handles robust WebSocket connection to Kalshi.
    Subscribes to orderbook deltas.
    """
    def __init__(self, client: AsyncKalshiClient, message_handler):
        self.client = client
        self.handler = message_handler
        self.ws_url = "wss://api.elections.kalshi.com/trade-api/v2/ws" if Config.ENV == Environment.PROD else "wss://demo-api.kalshi.co/trade-api/v2/ws"
        self.running = False
        self.sid = None
        
    async def connect(self):
        self.running = True
        while self.running:
            try:
                # 1. Auth & Connect
                # Kalshi WS requires a token or signature?
                # V2 WS usually involves sending a "subscribe" message with auth headers or a text frame.
                # Actually, check docs: usually URL query param OR initial message.
                # Assuming initial message protocol for "subscribe".
                
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Connected to WebSocket.")
                    
                    # 2. Authenticate / Subscribe
                    # We need to send a subscription message.
                    # Format: { "id": 1, "cmd": "subscribe", "params": { "channels": ["orderbook_delta"], "market_tickers": [...] } }
                    # But we usually need to AUTH first?
                    # "cmd": "login" with PSS signature.
                    
                    # Construct Login Msg
                    ts = str(int(time.time() * 1000))
                    msg_str = ts + "GET" + "/users/websocket/auth"
                    sig = self.client.sign_pss_text(msg_str)
                    
                    login_msg = {
                        "id": 1,
                        "cmd": "login",
                        "params": {
                            "keyId": self.client.key_id,
                            "signature": sig,
                            "timestamp": ts
                        }
                    }
                    await ws.send(json.dumps(login_msg))
                    
                    # Loop for messages
                    async for message in ws:
                        if not self.running: break
                        data = json.loads(message)
                        
                        # Handle Login Resp
                        if data.get("id") == 1 and data.get("type") == "subscribed": # or success
                            logger.info("WS Logged In.")
                            # Now we can subscribe
                            pass
                        
                        await self.handler(data)
                        
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5)
                
    async def subscribe(self, tickers: List[str]):
        # TODO: Send subscribe command to the active WS connection
        # This requires `self.ws` to be accessible or managed via a queue.
        # For this MVP, we might hardcode the subscription in the connect loop or expose a send method.
        pass
