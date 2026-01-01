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
