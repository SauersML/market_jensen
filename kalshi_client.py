import requests
import base64
import time
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timedelta
from enum import Enum
import json

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

class Environment(Enum):
    DEMO = "demo"
    PROD = "prod"

class KalshiBaseClient:
    """Base client class for interacting with the Kalshi API."""
    def __init__(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
        environment: Environment = Environment.DEMO,
    ):
        self.key_id = key_id
        self.private_key = private_key
        self.environment = environment
        self.last_api_call = datetime.now()

        if self.environment == Environment.DEMO:
            self.HTTP_BASE_URL = "https://demo-api.kalshi.co"
        elif self.environment == Environment.PROD:
            self.HTTP_BASE_URL = "https://api.elections.kalshi.com"
        else:
            raise ValueError("Invalid environment")

    def request_headers(self, method: str, path: str) -> Dict[str, Any]:
        """Generates the required authentication headers for API requests."""
        current_time_milliseconds = int(time.time() * 1000)
        timestamp_str = str(current_time_milliseconds)

        # The signature message should be timestamp + method + path (including query params)
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
            raise ValueError("RSA sign PSS failed") from e

class KalshiHttpClient(KalshiBaseClient):
    """Client for handling HTTP connections to the Kalshi API."""
    def __init__(
        self,
        key_id: str,
        private_key: rsa.RSAPrivateKey,
        environment: Environment = Environment.DEMO,
    ):
        super().__init__(key_id, private_key, environment)
        self.host = self.HTTP_BASE_URL
        self.base_api_path = "/trade-api/v2"

    def rate_limit(self) -> None:
        """Built-in rate limiter to prevent exceeding API rate limits."""
        THRESHOLD_IN_MILLISECONDS = 100 # Simple rate limit
        now = datetime.now()
        threshold_in_microseconds = 1000 * THRESHOLD_IN_MILLISECONDS
        threshold_in_seconds = THRESHOLD_IN_MILLISECONDS / 1000
        if now - self.last_api_call < timedelta(microseconds=threshold_in_microseconds):
            time.sleep(threshold_in_seconds)
        self.last_api_call = datetime.now()

    def raise_if_bad_response(self, response: requests.Response) -> None:
        """Raises an HTTPError if the response status code indicates an error."""
        if response.status_code not in range(200, 299):
            try:
                error_data = response.json()
                print(f"API Error: {error_data}")
            except:
                pass
            response.raise_for_status()

    def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None) -> Any:
        self.rate_limit()
        full_path = self.base_api_path + path

        # If there are params, we need to append them to the path for signing
        # requests.prepare_request would do this but we need to do it manually to sign before request
        if params:
            # We use requests to build the query string to match exactly what will be sent
            # This handles encoding correctly.
            req = requests.Request('GET', self.host + full_path, params=params)
            prepped = req.prepare()
            # extract path + query
            path_url = prepped.path_url
            # path_url includes the full path from root, e.g. /trade-api/v2/events?status=settled
            # But we just need to pass this to request_headers
            # Note: headers generation needs relative path or absolute?
            # Kalshi docs example shows path relative to host
            signing_path = path_url
        else:
            signing_path = full_path

        headers = self.request_headers(method, signing_path)

        url = self.host + full_path

        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=body, params=params)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=params)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=body, params=params)
        else:
            raise ValueError(f"Unsupported method {method}")

        self.raise_if_bad_response(response)
        return response.json()

    def get_balance(self) -> Dict[str, Any]:
        return self._request("GET", "/portfolio/balance")

    def get_exchange_status(self) -> Dict[str, Any]:
        return self._request("GET", "/exchange/status")

    def get_series(self, series_ticker: str) -> Dict[str, Any]:
        return self._request("GET", f"/series/{series_ticker}")

    def get_series_list(self, category: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if category:
            params['category'] = category
        return self._request("GET", "/series", params=params)

    def get_events(self, series_ticker: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        params = {'limit': limit}
        if series_ticker:
            params['series_ticker'] = series_ticker
        if status:
            params['status'] = status
        return self._request("GET", "/events", params=params)

    def get_markets(self, event_ticker: Optional[str] = None, series_ticker: Optional[str] = None, status: Optional[str] = None) -> Dict[str, Any]:
        params = {}
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

    def get_market_candlesticks(self, series_ticker: str, ticker: str, start_ts: int, end_ts: int, period_interval: int) -> Dict[str, Any]:
        # period_interval: 1, 60, or 1440 (minutes)
        params = {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'period_interval': period_interval
        }
        return self._request("GET", f"/series/{series_ticker}/markets/{ticker}/candlesticks", params=params)

    def batch_get_market_candlesticks(self, market_tickers: List[str], start_ts: int, end_ts: int, period_interval: int) -> Dict[str, Any]:
        """
        Batch request candlestick data for multiple markets.
        market_tickers: List of market tickers (max 100)
        """
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
