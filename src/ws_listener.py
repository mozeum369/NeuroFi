# ws_listener.py
# Coinbase Advanced Trade WebSocket listener (Python layout, robust reconnection)
# Supports channels: heartbeats, candles, status, ticker, ticker_batch, level2, market_trades
# Requires: websocket-client, PyJWT[crypto]

import os
import json
import time
import uuid
import ssl
import logging
import threading
import queue
import random
from typing import Callable, Dict, List, Optional

import PyJWT
import websocket-client

# ---- Logging ----------------------------------------------------------------
logger = logging.getLogger("coinbase_ws")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)

# ---- Constants from Coinbase docs -------------------------------------------
MARKET_DATA_URL = "wss://advanced-trade-ws.coinbase.com"  # public market data
USER_DATA_URL = "wss://advanced-trade-ws-user.coinbase.com"  # user-order data
# Subscribe must be sent within 5 seconds; one channel per sub message. (Docs)  # noqa

# Channel names (as per Advanced Trade WS docs)
VALID_CHANNELS = {
    "heartbeats",
    "candles",
    "status",
    "ticker",
    "ticker_batch",
    "level2",
    "market_trades",
    # "user", "futures_balance_summary" exist but not used here
}

# ---- Utilities ---------------------------------------------------------------

def _now_s() -> int:
    return int(time.time())

def _make_jwt_es256(api_key_name: str, signing_key_pem: str) -> str:
    """
    Generate an ES256 JWT for Coinbase Advanced Trade WS.
    - iss: "cdp"
    - nbf: now
    - exp: now + 120s
    - sub: API key name (organizations/{org_id}/apiKeys/{key_id})
    - header: kid=API key name, nonce=random hex
    Coinbase requires a fresh JWT for *each* websocket message (expires ~2 min).
    """
    now = _now_s()
    payload = {
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "sub": api_key_name,
    }
    headers = {
        "kid": api_key_name,
        "nonce": uuid.uuid4().hex,
    }
    token = jwt.encode(payload, signing_key_pem, algorithm="ES256", headers=headers)
    # PyJWT returns str for < 2.x and in 2.x returns str as well
    return token

def _warn_usdc(channel: str, product_ids: List[str]) -> None:
    """
    Warn if subscribing to *-USDC products on non-user channels (per docs).
    Exceptions: USDT-USDC and EURC-USDC allowed everywhere.
    """
    if channel == "user":
        return
    for pid in product_ids or []:
        if pid.endswith("-USDC") and pid not in ("USDT-USDC", "EURC-USDC"):
            logger.warning(
                "Per Coinbase docs, subscribing to %s on non-user channels "
                "will mirror *-USD data; prefer *-USD or use the user channel.", pid
            )

# ---- Main client -------------------------------------------------------------

class AdvancedTradeWS:
    """
    Python-layout WebSocket client using websocket-client, with:
    - Dynamic subscribe/unsubscribe (fresh JWT per message when auth is provided)
    - Resilient reconnect (exponential backoff + jitter)
    - Channel routing + user-defined handlers
    - Optional message queue for downstream consumers
    - Heartbeats subscription to keep other channels open
    """

    def __init__(
        self,
        product_ids: List[str],
        channels: List[str],
        api_key_name: Optional[str] = None,  # e.g., "organizations/{org_id}/apiKeys/{key_id}"
        signing_key_pem: Optional[str] = None,  # full EC private key PEM
        use_user_feed: bool = False,  # False for market data feed
        auto_heartbeats: bool = True,
        message_queue_size: int = 10000,
        ping_interval: int = 20,
        ping_timeout: int = 10,
        on_message_handlers: Optional[Dict[str, Callable[[dict], None]]] = None,
    ):
        self.product_ids = list(dict.fromkeys(product_ids))  # de-dupe, keep order
        self.channels = list(dict.fromkeys(channels))
        self.api_key_name = api_key_name
        self.signing_key_pem = signing_key_pem
        self.use_user_feed = use_user_feed
        self.auto_heartbeats = auto_heartbeats
        self.message_queue: "queue.Queue[dict]" = queue.Queue(maxsize=message_queue_size)
        self.on_message_handlers = on_message_handlers or {}

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._connected_event = threading.Event()

        self._url = USER_DATA_URL if use_user_feed else MARKET_DATA_URL
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # Sanity checks
        for ch in self.channels:
            if ch not in VALID_CHANNELS:
                raise ValueError(f"Unsupported channel: {ch}")

        # Auto-add heartbeats to keep connections open (docs recommend)
        if self.auto_heartbeats and "heartbeats" not in self.channels:
            self.channels.insert(0, "heartbeats")  # send this first

    # ---------------- Public API ----------------

    def start(self):
        """Start the WS connection in a background thread."""
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run_forever_loop, daemon=True)
        self._thread.start()
        # Optional: wait for initial connection or timeout
        self._connected_event.wait(timeout=10)

    def stop(self):
        """Stop the client and close the connection."""
        self._stop_flag.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def subscribe(self, channel: str, product_ids: Optional[List[str]] = None):
        """Dynamically subscribe to an additional channel/products."""
        if channel not in VALID_CHANNELS:
            raise ValueError(f"Unsupported channel: {channel}")
        self._send_subscribe(channel, product_ids or self.product_ids)

    def unsubscribe(self, channel: str, product_ids: Optional[List[str]] = None):
        """Dynamically unsubscribe from a channel/products."""
        if channel not in VALID_CHANNELS:
            raise ValueError(f"Unsupported channel: {channel}")
        self._send_unsubscribe(channel, product_ids or self.product_ids)

    def get_message(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Get the next message from the shared queue (or None if timeout)."""
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------------- Internal ----------------

    def _run_forever_loop(self):
        backoff = 1.0
        while not self._stop_flag.is_set():
            try:
                logger.info("Connecting to %s ...", self._url)
                self._connected_event.clear()
                self._ws = websocket.WebSocketApp(
                    self._url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Secure defaults
                ssl_opts = {"cert_reqs": ssl.CERT_REQUIRED}
                # run_forever will auto ping/pong; set intervals
                self._ws.run_forever(
                    sslopt=ssl_opts,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    ping_payload="ping",
                    origin=None,
                )
            except Exception as e:
                logger.error("WS exception: %s", e, exc_info=True)

            if self._stop_flag.is_set():
                break
            # Reconnect with backoff + jitter
            sleep_s = min(60.0, backoff) + random.uniform(0, 0.5)
            logger.warning("Reconnecting in %.1fs ...", sleep_s)
            time.sleep(sleep_s)
            backoff = min(60.0, backoff * 2.0)

    def _on_open(self, ws):
        logger.info("WebSocket opened.")
        self._connected_event.set()

        # Send subscribe messages — one channel per message (as required)
        for ch in self.channels:
            # heartbeats: no product_ids in docs; others typically include product_ids
            if ch == "heartbeats":
                self._send_subscribe(ch, product_ids=None)
            else:
                self._send_subscribe(ch, product_ids=self.product_ids)

    def _on_close(self, ws, status_code, msg):
        logger.warning("WebSocket closed: code=%s msg=%s", status_code, msg)

    def _on_error(self, ws, error):
        logger.error("WebSocket error: %s", error)

    def _on_message(self, ws, raw: str):
        try:
            data = json.loads(raw)
        except Exception:
            logger.debug("Non-JSON or parse error: %s", raw)
            return

        # Push into shared queue (drop on full to avoid backpressure)
        try:
            self.message_queue.put_nowait(data)
        except queue.Full:
            logger.warning("Message queue full; dropping message.")

        # Channel-based dispatch (Advanced Trade messages have 'channel')
        channel = data.get("channel")
        handler = self.on_message_handlers.get(channel)
        if handler:
            try:
                handler(data)
            except Exception:
                logger.exception("Handler for channel '%s' raised.", channel)

    # -- subscribe/unsubscribe helpers ----------------------------------------

    def _send_subscribe(self, channel: str, product_ids: Optional[List[str]]):
        msg = {
            "type": "subscribe",
            "channel": channel,
        }
        if product_ids:
            _warn_usdc(channel, product_ids)
            msg["product_ids"] = product_ids

        if self.api_key_name and self.signing_key_pem:
            msg["jwt"] = _make_jwt_es256(self.api_key_name, self.signing_key_pem)

        self._send(msg)

    def _send_unsubscribe(self, channel: str, product_ids: Optional[List[str]]):
        msg = {
            "type": "unsubscribe",
            "channel": channel,
        }
        if product_ids:
            msg["product_ids"] = product_ids

        if self.api_key_name and self.signing_key_pem:
            msg["jwt"] = _make_jwt_es256(self.api_key_name, self.signing_key_pem)

        self._send(msg)

    def _send(self, msg: dict):
        wire = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
        if not self._ws:
            logger.error("Attempted to send without active WS.")
            return
        try:
            self._ws.send(wire)
            logger.debug("Sent: %s", wire)
        except Exception as e:
            logger.error("Send failed: %s", e)
=======
import asyncio
import json
import websockets

URI = 'wss://ws-feed.exchange.coinbase.com'
active_subscriptions = {}
websocket_connection = None

def build_subscribe_message():
    channels = []
    for channel, product_ids in active_subscriptions.items():
        channels.append({
            "name": channel,
            "product_ids": list(product_ids)
        })
    return json.dumps({
        "type": "subscribe",
        "channels": channels
    })

def build_unsubscribe_message(channel, product_ids):
    return json.dumps({
        "type": "unsubscribe",
        "channels": [{
            "name": channel,
            "product_ids": product_ids
        }]
    })

async def subscribe_to_products(channel, product_ids):
    global websocket_connection
    if channel not in active_subscriptions:
        active_subscriptions[channel] = set()
    new_products = set(product_ids) - active_subscriptions[channel]
    if new_products:
        active_subscriptions[channel].update(new_products)
        message = json.dumps({
            "type": "subscribe",
            "channels": [{
                "name": channel,
                "product_ids": list(new_products)
            }]
        })
        if websocket_connection:
            await websocket_connection.send(message)

async def unsubscribe_from_products(channel, product_ids):
    global websocket_connection
    if channel in active_subscriptions:
        removed_products = set(product_ids) & active_subscriptions[channel]
        if removed_products:
            active_subscriptions[channel] -= removed_products
            message = build_unsubscribe_message(channel, list(removed_products))
            if websocket_connection:
                await websocket_connection.send(message)

async def websocket_listener():
    global websocket_connection
    while True:
        try:
            async with websockets.connect(URI, ping_interval=None) as websocket:
                websocket_connection = websocket
                if active_subscriptions:
                    await websocket.send(build_subscribe_message())

                while True:
                    response = await websocket.recv()
                    json_response = json.loads(response)
                    print(json_response)

        except Exception as e:
            print(f'WebSocket error: {e}, retrying...')
            await asyncio.sleep(1)

def start_listener():
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_listener())
    loop.run_forever()

