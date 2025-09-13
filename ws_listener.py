# ws_listener.py
# Multi-channel Coinbase WebSocket listener with Advanced Trade + Exchange support.
# Uses data_utils.log_message for logging and pushes raw JSON messages onto out_queue.

import asyncio
import json
from typing import Dict, List, Optional, Set, Union
import websockets

from data_utils import log_message

ApiStyle = Union["advanced_trade", "exchange"]

ADVANCED_TRADE_PUBLIC_WS = "wss://advanced-trade-ws.coinbase.com"
ADVANCED_TRADE_USER_WS = "wss://advanced-trade-ws-user.coinbase.com"  # requires JWT
EXCHANGE_PUBLIC_WS = "wss://ws-feed.exchange.coinbase.com"

# Channels you might use (not exhaustive):
# Advanced Trade public: heartbeats, candles, status, ticker, ticker_batch, level2, market_trades
# Exchange public: heartbeat, status, ticker, level2, matches, full, ... (older feed)

class WebSocketListener:
    """
    A unified listener that can speak either Coinbase Advanced Trade WS (recommended)
    or the older Exchange WS. It supports dynamic subscription management per channel.

    Examples:
        # Advanced Trade public market data:
        listener = WebSocketListener(
            out_queue=q,
            api_style="advanced_trade",
            ws_url=ADVANCED_TRADE_PUBLIC_WS,
            jwt=None,                         # optional but supported; required for user feed
            ensure_heartbeats=True
        )

        # Exchange (older) style:
        listener = WebSocketListener(
            out_queue=q,
            api_style="exchange",
            ws_url=EXCHANGE_PUBLIC_WS
        )
    """

    def __init__(
        self,
        out_queue: asyncio.Queue,
        api_style: str = "advanced_trade",
        ws_url: Optional[str] = None,
        jwt: Optional[str] = None,
        ensure_heartbeats: bool = True,
        ping_interval: float = 30.0,
        reconnect_base_delay: float = 2.0,
        reconnect_max_delay: float = 30.0,
    ):
        self.api_style = api_style  # "advanced_trade" | "exchange"
        self.ws_url = ws_url or (ADVANCED_TRADE_PUBLIC_WS if api_style == "advanced_trade" else EXCHANGE_PUBLIC_WS)
        self.jwt = jwt
        self.out_queue = out_queue
        self.ensure_heartbeats = ensure_heartbeats
        self.ping_interval = ping_interval
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay

        # Map[channel] -> set(product_ids) or {None} if channel doesn't need product_ids
        self.subscriptions: Dict[str, Set[Optional[str]]] = {}
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._stop = asyncio.Event()

    # -------------------- Public API --------------------

    async def run(self, initial_subscriptions: Optional[Dict[str, Optional[List[str]]]] = None):
        """
        Connect -> subscribe -> receive loop with automatic reconnect/backoff.
        initial_subscriptions: dict like {"ticker": ["BTC-USD", "ETH-USD"], "status": None, "candles": ["BTC-USD"]}
        """
        if initial_subscriptions:
            for ch, prods in initial_subscriptions.items():
                await self.add_subscription(ch, prods)

        # Heartbeats recommended to keep channels open on Advanced Trade;
        # On Exchange, "heartbeat" requires product_ids (we will infer from existing product subscriptions).
        # (Docs recommend heartbeats to maintain subscriptions when updates are sparse.)
        # We'll delay till just before we send during (re)connect.

        backoff = self.reconnect_base_delay

        while not self._stop.is_set():
            try:
                async with websockets.connect(self.ws_url, ping_interval=self.ping_interval) as ws:
                    self.websocket = ws
                    log_message(f"WebSocket connected: {self.ws_url}")

                    # On connect, (re)send all subscriptions
                    await self._send_all_current_subscriptions()

                    # Read loop
                    async for message in ws:
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            log_message(f"Non-JSON message: {message}", level="warning")
                            continue

                        # Push raw frame to downstream queue
                        await self.out_queue.put(data)

                    # If loop exits normally, trigger reconnect
                    log_message("WebSocket stream ended; reconnecting...", level="warning")
            except Exception as e:
                log_message(f"WebSocket error: {e}. Reconnecting...", level="error")

            # Exponential backoff (bounded)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.reconnect_max_delay)

        log_message("WebSocket listener stopped.")

    async def stop(self):
        self._stop.set()
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass

    async def add_subscription(self, channel: str, product_ids: Optional[List[str]] = None):
        """
        Add/merge a subscription (does not immediately send if not connected).
        """
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        if product_ids is None or len(product_ids) == 0:
            self.subscriptions[channel].add(None)
        else:
            for p in product_ids:
                self.subscriptions[channel].add(p)

        # If currently connected, send an incremental subscribe
        if self.websocket:
            await self._send_subscribe(channel, product_ids)

    async def remove_subscription(self, channel: str, product_ids: Optional[List[str]] = None):
        """
        Remove a subscription and (if connected) send unsubscribe.
        """
        if channel not in self.subscriptions:
            return

        # Compute which items to remove
        if product_ids is None:
            # Remove entire channel
            items = list(self.subscriptions[channel])
            self.subscriptions.pop(channel, None)
            await self._send_unsubscribe(channel, [x for x in items if x is not None] or None)
        else:
            # Remove specific product ids
            for p in product_ids:
                self.subscriptions[channel].discard(p)
            # If channel left empty, remove it
            if not self.subscriptions[channel]:
                self.subscriptions.pop(channel, None)
            await self._send_unsubscribe(channel, product_ids)

    # -------------------- Internal helpers --------------------

    async def _send_all_current_subscriptions(self):
        # Optionally ensure heartbeats:
        if self.ensure_heartbeats:
            await self._ensure_heartbeats_present()

        # Send each channel's subscription in **individual** messages (safer across both APIs)
        for channel, items in self.subscriptions.items():
            product_list = sorted([x for x in items if x is not None])
            await self._send_subscribe(channel, product_list or None)

    async def _ensure_heartbeats_present(self):
        """
        Advanced Trade: heartbeats without product_ids keeps connections open.
        Exchange: heartbeat requires product_ids per product. We add heartbeat for
        all products currently visible across your product-based channels.
        """
        if self.api_style == "advanced_trade":
            # Add channel with None
            if "heartbeats" not in self.subscriptions:
                await self.add_subscription("heartbeats", None)
        else:
            # Exchange style heartbeat per product
            all_products: Set[str] = set()
            for ch, items in self.subscriptions.items():
                if ch in ("ticker", "level2", "matches", "full", "ticker_batch"):
                    all_products.update([p for p in items if p is not None])
            if all_products:
                await self.add_subscription("heartbeat", sorted(all_products))

    async def _send_subscribe(self, channel: str, product_ids: Optional[List[str]] = None):
        if not self.websocket:
            return
        msg = self._build_message("subscribe", channel, product_ids)
        await self._ws_send(msg, f"Subscribed to {channel} {product_ids or ''}")

    async def _send_unsubscribe(self, channel: str, product_ids: Optional[List[str]] = None):
        if not self.websocket:
            return
        msg = self._build_message("unsubscribe", channel, product_ids)
        await self._ws_send(msg, f"Unsubscribed from {channel} {product_ids or ''}")

    async def _ws_send(self, msg: dict, success_log: str):
        try:
            await self.websocket.send(json.dumps(msg))
            log_message(success_log)
        except Exception as e:
            log_message(f"Failed to send WS message {msg}: {e}", level="error")

    def _build_message(self, action: str, channel: str, product_ids: Optional[List[str]]) -> dict:
        """
        Build the correct subscribe/unsubscribe message for the selected API style.
        """
        if self.api_style == "advanced_trade":
            # Advanced Trade format: single 'channel' field, plus optional product_ids, plus optional jwt.
            # e.g. {"type":"subscribe","channel":"ticker","product_ids":["BTC-USD"],"jwt":"..."}
            msg = {"type": action, "channel": channel}
            if product_ids:
                msg["product_ids"] = product_ids
            if self.jwt:
                msg["jwt"] = self.jwt
            return msg
        else:
            # Exchange format: array of channels with name/product_ids
            # e.g. {"type":"subscribe","channels":[{"name":"ticker","product_ids":["BTC-USD"]}]}
            entry = {"name": channel}
            if product_ids:
                entry["product_ids"] = product_ids
            return {"type": action, "channels": [entry]}
