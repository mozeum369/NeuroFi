import asyncio
import json
import logging
from typing import List, Optional
import websockets

logger = logging.getLogger("ws_listener")
logging.basicConfig(level=logging.INFO)

class WebSocketListener:
    def __init__(self, ws_url: str, out_queue: asyncio.Queue, top_movers_count: int = 20):
        self.ws_url = ws_url
        self.out_queue = out_queue
        self.top_movers_count = top_movers_count
        self.subscribed_symbols = set()
        self.websocket = None

    async def run(self, initial_symbols: Optional[List[str]] = None):
        if initial_symbols:
            self.subscribed_symbols.update(initial_symbols)

        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=None) as websocket:
                    self.websocket = websocket
                    if self.subscribed_symbols:
                        await self._send_subscribe(list(self.subscribed_symbols))

                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.out_queue.put(data)
                        except json.JSONDecodeError:
                            logger.warning("Received non-JSON message: %s", message)
            except Exception as e:
                logger.error("WebSocket error: %s. Reconnecting in 5 seconds...", e)
                await asyncio.sleep(5)

    async def subscribe(self, symbols: List[str]):
        new_symbols = set(symbols) - self.subscribed_symbols
        if new_symbols:
            self.subscribed_symbols.update(new_symbols)
            if self.websocket:
                await self._send_subscribe(list(new_symbols))

    async def _send_subscribe(self, symbols: List[str]):
        message = {
            "type": "subscribe",
            "channels": [{
                "name": "ticker",
                "product_ids": symbols
            }]
        }
        try:
            await self.websocket.send(json.dumps(message))
            logger.info("Subscribed to symbols: %s", symbols)
        except Exception as e:
            logger.error("Failed to send subscribe message: %s", e) 
