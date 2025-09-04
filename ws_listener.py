# Refactored ws_listener.py to use data_utils.py for logging

import asyncio
import json
from typing import List, Optional
import websockets
from data_utils import log_message

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
                            log_message(f"Received non-JSON message: {message}", level='warning')
            except Exception as e:
                log_message(f"WebSocket error: {e}. Reconnecting in 5 seconds...", level='error')
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
            log_message(f"Subscribed to symbols: {symbols}")
        except Exception as e:
            log_message(f"Failed to send subscribe message: {e}", level='error') 
