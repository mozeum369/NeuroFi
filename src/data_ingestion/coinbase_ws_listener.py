import asyncio
import json
import websockets
import nest_asyncio

nest_asyncio.apply()

COINBASE_WS_URL = "wss://ws-feed.advanced.trade.coinbase.com"
PRODUCT_IDS = ["BTC-USD", "ETH-USD", "SOL-USD"]

async def coinbase_ws_listener():
    async with websockets.connect(COINBASE_WS_URL) as websocket:
        subscribe_message = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": PRODUCT_IDS}]
        }
        await websocket.send(json.dumps(subscribe_message))
        print(f"Subscribed to ticker updates for: {PRODUCT_IDS}")

        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("type") == "ticker":
                    print(f"[{data.get('time')}] {data.get('product_id')} price: {data.get('price')}")
            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(coinbase_ws_listener())
