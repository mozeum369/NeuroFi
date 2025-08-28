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
