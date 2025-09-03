import asyncio
import logging
from ws_listener import WebSocketListener

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_ws_listener")

async def main():
    # Define the symbols and channels to subscribe to
    symbols = ["BTC-USD", "ETH-USD", "PEPE-USD"]
    channels = ["ticker", "candles", "market_trades"]

    # Create a queue to receive messages
    tick_queue = asyncio.Queue(maxsize=1000)

    # Initialize the WebSocketListener
    ws_listener = WebSocketListener(
        ws_url="wss://advanced-trade-ws.coinbase.com",
        out_queue=tick_queue,
        top_movers_count=3,
    )

    # Start the listener
    await ws_listener.run(initial_symbols=symbols)

    # Subscribe to the desired channels
    for channel in channels:
        await ws_listener.subscribe(channel, symbols)

    # Print incoming messages
    while True:
        msg = await tick_queue.get()
        print(msg)

# Run the test
if __name__ == "__main__":
    asyncio.run(main()) 
