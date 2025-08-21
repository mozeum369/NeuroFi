
import time

def run_bot():
    print("Cryptobot engine started. Running continuously...")
    try:
        while True:
            # Simulate bot activity
            print("Bot is analyzing market data...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Cryptobot engine stopped.")
