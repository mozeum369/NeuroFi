# Cryptobot.py
import time
import logging
import traceback
from ingestion import run_ingestion_pipeline
from sentiment import run_sentiment_analysis
from utils import analyze_market_patterns  # hypothetical utility

# Setup logging
logging.basicConfig(
    filename='bot_log.json',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def main_loop():
    retry_delay = 5  # initial delay in seconds
    max_delay = 300  # max delay of 5 minutes

    while True:
        try:
            logging.info("🔁 Starting new analysis cycle...")

            # Step 1: Ingest new data
            run_ingestion_pipeline()

            # Step 2: Run sentiment analysis
            run_sentiment_analysis()

            # Step 3: Analyze patterns and make predictions
            analyze_market_patterns()

            logging.info("✅ Cycle completed successfully.")
            retry_delay = 5  # reset delay after success

            # Wait before next cycle
            time.sleep(60)

        except Exception as e:
            logging.error("❌ Error in bot cycle: %s", str(e))
            logging.error(traceback.format_exc())

            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

            # Exponential backoff
            retry_delay = min(retry_delay * 2, max_delay)

if __name__ == "__main__":
    logging.info("🚀 Cryptobot started.")
    main_loop()
