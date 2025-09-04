import requests
import logging
import json
import datetime
from joblib import Memory

# Set up logging configuration
logging.basicConfig(filename='data_utils.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up joblib memory cache
memory = Memory(location='./cache', verbose=0)

def log_message(message, level='info'):
    """
    Logs a message to the data_utils.log file.
    """
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    else:
        logging.debug(message)

def safe_fetch_json(api_url):
    """
    Safely fetches JSON data from an API with error handling and logging.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        log_message(f"Successfully fetched data from {api_url}")
        return data
    except requests.exceptions.RequestException as e:
        log_message(f"API request failed for {api_url}: {e}", level='error')
        return None

@memory.cache
def cached_fetch_json(api_url):
    """
    Fetches and caches JSON data from an API.
    """
    return safe_fetch_json(api_url)

def save_data_snapshot(data, prefix='data_snapshot'):
    """
    Saves data to a JSON file with a UTC timestamp.
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        log_message(f"Data snapshot saved to {filename}")
    except Exception as e:
        log_message(f"Failed to save data snapshot: {e}", level='error') 
