from pathlib import Path
from loguru import logger
from config import config

def setup_logging():
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(Path(config.logs_dir) / "runtime.log", rotation="1 MB", retention=5, level="INFO", enqueue=True)
    logger.add(lambda m: print(m, end=""))
    logger.info("Logging initialized.")
    return logger
