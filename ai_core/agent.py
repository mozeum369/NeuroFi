# NeuroFi/src/ai_core/agent.py
# Cryptobot main loop with structured logs, retries, and graceful shutdown

import argparse
import json
import logging
import random
import signal
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Any, Dict
from threading import Event

# ==== Imports (prefer package imports; fallback for script mode with src/) ====
try:
    # When installed or run as a module: python -m ai_core.agent
    from ai_core.ingestion import run_ingestion_pipeline
    from ai_core.sentiment import run_sentiment_analysis
    from ai_core.utils import analyze_market_patterns
    from ai_core.paths import LOG_DIR
except ModuleNotFoundError:
    # Fallback when running directly from repo root without installing
    # Assumes this file lives at: .../NeuroFi/src/ai_core/agent.py
    repo_root = Path(__file__).resolve().parents[2]  # .../NeuroFi
    sys.path.insert(0, str(repo_root / "src"))
    from ai_core.ingestion import run_ingestion_pipeline
    from ai_core.sentiment import run_sentiment_analysis
    from ai_core.utils import analyze_market_patterns
    from ai_core.paths import LOG_DIR

# ==== Paths ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
STATUS_DIR = PROJECT_ROOT / "status"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "bot_log.json"
HEALTH_PATH = STATUS_DIR / "health.json"

# ==== Structured JSON logging ====
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Base payload
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extras if present
        for attr in ("cycle_id", "step", "duration_s", "attempt", "max_attempts"):
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        # Exception info
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def get_logger() -> logging.Logger:
    logger = logging.getLogger("cryptobot")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers in interactive sessions
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Also log a minimal line to console for operator visibility
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(console)
    return logger

logger = get_logger()

# ==== Utilities ====
def write_health(status: Dict[str, Any]) -> None:
    try:
        HEALTH_PATH.write_text(json.dumps(status, indent=2))
    except Exception:
        logger.exception("Failed to write health file")

def jittered_backoff(base: float, factor: float, attempt: int, max_delay: float) -> float:
    # backoff = min(base * (factor ** (attempt-1)), max_delay) ± 20% jitter
    delay = min(base * (factor ** max(0, attempt - 1)), max_delay)
    jitter = delay * random.uniform(-0.2, 0.2)
    return max(0.5, delay + jitter)

def run_step(name: str, func: Callable[[], Any], cycle_id: int,
             max_attempts: int = 3, base_delay: float = 2.0, factor: float = 2.0) -> Any:
    """Run a single step with local retries so transient failures don't abort the whole cycle."""
    attempt = 0
    while True:
        attempt += 1
        t0 = time.monotonic()
        try:
            logger.info(
                f"▶️  Step start: {name}",
                extra={"cycle_id": cycle_id, "step": name, "attempt": attempt, "max_attempts": max_attempts},
            )
            result = func()
            dt = round(time.monotonic() - t0, 3)
            logger.info(
                f"✅ Step success: {name} ({dt}s)",
                extra={"cycle_id": cycle_id, "step": name, "duration_s": dt, "attempt": attempt, "max_attempts": max_attempts},
            )
            return result
        except Exception:
            dt = round(time.monotonic() - t0, 3)
            logger.exception(
                f"❌ Step failed: {name} ({dt}s)",
                extra={"cycle_id": cycle_id, "step": name, "duration_s": dt, "attempt": attempt, "max_attempts": max_attempts},
            )
            if attempt >= max_attempts:
                raise
            delay = jittered_backoff(base_delay, factor, attempt, max_delay=60)
            logger.info(
                f"↩️  Retrying step '{name}' in {round(delay,2)}s",
                extra={"cycle_id": cycle_id, "step": name, "attempt": attempt, "max_attempts": max_attempts},
            )
            time.sleep(delay)

def main_loop(interval_s: int = 60, max_cycle_delay: int = 300, stop_event: Event | None = None):
    """
    interval_s: desired time between cycle starts.
    max_cycle_delay: cap for cycle-level exponential backoff after an unhandled error.
    """
    if stop_event is None:
        stop_event = Event()

    cycle_id = 0
    cycle_backoff_base = 5.0
    cycle_backoff_factor = 2.0
    cycle_backoff_attempt = 0
    logger.info("🚀 Cryptobot started.")
    write_health({"status": "starting", "last_success_ts": None, "last_error": None})

    while not stop_event.is_set():
        cycle_id += 1
        cycle_start_wall = time.time()
        cycle_start = time.monotonic()
        logger.info("🔁 Starting analysis cycle...", extra={"cycle_id": cycle_id})

        try:
            # --- Step 1: Ingest new data ---
            run_step("ingestion", run_ingestion_pipeline, cycle_id)

            # --- Step 2: Sentiment analysis ---
            run_step("sentiment", run_sentiment_analysis, cycle_id)

            # --- Step 3: Analyze patterns & predictions ---
            run_step("pattern_analysis", analyze_market_patterns, cycle_id)

            # Success housekeeping
            cycle_backoff_attempt = 0
            total_dt = round(time.monotonic() - cycle_start, 3)
            logger.info("🏁 Cycle completed successfully.", extra={"cycle_id": cycle_id, "duration_s": total_dt})
            write_health({
                "status": "ok",
                "last_success_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(cycle_start_wall)),
                "last_error": None,
                "last_cycle_duration_s": total_dt,
                "cycle_id": cycle_id,
            })

            # Sleep until next scheduled start (fixed-rate schedule)
            next_start = cycle_start + interval_s
            sleep_for = max(0.0, next_start - time.monotonic())
            if stop_event.wait(timeout=sleep_for):
                break

        except Exception:
            # Cycle-level failure (after step retries exhausted)
            cycle_backoff_attempt += 1
            total_dt = round(time.monotonic() - cycle_start, 3)
            exc_text = traceback.format_exc()
            logger.error(
                "‼️  Error in bot cycle",
                extra={"cycle_id": cycle_id, "duration_s": total_dt},
                exc_info=True
            )
            write_health({
                "status": "degraded",
                "last_success_ts": None,
                "last_error": {"cycle_id": cycle_id, "traceback": exc_text},
                "last_cycle_duration_s": total_dt,
            })
            delay = jittered_backoff(cycle_backoff_base, cycle_backoff_factor, cycle_backoff_attempt, max_delay=max_cycle_delay)
            logger.info(f"⏳ Retrying cycle in {round(delay,2)} seconds...", extra={"cycle_id": cycle_id})
            if stop_event.wait(timeout=delay):
                break

    logger.info("🛑 Cryptobot stopped.")
    write_health({"status": "stopped", "last_success_ts": None, "last_error": None})

def _install_signal_handlers(stop_event: Event):
    def _handler(signum, _frame):
        logger.info(f"Signal received ({signum}). Shutting down gracefully...")
        stop_event.set()
    signal.signal(signal.SIGINT, _handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _handler)  # Termination (systemd/Docker/K8s)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroFi Cryptobot loop")
    p.add_argument("--interval", type=int, default=60, help="Seconds between cycle starts (default: 60)")
    p.add_argument("--max-delay", type=int, default=300, help="Max cycle backoff delay in seconds (default: 300)")
    p.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    return p.parse_args()

def run_once():
    # For quick local testing
    stop = Event()
    _install_signal_handlers(stop)
    # Make interval large enough so we exit after one cycle
    main_loop(interval_s=3600, max_cycle_delay=300, stop_event=stop)

if __name__ == "__main__":
    args = parse_args()
    stop_event = Event()
    _install_signal_handlers(stop_event)
    if args.once:
        run_once()
    else:
        main_loop(interval_s=args.interval, max_cycle_delay=args.max_delay, stop_event=stop_event)
