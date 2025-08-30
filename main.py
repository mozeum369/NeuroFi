# File: main.py
import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# --- Ensure we can import from ./src ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Import your modules from src/ ---
try:
    from ws_listener import WebSocketListener      # expects async .run(initial_symbols=...) and .subscribe(symbols)
    from data_ingestor import DataIngestor         # expects async .backfill() and .run_periodic()
    from agent import Agent                        # expects async .run() and optionally .run_top_movers_refresh(cb)
except ImportError as e:
    print(f"[BOOT] Import error: {e}")
    print("Tip: run from repo root: cd ~/NeuroFi && python main.py run")
    # We do not sys.exit here because 'smoke' subcommand can run without these.
    WebSocketListener = None
    DataIngestor = None
    Agent = None

# --- Signal sink import (new) ---
try:
    from signal_sink import SignalSink, SignalSinkConfig
except ImportError as e:
    print(f"[BOOT] Missing src/signal_sink.py: {e}")
    SignalSink = None
    SignalSinkConfig = None


# -------------------- Configuration --------------------

@dataclass(frozen=True)
class Settings:
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Endpoints / API
    coinbase_ws_url: str = field(default_factory=lambda: os.getenv(
        "COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com"
    ))
    freecryptoapi_url: str = field(default_factory=lambda: os.getenv(
        "FREECRYPTOAPI_URL", "https://api.freecryptoapi.com"
    ))

    # Runtime behavior
    default_symbols: str = field(default_factory=lambda: os.getenv("DEFAULT_SYMBOLS", "auto"))  # "auto" or "BTC-USD,ETH-USD"
    top_movers_count: int = int(os.getenv("TOP_MOVERS_COUNT", "20"))
    backfill_interval_sec: int = int(os.getenv("BACKFILL_INTERVAL_SEC", "300"))  # 5 minutes

    # Reporting
    enable_daily_report: bool = os.getenv("ENABLE_DAILY_REPORT", "1") == "1"
    reports_dir: str = field(default_factory=lambda: os.getenv("REPORTS_DIR", str(ROOT / "reports")))
    report_cron_utc: str = field(default_factory=lambda: os.getenv("REPORT_CRON_UTC", "23:59"))

    # --- NEW: signals directory for the sink ---
    signals_dir: str = field(default_factory=lambda: os.getenv("SIGNALS_DIR", str(ROOT / "signals")))


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)


# -------------------- Orchestration --------------------

async def run_realtime(settings: Settings, symbols: Optional[List[str]] = None):
    if any(x is None for x in (WebSocketListener, DataIngestor, Agent, SignalSink, SignalSinkConfig)):
        raise SystemExit("Missing one or more required modules. Ensure src/ contains ws_listener.py, data_ingestor.py, agent.py, and signal_sink.py.")

    log = logging.getLogger("main.run_realtime")

    # Queues: shared plumbing between modules
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)   # real-time ticks
    ohlcv_queue: asyncio.Queue = asyncio.Queue(maxsize=4000)   # historical candles
    signal_queue: asyncio.Queue = asyncio.Queue(maxsize=4000)  # analysis signals / alerts

    # Instantiate modules
    ws = WebSocketListener(
        ws_url=settings.coinbase_ws_url,
        out_queue=tick_queue,
        top_movers_count=settings.top_movers_count,
    )

    ingestor = DataIngestor(
        base_url=settings.freecryptoapi_url,
        ohlcv_out=ohlcv_queue,
    )

    agent = Agent(
        tick_in=tick_queue,
        ohlcv_in=ohlcv_queue,
        signal_out=signal_queue,
        top_movers_count=settings.top_movers_count,
    )

    # --- NEW: Signal sink wiring ---
    sink_cfg = SignalSinkConfig(
        out_dir=settings.signals_dir,
        file_prefix="signals",
        rotate_daily=True,
        flush_each_write=True,
    )
    sink = SignalSink(signals_in=signal_queue, config=sink_cfg)

    # Determine symbols
    use_symbols = symbols
    if use_symbols is None and settings.default_symbols.strip().lower() != "auto":
        use_symbols = [s.strip().upper() for s in settings.default_symbols.split(",") if s.strip()]

    # Launch tasks
    tasks: List[asyncio.Task] = []
    tasks.append(asyncio.create_task(ws.run(initial_symbols=use_symbols), name="ws_listener"))
    tasks.append(asyncio.create_task(
        ingestor.run_periodic(interval_sec=settings.backfill_interval_sec, symbols=use_symbols),
        name="data_backfill"
    ))
    tasks.append(asyncio.create_task(agent.run(), name="agent"))
    tasks.append(asyncio.create_task(sink.run(), name="signal_sink"))  # start sink

    # Optional: Daily report
    if settings.enable_daily_report:
        try:
            from reporting import DailyReporter  # src/reporting.py (optional)
            reporter = DailyReporter(signals_in=signal_queue, out_dir=settings.reports_dir)
            tasks.append(asyncio.create_task(
                reporter.run_daily(cron=settings.report_cron_utc),
                name="daily_reporter"
            ))
        except ImportError:
            log.warning("Daily reporting enabled but src/reporting.py not found. Skipping.")

    # Optional: top-movers refresh loop if agent provides it
    if hasattr(agent, "run_top_movers_refresh"):
        tasks.append(asyncio.create_task(agent.run_top_movers_refresh(ws.subscribe), name="top_movers_refresh"))
    else:
        log.info("Agent has no 'run_top_movers_refresh'—skipping top-movers auto-refresh.")

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _graceful_shutdown():
        log.info("Shutdown signal received. Cancelling tasks…")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful_shutdown)
        except NotImplementedError:
            pass

    await stop_event.wait()

    for t in tasks:
        t.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks, return_exceptions=True)

    log.info("Shutdown complete.")


async def run_backfill(settings: Settings, symbols: List[str], days: int = 30):
    if DataIngestor is None:
        raise SystemExit("data_ingestor.py not found in src/")
    log = logging.getLogger("main.run_backfill")
    ingestor = DataIngestor(base_url=settings.freecryptoapi_url, ohlcv_out=None)
    await ingestor.backfill(symbols=symbols, days=days)
    log.info("Backfill complete.")


async def run_report(settings: Settings, period: str = "daily"):
    log = logging.getLogger("main.run_report")
    try:
        from reporting import DailyReporter
    except ImportError:
        log.error("src/reporting.py not found. Cannot run report.")
        return

    reporter = DailyReporter(signals_in=None, out_dir=settings.reports_dir)
    if period == "daily":
        await reporter.run_once()
    else:
        await reporter.run_period(period)
    log.info("Report complete.")


# -------------------- NEW: Smoke test --------------------

async def run_smoke(settings: Settings, duration_sec: int = 5) -> Path:
    """
    Runs a quick, offline smoke test with in-memory fakes:
      - Produces fake ticks and OHLCV
      - Emits a couple of fake signals
      - SignalSink writes a JSONL file
    Verifies at least one line is written. Returns the file path written.
    """
    if any(x is None for x in (SignalSink, SignalSinkConfig)):
        raise SystemExit("signal_sink.py is required for smoke test.")

    log = logging.getLogger("main.smoke")
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    ohlcv_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    signal_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

    # Start the sink
    sink_cfg = SignalSinkConfig(
        out_dir=os.getenv("SMOKE_SIGNALS_DIR", settings.signals_dir),
        file_prefix="signals-smoke",
        rotate_daily=True,
        flush_each_write=True,
    )
    sink = SignalSink(signals_in=signal_queue, config=sink_cfg)

    async def fake_ws_producer():
        # Emit mock ticks
        for i in range(20):
            await asyncio.sleep(0.05)
            await tick_queue.put({"symbol": "BTC-USD", "price": 50000 + i, "ts": None})
        # no sentinel; consumer not needed in smoke

    async def fake_ingestor_producer():
        # Emit mock OHLCV bars
        for i in range(10):
            await asyncio.sleep(0.1)
            await ohlcv_queue.put({"symbol": "BTC-USD", "open": 49900 + i, "high": 50100 + i,
                                   "low": 49800 + i, "close": 50000 + i, "volume": 123.45 + i})

    async def fake_agent_consumer_and_signal():
        # Read a few from each queue and emit signals
        consumed_ticks = 0
        consumed_bars = 0
        # Consume some data but don't block forever
        end_time = asyncio.get_event_loop().time() + duration_sec
        while asyncio.get_event_loop().time() < end_time:
            try:
                tick = await asyncio.wait_for(tick_queue.get(), timeout=0.2)
                consumed_ticks += 1
                tick_queue.task_done()
            except asyncio.TimeoutError:
                pass
            try:
                bar = await asyncio.wait_for(ohlcv_queue.get(), timeout=0.2)
                consumed_bars += 1
                ohlcv_queue.task_done()
            except asyncio.TimeoutError:
                pass

            # Emit a couple of signals
            if (consumed_ticks + consumed_bars) % 5 == 0:
                await signal_queue.put({
                    "symbol": "BTC-USD",
                    "side": "BUY" if (consumed_ticks % 2 == 0) else "SELL",
                    "strength": round(min(1.0, (consumed_ticks + consumed_bars) / 20.0), 3),
                    "reason": "smoke_test",
                })
            await asyncio.sleep(0.05)

        # Send sentinel to close sink nicely
        await signal_queue.put(None)

    tasks = [
        asyncio.create_task(sink.run(), name="sink"),
        asyncio.create_task(fake_ws_producer(), name="fake_ws"),
        asyncio.create_task(fake_ingestor_producer(), name="fake_ingestor"),
        asyncio.create_task(fake_agent_consumer_and_signal(), name="fake_agent"),
    ]

    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration_sec + 3)
    except asyncio.TimeoutError:
        log.warning("Smoke tasks timed out; cancelling.")
        for t in tasks:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)

    # Check output file
    from datetime import datetime, timezone
    yyyymmdd = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_file = Path(sink_cfg.out_dir) / f"{sink_cfg.file_prefix}-{yyyymmdd}.jsonl"
    if not out_file.exists() or out_file.stat().st_size == 0:
        raise SystemExit(f"Smoke test failed: no output written at {out_file}")

    log.info(f"Smoke test OK. Signals written to: {out_file}")
    return out_file


# -------------------- CLI --------------------

def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="NeuroFi Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run real-time system")
    p_run.add_argument("--symbols", type=str, default=None,
                       help="Comma-separated (e.g., BTC-USD,ETH-USD). Omit for auto top-movers.")
    p_run.add_argument("--log-level", type=str, default=None)

    p_back = sub.add_parser("backfill", help="Backfill historical OHLCV")
    p_back.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    p_back.add_argument("--days", type=int, default=30)
    p_back.add_argument("--log-level", type=str, default=None)

    p_rep = sub.add_parser("report", help="Generate report")
    p_rep.add_argument("--period", type=str, default="daily", choices=["daily", "weekly", "monthly"])
    p_rep.add_argument("--log-level", type=str, default=None)

    # --- NEW: smoke test ---
    p_smoke = sub.add_parser("smoke", help="Run an offline smoke test (no network)")
    p_smoke.add_argument("--duration", type=int, default=5, help="Approx seconds to run")
    p_smoke.add_argument("--log-level", type=str, default=None)

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    argv = argv if argv is not None else sys.argv[1:]
    args = parse_args(argv)

    settings = Settings()
    setup_logging(args.log_level or settings.log_level)
    logging.getLogger("main").info("Starting NeuroFi…")

    if args.cmd == "run":
        symbols = None
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        asyncio.run(run_realtime(settings, symbols))
    elif args.cmd == "backfill":
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        asyncio.run(run_backfill(settings, symbols=symbols, days=args.days))
    elif args.cmd == "report":
        asyncio.run(run_report(settings, period=args.period))
    elif args.cmd == "smoke":
        asyncio.run(run_smoke(settings, duration_sec=args.duration))
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
