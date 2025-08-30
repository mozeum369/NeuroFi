# File: src/signal_sink.py
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict


@dataclass
class SignalSinkConfig:
    out_dir: str
    file_prefix: str = "signals"
    rotate_daily: bool = True
    flush_each_write: bool = True


class SignalSink:
    """
    Consumes dict-like 'signal' objects from an asyncio.Queue and writes them to JSONL files.
    Expected signal structure (flexible, but recommended keys):
      {
        "ts": ISO8601 timestamp (UTC),
        "symbol": "BTC-USD",
        "side": "BUY" | "SELL" | "HOLD",
        "strength": float in [0, 1],
        "reason": "string",
        ... (any additional metadata)
      }
    """
    def __init__(self, signals_in: asyncio.Queue, config: SignalSinkConfig):
        self.signals_in = signals_in
        self.config = config
        self._dir = Path(config.out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[Any] = None
        self._open_date: Optional[str] = None  # YYYYMMDD

    def _file_path_for_date(self, yyyymmdd: str) -> Path:
        return self._dir / f"{self.config.file_prefix}-{yyyymmdd}.jsonl"

    def _open_for_today_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self._open_date != today:
            # rotate: close previous handle
            if self._fh:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
            self._open_date = today
            self._fh = self._file_path_for_date(today).open("a", encoding="utf-8")

    async def run(self):
        try:
            while True:
                item: Dict[str, Any] = await self.signals_in.get()
                # Allow None to signal a graceful stop if you want to use it
                if item is None:
                    self.signals_in.task_done()
                    break

                # Ensure a timestamp exists
                item.setdefault("ts", datetime.now(timezone.utc).isoformat())

                # Rotate daily if configured
                if self.config.rotate_daily:
                    self._open_for_today_if_needed()
                else:
                    if self._fh is None:
                        # Open a single static file if not rotating
                        if self._open_date is None:
                            self._open_date = "static"
                            self._fh = (self._dir / f"{self.config.file_prefix}.jsonl").open("a", encoding="utf-8")

                # Write JSONL
                assert self._fh is not None
                self._fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                if self.config.flush_each_write:
                    self._fh.flush()

                self.signals_in.task_done()
        except asyncio.CancelledError:
            # Cooperative cancel: best-effort flush
            pass
        finally:
            if self._fh:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
                self._fh = None
