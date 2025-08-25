import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# ============================================================
# SelfModifyingAgent
# Sandboxed, controlled self-modification + skill evolution
# ============================================================

@dataclass
class AgentConfig:
    allowed_dirs: List[str]
    max_children_per_generation: int = 4
    mutation_offsets: Tuple[int, ...] = (-2, -1, 1, 2)
    registry_path: str = "state/registry.json"
    events_log: str = "logs/events.jsonl"
    metrics_path: str = "state/metrics.json"
    experiments_dir: str = "experiments"
    skills_dir: str = "skills"
    heartbeat_block_begin: str = "# <HEARTBEAT>"
    heartbeat_block_end: str = "# </HEARTBEAT>"

class SelfModifyingAgent:
    def __init__(self, project_root: Path, name: str = "SelfModAI"):
        self.name = name
        self.root = project_root
        self.cfg = AgentConfig(
            allowed_dirs=[
                str(self.root / "skills"),
                str(self.root / "state"),
                str(self.root / "logs"),
                str(self.root / "experiments"),
            ]
        )
        self.paths = {
            "skills": self.root / self.cfg.skills_dir,
            "state": self.root / "state",
            "logs": self.root / "logs",
            "experiments": self.root / self.cfg.experiments_dir,
            "registry": self.root / self.cfg.registry_path,
            "events_log": self.root / self.cfg.events_log,
            "metrics": self.root / self.cfg.metrics_path,
        }

        self.generation = 0
        self._ensure_dirs()
        self._ensure_registry()

    # ---------------------- Bootstrap & Dirs ----------------------
    def _ensure_dirs(self):
        for p in self.paths.values():
            if isinstance(p, Path) and p.suffix == "":
                p.mkdir(parents=True, exist_ok=True)
        (self.root / self.cfg.skills_dir).mkdir(parents=True, exist_ok=True)

    def _ensure_registry(self):
        if not self.paths["registry"].exists():
            data = {
                "created_at": datetime.utcnow().isoformat(),
                "agent": self.name,
                "best_skill": None,
                "skills": {},
                "history": [],
                "generation": 0,
            }
            self._safe_write_json(self.paths["registry"], data)

    def bootstrap(self):
        registry = self._load_registry()
        if not registry["skills"]:
            # Create an initial MA skill with window 5
            base_name = "ma_w5"
            code = self._make_ma_skill_code(window=5)
            path = self._write_skill(base_name, code)
            score = self._evaluate_skill(base_name)
            registry["skills"][base_name] = {
                "path": str(path),
                "created_at": datetime.utcnow().isoformat(),
                "score": score,
            }
            registry["best_skill"] = base_name
            registry["generation"] = 0
            self._save_registry(registry)
            self._log_event("bootstrap", {"skill": base_name, "score": score})

    # ---------------------- Evolution Loop -----------------------
    def evolve_once(self) -> Dict[str, Any]:
        registry = self._load_registry()
        parent = registry["best_skill"]
        if parent is None:
            self.bootstrap()
            registry = self._load_registry()
            parent = registry["best_skill"]

        parent_score = registry["skills"][parent]["score"]

        # Propose children via mutation
        children: List[Tuple[str, float]] = []
        parent_w = self._extract_window_from_name(parent)

        offsets = list(self.cfg.mutation_offsets)
        random.shuffle(offsets)
        offsets = offsets[: self.cfg.max_children_per_generation]

        for off in offsets:
            new_w = max(2, parent_w + off)  # keep window >= 2
            child_name = f"ma_w{new_w}"
            if child_name in registry["skills"]:
                # already exists; just re-evaluate
                score = self._evaluate_skill(child_name)
            else:
                code = self._make_ma_skill_code(window=new_w)
                self._write_skill(child_name, code)
                score = self._evaluate_skill(child_name)

            children.append((child_name, score))

        # Decide best among parent + children (lower score is better)
        candidates = [(parent, parent_score)] + children
        best_child, best_score = min(candidates, key=lambda kv: kv[1])

        # Update registry
        for name, score in candidates:
            srec = registry["skills"].get(name, None)
            if srec is None:
                registry["skills"][name] = {
                    "path": str(self._skill_path(name)),
                    "created_at": datetime.utcnow().isoformat(),
                    "score": score,
                }
            else:
                # keep best observed score
                srec["score"] = min(srec.get("score", float("inf")), score)

        registry["best_skill"] = best_child
        registry["generation"] = registry.get("generation", 0) + 1
        self._save_registry(registry)

        self.generation = registry["generation"]
        self._log_event(
            "evolve",
            {
                "generation": self.generation,
                "parent": parent,
                "parent_score": parent_score,
                "candidates": candidates,
                "best": best_child,
                "best_score": best_score,
            },
        )
        self._update_metrics(best=best_child, score=best_score)

        return {
            "generation": self.generation,
            "best_skill": best_child,
            "best_score": best_score,
        }

    # ---------------------- Skill Creation -----------------------
    def _make_ma_skill_code(self, window: int) -> str:
        # Simple moving-average predictor skill
        return textwrap.dedent(
            f"""
            \"\"\"Auto-generated skill: Moving Average predictor (window={window})\"\"\"

            METADATA = {{
                "name": "ma_w{window}",
                "type": "moving_average",
                "window": {window}
            }}

            def predict(series):
                \"\"\"
                Predict next values using simple moving average with fixed window.
                Input: list[float] series
                Output: list[float] predictions (same length; first window elements are repeated first observed value)
                \"\"\"
                if not series:
                    return []
                w = {window}
                n = len(series)
                preds = []
                runsum = 0.0
                for i, x in enumerate(series):
                    runsum += x
                    if i >= w:
                        runsum -= series[i - w]
                    if i < w - 1:
                        preds.append(series[0])
                    else:
                        # average of last w elements including current
                        start = i - w + 1
                        s = 0.0
                        for j in range(start, i + 1):
                            s += series[j]
                        preds.append(s / w)
                return preds
            """
        ).strip() + "\n"

    def _write_skill(self, name: str, code: str) -> Path:
        path = self._skill_path(name)
        self._safe_write_text(path, code)
        return path

    def _skill_path(self, name: str) -> Path:
        return self.paths["skills"] / f"{name}.py"

    def _extract_window_from_name(self, name: str) -> int:
        m = re.search(r"ma_w(\\d+)", name)
        if not m:
            return 5
        return int(m.group(1))

    # ---------------------- Evaluation ---------------------------
    def _evaluate_skill(self, name: str) -> float:
        """
        Returns a loss score (lower is better).
        Uses synthetic dataset for reproducibility.
        """
        module = self._load_module(name, self._skill_path(name))
        series = self._make_synthetic_series(seed=42, length=200)
        preds = module.predict(series)

        # Mean Absolute Percentage Error (MAPE)-like metric
        eps = 1e-8
        total = 0.0
        count = 0
        for y, yhat in zip(series, preds):
            denom = max(eps, abs(y))
            total += abs(y - yhat) / denom
            count += 1
        score = total / max(1, count)
        return float(score)

    def _make_synthetic_series(self, seed: int, length: int) -> List[float]:
        random.seed(seed)
        # Trend + seasonality + noise
        series = []
        for t in range(length):
            trend = 0.01 * t
            season = 0.5 * (1 + __import__("math").sin(2 * __import__("math").pi * t / 24))
            noise = random.gauss(0, 0.05)
            series.append(1.0 + trend + season + noise)
        return series

    def _load_module(self, name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module: {name} from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    # ----------------- Controlled Self-Modification --------------
    def self_edit_heartbeat(self, note: str = ""):
        """
        Safely updates the HEARTBEAT block in THIS FILE with current best-skill info.
        Demonstrates self-modifying code without touching logic.
        """
        this_path = Path(__file__).resolve()
        text = this_path.read_text(encoding="utf-8")

        registry = self._load_registry()
        best = registry.get("best_skill")
        best_score = None
        if best:
            best_score = registry["skills"][best]["score"]

        stamp = datetime.utcnow().isoformat()
        payload = textwrap.dedent(
            f"""
            {self.cfg.heartbeat_block_begin}
            name: {self.name}
            timestamp_utc: {stamp}
            generation: {registry.get('generation', 0)}
            best_skill: {best}
            best_score: {best_score}
            note: {note}
            {self.cfg.heartbeat_block_end}
            """
        ).strip()

        # Replace or insert the block
        begin = self.cfg.heartbeat_block_begin
        end = self.cfg.heartbeat_block_end
        block_re = re.compile(
            rf"{re.escape(begin)}[\\s\\S]*?{re.escape(end)}", re.MULTILINE
        )

        if block_re.search(text):
            new_text = block_re.sub(payload, text)
        else:
            new_text = text + "\n\n" + payload + "\n"

        # Write back (self-modification!)
        this_path.write_text(new_text, encoding="utf-8")
        self._log_event("self_edit", {"file": str(this_path), "action": "heartbeat_update"})

    # ---------------------- Logging & State ----------------------
    def _load_registry(self) -> Dict[str, Any]:
        return self._safe_read_json(self.paths["registry"])

    def _save_registry(self, data: Dict[str, Any]):
        self._safe_write_json(self.paths["registry"], data)

    def _update_metrics(self, best: str, score: float):
        metrics = {"updated_at": datetime.utcnow().isoformat(), "best_skill": best, "best_score": score}
        self._safe_write_json(self.paths["metrics"], metrics)

    def _log_event(self, event: str, payload: Dict[str, Any]):
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.paths["events_log"].open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------- Safe I/O -----------------------------
    def _safe_path(self, p: Path) -> Path:
        p = p.resolve()
        allowed = any(str(p).startswith(d) for d in self.cfg.allowed_dirs)
        if not allowed:
            raise PermissionError(f"Write access denied outside sandbox: {p}")
        return p

    def _safe_write_text(self, path: Path, text: str):
        path = self._safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _safe_read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_write_json(self, path: Path, data: Dict[str, Any]):
        path = self._safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    # ---------------------- Git (Optional) -----------------------
    def git_autocommit(self, message: str = "autocommit"):
        git = shutil.which("git")
        if not git:
            return
        try:
            subprocess.run([git, "add", "-A"], cwd=str(self.root), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run([git, "commit", "-m", message], cwd=str(self.root), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            self._log_event("git_error", {"error": str(e)})

# <HEARTBEAT>
# (Populated at runtime)
# </HEARTBEAT>
