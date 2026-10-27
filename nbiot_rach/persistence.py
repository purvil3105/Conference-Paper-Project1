"""Portable checkpoints, logs and run manifests (plan §6, §J).

Checkpoints store the Q-table, visit counts, feasible action set, config snapshot,
action registry, state-bin edges and training history, so every reported value
traces back to a recorded run. Numeric arrays go in ``.npz``; metadata in JSON.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config
from .actions import ActionRegistry
from .agents import QLearningAgent


def _git_revision() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def runtime_metadata() -> dict:
    import numpy, scipy, matplotlib
    return {
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, agent: QLearningAgent, cfg: Config, bid: str,
                    policy_seed: int, history, action_mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, q=agent.q, visits=agent.visits, feasible=agent.feasible,
        action_mask=action_mask, epsilon=np.float64(agent.epsilon),
        frozen=np.bool_(agent.frozen))
    meta = {
        "bid": bid, "policy_seed": policy_seed, "config": cfg.to_dict(),
        "history": asdict(history) if hasattr(history, "__dataclass_fields__")
        else history,
        "runtime": runtime_metadata(),
        "q_checksum": agent.checksum(),
    }
    write_json(str(path).replace(".npz", ".json"), meta)


def load_checkpoint(path, cfg: Config) -> tuple[QLearningAgent, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    registry = ActionRegistry(cfg)
    n_states = (len(cfg.congestion_bins) - 1) * (len(cfg.load_bins) - 1) * (len(cfg.trend_bins) - 1)
    agent = QLearningAgent(n_states=n_states, n_actions=registry.n_actions,
                           feasible=data["feasible"], learning=cfg.learning,
                           exploration_rng=np.random.default_rng(0),
                           tiebreak_rng=np.random.default_rng(1))
    agent.load_state_dict({"q": data["q"], "visits": data["visits"],
                           "feasible": data["feasible"],
                           "epsilon": float(data["epsilon"]),
                           "frozen": bool(data["frozen"])})
    agent.freeze()
    return agent, data["action_mask"]


def write_csv(path, rows: list[dict], columns: list[str] | None = None) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
