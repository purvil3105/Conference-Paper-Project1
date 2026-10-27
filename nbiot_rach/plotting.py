"""Figure generation from recorded logs (plan §8 Phase 8).

All figures are built from the CSV tables produced by the evaluation pipeline
(``load_sweep.csv``, ``main_comparison.csv``), so every plotted value traces to a
recorded run through the single baseline registry. No synthetic or proxy curves.

Styling uses the Okabe-Ito colour-blind-safe palette and explicit axis labels
with units, legible at two-column width and in print.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind-safe qualitative palette
_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
            "#56B4E9", "#F0E442", "#000000"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("baseline", "name"):
                continue
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def _style(i):
    return dict(color=_PALETTE[i % len(_PALETTE)],
                marker=_MARKERS[i % len(_MARKERS)],
                markersize=5, linewidth=1.8)


def _baseline_order(rows):
    seen = []
    for r in rows:
        key = (r["baseline"], r["name"])
        if key not in seen:
            seen.append(key)
    return seen


def plot_load_sweep_metric(rows, metric, ylabel, title, out, ci=False,
                           yfrac=False, logy=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (bid, name) in enumerate(_baseline_order(rows)):
        sub = [r for r in rows if r["baseline"] == bid]
        sub.sort(key=lambda r: r["n_meters"])
        x = np.array([r["n_meters"] / 1000.0 for r in sub])
        y = np.array([r[f"{metric}_mean"] for r in sub])
        st = _style(i)
        ax.plot(x, y, label=f"{bid}: {name}", **st)
        if ci and f"{metric}_std" in sub[0]:
            sd = np.array([r.get(f"{metric}_std", np.nan) for r in sub])
            ax.fill_between(x, y - sd, y + sd, color=st["color"], alpha=0.12,
                            linewidth=0)
    ax.set_xlabel("Number of smart meters (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if yfrac:
        ax.set_ylim(-0.02, 1.02)
    if logy:
        ax.set_yscale("log")
    ax.legend(ncol=1, loc="best")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out


def plot_delay_percentiles(rows, baseline_id, out):
    """p50 / p95 / p99 arrival-to-completion delay vs load for one policy."""
    sub = [r for r in rows if r["baseline"] == baseline_id]
    if not sub:
        return None
    sub.sort(key=lambda r: r["n_meters"])
    name = sub[0]["name"]
    x = np.array([r["n_meters"] / 1000.0 for r in sub])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (pc, lab) in enumerate([("delay_median_s", "median"),
                                   ("delay_p95_s", "p95"), ("delay_p99_s", "p99")]):
        y = np.array([r[f"{pc}_mean"] for r in sub])
        ax.plot(x, y, label=lab, **_style(i))
    ax.set_xlabel("Number of smart meters (thousands)")
    ax.set_ylabel("Arrival-to-completion delay (s)\n(completed requests)")
    ax.set_title(f"Delay percentiles vs load — {baseline_id}: {name}", fontsize=10)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out


def plot_main_comparison_bar(rows, out):
    rows = sorted(rows, key=lambda r: r["baseline"])
    labels = [f"{r['baseline']}\n{r['name']}" for r in rows]
    means = [r["p_on_time_mean"] for r in rows]
    lo = [r["p_on_time_mean"] - r.get("p_on_time_ci_low", r["p_on_time_mean"]) for r in rows]
    hi = [r.get("p_on_time_ci_high", r["p_on_time_mean"]) - r["p_on_time_mean"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(rows))
    ax.bar(x, means, yerr=[lo, hi], capsize=3,
           color=[_PALETTE[i % len(_PALETTE)] for i in range(len(rows))])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("On-time completion fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("On-time completion at reference load (95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out


def plot_training_curves(history_jsons, out):
    import json
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for i, hj in enumerate(history_jsons):
        with open(hj, encoding="utf-8") as f:
            meta = json.load(f)
        h = meta["history"]
        bid = meta["bid"]
        st = _style(i)
        if h.get("episodes"):
            ax1.plot(h["episodes"], h["train_reward"],
                     label=f"{bid} s{meta['policy_seed']}", color=st["color"],
                     linewidth=1.2, alpha=0.8)
        if h.get("val_episode"):
            ax2.plot(h["val_episode"], h["val_p_on"], marker="o", markersize=4,
                     label=f"{bid} s{meta['policy_seed']}", color=st["color"],
                     linewidth=1.5)
    ax1.set_xlabel("Training episode"); ax1.set_ylabel("Episode reward (sum)")
    ax1.set_title("Training reward", fontsize=10)
    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Validation on-time fraction")
    ax2.set_title("Validation checkpoints", fontsize=10)
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out); fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out


def generate_all(tables_dir, figures_dir, logs_dir=None) -> list[str]:
    tables_dir = Path(tables_dir); figures_dir = Path(figures_dir)
    sweep = read_csv(tables_dir / "load_sweep.csv")
    made = []
    made.append(plot_load_sweep_metric(
        sweep, "p_on_time", "On-time completion fraction",
        "Deadline-compliant service vs load", figures_dir / "fig_on_time_vs_load.png",
        ci=True, yfrac=True))
    made.append(plot_load_sweep_metric(
        sweep, "preamble_success_prob", "Preamble success probability\n(attempt-conditioned)",
        "Access (preamble) success vs load",
        figures_dir / "fig_preamble_success_vs_load.png", ci=True, yfrac=True))
    made.append(plot_load_sweep_metric(
        sweep, "collision_prob_devices", "Collision probability\n(collided devices / attempts)",
        "Collision probability vs load",
        figures_dir / "fig_collision_vs_load.png", ci=True, yfrac=True))
    made.append(plot_load_sweep_metric(
        sweep, "delay_p99_s", "p99 arrival-to-completion delay (s)",
        "Tail delay vs load", figures_dir / "fig_p99_delay_vs_load.png", ci=False))
    made.append(plot_load_sweep_metric(
        sweep, "dropped_total", "Dropped requests (count)",
        "Terminal drops vs load", figures_dir / "fig_drops_vs_load.png", ci=True))
    made.append(plot_load_sweep_metric(
        sweep, "resource_utilisation", "Resource utilisation\n(reserved / budget)",
        "Resource use vs load", figures_dir / "fig_resource_vs_load.png",
        ci=True, yfrac=True))
    made.append(plot_load_sweep_metric(
        sweep, "p_on_time_event", "Event-class on-time fraction",
        "Event-triggered deadline compliance vs load",
        figures_dir / "fig_on_time_event_vs_load.png", ci=True, yfrac=True))
    # delay percentiles for the full joint controller (B5) if present
    if any(r["baseline"] == "B5" for r in sweep):
        made.append(plot_delay_percentiles(sweep, "B5",
                    figures_dir / "fig_delay_percentiles_B5.png"))
    # reference-load bar chart
    main = read_csv(tables_dir / "main_comparison.csv")
    if main:
        made.append(plot_main_comparison_bar(main, figures_dir / "fig_main_comparison_bar.png"))
    # training curves
    if logs_dir is not None:
        hjs = sorted(Path(logs_dir).glob("../checkpoints/B*_s*.json"))
        hjs = [p for p in Path(tables_dir).parent.glob("checkpoints/B*_s*.json")]
        if hjs:
            made.append(plot_training_curves(hjs, figures_dir / "fig_training_curves.png"))
    return [m for m in made if m]
