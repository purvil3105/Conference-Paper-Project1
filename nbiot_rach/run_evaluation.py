"""CLI: evaluate all baselines on held-out test traces (reference + load sweep).

Loads frozen checkpoints produced by ``run_training`` (trains them first if they
are missing). Writes ``main_comparison.csv``, ``load_sweep.csv`` and
``controller_cost.csv`` plus a manifest. Nothing here touches the training or
validation seeds.

Example:
    python -m nbiot_rach.run_evaluation --populations 10000 30000 50000 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import TEST_SEEDS, VAL_SEEDS
from .experiment import (ExperimentSettings, run_training_phase,
                         evaluate_reference, evaluate_load_sweep)
from .persistence import write_json, runtime_metadata
from .baseline_registry import baseline_specs


def main():
    ap = argparse.ArgumentParser(description="Evaluate NB-IoT RACH baselines")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--train-pop", type=int, default=30000)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--baselines", nargs="+",
                    default=["B0", "B1", "B2", "B3", "B4", "B5"])
    ap.add_argument("--populations", type=int, nargs="+",
                    default=[10000, 20000, 30000, 50000, 75000, 100000])
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--train-if-missing", action="store_true", default=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    st = ExperimentSettings(
        out_dir=args.out_dir, episodes=args.episodes, train_pop=args.train_pop,
        policy_seeds=tuple(args.policy_seeds), baselines=tuple(args.baselines),
        populations=tuple(args.populations), test_seeds=TEST_SEEDS[:args.n_test],
        val_seeds=VAL_SEEDS[:args.n_val])

    # train learning baselines if their checkpoints are missing
    specs = baseline_specs()
    need_train = False
    for bid in st.baselines:
        if specs[bid].is_learning and specs[bid].implemented:
            for ps in st.policy_seeds:
                if not (st.paths()["ckpt"] / f"{bid}_s{ps}.npz").exists():
                    need_train = True
    if need_train and args.train_if_missing:
        print("[eval] missing checkpoints -> training first")
        run_training_phase(st, verbose=not args.quiet)

    print("[eval] reference-load comparison ...")
    ref = evaluate_reference(st, verbose=not args.quiet)
    print("[eval] load (device-count) sweep ...")
    sweep = evaluate_load_sweep(st, verbose=not args.quiet)

    manifest = {
        "settings": {k: getattr(st, k) for k in
                     ("out_dir", "train_pop", "episodes", "policy_seeds",
                      "baselines", "populations", "test_seeds", "val_seeds",
                      "label")},
        "b0_action_id": ref["b0_action_id"],
        "runtime": runtime_metadata(),
        "note": ("PILOT results unless episodes/seeds were scaled up. Not a full "
                 "research experiment. See docs/model_assumptions.md for scope."),
    }
    write_json(st.paths()["root"] / "manifest.json", manifest)
    print("evaluation complete -> tables in", st.paths()["tables"])


if __name__ == "__main__":
    main()
