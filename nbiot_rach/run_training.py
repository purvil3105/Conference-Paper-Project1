"""CLI: train the tabular-Q baselines and save frozen checkpoints + histories.

Example:
    python -m nbiot_rach.run_training --episodes 150 --train-pop 30000 \
        --policy-seeds 0 1 --baselines B2 B3 B4 B5
"""
from __future__ import annotations

import argparse

from .experiment import ExperimentSettings, run_training_phase


def main():
    ap = argparse.ArgumentParser(description="Train NB-IoT RACH tabular-Q baselines")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--train-pop", type=int, default=30000)
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--baselines", nargs="+", default=["B2", "B3", "B4", "B5"])
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--n-val", type=int, default=5, help="number of validation traces")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from .config import VAL_SEEDS
    st = ExperimentSettings(
        out_dir=args.out_dir, episodes=args.episodes, train_pop=args.train_pop,
        policy_seeds=tuple(args.policy_seeds), baselines=tuple(args.baselines),
        val_seeds=VAL_SEEDS[:args.n_val], val_every=args.val_every)
    run_training_phase(st, verbose=not args.quiet)
    print("training complete -> checkpoints in", st.paths()["ckpt"])


if __name__ == "__main__":
    main()
