"""CLI: the small, reproducible end-to-end PILOT (smoke -> train -> eval -> plots).

This is the "first useful milestone" (plan §15): one reproducible trace set, a
conservation-checked ledger, tuned fixed + adaptive baselines, trained+frozen
tabular policies, and a comparison table/figures generated entirely from logs.

It is a PILOT (small training budget), explicitly NOT the full research
experiment. Scale it up with the flags for a real run.

    python -m nbiot_rach.run_pilot                 # default pilot
    python -m nbiot_rach.run_pilot --episodes 2000 --policy-seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import time

from .config import TEST_SEEDS, VAL_SEEDS
from .experiment import (ExperimentSettings, run_training_phase,
                         evaluate_reference, evaluate_load_sweep)
from .persistence import write_json, runtime_metadata
from . import run_smoke_tests, make_paper_outputs


def main():
    ap = argparse.ArgumentParser(description="NB-IoT RACH reproducible pilot")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--train-pop", type=int, default=30000)
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--baselines", nargs="+",
                    default=["B0", "B1", "B2", "B3", "B4", "B5"])
    ap.add_argument("--populations", type=int, nargs="+",
                    default=[10000, 20000, 30000, 50000, 75000, 100000])
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=5)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    if not args.skip_smoke:
        print("=" * 70, "\n[1/4] smoke tests\n" + "=" * 70)
        rc = run_smoke_tests.run()
        if rc != 0:
            raise SystemExit("smoke tests failed; aborting pilot")

    st = ExperimentSettings(
        out_dir=args.out_dir, episodes=args.episodes, train_pop=args.train_pop,
        policy_seeds=tuple(args.policy_seeds), baselines=tuple(args.baselines),
        populations=tuple(args.populations), test_seeds=TEST_SEEDS[:args.n_test],
        val_seeds=VAL_SEEDS[:args.n_val])

    print("=" * 70, "\n[2/4] training tabular-Q baselines\n" + "=" * 70)
    run_training_phase(st, verbose=True)

    print("=" * 70, "\n[3/4] evaluation (reference + load sweep)\n" + "=" * 70)
    ref = evaluate_reference(st, verbose=True)
    evaluate_load_sweep(st, verbose=True)

    print("=" * 70, "\n[4/4] figures and tables\n" + "=" * 70)
    import sys
    sys.argv = ["make_paper_outputs", "--out-dir", args.out_dir]
    make_paper_outputs.main()

    dur = time.time() - t0
    write_json(st.paths()["root"] / "manifest.json", {
        "label": "pilot", "wall_clock_s": dur,
        "settings": {k: getattr(st, k) for k in
                     ("train_pop", "episodes", "policy_seeds", "baselines",
                      "populations", "test_seeds", "val_seeds")},
        "b0_action_id": ref["b0_action_id"],
        "runtime": runtime_metadata(),
        "note": "PILOT (small budget). Not a full research experiment.",
    })
    print(f"\nPILOT complete in {dur:.1f}s -> {st.paths()['root']}")


if __name__ == "__main__":
    main()
