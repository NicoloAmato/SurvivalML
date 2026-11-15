"""Train every model on every feature set and save test-set predictions and metrics.

Usage (from the `revisited` folder):
    python run_experiments.py                  # everything
    python run_experiments.py --models rsf cox --feature-sets full
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from survml.common import FEATURE_SETS, PREDICTIONS_DIR, RESULTS_DIR, evaluation_times, load_split
from survml.metrics import METRICS, evaluate
from survml.models import MODELS

warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    args = parser.parse_args()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for feature_set in args.feature_sets:
        X_train, X_test, y_train, y_test = load_split(feature_set)
        times = evaluation_times(y_test)
        for name in args.models:
            label, fit = MODELS[name]
            start = time.time()
            pred = fit(X_train, y_train, X_test, times)
            per_run = [evaluate(y_train, y_test, r, s, times) for r, s in zip(pred.risks, pred.survs)]
            metrics = {m: float(np.mean([p[m] for p in per_run])) for m in METRICS}
            spread = {m: float(np.std([p[m] for p in per_run])) for m in METRICS}
            np.savez_compressed(PREDICTIONS_DIR / f"{feature_set}__{name}.npz", risks=pred.risks, survs=pred.survs)
            record = {"model": name, "label": label, "feature_set": feature_set, "n_features": int(X_train.shape[1]),
                      "metrics": metrics, "std_over_runs": spread, "info": pred.info,
                      "seconds": round(time.time() - start, 1)}
            (RESULTS_DIR / f"metrics__{feature_set}__{name}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
            print(f"[{feature_set:12s}] {label:26s} C-index {metrics['c_index']:.4f} | IBS {metrics['ibs']:.4f} | "
                  f"AUC 1/3/5y {metrics['auc_1y']:.3f}/{metrics['auc_3y']:.3f}/{metrics['auc_5y']:.3f} | "
                  f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
