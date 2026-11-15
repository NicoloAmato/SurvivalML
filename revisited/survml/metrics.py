"""Evaluation metrics shared by every model, and their bootstrap distribution."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sksurv.metrics import (concordance_index_censored, cumulative_dynamic_auc,
                            integrated_brier_score)

from .common import HORIZONS

METRICS = ["c_index", "ibs", *HORIZONS, "auc_mean"]
HIGHER_IS_BETTER = {m: m != "ibs" for m in METRICS}


def evaluate(y_train, y_test, risk: np.ndarray, surv: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """Compute all metrics for one set of predictions.

    risk: one score per test patient, higher means shorter expected survival.
    surv: survival probabilities, shape (n_test, len(times)).
    A metric that cannot be computed on the given sample is returned as NaN.
    """
    out: dict[str, float] = {}
    out["c_index"] = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]
    try:
        out["ibs"] = integrated_brier_score(y_train, y_test, np.clip(surv, 1e-6, 1 - 1e-6), times)
    except ValueError:
        out["ibs"] = np.nan
    aucs = []
    for name, horizon in HORIZONS.items():
        try:
            auc = cumulative_dynamic_auc(y_train, y_test, risk, np.array([horizon]))[0][0]
        except ValueError:
            auc = np.nan
        out[name] = auc
        aucs.append(auc)
    out["auc_mean"] = float(np.mean(aucs))
    return out


def _evaluate_runs(y_train, y_test, risks, survs, times) -> dict[str, float]:
    """Average the metrics over the runs of a model (a single run for most models)."""
    per_run = [evaluate(y_train, y_test, r, s, times) for r, s in zip(risks, survs)]
    return {m: float(np.nanmean([p[m] for p in per_run])) for m in METRICS}


def bootstrap_metrics(y_train, y_test, predictions: dict, times: np.ndarray,
                      n_boot: int = 1000, seed: int = 0, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Bootstrap the test set and recompute every metric for every model.

    predictions maps a model name to (risks, survs) with shapes (n_runs, n_test) and
    (n_runs, n_test, len(times)). The same resampled patients are used for all models,
    so differences between models can be read as paired differences.
    Returns, for each model, an array of shape (n_boot, len(METRICS)).
    """
    rng = np.random.default_rng(seed)
    n = len(y_test)
    samples = [rng.integers(0, n, n) for _ in range(n_boot)]

    def one(idx):
        yb = y_test[idx]
        return {name: [_evaluate_runs(y_train, yb, risks[:, idx], survs[:, idx], times)[m] for m in METRICS]
                for name, (risks, survs) in predictions.items()}

    rows = Parallel(n_jobs=n_jobs)(delayed(one)(idx) for idx in samples)
    return {name: np.array([r[name] for r in rows]) for name in predictions}


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile confidence interval that ignores resamples where the metric was undefined."""
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return float(low), float(high)
