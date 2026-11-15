"""Bootstrap confidence intervals and paired model comparisons from the saved predictions.

Run `python run_experiments.py` first. Writes three tables to `results/`:
    summary.csv           point estimate and 95% interval of every metric, model and feature set
    paired_vs_rsf.csv     difference between each model and the Random Survival Forest
    feature_set_effect.csv  change of each model when treatment and intent features are removed
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from survml.common import (FEATURE_SET_LABELS, FEATURE_SETS, PREDICTIONS_DIR, RESULTS_DIR,
                           evaluation_times, load_split)
from survml.metrics import METRICS, bootstrap_metrics, percentile_interval
from survml.models import MODELS

N_BOOT = 1000
REFERENCE_MODEL = "rsf"
REFERENCE_FEATURES = "full"


def difference_rows(samples_a, samples_b, point_a, point_b, **labels) -> list[dict]:
    """Paired bootstrap difference a - b for every metric."""
    rows = []
    for i, metric in enumerate(METRICS):
        diff = samples_a[:, i] - samples_b[:, i]
        low, high = percentile_interval(diff)
        rows.append({**labels, "metric": metric, "difference": point_a[metric] - point_b[metric],
                     "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0)})
    return rows


def main() -> None:
    _, _, y_train, y_test = load_split("full")  # the patients are the same for every feature set
    times = evaluation_times(y_test)

    predictions, points = {}, {}
    for feature_set in FEATURE_SETS:
        for model in MODELS:
            saved = np.load(PREDICTIONS_DIR / f"{feature_set}__{model}.npz")
            predictions[(feature_set, model)] = (saved["risks"], saved["survs"])
            record = json.loads((RESULTS_DIR / f"metrics__{feature_set}__{model}.json").read_text())
            points[(feature_set, model)] = record["metrics"]

    samples = bootstrap_metrics(y_train, y_test, predictions, times, n_boot=N_BOOT)

    summary = []
    for (feature_set, model), values in samples.items():
        for i, metric in enumerate(METRICS):
            low, high = percentile_interval(values[:, i])
            summary.append({"feature_set": feature_set, "feature_set_label": FEATURE_SET_LABELS[feature_set],
                            "model": model, "model_label": MODELS[model][0], "metric": metric,
                            "estimate": points[(feature_set, model)][metric], "ci_low": low, "ci_high": high})
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "summary.csv", index=False)

    paired = []
    for feature_set in FEATURE_SETS:
        ref = (feature_set, REFERENCE_MODEL)
        for model in MODELS:
            if model != REFERENCE_MODEL:
                key = (feature_set, model)
                paired += difference_rows(samples[key], samples[ref], points[key], points[ref],
                                          feature_set=feature_set, model=model, reference=REFERENCE_MODEL)
    pd.DataFrame(paired).to_csv(RESULTS_DIR / "paired_vs_rsf.csv", index=False)

    effect = []
    for model in MODELS:
        ref = (REFERENCE_FEATURES, model)
        for feature_set in FEATURE_SETS:
            if feature_set != REFERENCE_FEATURES:
                key = (feature_set, model)
                effect += difference_rows(samples[key], samples[ref], points[key], points[ref],
                                          model=model, feature_set=feature_set, reference=REFERENCE_FEATURES)
    pd.DataFrame(effect).to_csv(RESULTS_DIR / "feature_set_effect.csv", index=False)
    print(f"{N_BOOT} bootstrap resamples of {len(y_test)} test patients -> results/*.csv")


if __name__ == "__main__":
    main()
