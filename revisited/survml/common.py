"""Shared configuration, data loading and the single train/test split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.util import Surv

REVISITED_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = REVISITED_DIR.parent / "thesis" / "data" / "dataset.csv"
RESULTS_DIR = REVISITED_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

TIME_COL = "follow_ups.days_to_follow_up"
EVENT_COL = "demographic.vital_status_DEAD"

SEED = 25          # same split as the RSF and XGBSE notebooks of the thesis
TEST_SIZE = 0.25
HORIZONS = {"auc_1y": 365.0, "auc_3y": 1095.0, "auc_5y": 1825.0}
N_IBS_TIMES = 20

# Treatments actually delivered: they happen after diagnosis.
TREATMENT_COLS = ["Trt_PHARMACEUTICAL", "Trt_RADIATION", "Trt_SURGERY"]
# Adjuvant therapy follows a completed primary treatment, so it is not known at diagnosis either.
POST_TREATMENT_INTENT_COLS = ["Intent_ADJUVANT"]
INTENT_COLS = ["Intent_ADJUVANT", "Intent_CURE", "Intent_BEFORETRT", "Intent_PALLIATIVE"]

FEATURE_SETS: dict[str, list[str]] = {
    # every feature, as in the thesis
    "full": [],
    # without delivered treatments and adjuvant intent
    "no_treatment": TREATMENT_COLS + POST_TREATMENT_INTENT_COLS,
    # demographics, tumour and exposure only
    "baseline": TREATMENT_COLS + INTENT_COLS,
}
FEATURE_SET_LABELS = {
    "full": "All features (thesis)",
    "no_treatment": "No delivered treatments",
    "baseline": "No treatment or intent",
}


def load_dataset() -> pd.DataFrame:
    """Read the thesis dataset and turn the -1 placeholders into NaN."""
    return pd.read_csv(DATA_PATH).replace(-1, np.nan)


def load_split(feature_set: str = "full"):
    """Return X_train, X_test, y_train, y_test for one feature set.

    The split depends only on the seed and on the event column, so every model and every
    feature set is trained and tested on exactly the same patients.
    """
    df = load_dataset()
    dropped = FEATURE_SETS[feature_set]
    X = df.drop(columns=[TIME_COL, EVENT_COL] + dropped)
    y = Surv.from_arrays(event=df[EVENT_COL].astype(bool), time=df[TIME_COL])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=df[EVENT_COL])


def evaluation_times(y_test) -> np.ndarray:
    """Time grid for the Integrated Brier Score: 10th to 90th percentile of the test times."""
    lower, upper = np.percentile(y_test["time"], [10, 90])
    return np.linspace(lower, upper, N_IBS_TIMES)
