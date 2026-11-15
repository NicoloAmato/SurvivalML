"""The four survival pipelines. Each one is fitted on the training set only and returns
its test-set predictions in a common format."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

from .common import SEED


@dataclass
class Prediction:
    """Test-set output of a pipeline.

    risks: shape (n_runs, n_test); survs: shape (n_runs, n_test, n_times).
    """
    risks: np.ndarray
    survs: np.ndarray
    info: dict = field(default_factory=dict)


def _step_functions_to_matrix(functions, times: np.ndarray) -> np.ndarray:
    return np.vstack([fn(times) for fn in functions])


def _interpolate(curves: pd.DataFrame, grid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Interpolate survival curves (one row per patient, columns on `grid`) at `times`."""
    return np.array([np.interp(times, grid, row) for row in curves.to_numpy()])


# --------------------------------------------------------------------------- Cox baseline
def fit_cox(X_train, y_train, X_test, times) -> Prediction:
    """Ridge-penalised Cox regression: the classical reference model."""
    pipeline = Pipeline([
        ("impute", IterativeImputer(random_state=SEED, max_iter=15)),
        ("scale", StandardScaler()),
        ("cox", CoxPHSurvivalAnalysis()),
    ])
    search = GridSearchCV(pipeline, {"cox__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]}, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    surv = _step_functions_to_matrix(model.predict_survival_function(X_test), times)
    info = {"cv_c_index": float(search.best_score_), "alpha": search.best_params_["cox__alpha"],
            "train_c_index": float(model.score(X_train, y_train))}
    return Prediction(model.predict(X_test)[None], surv[None], info)


# --------------------------------------------------------------------------- Random Survival Forest
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Derived features of the thesis; each one is added only if its inputs are present."""
    X = X.copy()
    if {"Final_T_enc", "Final_N_enc"} <= set(X.columns):
        X["TNM_Burden"] = X["Final_T_enc"] + X["Final_N_enc"]
        X["TNM_Interaction"] = X["Final_T_enc"] * X["Final_N_enc"]
    treatments = [c for c in ("Trt_PHARMACEUTICAL", "Trt_RADIATION", "Trt_SURGERY") if c in X.columns]
    if treatments:
        X["Total_Treatments"] = X[treatments].sum(axis=1)
    if {"demographic.age_at_index", "TNM_Burden"} <= set(X.columns):
        X["Age_TNM_Risk"] = X["demographic.age_at_index"] / 10 * X["TNM_Burden"]
    return X


def _rsf_preprocess(X_fit, y_fit, X_apply, top_k=30, n_repeats=5):
    """MICE imputation, feature engineering and permutation-based selection, fitted on X_fit."""
    imputer = IterativeImputer(random_state=SEED, max_iter=15)
    fit = engineer_features(pd.DataFrame(imputer.fit_transform(X_fit), columns=X_fit.columns))
    apply = engineer_features(pd.DataFrame(imputer.transform(X_apply), columns=X_apply.columns))
    selector = RandomSurvivalForest(n_estimators=100, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    selector.fit(fit, y_fit)
    importance = permutation_importance(selector, fit, y_fit, n_repeats=n_repeats, random_state=SEED, n_jobs=-1)
    keep = fit.columns[np.argsort(importance.importances_mean)[::-1][:min(top_k, fit.shape[1])]]
    return fit[keep], apply[keep]


RSF_SEARCH_SPACE = {
    "n_estimators": [300, 500],
    "max_depth": [3, 4, 5],
    "min_samples_split": [30, 40, 50],
    "min_samples_leaf": [15, 25, 35],
    "max_features": ["sqrt", "log2", 0.5],
}


def fit_rsf(X_train, y_train, X_test, times) -> Prediction:
    """Random Survival Forest pipeline of the thesis, unchanged."""
    train, test = _rsf_preprocess(X_train, y_train, X_test)
    search = RandomizedSearchCV(
        RandomSurvivalForest(random_state=SEED, n_jobs=-1, max_samples=0.75),
        RSF_SEARCH_SPACE, n_iter=40, cv=5, n_jobs=-1, random_state=SEED,
    )
    search.fit(train, y_train)
    model = search.best_estimator_

    # Leakage-free internal estimate: the whole preprocessing is refitted inside every fold.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for fit_idx, val_idx in folds.split(X_train, y_train["event"]):
        fold_fit, fold_val = _rsf_preprocess(X_train.iloc[fit_idx], y_train[fit_idx], X_train.iloc[val_idx], n_repeats=3)
        fold_model = RandomSurvivalForest(random_state=SEED, n_jobs=-1, max_samples=0.75, **search.best_params_)
        scores.append(fold_model.fit(fold_fit, y_train[fit_idx]).score(fold_val, y_train[val_idx]))

    surv = _step_functions_to_matrix(model.predict_survival_function(test), times)
    info = {"cv_c_index": float(np.mean(scores)), "cv_c_index_std": float(np.std(scores)),
            "train_c_index": float(model.score(train, y_train)), "best_params": search.best_params_,
            "n_features": int(train.shape[1])}
    return Prediction(model.predict(test)[None], surv[None], info)


# --------------------------------------------------------------------------- XGBSE
class TunableXGBSE(BaseEstimator):
    """scikit-learn wrapper that makes the XGBSE hyperparameters take effect.

    XGBSEDebiasedBCE builds its XGBoost feature extractor in the constructor, so parameters
    set later through `set_params` (as RandomizedSearchCV does) never reach it. This wrapper
    creates the estimator inside `fit`, and passes `n_estimators` as the number of boosting
    rounds, which the native XGBoost API does not read from the parameter dictionary.
    """

    def __init__(self, xgb_params: dict | None = None):
        self.xgb_params = xgb_params

    def fit(self, X, y):
        from xgbse import XGBSEDebiasedBCE

        params = dict(self.xgb_params or {})
        rounds = params.pop("n_estimators", 10)
        self.model_ = XGBSEDebiasedBCE(xgb_params=params)
        self.model_.fit(X, y, num_boost_round=rounds)
        return self

    def predict(self, X) -> pd.DataFrame:
        return self.model_.predict(X)


def risk_from_survival(curves: pd.DataFrame) -> np.ndarray:
    """Scalar risk score: one minus the mean survival probability over the time grid."""
    return 1.0 - curves.to_numpy().mean(axis=1)


def _xgbse_score(estimator, X, y) -> float:
    return concordance_index_censored(y["event"], y["time"], risk_from_survival(estimator.predict(X)))[0]


def xgbse_search_space(n_configs: int = 150) -> list[dict]:
    """The 150 configurations of the thesis, with one change: `tree_method` is "hist"
    because XGBoost 2.1 does not support column sampling by node with "exact"."""
    rng = random.Random(SEED)
    configs = []
    for _ in range(n_configs):
        configs.append({
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": "hist",
            "n_estimators": rng.choice([200, 300, 400, 500]),
            "learning_rate": rng.choice([0.005, 0.01, 0.02, 0.03]),
            "max_depth": rng.choice([2, 3, 4]),
            "min_child_weight": rng.choice([5, 10, 15, 20, 30]),
            "gamma": rng.choice([0.5, 1, 2, 5]),
            "subsample": rng.choice([0.5, 0.6, 0.7, 0.8]),
            "colsample_bynode": rng.choice([0.3, 0.5, 0.7, 1.0]),
            "reg_alpha": rng.choice([0.5, 1, 5, 10, 20]),
            "reg_lambda": rng.choice([1, 5, 10, 20, 50]),
            "seed": SEED,
        })
    return configs


def _xgbse_prediction(model, X_train, y_train, X_test, times, info) -> Prediction:
    curves = model.predict(X_test)
    surv = _interpolate(curves, curves.columns.to_numpy(dtype=float), times)
    info["train_c_index"] = float(_xgbse_score(model, X_train, y_train))
    return Prediction(risk_from_survival(curves)[None], surv[None], info)


def fit_xgbse(X_train, y_train, X_test, times) -> Prediction:
    """XGBSE with a hyperparameter search that really changes the model."""
    space = xgbse_search_space()
    search = RandomizedSearchCV(TunableXGBSE(), {"xgb_params": space}, n_iter=len(space), cv=5,
                                scoring=_xgbse_score, random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    scores = search.cv_results_["mean_test_score"]
    info = {"cv_c_index": float(search.best_score_), "best_params": search.best_params_["xgb_params"],
            "distinct_cv_scores": int(len(np.unique(np.round(scores, 10)))),
            "cv_score_range": [float(np.nanmin(scores)), float(np.nanmax(scores))],
            "failed_configs": int(np.isnan(scores).sum())}
    return _xgbse_prediction(search.best_estimator_, X_train, y_train, X_test, times, info)


def fit_xgbse_default(X_train, y_train, X_test, times) -> Prediction:
    """XGBSE with the library defaults: the model the thesis actually trained."""
    model = TunableXGBSE().fit(X_train, y_train)
    return _xgbse_prediction(model, X_train, y_train, X_test, times, {})


# --------------------------------------------------------------------------- DeepSurv
EXPOSURE_COLS = ["exposures.alcohol_days_per_week", "exposures.alcohol_drinks_per_day",
                 "exposures.pack_years_smoked"]


def fit_deepsurv(X_train, y_train, X_test, times, n_runs: int = 5) -> Prediction:
    """DeepSurv as in the thesis, trained `n_runs` times with different seeds.

    Unlike the thesis notebook, the imputer and the scaler are refitted on the training part
    of every run, so the validation patients used for early stopping never influence them.
    """
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH

    def add_missing_indicators(X):
        X = X.copy()
        for col in EXPOSURE_COLS:
            if col in X.columns:
                X[f"{col}_is_missing"] = X[col].isna().astype(float)
        return X

    X_train, X_test = add_missing_indicators(X_train), add_missing_indicators(X_test)
    binary = [c for c in X_train.columns if X_train[c].nunique() == 2]
    sparse = [c for c in binary if X_train[c].sum() < 10]
    X_train, X_test = X_train.drop(columns=sparse), X_test.drop(columns=sparse)

    def target(y):
        return y["time"].astype("float32"), y["event"].astype("float32")

    risks, survs, epochs = [], [], []
    for run in range(n_runs):
        seed = SEED + run
        fit_idx, val_idx = train_test_split(np.arange(len(X_train)), test_size=0.15, random_state=seed,
                                            stratify=y_train["event"])
        imputer = IterativeImputer(random_state=SEED, max_iter=10).fit(X_train.iloc[fit_idx])
        scaler = StandardScaler().fit(imputer.transform(X_train.iloc[fit_idx]))

        def prepare(X):
            return scaler.transform(imputer.transform(X)).astype("float32")

        torch.manual_seed(seed)
        np.random.seed(seed)
        net = tt.practical.MLPVanilla(X_train.shape[1], [64, 32, 16], 1, batch_norm=True, dropout=0.4)
        model = CoxPH(net, tt.optim.Adam(lr=0.001, weight_decay=0.01))
        log = model.fit(prepare(X_train.iloc[fit_idx]), target(y_train[fit_idx]), batch_size=64, epochs=500,
                        callbacks=[tt.callbacks.EarlyStopping(patience=25)], verbose=False,
                        val_data=(prepare(X_train.iloc[val_idx]), target(y_train[val_idx])))
        model.compute_baseline_hazards(prepare(X_train), target(y_train))

        test = prepare(X_test)
        curves = model.predict_surv_df(test)  # rows: times, columns: patients
        risks.append(model.predict(test).flatten())
        survs.append(_interpolate(curves.T, curves.index.to_numpy(dtype=float), times))
        epochs.append(len(log.to_pandas()))

    info = {"n_runs": n_runs, "epochs": epochs, "n_features": int(X_train.shape[1]), "dropped_sparse": sparse}
    return Prediction(np.array(risks), np.array(survs), info)


MODELS = {
    "cox": ("Cox PH (ridge)", fit_cox),
    "rsf": ("Random Survival Forest", fit_rsf),
    "xgbse": ("XGBSE (tuned)", fit_xgbse),
    "xgbse_default": ("XGBSE (library defaults)", fit_xgbse_default),
    "deepsurv": ("DeepSurv (5 runs)", fit_deepsurv),
}
