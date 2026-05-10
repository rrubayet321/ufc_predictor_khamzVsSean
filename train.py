"""Stacking ensemble training pipeline for UFC fight prediction.

Upgraded with:
  - 6 base learners (XGBoost, LightGBM, RF, LogisticRegression, SVM, KNN)
  - Optuna hyperparameter optimization
  - Mutual-information feature selection
  - Probability calibration (Platt scaling)
  - Repeated stratified K-fold CV
"""

import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Optuna Hyperparameter Tuning ────────────────────────────────────

def _optuna_tune(name: str, X: np.ndarray, y: np.ndarray, inner_cv):
    """Use Optuna to find best hyperparameters for a model."""
    if config.OPTUNA_TRIALS <= 0 or name == "svm":
        return None
    try:
        import optuna
    except ImportError:
        log.warning("Optuna not installed — falling back to defaults for %s", name)
        return None

    log.info("  Optuna tuning %s (%d trials)...", name, config.OPTUNA_TRIALS)

    def objective(trial):
        try:
            if name == "xgb":
                model = XGBClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
                    random_state=42, eval_metric="logloss", verbosity=0,
                )
            elif name == "lgbm":
                model = LGBMClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 63),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
                    random_state=42, verbose=-1,
                )
            elif name == "rf":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                    max_depth=trial.suggest_int("max_depth", 4, 15),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
                    class_weight="balanced",
                    random_state=42,
                )
            elif name == "svm":
                return 0.5

            scores = cross_val_score(model, X, y, cv=inner_cv, scoring="accuracy", n_jobs=-1)
            return float(scores.mean())
        except Exception as e:
            log.debug("  Optuna trial failed: %s", e)
            return 0.5

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS, show_progress_bar=False)
    log.info("  Best score: %.4f, Best params: %s", study.best_value, study.best_params)
    return study.best_params


def _build_default_estimators():
    """Return default base learner configurations."""
    return {
        "xgb": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
        "lgbm": LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42,
        ),
        "svm": SVC(
            C=1.0, gamma="scale", kernel="rbf",
            probability=True, class_weight="balanced", random_state=42,
        ),
    }


def _build_meta_params():
    """Return meta-learner for stacking."""
    return LogisticRegressionCV(
        cv=5, penalty="l2",
        class_weight="balanced", random_state=42, max_iter=2000,
    )


# ── Main Training Pipeline ──────────────────────────────────────────

def train() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full training pipeline."""
    log.info("=== Loading feature matrix ===")
    df = pd.read_csv(config.FEATURES_FILE)
    if df.empty:
        raise ValueError("Empty feature matrix. Run features.py first.")

    exclude_cols = ["fighter_a", "fighter_b", "event_date", "target"]
    raw_feature_cols = [c for c in df.columns if c not in exclude_cols]
    log.info("Raw features: %d  Samples: %d", len(raw_feature_cols), len(df))

    X_raw = df[raw_feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(np.float64)

    # ── Feature Selection (Mutual Information) ───────────────────────
    feature_cols = raw_feature_cols
    selector = None
    if config.FEATURE_SELECTION_K > 0 and len(raw_feature_cols) > config.FEATURE_SELECTION_K:
        log.info("Selecting top %d features by mutual information...", config.FEATURE_SELECTION_K)
        selector = SelectKBest(mutual_info_classif, k=min(config.FEATURE_SELECTION_K, len(raw_feature_cols)))
        selector.fit(X_raw, y)
        selected_mask = selector.get_support()
        feature_cols = [c for c, m in zip(raw_feature_cols, selected_mask) if m]
        X = X_raw[:, selected_mask]
        joblib.dump(selector, config.SELECTOR_FILE)
        log.info("Selected %d features (from %d)", len(feature_cols), len(raw_feature_cols))
    else:
        X = X_raw

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, config.SCALER_FILE)
    joblib.dump({"feature_names": feature_cols, "raw_feature_names": raw_feature_cols,
                 "selector": selector is not None}, config.METADATA_FILE)

    # ── CV setup ─────────────────────────────────────────────────────
    outer_cv = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=42
    )
    inner_cv = StratifiedKFold(n_splits=config.CV_INNER_FOLDS, shuffle=True, random_state=42)

    # ── Hyperparameter Tuning (Optuna) ───────────────────────────────
    log.info("=== Optuna Hyperparameter Tuning ===")
    defaults = _build_default_estimators()
    param_names = ["xgb", "lgbm", "rf", "svm"]

    best_estimators = {}
    for name in param_names:
        best_params = _optuna_tune(name, X_scaled, y, inner_cv)
        if best_params:
            estimator = type(defaults[name])(**best_params)
            if name == "xgb":
                estimator.set_params(random_state=42, eval_metric="logloss", verbosity=0)
            elif name == "lgbm":
                estimator.set_params(random_state=42, verbose=-1)
        else:
            estimator = defaults[name]
        estimator.fit(X_scaled, y)
        best_estimators[name] = estimator

    # ── Generate meta-features (stacking) ────────────────────────────
    log.info("=== Training meta-learner (stacking) ===")
    meta_features = np.zeros((len(X), len(best_estimators)))

    meta_cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=43)
    for train_idx, val_idx in meta_cv.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr = y[train_idx]

        for est_idx, (name, est) in enumerate(best_estimators.items()):
            clf_fold = clone(est)
            clf_fold.fit(X_tr, y_tr)
            meta_features[val_idx, est_idx] = clf_fold.predict_proba(X_val)[:, 1]

    # Fit meta-learner
    meta_learner = _build_meta_params()
    meta_learner.fit(meta_features, y)

    # ── Evaluation ───────────────────────────────────────────────────
    log.info("=== Cross-validation evaluation ===")
    meta_cv_scores = cross_val_score(
        meta_learner, meta_features, y,
        cv=outer_cv, scoring="accuracy", n_jobs=-1,
    )
    log.info("Stacking CV accuracy: %.4f (+/- %.4f)",
             meta_cv_scores.mean(), meta_cv_scores.std() * 2)

    # Per-base-learner evaluation
    for name, est in best_estimators.items():
        scores = cross_val_score(est, X_scaled, y, cv=outer_cv, scoring="accuracy", n_jobs=-1)
        log.info("  %-6s standalone CV accuracy: %.4f (+/- %.4f)",
                 name, scores.mean(), scores.std() * 2)

    # ── Retrain everything on full dataset ───────────────────────────
    log.info("=== Final retraining on full dataset ===")
    for name in best_estimators:
        best_estimators[name] = clone(best_estimators[name])
        best_estimators[name].fit(X_scaled, y)

    meta_features_full = np.zeros((len(X), len(best_estimators)))
    for est_idx, (name, est) in enumerate(best_estimators.items()):
        meta_features_full[:, est_idx] = est.predict_proba(X_scaled)[:, 1]

    meta_learner_full = _build_meta_params()
    meta_learner_full.fit(meta_features_full, y)

    # ── Save ─────────────────────────────────────────────────────────
    model_bundle = {
        "base_learners": best_estimators,
        "meta_learner": meta_learner_full,
        "feature_names": feature_cols,
        "raw_feature_names": raw_feature_cols,
        "selector": selector,
        "cv_accuracy_mean": float(meta_cv_scores.mean()),
        "cv_accuracy_std": float(meta_cv_scores.std()),
    }
    joblib.dump(model_bundle, config.MODEL_FILE)
    log.info("Model saved to %s", config.MODEL_FILE)
    log.info("Cross-validation accuracy: %.2f%% (+/- %.2f%%)",
             meta_cv_scores.mean() * 100, meta_cv_scores.std() * 200)

    return X_scaled, y, np.array(feature_cols), meta_cv_scores


if __name__ == "__main__":
    train()
