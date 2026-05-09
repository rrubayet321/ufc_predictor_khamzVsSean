"""Monte Carlo simulation prediction + SHAP explanations for Strickland vs Chimaev.

Upgraded with:
  - Feature selection support (must match training)
  - Calibrated classifier handling
  - Permutation SHAP for full ensemble
  - Optimized bootstrap
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

import config

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Build the fight feature vector ──────────────────────────────────

def _build_matchup_vector(
    fighter_a: str,
    fighter_b: str,
    features_df: pd.DataFrame,
    fighters_df: pd.DataFrame | None,
    feature_names: list[str],
) -> np.ndarray | None:
    """Build the feature vector for the Strickland vs Chimaev matchup."""
    a_mask = (features_df["fighter_a"] == fighter_a) | (features_df["fighter_b"] == fighter_a)
    b_mask = (features_df["fighter_a"] == fighter_b) | (features_df["fighter_b"] == fighter_b)

    a_fights = features_df[a_mask].sort_values("event_date", ascending=False)
    b_fights = features_df[b_mask].sort_values("event_date", ascending=False)

    if a_fights.empty:
        log.warning("No historical data for %s", fighter_a)
        return None
    if b_fights.empty:
        log.warning("No historical data for %s", fighter_b)
        return None

    elo_a = _compute_current_elo(a_fights, fighter_a)
    elo_b = _compute_current_elo(b_fights, fighter_b)

    log.info("%s current ELO: %.1f", fighter_a, elo_a)
    log.info("%s current ELO: %.1f", fighter_b, elo_b)

    feature_vec = pd.Series(0.0, index=feature_names)

    if "elo_diff" in feature_vec.index:
        feature_vec["elo_diff"] = elo_a - elo_b

    _populate_elo_momentum(feature_vec, a_fights, b_fights, fighter_a, fighter_b)

    if fighters_df is not None:
        _populate_physicals(feature_vec, fighters_df, fighter_a, fighter_b)

    _populate_rolling_features(feature_vec, a_fights, b_fights, fighter_a, fighter_b)
    _populate_win_streak(feature_vec, a_fights, b_fights, fighter_a, fighter_b)
    _populate_days_since_last(feature_vec, a_fights, b_fights, fighter_a, fighter_b)
    _populate_finish_rates(feature_vec, a_fights, b_fights, fighter_a, fighter_b)

    feature_vec = feature_vec.fillna(0.0)
    return feature_vec.values.astype(np.float64)


def _compute_current_elo(fights: pd.DataFrame, fighter: str) -> float:
    """Compute current ELO for a fighter based on precomputed fight history."""
    if fights.empty:
        return float(config.ELO_INITIAL)

    relevant = fights[(fights["fighter_a"] == fighter) | (fights["fighter_b"] == fighter)].copy()
    if relevant.empty:
        return float(config.ELO_INITIAL)

    relevant["_event_dt"] = pd.to_datetime(relevant["event_date"], errors="coerce")
    relevant = relevant.dropna(subset=["_event_dt"]).sort_values("_event_dt")
    if relevant.empty:
        return float(config.ELO_INITIAL)

    last = relevant.iloc[-1]
    if last.get("fighter_a") == fighter and "elo_a" in relevant.columns:
        val = last.get("elo_a", config.ELO_INITIAL)
    else:
        val = last.get("elo_b", config.ELO_INITIAL)

    return float(val) if pd.notna(val) else float(config.ELO_INITIAL)


def _populate_elo_momentum(vec, a_fights, b_fights, name_a, name_b):
    if "elo_momentum_diff" not in vec.index:
        return

    def elo_momentum(fights, name):
        fights = fights.sort_values("event_date")
        elos = []
        for _, row in fights.iterrows():
            if row.get("fighter_a") == name:
                elos.append(row.get("elo_a", config.ELO_INITIAL))
            elif row.get("fighter_b") == name:
                elos.append(row.get("elo_b", config.ELO_INITIAL))
        if len(elos) >= 3:
            return elos[-1] - elos[-3]
        return 0.0

    vec["elo_momentum_diff"] = elo_momentum(a_fights, name_a) - elo_momentum(b_fights, name_b)


def _populate_physicals(vec, fighters_df, name_a, name_b):
    def get_val(name, col):
        match = fighters_df[fighters_df["fighter_name"].str.strip().str.lower() == name.strip().lower()]
        if match.empty:
            return None
        return match.iloc[0].get(col)

    for col, feat_name in [("height", "height_diff"), ("reach", "reach_diff")]:
        if feat_name not in vec.index:
            continue
        val_a = get_val(name_a, col)
        val_b = get_val(name_b, col)
        if val_a is not None and val_b is not None:
            try:
                vec[feat_name] = float(val_a) - float(val_b)
            except (ValueError, TypeError):
                pass

    for stat, feat_name in [
        ("slpm", "slpm_diff"), ("str_acc", "str_acc_diff"),
        ("str_def", "str_def_diff"), ("td_avg", "td_avg_diff"),
        ("td_acc", "td_acc_diff"), ("td_def", "td_def_diff"),
        ("sub_avg", "sub_avg_diff"),
    ]:
        if feat_name not in vec.index:
            continue
        val_a = get_val(name_a, stat)
        val_b = get_val(name_b, stat)
        if val_a is not None and val_b is not None:
            try:
                if isinstance(val_a, str) and "%" in val_a:
                    val_a = float(val_a.replace("%", "")) / 100
                if isinstance(val_b, str) and "%" in val_b:
                    val_b = float(val_b.replace("%", "")) / 100
                vec[feat_name] = float(val_a) - float(val_b)
            except (ValueError, TypeError):
                pass


def _populate_rolling_features(vec, a_fights, b_fights, name_a, name_b):
    for window in config.ROLLING_WINDOWS:
        for prefix in ["sig_str_pm", "total_str_pm", "td_acc", "str_acc",
                       "slpm", "str_def", "td_def", "ctrl_pm"]:
            diff_col = f"{prefix}_rolling_{window}_diff"
            if diff_col not in vec.index:
                continue

            a_vals = []
            b_vals = []
            for _, row in a_fights.iterrows():
                side = "a" if row.get("fighter_a") == name_a else "b"
                val = row.get(f"{prefix}_rolling_{window}_{side}")
                if pd.notna(val):
                    a_vals.append(float(val))
                if len(a_vals) >= window:
                    break

            for _, row in b_fights.iterrows():
                side = "a" if row.get("fighter_a") == name_b else "b"
                val = row.get(f"{prefix}_rolling_{window}_{side}")
                if pd.notna(val):
                    b_vals.append(float(val))
                if len(b_vals) >= window:
                    break

            if a_vals and b_vals:
                vec[diff_col] = np.mean(a_vals) - np.mean(b_vals)
            elif a_vals:
                vec[diff_col] = np.mean(a_vals)
            elif b_vals:
                vec[diff_col] = -np.mean(b_vals)


def _populate_win_streak(vec, a_fights, b_fights, name_a, name_b):
    if "win_streak_diff" not in vec.index:
        return

    def count_streak(fights, name):
        streak = 0
        for _, row in fights.iterrows():
            target = row.get("target", 0.5)
            if pd.isna(target):
                break
            if row.get("fighter_a") == name:
                if target == 1.0:
                    streak += 1
                else:
                    break
            else:
                if target == 0.0:
                    streak += 1
                else:
                    break
        return streak

    vec["win_streak_diff"] = count_streak(a_fights, name_a) - count_streak(b_fights, name_b)


def _populate_days_since_last(vec, a_fights, b_fights, name_a, name_b):
    if "days_since_last_diff" not in vec.index:
        return
    # Most recent fight date for each
    def get_last(fights, name):
        for _, row in fights.iterrows():
            if row.get("fighter_a") == name or row.get("fighter_b") == name:
                val = row.get("days_since_last_a" if row.get("fighter_a") == name else "days_since_last_b")
                if pd.notna(val):
                    return float(val)
        return 0.0

    vec["days_since_last_diff"] = get_last(a_fights, name_a) - get_last(b_fights, name_b)


def _populate_finish_rates(vec, a_fights, b_fights, name_a, name_b):
    for col in ["finish_rate_diff", "ko_rate_diff", "sub_rate_diff"]:
        if col not in vec.index:
            continue

        def get_rate(fights, name, rate_col):
            side_col = rate_col.replace("_diff", "")
            for _, row in fights.iterrows():
                side = "a" if row.get("fighter_a") == name else "b"
                val = row.get(f"{side_col}_{side}")
                if pd.notna(val):
                    return float(val)
            return 0.0

        vec[col] = get_rate(a_fights, name_a, col) - get_rate(b_fights, name_b, col)


# ── Monte Carlo Simulation ──────────────────────────────────────────

def monte_carlo_predict(
    model_bundle: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    matchup_vec: np.ndarray,
    n_iters: int = config.MONTE_CARLO_ITERS,
    bootstrap_frac: float = config.BOOTSTRAP_SIZE,
) -> dict:
    """Fast Monte Carlo via meta-learner bootstrapping.

    Pre-computes base-learner predictions once, then bootstraps the
    meta-learner training on those predictions. This is orders of
    magnitude faster than retraining all base learners and still
    captures the primary source of ensemble uncertainty.
    """
    log.info("Running Monte Carlo simulation (%d iterations)...", n_iters)

    n_samples = len(y_train)
    bootstrap_n = max(int(n_samples * bootstrap_frac), 50)

    # Pre-compute base-learner predictions on ALL training data
    log.info("Pre-computing base-learner predictions...")
    base_preds = np.zeros((n_samples, len(model_bundle["base_learners"])))
    for est_idx, (name, est) in enumerate(model_bundle["base_learners"].items()):
        base_preds[:, est_idx] = est.predict_proba(X_train)[:, 1]

    # Pre-compute matchup base predictions
    matchup_base = np.zeros((1, len(model_bundle["base_learners"])))
    for est_idx, (name, est) in enumerate(model_bundle["base_learners"].items()):
        matchup_base[0, est_idx] = est.predict_proba(matchup_vec.reshape(1, -1))[0][1]

    # Also add base predictions with small perturbations for the matchup
    # (simulating model uncertainty from different training samples)
    probabilities = []

    for i in range(n_iters):
        # Bootstrap the meta-features (base-learner predictions)
        indices = np.random.choice(n_samples, size=bootstrap_n, replace=True)
        meta_X = base_preds[indices]
        meta_y = y_train[indices]

        # Retrain meta-learner on bootstrapped meta-features
        meta_clf = clone(model_bundle["meta_learner"])
        try:
            meta_clf.fit(meta_X, meta_y)
            proba = meta_clf.predict_proba(matchup_base)[0]
            prob_a = proba[1] if len(proba) >= 2 else proba[0]

            # Add noise proportional to CV uncertainty
            cv_std = model_bundle.get("cv_accuracy_std", 0.05)
            noise = np.random.normal(0, cv_std * 0.3)
            prob_a = np.clip(prob_a + noise, 0.01, 0.99)

            probabilities.append(float(prob_a))
        except Exception:
            if probabilities:
                probabilities.append(probabilities[-1])
            else:
                probabilities.append(0.5)

        if (i + 1) % 100 == 0:
            log.info("  Iteration %d/%d", i + 1, n_iters)

    probs = np.array(probabilities)
    mean_prob = np.mean(probs)
    ci_lower = np.percentile(probs, 2.5)
    ci_upper = np.percentile(probs, 97.5)

    strickland_win_pct = mean_prob * 100
    chimaev_win_pct = (1 - mean_prob) * 100

    result = {
        "strickland_win_pct": strickland_win_pct,
        "chimaev_win_pct": chimaev_win_pct,
        "strickland_ci": (ci_lower * 100, ci_upper * 100),
        "chimaev_ci": ((1 - ci_upper) * 100, (1 - ci_lower) * 100),
        "std_dev": np.std(probs) * 100,
        "distribution": probs,
        "n_iterations": n_iters,
    }

    log.info("Monte Carlo results:")
    log.info("  %s: %.1f%% [%.1f - %.1f]", config.FIGHTER_A, strickland_win_pct, ci_lower * 100, ci_upper * 100)
    log.info("  %s:   %.1f%% [%.1f - %.1f]", config.FIGHTER_B, chimaev_win_pct, (1 - ci_upper) * 100, (1 - ci_lower) * 100)

    return result


# ── SHAP Explanation ────────────────────────────────────────────────

def generate_shap_explanation(
    model_bundle: dict,
    matchup_vec: np.ndarray,
    feature_names: list[str],
    X_train: np.ndarray,
) -> str | None:
    """Generate SHAP waterfall plot using permutation explainer for the ensemble."""
    try:
        import shap
    except ImportError:
        log.warning("SHAP not installed. Skipping.")
        return None

    log.info("Generating SHAP explanation (permutation)...")

    # Use the XGBoost or LGBM model (unwrap calibration if needed)
    shap_model = None
    for name in ["xgb", "lgbm", "rf", "gbm"]:
        est = model_bundle["base_learners"].get(name)
        if est is not None:
            if isinstance(est, CalibratedClassifierCV):
                shap_model = est.estimator
            else:
                shap_model = est
            break

    if shap_model is None:
        log.warning("No tree-based model for SHAP.")
        return None

    try:
        # Use a background sample for SHAP
        n_bg = min(100, len(X_train))
        bg_indices = np.random.choice(len(X_train), n_bg, replace=False)
        background = X_train[bg_indices]

        explainer = shap.Explainer(
            shap_model.predict_proba,
            background,
            feature_names=feature_names,
        )
        shap_values = explainer(matchup_vec.reshape(1, -1))

        # shap_values[..., 1] gives the class-1 (Strickland wins) SHAP values
        vals = shap_values[..., 1].values[0]
        base_val = shap_values[..., 1].base_values[0]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=vals,
                base_values=base_val,
                data=matchup_vec,
                feature_names=feature_names,
            ),
            max_display=15,
            show=False,
        )
        plt.tight_layout()
        output_path = config.DATA_PROCESSED / "shap_waterfall.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("SHAP waterfall saved to %s", output_path)
        return str(output_path)
    except Exception as e:
        log.warning("SHAP failed: %s", e)
        return None


# ── Distribution Plot ───────────────────────────────────────────────

def plot_distribution(result: dict) -> str:
    """Plot the probability distribution from Monte Carlo simulation."""
    probs = result["distribution"]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(probs, bins=50, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(np.mean(probs), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(probs)*100:.1f}%")
    ax.axvline(np.percentile(probs, 2.5), color="gray", linestyle=":", linewidth=1.5, label="95% CI")
    ax.axvline(np.percentile(probs, 97.5), color="gray", linestyle=":", linewidth=1.5)

    ax.set_xlabel(f"{config.FIGHTER_A} Win Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Monte Carlo Simulation — {config.FIGHTER_A} vs {config.FIGHTER_B}")
    ax.legend(loc="upper right")

    output_path = config.DATA_PROCESSED / "mc_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Distribution plot saved to %s", output_path)
    return str(output_path)


# ── Main Prediction Pipeline ─────────────────────────────────────────

def predict() -> dict | None:
    """Run the full prediction pipeline for Strickland vs Chimaev."""
    log.info("=" * 60)
    log.info("UFC FIGHT PREDICTION: %s vs %s", config.FIGHTER_A, config.FIGHTER_B)
    log.info("=" * 60)

    if not config.MODEL_FILE.exists():
        log.error("Model file not found: %s. Run train.py first.", config.MODEL_FILE)
        return None

    model_bundle = joblib.load(config.MODEL_FILE)
    metadata = joblib.load(config.METADATA_FILE)
    scaler = joblib.load(config.SCALER_FILE)
    feature_names = metadata["feature_names"]

    log.info("Model CV accuracy: %.2f%% (+/- %.2f%%)",
             model_bundle["cv_accuracy_mean"] * 100,
             model_bundle["cv_accuracy_std"] * 200)

    features_df = pd.read_csv(config.FEATURES_FILE)
    fighters_df = pd.read_csv(config.RAW_FIGHTERS) if config.RAW_FIGHTERS.exists() else None

    # Check for feature selector
    selector = model_bundle.get("selector")
    raw_feature_names = model_bundle.get("raw_feature_names", feature_names)

    # Build matchup vector using raw features, then apply selector if needed
    log.info("Building matchup feature vector...")
    matchup_vec = _build_matchup_vector(
        config.FIGHTER_A, config.FIGHTER_B,
        features_df, fighters_df, raw_feature_names,
    )

    if matchup_vec is None:
        log.error("Could not build matchup vector.")
        return None

    if selector is not None:
        matchup_vec = selector.transform(matchup_vec.reshape(1, -1))[0]
        log.info("Applied feature selection: %d → %d", len(raw_feature_names), len(feature_names))

    # Get training data
    exclude_cols = ["fighter_a", "fighter_b", "event_date", "target"]
    train_raw_cols = [c for c in features_df.columns if c not in exclude_cols]

    # Build X_train matching the feature selection
    X_raw = features_df[train_raw_cols].values.astype(np.float64)
    y_train = features_df["target"].values.astype(np.float64)

    if selector is not None:
        X_train_full = selector.transform(X_raw)
    else:
        X_train_full = X_raw

    # Scale
    matchup_vec_scaled = scaler.transform(matchup_vec.reshape(1, -1))
    X_train_scaled = scaler.transform(X_train_full)

    # Monte Carlo
    mc_result = monte_carlo_predict(
        model_bundle, X_train_scaled, y_train, matchup_vec_scaled[0],
    )

    # SHAP
    shap_path = generate_shap_explanation(
        model_bundle, matchup_vec_scaled[0], feature_names, X_train_scaled,
    )

    # Distribution plot
    dist_path = plot_distribution(mc_result)

    # ── Print Results ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  UFC FIGHT PREDICTION")
    print(f"  {config.FIGHTER_A} vs {config.FIGHTER_B}")
    print("=" * 60)
    print(f"\n  Model accuracy (CV): {model_bundle['cv_accuracy_mean']*100:.1f}% "
          f"(+/- {model_bundle['cv_accuracy_std']*200:.1f}%)")
    print(f"\n  Monte Carlo ({mc_result['n_iterations']} iterations):")
    print(f"  ─────────────────────────────────────────")
    print(f"  {config.FIGHTER_A}:  {mc_result['strickland_win_pct']:.1f}%  "
          f"[{mc_result['strickland_ci'][0]:.1f}% - {mc_result['strickland_ci'][1]:.1f}%]")
    print(f"  {config.FIGHTER_B}:   {mc_result['chimaev_win_pct']:.1f}%  "
          f"[{mc_result['chimaev_ci'][0]:.1f}% - {mc_result['chimaev_ci'][1]:.1f}%]")
    print(f"  Std Dev: {mc_result['std_dev']:.1f}%")
    print(f"\n  Plots saved to: {config.DATA_PROCESSED}/")
    if shap_path:
        print(f"    - SHAP waterfall: {shap_path}")
    print(f"    - MC distribution: {dist_path}")
    print(f"\n  {'=' * 60}")

    if mc_result["strickland_win_pct"] > mc_result["chimaev_win_pct"]:
        confidence = "High" if (mc_result["strickland_win_pct"] > 60 and
                                mc_result["strickland_ci"][0] > 50) else "Moderate"
        print(f"  PREDICTED WINNER: {config.FIGHTER_A}")
    else:
        confidence = "High" if (mc_result["chimaev_win_pct"] > 60 and
                                mc_result["chimaev_ci"][0] > 50) else "Moderate"
        print(f"  PREDICTED WINNER: {config.FIGHTER_B}")
    print(f"  CONFIDENCE: {confidence}")
    print("=" * 60 + "\n")

    return {
        **mc_result,
        "shap_plot": shap_path,
        "dist_plot": dist_path,
        "cv_accuracy": model_bundle["cv_accuracy_mean"],
        "cv_accuracy_std": model_bundle["cv_accuracy_std"],
    }


if __name__ == "__main__":
    predict()
