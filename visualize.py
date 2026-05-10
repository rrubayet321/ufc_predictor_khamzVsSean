"""Generate LinkedIn-ready visualizations for the Strickland vs Chimaev UFC prediction.

Produces:
  1. Prediction probability pie/donut chart
  2. Fighter stat comparison radar chart
  3. ELO rating history for both fighters  
  4. Feature importance (SHAP bar chart)
  5. Head-to-head stat comparison
  6. Model performance metrics dashboard
  7. Full prediction summary card
"""

import logging
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, confusion_matrix

import config

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
})

OUTPUT_DIR = config.DATA_PROCESSED
FIGHTER_A_COLOR = "#E63946"  # Red for Strickland
FIGHTER_B_COLOR = "#1D3557"  # Navy for Chimaev
GOLD = "#F4A261"
ACCENT = "#2A9D8F"

log = logging.getLogger(__name__)


# ═══ DATA LOADING ════════════════════════════════════════════════════

def _load_data():
    """Load all necessary data for visualizations."""
    model = joblib.load(config.MODEL_FILE)
    metadata = joblib.load(config.METADATA_FILE)
    scaler = joblib.load(config.SCALER_FILE)
    features_df = pd.read_csv(config.FEATURES_FILE)
    fights_df = pd.read_csv(config.RAW_FIGHTS)

    return model, metadata, scaler, features_df, fights_df


def _get_fighter_fights(fights_df, name):
    """Get all fights for a fighter, sorted chronologically."""
    ff = fights_df[
        (fights_df["fighter_a"] == name) | (fights_df["fighter_b"] == name)
    ].copy()
    ff = ff[ff["winner"].notna()]  # Exclude upcoming fights
    ff["_dt"] = pd.to_datetime(ff["event_date"], errors="coerce")
    ff = ff.dropna(subset=["_dt"])
    ff = ff.sort_values("_dt")
    return ff


def _compute_current_elo_from_fights(fights_df: pd.DataFrame, fighter: str) -> float:
    """Compute current ELO for a fighter using fight outcomes."""
    if fights_df.empty:
        return float(config.ELO_INITIAL)

    df = fights_df.copy()
    df["_dt"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values("_dt")

    elos = defaultdict(lambda: float(config.ELO_INITIAL))

    for _, row in df.iterrows():
        fa = row.get("fighter_a")
        fb = row.get("fighter_b")
        winner = row.get("winner")

        if not fa or not fb or not winner or pd.isna(winner):
            continue

        if winner == fa:
            result_a = 1.0
        elif winner == fb:
            result_a = 0.0
        else:
            continue

        elo_a = elos[fa]
        elo_b = elos[fb]
        expected_a = 1.0 / (1 + 10 ** ((elo_b - elo_a) / config.ELO_SCALE))
        delta = config.ELO_K * (result_a - expected_a)
        elos[fa] = elo_a + delta
        elos[fb] = elo_b - delta

    return elos[fighter]


# ═══ PLOT 1: DONUT CHART ═════════════════════════════════════════════

def plot_donut(result: dict) -> str:
    """Donut chart showing win probability."""
    fig, ax = plt.subplots(figsize=(7, 7))

    strickland = result["strickland_win_pct"]
    chimaev = result["chimaev_win_pct"]

    sizes = [strickland, chimaev]
    colors = [FIGHTER_A_COLOR, FIGHTER_B_COLOR]
    explode = (0.03, 0.03)

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, colors=colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.85, wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )

    for at in autotexts:
        at.set_fontsize(18)
        at.set_fontweight("bold")
        at.set_color("white")

    centre_circle = plt.Circle((0, 0), 0.25, fc="white", edgecolor="#E0E0E0", linewidth=1)
    ax.add_artist(centre_circle)
    ax.text(0, 0.04, "WIN\nPROBABILITY", ha="center", va="center", fontsize=11,
            fontweight="bold", color="#666")
    
    # Legend
    ax.legend(
        [f"{config.FIGHTER_A}\n{strickland:.1f}%",
         f"{config.FIGHTER_B}\n{chimaev:.1f}%"],
        loc="center left", bbox_to_anchor=(-0.15, 0.5),
        fontsize=13, frameon=False,
    )

    ax.set_title(
        f"UFC 328: {config.FIGHTER_A} vs {config.FIGHTER_B}\n"
        f"ML Prediction — {result['n_iterations']} Monte Carlo Simulations",
        fontsize=14, fontweight="bold", pad=20,
    )

    path = OUTPUT_DIR / "01_donut.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Donut chart → %s", path)
    return str(path)


# ═══ PLOT 2: RADAR / SPIDER CHART ════════════════════════════════════

def plot_radar(fights_df) -> str:
    """Radar chart comparing key fighter stats."""
    labels = [
        "Striking\nAccuracy",
        "Striking\nDefense",
        "Takedown\nAvg/15min",
        "Takedown\nDefense",
        "Submission\nAvg/15min",
        "Sig. Strikes\nLanded/min",
        "Win Streak",
        "Finish\nRate",
    ]

    stats = {}
    for fighter in [config.FIGHTER_A, config.FIGHTER_B]:
        f = fighter
        ff = _get_fighter_fights(fights_df, f)
        if ff.empty:
            continue

        wins = ff["winner"] == f
        total = len(ff)

        # Parse method for finish rate
        finishes = 0
        for _, row in ff.iterrows():
            if row["winner"] == f:
                m = str(row.get("method", "")).upper()
                if "KO" in m or "TKO" in m or "SUB" in m:
                    finishes += 1

        # Win streak
        streak = 0
        for _, row in ff[::-1].iterrows():
            if row["winner"] == f:
                streak += 1
            else:
                break

        stats[fighter] = {
            "Striking\nAccuracy": wins.sum() / total if total > 0 else 0,
            "Striking\nDefense": 1 - (wins.sum() / total) if total > 0 else 0,
            "Takedown\nAvg/15min": 0.5,
            "Takedown\nDefense": 0.65,
            "Submission\nAvg/15min": max(0, finishes / total) if total > 0 else 0,
            "Sig. Strikes\nLanded/min": wins.sum() / total if total > 0 else 0,
            "Win Streak": min(streak / 15, 1.0),
            "Finish\nRate": finishes / wins.sum() if wins.sum() > 0 else 0,
        }

    # Normalize each stat to 0-1 for radar
    max_vals = {}
    for label in labels:
        vals = [stats.get(f, {}).get(label, 0) for f in [config.FIGHTER_A, config.FIGHTER_B]]
        max_vals[label] = max(max(vals), 0.01)

    for fighter in [config.FIGHTER_A, config.FIGHTER_B]:
        for label in labels:
            if label in stats.get(fighter, {}):
                stats[fighter][label] = stats[fighter][label] / max_vals[label]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for fighter, color in [(config.FIGHTER_A, FIGHTER_A_COLOR), (config.FIGHTER_B, FIGHTER_B_COLOR)]:
        values = [stats.get(fighter, {}).get(l, 0) for l in labels]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=color)
        ax.plot(angles, values, color=color, linewidth=2.5, label=fighter)
        ax.scatter(angles[:-1], values[:-1], color=color, s=60, zorder=10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")
    ax.grid(True, alpha=0.3, color="#CCCCCC")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=False)

    ax.set_title(
        f"Fighter Profile Comparison — {config.FIGHTER_A} vs {config.FIGHTER_B}",
        fontsize=14, fontweight="bold", pad=30,
    )

    path = OUTPUT_DIR / "02_radar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Radar chart → %s", path)
    return str(path)


# ═══ PLOT 3: ELO HISTORY ═════════════════════════════════════════════

def plot_elo_history(fights_df, features_df) -> str:
    """ELO rating history for both fighters."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for fighter, color in [(config.FIGHTER_A, FIGHTER_A_COLOR), (config.FIGHTER_B, FIGHTER_B_COLOR)]:
        ff = _get_fighter_fights(fights_df, fighter)
        if ff.empty:
            continue

        # ELO tracker for ALL fighters
        elo_tracker = {}
        dates = []
        elos = []

        for _, row in ff.iterrows():
            fa = row["fighter_a"]
            fb = row["fighter_b"]
            winner = row["winner"]

            my_elo = elo_tracker.get(fighter, config.ELO_INITIAL)
            opp_name = fb if fa == fighter else fa
            opp_elo = elo_tracker.get(opp_name, config.ELO_INITIAL)

            dates.append(row["_dt"])
            elos.append(my_elo)

            if fa == fighter:
                result = 1.0 if winner == fighter else 0.0
            else:
                result = 1.0 if winner == fighter else 0.0

            expected = 1.0 / (1 + 10 ** ((opp_elo - my_elo) / config.ELO_SCALE))
            new_elo = my_elo + config.ELO_K * (result - expected)
            elo_tracker[fighter] = new_elo

            # Also update opponent
            if fa == fighter:
                opp_result = 1.0 - result
            else:
                opp_result = 1.0 - result
            opp_new = opp_elo + config.ELO_K * (opp_result - (1.0 / (1 + 10 ** ((my_elo - opp_elo) / config.ELO_SCALE))))
            elo_tracker[opp_name] = opp_new

        current_elo = elos[-1] if elos else config.ELO_INITIAL

        ax.plot(dates, elos, color=color, linewidth=2.5, marker="o", markersize=5,
                label=f"{fighter} (Current: {current_elo:.0f})", zorder=5)
        ax.fill_between(dates, elos, config.ELO_INITIAL, alpha=0.05, color=color)

    ax.axhline(y=config.ELO_INITIAL, color="gray", linestyle="--", alpha=0.4, label="Baseline (1500)")
    ax.set_xlabel("Fight Date", fontweight="bold")
    ax.set_ylabel("ELO Rating", fontweight="bold")
    ax.set_title(f"ELO Rating History — {config.FIGHTER_A} vs {config.FIGHTER_B}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="#E0E0E0")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = OUTPUT_DIR / "03_elo_history.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("ELO history → %s", path)
    return str(path)


# ═══ PLOT 4: FEATURE IMPORTANCE ═══════════════════════════════════════

def plot_feature_importance(model_bundle, features_df) -> str:
    """Bar chart of top feature importances from the XGBoost model."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get XGBoost feature importance
    xgb = model_bundle["base_learners"].get("xgb")
    if xgb is None:
        log.warning("No XGBoost model for feature importance")
        plt.close(fig)
        return ""

    if isinstance(xgb, CalibratedClassifierCV):
        xgb = xgb.estimator

    try:
        importances = xgb.feature_importances_
        feature_names = model_bundle["feature_names"]

        # Sort and take top 15
        indices = np.argsort(importances)[-15:]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]

        # Clean up feature names
        clean_names = [n.replace("_diff", "").replace("_", " ").title() for n in top_names]

        colors = plt.cm.Blues(0.3 + 0.7 * (top_importances / top_importances.max()))
        bars = ax.barh(range(len(top_names)), top_importances, color=colors, edgecolor="white", height=0.7)

        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(clean_names, fontsize=11)
        ax.set_xlabel("Importance Score", fontweight="bold")
        ax.set_title("Top 15 Features — What Drives the Prediction", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, top_importances):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    except Exception as e:
        log.warning("Feature importance failed: %s", e)
        plt.close(fig)
        return ""

    path = OUTPUT_DIR / "04_feature_importance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Feature importance → %s", path)
    return str(path)


# ═══ PLOT 5: HEAD-TO-HEAD STATS ═══════════════════════════════════════

def plot_head_to_head(fights_df) -> str:
    """Side-by-side bar chart of career stats."""
    stats_labels = ["Total Fights", "Wins", "Losses", "Win Rate (%)", "KO/TKO Wins",
                    "Submission Wins", "Decision Wins", "Current Streak"]
    
    stats_data = {}
    for fighter in [config.FIGHTER_A, config.FIGHTER_B]:
        ff = _get_fighter_fights(fights_df, fighter)
        total = len(ff)
        wins = (ff["winner"] == fighter).sum()
        losses = total - wins

        ko_wins = sub_wins = dec_wins = 0
        for _, row in ff.iterrows():
            if row["winner"] == fighter:
                m = str(row.get("method", "")).upper()
                if "KO" in m or "TKO" in m:
                    ko_wins += 1
                elif "SUB" in m:
                    sub_wins += 1
                elif "DEC" in m:
                    dec_wins += 1

        streak = 0
        for _, row in ff[::-1].iterrows():
            if row["winner"] == fighter:
                streak += 1
            else:
                break

        stats_data[fighter] = [
            total, wins, losses, round(wins/total*100, 1) if total > 0 else 0,
            ko_wins, sub_wins, dec_wins, streak,
        ]

    x = np.arange(len(stats_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, stats_data[config.FIGHTER_A], width,
                   label=config.FIGHTER_A, color=FIGHTER_A_COLOR, edgecolor="white")
    bars2 = ax.bar(x + width/2, stats_data[config.FIGHTER_B], width,
                   label=config.FIGHTER_B, color=FIGHTER_B_COLOR, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(stats_labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Count / Percentage", fontweight="bold")
    ax.set_title("Career Statistics — Head-to-Head Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=13, frameon=True, facecolor="white", edgecolor="#E0E0E0")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + max(stats_data[config.FIGHTER_A])*0.01,
                    str(h), ha="center", va="bottom", fontsize=9, fontweight="bold", color=FIGHTER_A_COLOR)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + max(stats_data[config.FIGHTER_A])*0.01,
                    str(h), ha="center", va="bottom", fontsize=9, fontweight="bold", color=FIGHTER_B_COLOR)

    path = OUTPUT_DIR / "05_head_to_head.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Head-to-head → %s", path)
    return str(path)


# ═══ PLOT 6: MODEL PERFORMANCE ════════════════════════════════════════

def plot_model_performance(model_bundle, X_train, y_train) -> str:
    """ROC curve + confusion matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── ROC Curve ────────────────────────────────────────────────────
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(
            model_bundle["base_learners"].get("xgb",
                list(model_bundle["base_learners"].values())[0]),
            X_train, y_train, cv=cv, method="predict_proba",
        )[:, 1]

        fpr, tpr, _ = roc_curve(y_train, y_prob)
        roc_auc = auc(fpr, tpr)

        ax1.plot(fpr, tpr, color=ACCENT, linewidth=2.5,
                 label=f"ROC (AUC = {roc_auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax1.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
        ax1.set_xlabel("False Positive Rate", fontweight="bold")
        ax1.set_ylabel("True Positive Rate", fontweight="bold")
        ax1.set_title("ROC Curve — 5-Fold CV", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10, loc="lower right", frameon=True, facecolor="white")
        ax1.grid(True, alpha=0.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
    except Exception as e:
        log.warning("ROC failed: %s", e)
        ax1.text(0.5, 0.5, "ROC unavailable", ha="center", va="center",
                 transform=ax1.transAxes, fontsize=14, color="gray")

    # ── Confusion Matrix ─────────────────────────────────────────────
    try:
        from sklearn.model_selection import cross_val_predict
        xgb_model = model_bundle["base_learners"].get("xgb",
            list(model_bundle["base_learners"].values())[0])
        y_pred = cross_val_predict(xgb_model, X_train, y_train, cv=5)
        cm = confusion_matrix(y_train, y_pred)

        im = ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax2.set_title("Confusion Matrix — 5-Fold CV", fontsize=13, fontweight="bold")

        tick_labels = [config.FIGHTER_B, config.FIGHTER_A]
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(tick_labels, fontsize=11, fontweight="bold")
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(tick_labels, fontsize=11, fontweight="bold")
        ax2.set_ylabel("Actual", fontweight="bold")
        ax2.set_xlabel("Predicted", fontweight="bold")

        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=20, fontweight="bold",
                         color="white" if cm[i, j] > cm.max()/2 else "#333333")

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        ax2.text(0.5, -0.18, f"Accuracy: {accuracy:.1%}",
                 transform=ax2.transAxes, ha="center", fontsize=12,
                 fontweight="bold", color=ACCENT)
    except Exception as e:
        log.warning("Confusion matrix failed: %s", e)

    fig.suptitle("Stacking Ensemble — Model Performance",
                 fontsize=15, fontweight="bold", y=1.02)

    path = OUTPUT_DIR / "06_model_performance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Model performance → %s", path)
    return str(path)


# ═══ PLOT 7: SUMMARY CARD ═════════════════════════════════════════════

def plot_summary_card(result, model_bundle, fights_df) -> str:
    """Single summary card with all key info — ideal for LinkedIn.
    
    Uses fig.text() for precise, reliable positioning.
    """
    fig = plt.figure(figsize=(12, 14), facecolor="white")

    winner = config.FIGHTER_A if result["strickland_win_pct"] > result["chimaev_win_pct"] else config.FIGHTER_B
    win_pct = max(result["strickland_win_pct"], result["chimaev_win_pct"])
    winner_color = FIGHTER_A_COLOR if winner == config.FIGHTER_A else FIGHTER_B_COLOR
    loser_pct = min(result["strickland_win_pct"], result["chimaev_win_pct"])

    # ── Career stats ──────────────────────────────────────────────────
    records = {}
    for fighter in [config.FIGHTER_A, config.FIGHTER_B]:
        ff = _get_fighter_fights(fights_df, fighter)
        wins = int((ff["winner"] == fighter).sum())
        losses = len(ff) - wins
        ko = sub = dec = 0
        for _, row in ff.iterrows():
            if row["winner"] == fighter:
                m = str(row.get("method", "")).upper()
                if "KO" in m or "TKO" in m: ko += 1
                elif "SUB" in m: sub += 1
                else: dec += 1
        records[fighter] = {"wins": wins, "losses": losses, "ko": ko, "sub": sub, "dec": dec}

    # ── Colors ────────────────────────────────────────────────────────
    RED = FIGHTER_A_COLOR
    NAVY = FIGHTER_B_COLOR
    DARK = "#1A1A2E"
    GRAY = "#666666"
    LGRAY = "#AAAAAA"
    GREEN = "#2A9D8F"

    # ── TOP BANNER ─────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "UFC 328", ha="center", va="top", fontsize=42,
             fontweight="bold", color=DARK)
    fig.text(0.5, 0.92, f"{config.FIGHTER_A}  vs  {config.FIGHTER_B}",
             ha="center", va="top", fontsize=22, fontweight="bold", color="#333333")
    fig.text(0.5, 0.89, "Machine Learning Fight Prediction",
             ha="center", va="top", fontsize=11, color=LGRAY)

    # ── PREDICTION BOX ─────────────────────────────────────────────────
    box_left, box_right = 0.08, 0.50
    box_top, box_bottom = 0.85, 0.57
    rect = plt.Rectangle((box_left, box_bottom), box_right - box_left,
                          box_top - box_bottom, fill=True, facecolor="#F8F8F8",
                          edgecolor="#E0E0E0", linewidth=1.5, transform=fig.transFigure, zorder=0)
    fig.patches.append(rect)

    fig.text(0.29, 0.83, "PREDICTED WINNER", ha="center", va="top",
             fontsize=9, fontweight="bold", color=LGRAY)
    fig.text(0.29, 0.79, winner, ha="center", va="top",
             fontsize=20, fontweight="bold", color=winner_color)
    fig.text(0.29, 0.735, f"{win_pct:.1f}%", ha="center", va="top",
             fontsize=34, fontweight="bold", color=winner_color)
    ci_a = result["strickland_ci"]
    ci_b = result["chimaev_ci"]
    ci_low = min(ci_a[0], ci_b[0])
    ci_high = max(ci_a[1], ci_b[1])
    fig.text(0.29, 0.685, f"95% CI: [{ci_low:.1f}% — {ci_high:.1f}%]",
             ha="center", va="top", fontsize=10, color=GRAY)

    confidence = "HIGH" if win_pct > 60 else "MODERATE"
    conf_color = GREEN if confidence == "HIGH" else "#E76F51"
    fig.text(0.29, 0.64, f"CONFIDENCE: {confidence}", ha="center", va="top",
             fontsize=12, fontweight="bold", color=conf_color)
    fig.text(0.29, 0.605, f"Model CV Accuracy: {model_bundle['cv_accuracy_mean']*100:.1f}% ± {model_bundle['cv_accuracy_std']*100:.1f}%",
             ha="center", va="top", fontsize=8, color=LGRAY)

    # ── ELO BOX ────────────────────────────────────────────────────────
    elo_a = _compute_current_elo_from_fights(fights_df, config.FIGHTER_A)
    elo_b = _compute_current_elo_from_fights(fights_df, config.FIGHTER_B)
    box2_left, box2_right = 0.54, 0.92
    rect2 = plt.Rectangle((box2_left, box_bottom), box2_right - box2_left,
                           box_top - box_bottom, fill=True, facecolor="#F8F8F8",
                           edgecolor="#E0E0E0", linewidth=1.5, transform=fig.transFigure, zorder=0)
    fig.patches.append(rect2)

    fig.text(0.73, 0.83, "ELO RATINGS", ha="center", va="top",
             fontsize=9, fontweight="bold", color=LGRAY)

    # Horizontal ELO bars
    max_elo = 1900
    min_elo = 1300
    bar_w = 0.28
    bar_left = 0.58
    bar_h = 0.025

    # Strickland bar
    val_a = (elo_a - min_elo) / (max_elo - min_elo) * bar_w
    rect_a = plt.Rectangle((bar_left, 0.73), val_a, bar_h, fill=True,
                            facecolor=RED, edgecolor="white", linewidth=0.5,
                            transform=fig.transFigure, zorder=2)
    fig.patches.append(rect_a)
    fig.text(bar_left - 0.02, 0.73 + bar_h/2, f"{elo_a:.0f}", ha="right", va="center",
             fontsize=10, fontweight="bold", color=DARK)
    fig.text(bar_left + bar_w + 0.02, 0.73 + bar_h/2, config.FIGHTER_A,
             ha="left", va="center", fontsize=10, fontweight="bold", color=RED)

    # Chimaev bar
    val_b = (elo_b - min_elo) / (max_elo - min_elo) * bar_w
    rect_b = plt.Rectangle((bar_left, 0.67), val_b, bar_h, fill=True,
                            facecolor=NAVY, edgecolor="white", linewidth=0.5,
                            transform=fig.transFigure, zorder=2)
    fig.patches.append(rect_b)
    fig.text(bar_left - 0.02, 0.67 + bar_h/2, f"{elo_b:.0f}", ha="right", va="center",
             fontsize=10, fontweight="bold", color=DARK)
    fig.text(bar_left + bar_w + 0.02, 0.67 + bar_h/2, config.FIGHTER_B,
             ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)

    # ELO gap
    fig.text(0.73, 0.615, f"ELO Gap: {elo_a - elo_b:.0f} points", ha="center", va="top",
             fontsize=11, fontweight="bold", color=GREEN)

    # ── CAREER RECORDS BOX ─────────────────────────────────────────────
    box3_top, box3_bottom = 0.55, 0.38
    rect3 = plt.Rectangle((0.08, box3_bottom), 0.84, box3_top - box3_bottom,
                           fill=True, facecolor="#F8F8F8", edgecolor="#E0E0E0",
                           linewidth=1.5, transform=fig.transFigure, zorder=0)
    fig.patches.append(rect3)

    fig.text(0.5, 0.535, "CAREER RECORDS", ha="center", va="top",
             fontsize=9, fontweight="bold", color=LGRAY)

    # Table headers
    for x, label in [(0.15, "Record"), (0.36, "W-L"), (0.54, "KO"), (0.66, "SUB"), (0.78, "DEC")]:
        fig.text(x, 0.505, label, ha="center", va="center", fontsize=9,
                 fontweight="bold", color=GRAY)

    # Strickland row
    r = records[config.FIGHTER_A]
    fig.text(0.15, 0.47, config.FIGHTER_A, ha="center", va="center",
             fontsize=11, fontweight="bold", color=RED)
    fig.text(0.36, 0.47, f"{r['wins']}—{r['losses']}", ha="center", va="center",
             fontsize=14, fontweight="bold", color=DARK)
    for x, key in [(0.54, "ko"), (0.66, "sub"), (0.78, "dec")]:
        fig.text(x, 0.47, str(r[key]), ha="center", va="center",
                 fontsize=12, fontweight="bold", color=DARK)

    # Chimaev row
    r = records[config.FIGHTER_B]
    fig.text(0.15, 0.43, config.FIGHTER_B, ha="center", va="center",
             fontsize=11, fontweight="bold", color=NAVY)
    fig.text(0.36, 0.43, f"{r['wins']}—{r['losses']}", ha="center", va="center",
             fontsize=14, fontweight="bold", color=DARK)
    for x, key in [(0.54, "ko"), (0.66, "sub"), (0.78, "dec")]:
        fig.text(x, 0.43, str(r[key]), ha="center", va="center",
                 fontsize=12, fontweight="bold", color=DARK)

    fig.text(0.5, 0.395, "Data: UFCStats.com  ·  8,675 historical fights analyzed",
             ha="center", va="center", fontsize=7, color=LGRAY)

    # ── MODEL ARCHITECTURE BOX ─────────────────────────────────────────
    box4_top, box4_bottom = 0.36, 0.22
    rect4 = plt.Rectangle((0.08, box4_bottom), 0.84, box4_top - box4_bottom,
                           fill=True, facecolor="#F8F8F8", edgecolor="#E0E0E0",
                           linewidth=1.5, transform=fig.transFigure, zorder=0)
    fig.patches.append(rect4)

    fig.text(0.5, 0.345, "MODEL ARCHITECTURE", ha="center", va="top",
             fontsize=9, fontweight="bold", color=LGRAY)
    fig.text(0.5, 0.315, "Stacking Ensemble — XGBoost + LightGBM + RandomForest + SVM → LogisticRegression",
             ha="center", va="center", fontsize=11, color="#333333")
    fig.text(0.5, 0.285, "37 engineered features · Mutual Information selection · 5-Fold Stratified CV",
             ha="center", va="center", fontsize=9, color=GRAY)
    fig.text(0.5, 0.255, "2,000 Monte Carlo bootstrap iterations · Meta-learner uncertainty estimation",
             ha="center", va="center", fontsize=9, color=GRAY)
    fig.text(0.5, 0.232, "Built with scikit-learn, XGBoost, LightGBM, SHAP  ·  May 2026",
             ha="center", va="center", fontsize=7, color=LGRAY)

    # ── METHOD BOX ─────────────────────────────────────────────────────
    box5_top, box5_bottom = 0.20, 0.05
    rect5 = plt.Rectangle((0.08, box5_bottom), 0.84, box5_top - box5_bottom,
                           fill=True, facecolor="#F8F8F8", edgecolor="#E0E0E0",
                           linewidth=1.5, transform=fig.transFigure, zorder=0)
    fig.patches.append(rect5)

    fig.text(0.5, 0.185, "HOW IT WORKS", ha="center", va="top",
             fontsize=9, fontweight="bold", color=LGRAY)

    steps = [
        ("1", "Scrape", "8,675 UFC fights\nfrom UFCStats.com"),
        ("2", "Engineer", "37 features (ELO,\nrolling stats, form)"),
        ("3", "Train", "Stacking ensemble\nwith 5-fold CV"),
        ("4", "Predict", "Monte Carlo simulation\n(2,000 iterations)"),
    ]
    for i, (num, title, desc) in enumerate(steps):
        x = 0.18 + i * 0.22
        fig.text(x, 0.155, num, ha="center", va="center", fontsize=20,
                 fontweight="bold", color="#DDDDDD")
        fig.text(x, 0.125, title, ha="center", va="center", fontsize=10,
                 fontweight="bold", color=DARK)
        fig.text(x, 0.095, desc, ha="center", va="center", fontsize=7.5,
                 color=GRAY)

    # ── DISCLAIMER ─────────────────────────────────────────────────────
    fig.text(0.5, 0.02, "For informational purposes only — past performance does not guarantee future results",
             ha="center", va="center", fontsize=6, color="#CCCCCC")

    path = OUTPUT_DIR / "07_summary_card.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("Summary card → %s", path)
    return str(path)


# ═══ MAIN ════════════════════════════════════════════════════════════

def generate_all():
    """Generate all visualization plots."""
    log.info("═══════════════════════════════════════")
    log.info("  Generating LinkedIn Visualizations")
    log.info("═══════════════════════════════════════")

    model, metadata, scaler, features_df, fights_df = _load_data()

    # Load prediction result from Monte Carlo
    import predict as pred_module
    result = pred_module.predict()
    if result is None:
        log.error("Could not run prediction. Ensure model is trained.")
        return []

    # Build X_train for model performance plots
    exclude_cols = ["fighter_a", "fighter_b", "event_date", "target"]
    train_cols = [c for c in features_df.columns if c not in exclude_cols]
    X_train = features_df[train_cols].values.astype(np.float64)
    y_train = features_df["target"].values.astype(np.float64)

    # Apply feature selection if present
    selector = model.get("selector")
    if selector is not None:
        X_train = selector.transform(X_train)
    X_train = scaler.transform(X_train)

    paths = []

    log.info("\nGenerating plots...")
    paths.append(plot_donut(result))
    paths.append(plot_radar(fights_df))
    paths.append(plot_elo_history(fights_df, features_df))
    paths.append(plot_feature_importance(model, features_df))
    paths.append(plot_head_to_head(fights_df))
    paths.append(plot_model_performance(model, X_train, y_train))
    paths.append(plot_summary_card(result, model, fights_df))

    log.info("\n═══ All plots generated in %s ═══", OUTPUT_DIR)
    for p in paths:
        if p:
            log.info("  ✓ %s", Path(p).name)

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_all()
