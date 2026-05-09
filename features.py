"""Feature engineering — ELO ratings, rolling averages, physical differentials,
age, fight frequency, opponent quality, method encoding, and more.

Produces a single feature matrix where each row is a fight with:
  - target = 1 if fighter_a wins, 0 if fighter_b wins
  - features are differentials / ratios between the two fighters
"""

import logging
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────

def _parse_height(inches_str: str) -> float | None:
    """Parse '5 11 \" ' or '6 2 \"' into inches."""
    if pd.isna(inches_str) or not inches_str:
        return None
    parts = inches_str.replace('"', "").replace("'", "").strip().split()
    if len(parts) >= 2:
        try:
            return float(parts[0]) * 12 + float(parts[1])
        except ValueError:
            return None
    return None


def _parse_reach(inches_str: str) -> float | None:
    """Parse '72.0\"' into float inches."""
    if pd.isna(inches_str) or not inches_str:
        return None
    try:
        return float(inches_str.replace('"', "").strip())
    except ValueError:
        return None


def _parse_pct(pct_str: str) -> float | None:
    """Parse '45%' into 0.45."""
    if pd.isna(pct_str) or not pct_str:
        return None
    try:
        return float(pct_str.replace("%", "").strip()) / 100
    except ValueError:
        return None


def _parse_int_or_float(val_str: str) -> float | None:
    """Parse a raw integer/float from a string."""
    if pd.isna(val_str) or not val_str:
        return None
    try:
        return float(str(val_str).replace(",", ""))
    except ValueError:
        return None


def _parse_dob_to_age(dob_str: str, fight_date: str) -> float | None:
    """Parse DOB like 'Jan 12, 1990' and compute age at fight date."""
    if pd.isna(dob_str) or not dob_str:
        return None
    try:
        dob = datetime.strptime(str(dob_str).strip(), "%b %d, %Y")
        fd = pd.to_datetime(fight_date).to_pydatetime()
        age = (fd - dob).days / 365.25
        return age
    except Exception:
        return None


# ── ELO System ──────────────────────────────────────────────────────

def _expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for fighter A given ELO ratings."""
    return 1.0 / (1 + 10 ** ((elo_b - elo_a) / config.ELO_SCALE))


def _update_elo(elo_a: float, elo_b: float, result_a: float, k: float = config.ELO_K) -> tuple[float, float]:
    """Update ELO for both fighters. result_a = 1 if A wins, 0 if B wins, 0.5 draw."""
    expected = _expected_score(elo_a, elo_b)
    delta = k * (result_a - expected)
    return elo_a + delta, elo_b - delta


def compute_elo(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ELO ratings historically for each fighter before every fight.

    Also tracks ELO momentum (change over last 3 fights).
    """
    log.info("Computing historical ELO ratings...")

    elo = defaultdict(lambda: config.ELO_INITIAL)
    elo_history = defaultdict(list)  # fighter -> list of (date, elo)

    elos_a = []
    elos_b = []
    elos_diff = []
    elo_momentum_a = []
    elo_momentum_b = []

    fights_df = fights_df.copy()
    fights_df["_event_dt"] = pd.to_datetime(fights_df["event_date"], errors="coerce")
    fights_df.sort_values("_event_dt", inplace=True)

    for _, fight in fights_df.iterrows():
        fa = fight.get("fighter_a", "")
        fb = fight.get("fighter_b", "")
        target = fight.get("target", 0.5)

        elo_a = elo[fa]
        elo_b = elo[fb]
        elos_a.append(elo_a)
        elos_b.append(elo_b)
        elos_diff.append(elo_a - elo_b)

        # ELO momentum: change over last 3 fights
        hist_a = [e for _, e in elo_history.get(fa, [])[-3:]]
        hist_b = [e for _, e in elo_history.get(fb, [])[-3:]]
        mom_a = (elo_a - hist_a[0]) if len(hist_a) >= 3 else 0.0
        mom_b = (elo_b - hist_b[0]) if len(hist_b) >= 3 else 0.0
        elo_momentum_a.append(mom_a)
        elo_momentum_b.append(mom_b)

        # Store current ELO before update
        elo_history[fa].append((fight.get("event_date", ""), elo_a))
        elo_history[fb].append((fight.get("event_date", ""), elo_b))

        # Update ELO
        target_val = float(target) if pd.notna(target) else 0.5
        elo[fa], elo[fb] = _update_elo(elo_a, elo_b, target_val)

    fights_df["elo_a"] = elos_a
    fights_df["elo_b"] = elos_b
    fights_df["elo_diff"] = elos_diff
    fights_df["elo_momentum_a"] = elo_momentum_a
    fights_df["elo_momentum_b"] = elo_momentum_b
    fights_df["elo_momentum_diff"] = fights_df["elo_momentum_a"] - fights_df["elo_momentum_b"]

    return fights_df


# ── Rolling Features ────────────────────────────────────────────────

def _rolling_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int,
    prefix: str,
) -> pd.DataFrame:
    """Compute rolling average over the last `window` fights for a stat.
    Lookback is per-fighter, sorted chronologically.
    Adds diff and ratio columns.
    """
    log.info("Computing rolling feature: %s (window=%d)", prefix, window)

    records = []
    for _, row in df.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]
        date = row["event_date"]
        val_a = row.get(col_a)
        val_b = row.get(col_b)

        if val_a is not None and pd.notna(val_a):
            records.append({"fighter": fa, "date": date, "value": float(val_a), "fight_idx": row.name, "side": "a"})
        if val_b is not None and pd.notna(val_b):
            records.append({"fighter": fb, "date": date, "value": float(val_b), "fight_idx": row.name, "side": "b"})

    rdf = pd.DataFrame(records)
    if rdf.empty:
        return df

    rdf.sort_values(["fighter", "date"], inplace=True)

    rdf["rolling_mean"] = rdf.groupby("fighter")["value"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    rdf["rolling_std"] = rdf.groupby("fighter")["value"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )

    rolling_a = rdf[rdf["side"] == "a"].set_index("fight_idx")["rolling_mean"]
    rolling_b = rdf[rdf["side"] == "b"].set_index("fight_idx")["rolling_mean"]
    std_a = rdf[rdf["side"] == "a"].set_index("fight_idx")["rolling_std"]
    std_b = rdf[rdf["side"] == "b"].set_index("fight_idx")["rolling_std"]

    df[f"{prefix}_rolling_{window}_a"] = rolling_a
    df[f"{prefix}_rolling_{window}_b"] = rolling_b
    df[f"{prefix}_rolling_{window}_diff"] = rolling_a - rolling_b
    df[f"{prefix}_rolling_{window}_ratio"] = (rolling_a + 1e-6) / (rolling_b + 1e-6)
    df[f"{prefix}_rolling_{window}_std_diff"] = std_a.fillna(0) - std_b.fillna(0)

    return df


# ── Days Since Last Fight ───────────────────────────────────────────

def _days_since_last(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute days since last fight for each fighter before each fight."""
    log.info("Computing days since last fight...")

    last_dates = {}
    days_a = []
    days_b = []

    for _, row in fights_df.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]
        try:
            fight_date = pd.to_datetime(row["event_date"])
        except Exception:
            days_a.append(0)
            days_b.append(0)
            continue

        for fighter, day_list in [(fa, days_a), (fb, days_b)]:
            if fighter in last_dates:
                delta = (fight_date - last_dates[fighter]).days
                day_list.append(delta)
            else:
                day_list.append(0)
            last_dates[fighter] = fight_date

    fights_df["days_since_last_a"] = days_a
    fights_df["days_since_last_b"] = days_b
    fights_df["days_since_last_diff"] = fights_df["days_since_last_a"] - fights_df["days_since_last_b"]
    return fights_df


# ── Fight Frequency ─────────────────────────────────────────────────

def _fight_frequency(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute fights per year (career fight rate) for each fighter."""
    log.info("Computing fight frequency...")

    career_stats = defaultdict(list)  # fighter -> [(date, cumulative_fights)]

    # Build career timeline
    for _, row in fights_df.iterrows():
        for side, name_col in [("a", "fighter_a"), ("b", "fighter_b")]:
            fighter = row[name_col]
            try:
                date = pd.to_datetime(row["event_date"])
            except Exception:
                date = None
            career_stats[fighter].append((date, 0))  # placeholder

    # Compute cumulative fights over time
    freq_a = []
    freq_b = []

    for _, row in fights_df.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]
        try:
            fd = pd.to_datetime(row["event_date"])
        except Exception:
            freq_a.append(0)
            freq_b.append(0)
            continue

        for fighter, freq_list in [(fa, freq_a), (fb, freq_b)]:
            # Count how many fights this fighter had up to this point
            dates = career_stats.get(fighter, [])
            past_fights = sum(1 for d, _ in dates if d is not None and d < fd)
            if past_fights > 0:
                # Estimate career length in years from first fight to now
                first_date = min(d for d, _ in dates if d is not None)
                years = max((fd - first_date).days / 365.25, 0.25)
                rate = past_fights / years
            else:
                rate = 0
            freq_list.append(rate)

    fights_df["fight_rate_a"] = freq_a
    fights_df["fight_rate_b"] = freq_b
    fights_df["fight_rate_diff"] = fights_df["fight_rate_a"] - fights_df["fight_rate_b"]
    return fights_df


# ── Opponent Quality ────────────────────────────────────────────────

def _opponent_quality(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute average ELO of last N opponents (quality of opposition)."""
    log.info("Computing opponent quality...")

    window = config.OPPONENT_QUALITY_WINDOW
    opp_elo_history = defaultdict(list)

    qual_a = []
    qual_b = []

    for _, row in fights_df.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]

        # Average ELO of last N opponents for fighter A
        past_opps_a = opp_elo_history.get(fa, [])[-window:]
        qual_a.append(np.mean(past_opps_a) if past_opps_a else config.ELO_INITIAL)

        past_opps_b = opp_elo_history.get(fb, [])[-window:]
        qual_b.append(np.mean(past_opps_b) if past_opps_b else config.ELO_INITIAL)

        # Record opponent ELO for future fights
        opp_elo_history[fa].append(row.get("elo_b", config.ELO_INITIAL))
        opp_elo_history[fb].append(row.get("elo_a", config.ELO_INITIAL))

    fights_df["opp_quality_a"] = qual_a
    fights_df["opp_quality_b"] = qual_b
    fights_df["opp_quality_diff"] = fights_df["opp_quality_a"] - fights_df["opp_quality_b"]
    return fights_df


# ── Win Methods & Finish Rate ───────────────────────────────────────

def _win_method_features(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute finish rate, method preferences."""
    log.info("Computing win method features...")

    career_wins = defaultdict(list)  # fighter -> [(method_type)]

    finish_rate_a = []
    finish_rate_b = []
    ko_rate_a = []
    ko_rate_b = []
    sub_rate_a = []
    sub_rate_b = []

    def classify_method(method_str: str) -> str:
        m = str(method_str).upper()
        if "KO" in m or "TKO" in m:
            return "KO"
        elif "SUB" in m:
            return "SUB"
        elif "DEC" in m:
            return "DEC"
        return "OTHER"

    for _, row in fights_df.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]
        target = row.get("target", 0.5)
        method = str(row.get("method", ""))

        for fighter, rate_list, ko_list, sub_list, is_winner in [
            (fa, finish_rate_a, ko_rate_a, sub_rate_a, target == 1.0),
            (fb, finish_rate_b, ko_rate_b, sub_rate_b, target == 0.0),
        ]:
            past = career_wins.get(fighter, [])
            total = len(past)
            finishes = sum(1 for m in past if m in ("KO", "SUB"))
            kos = sum(1 for m in past if m == "KO")
            subs = sum(1 for m in past if m == "SUB")

            rate_list.append(finishes / total if total > 0 else 0)
            ko_list.append(kos / total if total > 0 else 0)
            sub_list.append(subs / total if total > 0 else 0)

            # Record this fight's method if they won
            if is_winner:
                career_wins[fighter].append(classify_method(method))

    fights_df["finish_rate_a"] = finish_rate_a
    fights_df["finish_rate_b"] = finish_rate_b
    fights_df["ko_rate_a"] = ko_rate_a
    fights_df["ko_rate_b"] = ko_rate_b
    fights_df["sub_rate_a"] = sub_rate_a
    fights_df["sub_rate_b"] = sub_rate_b

    fights_df["finish_rate_diff"] = fights_df["finish_rate_a"] - fights_df["finish_rate_b"]
    fights_df["ko_rate_diff"] = fights_df["ko_rate_a"] - fights_df["ko_rate_b"]
    fights_df["sub_rate_diff"] = fights_df["sub_rate_a"] - fights_df["sub_rate_b"]
    return fights_df


# ── Winner Determination ────────────────────────────────────────────

def _determine_winner(fights_df: pd.DataFrame) -> pd.DataFrame:
    """Determine the winner from the scraped 'winner' column."""
    winners = []
    targets = []

    for _, row in fights_df.iterrows():
        fa = str(row.get("fighter_a", "")).strip()
        fb = str(row.get("fighter_b", "")).strip()
        winner_col = str(row.get("winner", "")).strip()

        winner_name = ""
        if winner_col and winner_col != "nan":
            # winner column contains the winning fighter's name
            if winner_col == fa:
                winner_name = fa
            elif winner_col == fb:
                winner_name = fb
            else:
                # Try case-insensitive match
                if winner_col.lower() == fa.lower():
                    winner_name = fa
                elif winner_col.lower() == fb.lower():
                    winner_name = fb

        target = 1.0 if winner_name == fa else (0.0 if winner_name == fb else 0.5)
        winners.append(winner_name)
        targets.append(target)

    fights_df["winner_name"] = winners
    fights_df["target"] = targets
    return fights_df


# ── Main Feature Builder ────────────────────────────────────────────

def build_features() -> pd.DataFrame:
    """Main feature engineering pipeline — builds comprehensive feature matrix."""
    log.info("=== Feature Engineering ===")

    fights = pd.read_csv(config.RAW_FIGHTS, low_memory=False)
    fighters = pd.read_csv(config.RAW_FIGHTERS, low_memory=False) if config.RAW_FIGHTERS.exists() else pd.DataFrame()

    if fights.empty:
        log.error("No fight data found. Run scraper first.")
        return pd.DataFrame()

    # Determine winner
    fights = _determine_winner(fights)

    # Filter to decisive results
    fights = fights[fights["target"].isin([0.0, 1.0])].copy()
    log.info("Fights with decisive results: %d", len(fights))

    if fights.empty:
        log.error("No decisive fights found.")
        return fights

    # ═══ RANDOMIZE FIGHTER ORDERING ═══════════════════════════════════
    # The scraper puts the winner first in every row (fighter_a = winner).
    # We must shuffle so fighter_a is NOT always the winner, otherwise
    # the model trivially learns "fighter_a always wins" (98% accuracy).
    # After randomization, ~50% of targets will be 1 and ~50% will be 0.
    np.random.seed(42)
    swap_mask = np.random.choice([True, False], size=len(fights))
    for idx in fights[swap_mask].index:
        # Swap fighter names
        fa = fights.at[idx, "fighter_a"]
        fb = fights.at[idx, "fighter_b"]
        fights.at[idx, "fighter_a"] = fb
        fights.at[idx, "fighter_b"] = fa
        # Swap all _a / _b columns
        for col in fights.columns:
            if col.endswith("_a"):
                base = col[:-2]
                col_b = f"{base}_b"
                if col_b in fights.columns:
                    tmp = fights.at[idx, col]
                    fights.at[idx, col] = fights.at[idx, col_b]
                    fights.at[idx, col_b] = tmp
        # Flip target
        fights.at[idx, "target"] = 1.0 - fights.at[idx, "target"]
        # Swap winner name if present
        if "winner_name" in fights.columns:
            wn = fights.at[idx, "winner_name"]
            if wn == fa:
                fights.at[idx, "winner_name"] = fb
            elif wn == fb:
                fights.at[idx, "winner_name"] = fa

    log.info("After randomization — target distribution: 1=%.1f%%  0=%.1f%%",
             fights["target"].mean() * 100, (1 - fights["target"].mean()) * 100)

    # Parse numeric columns — skip fighter names
    for col in fights.columns:
        col_lower = col.lower()
        if col_lower in ("fighter_a", "fighter_b"):
            continue
        if col_lower.endswith("_a") or col_lower.endswith("_b"):
            try:
                fights[col] = fights[col].apply(_parse_int_or_float)
            except Exception:
                pass

    # Compute ELO (must be first — other features depend on it)
    fights = compute_elo(fights)

    # Days since last fight
    fights = _days_since_last(fights)

    # Fight frequency
    fights = _fight_frequency(fights)

    # Opponent quality
    fights = _opponent_quality(fights)

    # Win method features
    fights = _win_method_features(fights)

    # Per-fight stat ratios
    for stat_prefix in ["sig_str", "total_str", "td"]:
        landed_a = f"{stat_prefix}_landed_a"
        attempted_a = f"{stat_prefix}_attempted_a"
        landed_b = f"{stat_prefix}_landed_b"
        attempted_b = f"{stat_prefix}_attempted_b"

        if landed_a in fights.columns and attempted_a in fights.columns:
            fights[f"{stat_prefix}_acc_a"] = fights[landed_a] / (fights[attempted_a] + 1e-6)
        if landed_b in fights.columns and attempted_b in fights.columns:
            fights[f"{stat_prefix}_acc_b"] = fights[landed_b] / (fights[attempted_b] + 1e-6)

    # Fight time and per-minute stats
    try:
        fights["fight_minutes"] = fights["time"].apply(
            lambda t: sum(int(x) * (60 if i == 0 else 1)
                         for i, x in enumerate(reversed(str(t).split(":"))))
            if pd.notna(t) and ":" in str(t) else 15.0
        ) / 60.0
    except Exception:
        fights["fight_minutes"] = 15.0

    for side in ["a", "b"]:
        for landed_prefix in ["sig_str_landed", "total_str_landed"]:
            landed_col = f"{landed_prefix}_{side}"
            if landed_col in fights.columns:
                pm_col = landed_prefix.replace("_landed", "_pm")
                fights[f"{pm_col}_{side}"] = fights[landed_col] / (fights["fight_minutes"] + 1e-6)

    # Control time per minute
    ctrl_cols = [c for c in fights.columns if "ctrl" in c.lower()]
    for col in ctrl_cols:
        side = col[-1] if col.endswith("_a") or col.endswith("_b") else None
        if side:
            ctrl_pm = f"ctrl_pm_{side}"
            fights[ctrl_pm] = fights[col] / (fights["fight_minutes"] + 1e-6)

    # Knockdown differential
    kd_cols = [c for c in fights.columns if "kd" in c.lower() or "knockdown" in c.lower()]
    if len(kd_cols) >= 2:
        kd_a = kd_cols[0] if kd_cols[0].endswith("_a") else kd_cols[1]
        kd_b = kd_cols[1] if kd_cols[1].endswith("_b") else kd_cols[0]
        fights["kd_diff"] = fights[kd_a] - fights[kd_b]

    # ── Merge fighter physical attributes ────────────────────────────
    if not fighters.empty:
        phys_lookup = {}
        for _, frow in fighters.iterrows():
            name = str(frow.get("fighter_name", ""))
            phys_lookup[name] = frow.to_dict()

        for side, sfx in [("a", "fighter_a"), ("b", "fighter_b")]:
            for phys_col in ["height", "reach", "stance", "dob", "slpm",
                             "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]:
                col_name = f"{phys_col}_{side}"
                fights[col_name] = fights[sfx].map(
                    lambda n, pc=phys_col: phys_lookup.get(str(n), {}).get(pc)
                )

    # Parse physical attributes to numeric
    for side in ["a", "b"]:
        fights[f"height_inches_{side}"] = fights.get(f"height_{side}", pd.Series()).apply(_parse_height)
        fights[f"reach_inches_{side}"] = fights.get(f"reach_{side}", pd.Series()).apply(_parse_reach)
        fights[f"slpm_{side}"] = fights.get(f"slpm_{side}", pd.Series()).apply(_parse_int_or_float)
        fights[f"str_acc_{side}"] = fights.get(f"str_acc_{side}", pd.Series()).apply(_parse_pct)
        fights[f"str_def_{side}"] = fights.get(f"str_def_{side}", pd.Series()).apply(_parse_pct)
        fights[f"td_avg_{side}"] = fights.get(f"td_avg_{side}", pd.Series()).apply(_parse_int_or_float)
        fights[f"td_acc_{side}"] = fights.get(f"td_acc_{side}", pd.Series()).apply(_parse_pct)
        fights[f"td_def_{side}"] = fights.get(f"td_def_{side}", pd.Series()).apply(_parse_pct)
        fights[f"sub_avg_{side}"] = fights.get(f"sub_avg_{side}", pd.Series()).apply(_parse_int_or_float)

        # Age
        if "dob" in fights.columns:
            dob_col = f"dob_{side}"
            if dob_col in fights.columns:
                fights[f"age_{side}"] = fights.apply(
                    lambda r: _parse_dob_to_age(
                        r.get(dob_col), r.get("event_date", "")
                    ), axis=1
                )

    # Physical differentials
    fights["height_diff"] = fights["height_inches_a"] - fights["height_inches_b"]
    fights["reach_diff"] = fights["reach_inches_a"] - fights["reach_inches_b"]
    fights["age_diff"] = fights.get("age_a", 0) - fights.get("age_b", 0)
    fights["slpm_diff"] = fights["slpm_a"] - fights["slpm_b"]
    fights["str_acc_diff"] = fights["str_acc_a"] - fights["str_acc_b"]
    fights["str_def_diff"] = fights["str_def_a"] - fights["str_def_b"]
    fights["td_avg_diff"] = fights["td_avg_a"] - fights["td_avg_b"]
    fights["td_acc_diff"] = fights["td_acc_a"] - fights["td_acc_b"]
    fights["td_def_diff"] = fights["td_def_a"] - fights["td_def_b"]
    fights["sub_avg_diff"] = fights["sub_avg_a"] - fights["sub_avg_b"]

    # Stance matchup encoding
    fights["stance_matchup"] = fights.apply(
        lambda r: f"{r.get('stance_a', '?')}_vs_{r.get('stance_b', '?')}", axis=1
    )
    stance_dummies = pd.get_dummies(fights["stance_matchup"], prefix="stance")

    # Same stance flag
    if "stance_a" in fights.columns and "stance_b" in fights.columns:
        fights["same_stance"] = (fights["stance_a"] == fights["stance_b"]).astype(float)
    else:
        fights["same_stance"] = 0.0

    # ── Compute rolling features ─────────────────────────────────────
    rolling_stats = ["sig_str_pm", "total_str_pm", "td_acc", "str_acc",
                     "slpm", "str_def", "td_def", "ctrl_pm"]
    for window in config.ROLLING_WINDOWS:
        for stat in rolling_stats:
            col_a = f"{stat}_a"
            col_b = f"{stat}_b"
            if col_a in fights.columns and col_b in fights.columns:
                fights = _rolling_feature(fights, col_a, col_b, window, stat)

    # ── Win streak ───────────────────────────────────────────────────
    streaks = defaultdict(int)
    win_streak_a = []
    win_streak_b = []
    for idx, row in fights.iterrows():
        fa = row["fighter_a"]
        fb = row["fighter_b"]
        win_streak_a.append(streaks.get(fa, 0))
        win_streak_b.append(streaks.get(fb, 0))
        target = row["target"]
        if target == 1.0:
            streaks[fa] = streaks.get(fa, 0) + 1
            streaks[fb] = 0
        else:
            streaks[fb] = streaks.get(fb, 0) + 1
            streaks[fa] = 0

    fights["win_streak_a"] = win_streak_a
    fights["win_streak_b"] = win_streak_b
    fights["win_streak_diff"] = fights["win_streak_a"] - fights["win_streak_b"]

    # ── Weight class encoding ────────────────────────────────────────
    if "weight_class" in fights.columns:
        weight_dummies = pd.get_dummies(fights["weight_class"], prefix="wc")

    # ── Build final feature columns ──────────────────────────────────
    core_features = [
        "elo_diff",
        "elo_momentum_diff",
        "height_diff",
        "reach_diff",
        "age_diff",
        "slpm_diff",
        "str_acc_diff",
        "str_def_diff",
        "td_avg_diff",
        "td_acc_diff",
        "td_def_diff",
        "sub_avg_diff",
        "days_since_last_diff",
        "fight_rate_diff",
        "opp_quality_diff",
        "finish_rate_diff",
        "ko_rate_diff",
        "sub_rate_diff",
        "win_streak_diff",
        "same_stance",
    ]

    if "kd_diff" in fights.columns:
        core_features.insert(-1, "kd_diff")

    feature_cols = list(core_features)

    # Add rolling features
    for window in config.ROLLING_WINDOWS:
        for stat in rolling_stats:
            diff_col = f"{stat}_rolling_{window}_diff"
            if diff_col in fights.columns:
                feature_cols.append(diff_col)

    # Add stance dummies
    for col in stance_dummies.columns:
        if col not in feature_cols:
            fights[col] = stance_dummies[col].values
            feature_cols.append(col)

    # Add weight class dummies
    if "weight_class" in fights.columns:
        for col in weight_dummies.columns:
            if col not in feature_cols:
                fights[col] = weight_dummies[col].values
                feature_cols.append(col)

    available_features = [c for c in feature_cols if c in fights.columns]
    log.info("Available features: %d", len(available_features))

    # Build result
    result = fights[["fighter_a", "fighter_b", "event_date", "target"] + available_features].copy()
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.dropna(subset=["target"], inplace=True)

    # Drop rows with >50% NaN features
    threshold = len(available_features) * 0.5
    result = result[result[available_features].isna().sum(axis=1) <= threshold]
    result[available_features] = result[available_features].fillna(0)

    result.to_csv(config.FEATURES_FILE, index=False)
    log.info("Feature matrix saved: %d rows, %d features → %s",
             len(result), len(available_features), config.FEATURES_FILE)

    return result


if __name__ == "__main__":
    build_features()
