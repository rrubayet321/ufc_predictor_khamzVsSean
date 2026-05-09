"""Configuration for UFC Predictor — Strickland vs Chimaev."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ── UFCStats.com URLs ──────────────────────────────────────────────
UFOSTATS_BASE = "http://ufcstats.com"
UFOSTATS_EVENTS = f"{UFOSTATS_BASE}/statistics/events/completed?page=all"
UFOSTATS_FIGHTERS = f"{UFOSTATS_BASE}/statistics/fighters"

# ── Fighters for prediction ────────────────────────────────────────
FIGHTER_A = "Sean Strickland"
FIGHTER_B = "Khamzat Chimaev"

# ── ELO parameters ─────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K = 32
ELO_SCALE = 400  # Logistic scale factor

# ── Feature settings ───────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 8]  # Last-N fight windows for rolling features
OPPONENT_QUALITY_WINDOW = 3   # Last N opponents to gauge quality

# ── Monte Carlo / Prediction settings ──────────────────────────────
MONTE_CARLO_ITERS = 2000
BOOTSTRAP_SIZE = 0.7

# ── Model CV parameters ────────────────────────────────────────────
CV_FOLDS = 5
CV_INNER_FOLDS = 3
CV_REPEATS = 1  # Use RepeatedStratifiedKFold
OPTUNA_TRIALS = 0  # 0 = skip Optuna, use defaults

# ── Feature selection ──────────────────────────────────────────────
FEATURE_SELECTION_K = 30  # Top K features by mutual information (0 = no selection)

# ── Class labels ───────────────────────────────────────────────────
FIGHTER_A_LABEL = 1  # Strickland wins
FIGHTER_B_LABEL = 0  # Chimaev wins

# ── File paths ─────────────────────────────────────────────────────
RAW_FIGHTERS = DATA_RAW / "fighters.csv"
RAW_FIGHTS = DATA_RAW / "fights.csv"
RAW_EVENTS = DATA_RAW / "events.csv"
FEATURES_FILE = DATA_PROCESSED / "features.csv"
MODEL_FILE = DATA_PROCESSED / "model.joblib"
SCALER_FILE = DATA_PROCESSED / "scaler.joblib"
METADATA_FILE = DATA_PROCESSED / "metadata.joblib"
SELECTOR_FILE = DATA_PROCESSED / "selector.joblib"
