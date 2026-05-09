"""One-command orchestrator — scrape, train, and predict in one go.

Usage:
    python run.py                  # Full pipeline: scrape → features → train → predict
    python run.py --features       # Features only (requires raw data)
    python run.py --train          # Train only (requires features)
    python run.py --predict        # Predict only (requires trained model)
    python run.py --train --predict  # Train + predict
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run")


def _check_file(path: Path, name: str) -> bool:
    if not path.exists():
        log.warning("%s not found at %s", name, path)
        return False
    return True


def cmd_scrape():
    log.info(">>> PHASE 1: Scraping UFCStats.com <<<")
    from scraper import run as scrape_run
    scrape_run()


def cmd_features():
    log.info(">>> PHASE 2: Feature Engineering <<<")
    if not _check_file(config.RAW_FIGHTS, "Raw fights CSV"):
        log.error("Run 'python run.py --scrape' first to collect fight data.")
        return False
    from features import build_features
    build_features()
    return True


def cmd_train():
    log.info(">>> PHASE 3: Model Training <<<")
    if not _check_file(config.FEATURES_FILE, "Features CSV"):
        log.error("Run 'python run.py --features' first to build the feature matrix.")
        return False
    from train import train as train_model
    train_model()
    return True


def cmd_predict():
    log.info(">>> PHASE 4: Prediction <<<")
    if not _check_file(config.MODEL_FILE, "Trained model"):
        log.error("Run 'python run.py --train' first to train the model.")
        return False
    from predict import predict as run_prediction
    run_prediction()
    return True


def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor — Strickland vs Chimaev")
    parser.add_argument("--scrape", action="store_true", help="Run only data scraping")
    parser.add_argument("--features", action="store_true", help="Run only feature engineering")
    parser.add_argument("--train", action="store_true", help="Run only model training")
    parser.add_argument("--predict", action="store_true", help="Run only prediction")
    args = parser.parse_args()

    # If no specific flags, run the full pipeline
    run_all = not any([args.scrape, args.features, args.train, args.predict])

    t_start = time.time()

    if run_all:
        log.info("=" * 60)
        log.info("  UFC FIGHT PREDICTOR — FULL PIPELINE")
        log.info("  %s vs %s", config.FIGHTER_A, config.FIGHTER_B)
        log.info("=" * 60)

        # Phase 1
        if not cmd_scrape():
            sys.exit(1)

        # Phase 2
        if not cmd_features():
            sys.exit(1)

        # Phase 3
        if not cmd_train():
            sys.exit(1)

        # Phase 4
        if not cmd_predict():
            sys.exit(1)

    else:
        if args.scrape:
            cmd_scrape()

        if args.features:
            if not cmd_features():
                sys.exit(1)

        if args.train:
            if not cmd_train():
                sys.exit(1)

        if args.predict:
            if not cmd_predict():
                sys.exit(1)

    elapsed = time.time() - t_start
    log.info("Pipeline completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
