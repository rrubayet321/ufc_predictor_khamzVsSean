# UFC Fight Predictor — Strickland vs Chimaev

End-to-end UFC fight prediction system that scrapes UFCStats, engineers advanced features, trains a stacking ensemble, runs Monte Carlo simulations for uncertainty, and generates publication-ready visuals.

## Highlights
- **Full pipeline**: scrape → engineer features → train → predict → visualize
- **Feature-rich modeling**: ELO, rolling performance, physical differentials, opponent quality, win streaks
- **Stacking ensemble**: XGBoost + LightGBM + RandomForest + SVM → Logistic Regression
- **Uncertainty modeling**: Monte Carlo bootstrap on the meta-learner
- **Professional visuals**: donut, radar, ELO history, feature importance, performance dashboard, summary card

## Project structure
- `scraper.py` — collects events, fights, fighter profiles, and detailed round stats
- `features.py` — builds a fight-level feature matrix (`data/processed/features.csv`)
- `train.py` — trains the stacking ensemble and saves the model bundle
- `predict.py` — produces prediction + MC distribution + SHAP explanation
- `visualize.py` — generates all figures for reporting
- `run.py` — orchestration entry point

## Quick start
1. Install dependencies
2. Run the full pipeline

The pipeline automatically scrapes data, builds features, trains the model, and generates predictions and visuals.

## Usage
Run the full pipeline:
- `python run.py`

Run stages independently:
- `python run.py --scrape`
- `python run.py --features`
- `python run.py --train`
- `python run.py --predict`

## Prediction output
`predict.py` prints a summary including:
- Estimated win probabilities for both fighters
- 95% confidence intervals from Monte Carlo simulation
- Model accuracy (CV mean ± std)
- Saved plots for SHAP and probability distribution

Prediction artifacts are written to `data/processed/`:
- `mc_distribution.png`
- `shap_waterfall.png` (if SHAP is installed)

## Visualizations
`visualize.py` produces a full set of report-ready graphics:
1. `01_donut.png` — win probability donut chart
2. `02_radar.png` — fighter profile radar comparison
3. `03_elo_history.png` — ELO rating history
4. `04_feature_importance.png` — top model drivers
5. `05_head_to_head.png` — career comparison bars
6. `06_model_performance.png` — ROC + confusion matrix
7. `07_summary_card.png` — LinkedIn-style summary panel

All figures are saved under `data/processed/`.

## Outputs
- `data/processed/features.csv` — model-ready feature matrix
- `data/processed/model.joblib` — trained ensemble bundle
- `data/processed/scaler.joblib` — feature scaler
- `data/processed/selector.joblib` — optional feature selector
- `data/processed/metadata.joblib` — feature metadata
- Visualizations: `01_donut.png` … `07_summary_card.png`

## Notes
- Scraping all fight details can take time; UFCStats may throttle requests.
- If raw data already exists in `data/raw/`, you can start from `--features`.

## Troubleshooting
- If `features.csv` is empty, ensure `data/raw/fights.csv` exists and has decisive results.
- If prediction fails, re-run `python run.py --train` to regenerate the model bundle.
