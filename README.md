# UFC Fight Predictor — Strickland vs Chimaev

End-to-end UFC fight prediction pipeline that scrapes UFCStats, engineers features, trains a stacking ensemble, runs Monte Carlo simulations, and generates publication-ready visuals.

## What’s inside
- **Scraping**: events, fights, fighter profiles, and detailed fight stats from UFCStats.com
- **Features**: ELO, rolling stats, opponent quality, physical differentials, win streaks, and more
- **Model**: stacking ensemble (XGBoost + LightGBM + RandomForest + SVM → Logistic Regression)
- **Prediction**: Monte Carlo bootstrap for uncertainty + SHAP explanations
- **Visuals**: donut, radar, ELO history, feature importance, model performance, and summary card

## Quick start
1. Install dependencies
2. Run the full pipeline

The pipeline automatically scrapes data, builds features, trains the model, and generates predictions + visualizations.

## Pipeline usage
- Full run:
  - `python run.py`
- Individual stages:
  - `python run.py --scrape`
  - `python run.py --features`
  - `python run.py --train`
  - `python run.py --predict`

## Outputs
Artifacts are written to `data/processed/`, including:
- `features.csv` — model-ready feature matrix
- `model.joblib`, `scaler.joblib`, `selector.joblib`, `metadata.joblib`
- Visuals: `01_donut.png` … `07_summary_card.png`

## Notes
- Scraping **all** fight details can take a while; UFCStats may throttle requests.
- If you already have `data/raw/*`, you can skip scraping and start at feature generation.

## Troubleshooting
- If `features.csv` is empty, ensure `data/raw/fights.csv` exists and has decisive results.
- If the model can’t load, re-run `python run.py --train`.
