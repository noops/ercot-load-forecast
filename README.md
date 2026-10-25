# ERCOT Load Forecast

Short-horizon (24h-ahead) load forecasting for the Texas grid, benchmarked
against a naive seasonal baseline.

Status: **v0 — working prototype.** Part of [Bat City AI](https://batcityai.com)'s
Year One programs.

## Result (most recent run)

24-hour-ahead forecast vs. a naive "same hour, one week prior" baseline,
evaluated on the trailing 7 days held out of training:

| Model | MAE | MAPE |
|---|---|---|
| Naive (1 week prior) | 11,847 MW | 23.34% |
| XGBoost (24h ahead) | 2,151 MW | 4.43% |

![Forecast vs actual](results/forecast_vs_actual.png)

The naive baseline overshoots badly here because temperatures were dropping
week-over-week into fall — a pattern a same-hour-last-week lookup can't see,
but the model's temperature feature can.

## How it works

1. `scripts/fetch_data.py` pulls ERCOT's public "Actual System Load by
   Weather Zone" report (via [gridstatus](https://github.com/kmax12/gridstatus),
   no auth required) and matching Austin temperature from
   [Open-Meteo](https://open-meteo.com)'s free historical archive.
2. `scripts/train_baseline.py` builds lag + calendar + temperature features,
   trains an XGBoost regressor to predict load 24 hours out, and compares it
   to the naive baseline on a held-out week.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS only — xgboost needs the OpenMP runtime
brew install libomp

python3 scripts/fetch_data.py
python3 scripts/train_baseline.py
```

## Known limitations (v1 candidates)

- **30-day data window.** ERCOT's live document feed only retains ~30 days
  of history for this report, so the model trains on ~3 weeks of hourly
  data after feature lags are trimmed. No way yet to see full seasonal
  (summer/winter) patterns. ERCOT publishes static annual historical load
  archives separately — pulling those in is the highest-leverage v1 change.
- **Temperature is not a forecast.** The model uses the *actual* recorded
  temperature at the target hour as a stand-in for a weather forecast,
  since this is a historical-data pipeline. A real deployment would need a
  genuine weather forecast feed, which has its own error and would make
  the model's real-world accuracy somewhat worse than shown here.
- **Single weather station.** Austin only; ERCOT spans multiple climate
  zones (Houston humidity, West Texas heat, Panhandle cold snaps).

## Data sources

- ERCOT public load data — [ERCOT MIS](https://www.ercot.com), via `gridstatus`
- Weather — [Open-Meteo](https://open-meteo.com) historical archive (free, no key)

## License

MIT
