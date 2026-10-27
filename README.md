# ERCOT Load Forecast

Short-horizon (24h-ahead) load forecasting for the Texas grid, benchmarked
against a naive seasonal baseline.

Status: **v1 — backtested across a seasonal transition.** Part of
[Bat City AI](https://batcityai.com)'s Year One programs.

## Result (most recent run)

Trained on ~2.5 years of hourly data (Jan 2024 – Jul 2026), evaluated on the
trailing 90 days held out of training — a window that itself spans the
transition from summer heat into fall:

| Model | MAE | MAPE |
|---|---|---|
| Naive (1 week prior) | 3,842 MW | 6.30% |
| XGBoost (24h ahead) | 2,407 MW | 3.53% |

![Forecast error by month](results/mape_by_month.png)

The monthly breakdown is the interesting part: in July and August, when the
weather is just "hot," the naive baseline is roughly as good as the model —
same-hour-last-week is a solid predictor when nothing is changing. The gap
opens up hard in October (14.99% vs. 4.15% MAPE) as temperatures start
swinging week to week. The naive baseline has no way to see that; the
model's temperature feature does.

![Forecast vs actual, last 14 days](results/forecast_vs_actual.png)

## How it works

1. `scripts/fetch_data.py` combines ERCOT's static annual
   ["Native Load" archives](https://www.ercot.com/gridinfo/load/load_hist)
   (full years, posted with a settlement lag) with the live "Actual System
   Load by Weather Zone" feed (via [gridstatus](https://github.com/kmax12/gridstatus))
   for the trailing days not yet archived — giving continuous history from
   Jan 2024 through yesterday instead of the ~30 days the live feed alone
   retains. Matching Austin temperature comes from
   [Open-Meteo](https://open-meteo.com)'s free historical archive.
2. `scripts/train_baseline.py` builds lag + calendar + temperature features,
   trains an XGBoost regressor to predict load 24 hours out, and compares it
   to the naive baseline on a 90-day held-out test window, with a monthly
   error breakdown.

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

## Known limitations (v2 candidates)

- **Temperature is not a forecast.** The model uses the *actual* recorded
  temperature at the target hour as a stand-in for a weather forecast,
  since this is a historical-data pipeline. A real deployment would need a
  genuine weather forecast feed, which has its own error and would make
  the model's real-world accuracy somewhat worse than shown here.
- **Single weather station.** Austin only; ERCOT spans multiple climate
  zones (Houston humidity, West Texas heat, Panhandle cold snaps). The
  data already includes per-weather-zone load (Coast, East, Far West,
  North, North Central, South, South Central, West) — adding a matching
  station per zone is the next obvious lever.
- **ERCOT archive URLs aren't stable.** The static yearly files are
  hand-coded in `fetch_data.py`; the current year's filename has an
  inconsistent versioning suffix, so the URL list needs occasional manual
  updates.

## Data sources

- ERCOT public load data — [ERCOT MIS](https://www.ercot.com)
  ([hourly load archives](https://www.ercot.com/gridinfo/load/load_hist) +
  live feed via `gridstatus`)
- Weather — [Open-Meteo](https://open-meteo.com) historical archive (free, no key)

## License

MIT
