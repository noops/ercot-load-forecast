# ERCOT Load Forecast

Short-horizon (24h-ahead) load forecasting for the Texas grid: naive
baseline, ARIMA, SARIMA, and XGBoost, benchmarked against each other on
the same held-out window.

Status: **v1 — four models backtested across a seasonal transition.**
Part of [Bat City AI](https://batcityai.com)'s Year One programs.

## Result (most recent run)

Four models, same 90-day held-out test window (Jul 28 – Oct 25, 2026 — a
window that itself spans the transition from summer heat into fall).
ARIMA/SARIMA are backtested from 6 different hours-of-day (00:00, 04:00,
08:00, 12:00, 16:00, 20:00 — 535 origins total), not just one:

| Model | MAPE |
|---|---|
| Naive (1 week prior) | 6.30% |
| ARIMA(2,1,2) | 3.79% |
| XGBoost + Fourier (24h ahead) | 3.45% |
| **SARIMA(1,0,1)(1,1,1,24)** | **3.07%** |

**SARIMA still wins, but the margin is much smaller than it first looked.**
An earlier version of this backtest only ever forecast from midnight (a
bug in how the rolling origins were spaced — see commit history) and made
SARIMA look better than it is: 2.65% vs. XGBoost's 3.53%, a 0.8-point gap.
Spread across 6 hours-of-day instead, SARIMA's own score gets *worse*
(2.65% → 3.07%) and the gap to XGBoost+Fourier on the identical matched
timestamps narrows to 0.28 points (3.07% vs. 3.35%). The ranking didn't
flip, but "SARIMA clearly wins" became "SARIMA has a real but modest edge"
once the easier evaluation slice was corrected. Worth remembering before
trusting any single-slice backtest.

![Forecast error by month](results/mape_by_month.png)

The monthly breakdown (XGBoost vs. naive) shows why the naive baseline is
dangerous to trust blindly: in July and August, when the weather is just
"hot," same-hour-last-week is nearly as good as any model. The gap opens up
hard in October (14.99% vs. 4.00% MAPE) once temperatures start swinging
week to week.

![Forecast vs actual, last 14 days](results/forecast_vs_actual.png)

### ARIMA vs. SARIMA by month and hour-of-day

Rolling refit on a trailing 21-day window at each origin, forecasting 24
hours ahead (`scripts/compare_classical.py`):

| Month | n origins | ARIMA | SARIMA |
|---|---|---|---|
| 2026-07 | 24 | 2.18% | 1.70% |
| 2026-08 | 186 | 2.44% | 2.31% |
| 2026-09 | 180 | 3.73% | 2.59% |
| 2026-10 | 145 | 5.88% | 4.88% |

![24h-ahead accuracy by hour-of-day](results/classical_by_hour.png)

SARIMA beats ARIMA at every hour-of-day we tested, but the margin varies —
near-identical at 16:00, widest at 00:00 and 20:00.

![Classical model accuracy vs. forecast horizon](results/classical_horizon_profile.png)

This horizon chart is the clearest explanation of *why* SARIMA wins: plain
ARIMA has no way to represent the daily load cycle, so its error balloons
as the horizon stretches toward 12-15 hours out before partially recovering
as the forecast wraps back into phase with a similar time of day. SARIMA's
seasonal term keeps the daily cycle in view the whole time, so its error
climbs far more gently across the full 24-hour horizon.

## How it works

1. `scripts/fetch_data.py` combines ERCOT's static annual
   ["Native Load" archives](https://www.ercot.com/gridinfo/load/load_hist)
   (full years, posted with a settlement lag) with the live "Actual System
   Load by Weather Zone" feed (via [gridstatus](https://github.com/kmax12/gridstatus))
   for the trailing days not yet archived — giving continuous history from
   Jan 2024 through yesterday instead of the ~30 days the live feed alone
   retains. Matching Austin temperature comes from
   [Open-Meteo](https://open-meteo.com)'s free historical archive.
2. `scripts/train_baseline.py` builds lag + calendar + temperature +
   Fourier (daily/weekly sin-cos) features, trains an XGBoost regressor to
   predict load 24 hours out, and compares it to the naive baseline on a
   90-day held-out test window, with a monthly error breakdown and a
   head-to-head against ARIMA/SARIMA on their matched timestamps.
3. `scripts/compare_classical.py` runs a rolling-origin backtest over the
   same 90-day window from 6 different hours-of-day: fit ARIMA and SARIMA
   on the trailing 21 days of actual load, forecast 24 hours ahead, record
   the result, advance one day, repeat — 535 origins total. Also produces
   the horizon-degradation and by-hour-of-day accuracy charts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS only — xgboost needs the OpenMP runtime
brew install libomp

python3 scripts/fetch_data.py
python3 scripts/train_baseline.py
python3 scripts/compare_classical.py  # ~20-25 min: 535 ARIMA + SARIMA refits (6 hours-of-day x 90 days)
```

## Known limitations (v2 candidates)

- **Still not literally every hour.** 6 of 24 hours-of-day is a much more
  representative sample than 1, but not exhaustive — a full 24-hour sweep
  would take ~90 minutes of fitting and wasn't run.
- **Fourier terms closed some, not all, of the gap to SARIMA.** Residual
  diagnostics on SARIMA's fit, or feeding SARIMA's own forecast as an
  XGBoost feature, are both reasonable next tries.
- **Classical models use a short, fixed 21-day trailing window**, refit
  daily. Different window sizes or refit cadences weren't swept.
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
