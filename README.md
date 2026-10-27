# ERCOT Load Forecast

Short-horizon (24h-ahead) load forecasting for the Texas grid: naive
baseline, ARIMA, SARIMA, and XGBoost, benchmarked against each other on
the same held-out window.

Status: **v1 — four models backtested across a seasonal transition.**
Part of [Bat City AI](https://batcityai.com)'s Year One programs.

## Result (most recent run)

Four models, same 90-day held-out test window (Jul 28 – Oct 25, 2026 — a
window that itself spans the transition from summer heat into fall):

| Model | MAE | MAPE |
|---|---|---|
| Naive (1 week prior) | 3,842 MW | 6.30% |
| ARIMA(2,1,2) | 2,532 MW | 4.07% |
| XGBoost + Fourier (24h ahead, all hours) | 2,359 MW | 3.45% |
| **SARIMA(1,0,1)(1,1,1,24)** | **1,632 MW** | **2.65%** |

**SARIMA still wins**, even after giving XGBoost the same seasonal signal
SARIMA gets for free. Adding daily + weekly Fourier terms (sin/cos features
at the 24h and 168h cycles — `scripts/train_baseline.py`) closed some of
the gap but didn't close all of it. On the exact 89 midnight timestamps
ARIMA/SARIMA were evaluated on (the fair, matched comparison — see caveat
below), XGBoost+Fourier scored 3.04% MAPE: clearly ahead of plain ARIMA's
4.07%, but still behind SARIMA's 2.65%. The seasonal ARIMA term is doing
something more structurally correct for this series than engineered
sinusoidal features give a tree model — a real, if modest, finding, not a
feature-engineering oversight.

**Caveat on that comparison:** the ARIMA/SARIMA backtest only ever forecasts
from midnight (its rolling origins step by exactly 24h, so every origin
lands on the same hour-of-day) — 89-90 samples, one hour-of-day only. The
3.53%/3.45% XGBoost numbers above are averaged across *all* 2,160 hourly
test points, a harder and more representative task. Extending the classical
backtest to multiple hours-of-day is the obvious next step before trusting
this ranking much further (see Known Limitations).

![Forecast error by month](results/mape_by_month.png)

The monthly breakdown (XGBoost vs. naive) shows why the naive baseline is
dangerous to trust blindly: in July and August, when the weather is just
"hot," same-hour-last-week is nearly as good as any model. The gap opens up
hard in October (14.99% vs. 4.15% MAPE) once temperatures start swinging
week to week.

![Forecast vs actual, last 14 days](results/forecast_vs_actual.png)

### ARIMA vs. SARIMA by month

Rolling daily refit on a trailing 21-day window, forecasting 24 hours ahead
each time (`scripts/compare_classical.py`):

| Month | n days | ARIMA | SARIMA |
|---|---|---|---|
| 2026-07 | 4 (too few to trust) | 0.50% | 0.56% |
| 2026-08 | 31 | 2.56% | 2.22% |
| 2026-09 | 30 | 4.52% | 2.07% |
| 2026-10 | 25 | 5.99% | 4.23% |

![Classical model accuracy vs. forecast horizon](results/classical_horizon_profile.png)

This horizon chart is the clearest explanation of *why* SARIMA wins: plain
ARIMA has no way to represent the daily load cycle, so its error balloons
as the horizon stretches past a few hours (peaking near hour 16-17) before
partially recovering as the forecast wraps back into phase with a similar
time of day. SARIMA's seasonal term keeps the daily cycle in view the whole
time, so its error stays flat and low across the full 24-hour horizon.

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
   same 90-day window: at the start of each test day, fit ARIMA and SARIMA
   on the trailing 21 days of actual load, forecast 24 hours ahead, record
   the result, advance one day, repeat. This is also what produces the
   horizon-degradation chart (accuracy at each of the 24 forecast steps).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS only — xgboost needs the OpenMP runtime
brew install libomp

python3 scripts/fetch_data.py
python3 scripts/train_baseline.py
python3 scripts/compare_classical.py  # ~4 min: 90 daily ARIMA + SARIMA refits
```

## Known limitations (v2 candidates)

- **ARIMA/SARIMA were only backtested at one hour-of-day (midnight).**
  Their rolling origins step by exactly 24h, so all 89-90 samples share an
  hour-of-day — likely an easier target than, say, a 6pm demand peak.
  Running the same rolling backtest from multiple hours-of-day (and
  re-running the XGBoost+Fourier head-to-head against that richer sample)
  is the highest-value next step before trusting the SARIMA-wins result
  much further.
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
