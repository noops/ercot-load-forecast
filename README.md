# ERCOT Load Forecast

A real, runnable next-24-hour load forecast for the Texas grid, built on
fully public data. Not a benchmark notebook you have to interpret —
`python3 forecast.py` gets you tomorrow's hourly forecast.

Status: **v4 — a real tool, not just a research comparison.** Part of
[Bat City AI](https://batcityai.com)'s Year One programs.

## Quickstart

```bash
git clone https://github.com/noops/ercot-load-forecast
cd ercot-load-forecast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install libomp  # macOS only — xgboost needs the OpenMP runtime

python3 forecast.py
```

Takes about a minute (it refetches fresh ERCOT load history and a live
weather forecast every run — see [How it works](#how-it-works)). Example
output from an actual run:

```
Forecast made at 2026-10-26 23:00:00 (most recent known load: 49,574 MW)
Next 24h peak:   60,178 MW at 2026-10-27 16:00:00
Next 24h trough: 42,322 MW at 2026-10-27 03:00:00
Saved hourly forecast to results/next_24h_forecast.csv
Saved plot to results/next_24h_forecast.png
```

No API keys, no paid data. Output is a 24-row CSV (`target_time`,
`predicted_load_mw`) and a PNG chart of the next day's hourly shape.

## How it works

`forecast.py`:

1. Refreshes ERCOT load history via `scripts/fetch_data.py` (static
   annual archives + the live feed, combined — see below) and fetches a
   real weather forecast (not historical reanalysis) for the next 2 days
   across all 8 ERCOT weather zones from
   [Open-Meteo](https://open-meteo.com).
2. Trains one XGBoost model across *every* horizon from 1 to 24 hours
   ahead (horizon is itself a feature), so it produces a full next-day
   hourly profile from "right now" — not just a single point 24 hours out
   like the research backtest below.
3. Builds the next-24-hour feature rows using the latest known actual
   load plus the live weather forecast, predicts, and saves a CSV + plot.

One real wrinkle worth knowing about: **ERCOT's own settlement data lags
"now" by roughly a day.** The forecast origin is whatever the most recent
*actual* load reading is — which might be yesterday, not this exact
minute. To avoid a gap between "latest actual load" and "earliest weather
forecast," the weather fetch uses Open-Meteo's `past_days` parameter to
pull a few days of recent-past weather in the same call as the forecast,
bridging it cleanly.

## First real-world check

The backtests below are all walk-forward *on history that already
happened*. The first genuinely live test — forecast made at the end of
Oct 25 for Oct 26, checked against what ERCOT actually reported once Oct
26 was over — came in at **5.41% MAPE**, with every single hour
under-predicted by roughly 5-8%. The daily *shape* was right (trough near
4am, peak near 4pm); the whole day just ran hotter than predicted.

This is one data point, not a trend — but it's already more honest than
the backtested ~2.2-2.3% numbers below, which were walk-forward over 90
days, not a single real morning when you don't yet know the answer.
Worth tracking over more days before drawing conclusions. (It also caught
a real bug: `fetch_data.py` was requesting one day less than intended
from the live feed, because `gridstatus` treats its `end` date as
exclusive — fixed now, but worth knowing the forecast-vs-check dates
above predate that fix.) See
[Known Limitations](#known-limitations).

## Research & validation

The live tool above exists because of this benchmark work — same data,
same features, evaluated by deliberately holding out history instead of
forecasting the real future.

Four models, same 90-day held-out test window (Jul 28 – Oct 25, 2026 — a
window that itself spans the transition from summer heat into fall).
ARIMA/SARIMA are backtested from 6 different hours-of-day (00:00, 04:00,
08:00, 12:00, 16:00, 20:00 — 535 origins total), not just one:

| Model | MAPE |
|---|---|
| Naive (1 week prior) | 6.30% |
| ARIMA(2,1,2) | 3.79% |
| SARIMA(1,0,1)(1,1,1,24) | 3.07% |
| **XGBoost + Fourier + 8-zone weather (24h ahead)** | **2.70%** |

**XGBoost wins outright** against plain SARIMA. The previous round gave
XGBoost Fourier terms for the daily/weekly cycle and it still lost
(3.45% vs. 3.07%). The missing piece turned out to be simpler than a
better seasonal representation: **ERCOT spans multiple climates, and the
model only ever saw Austin's temperature.** `fetch_data.py` pulls one
weather station per ERCOT weather zone (Houston for Coast, Midland for
Far West, Wichita Falls for North, etc. — 8 zones total) instead of one
station for the whole state. That alone dropped full-test MAPE from
3.45% to 2.70%, and the improvement held in every month, not one lucky
stretch (see table below).

This is the more interesting finding than "model X beats model Y": the
bottleneck wasn't model architecture or feature-engineering cleverness —
it was that the input data didn't represent the geography of the thing
being forecast. Worth remembering before reaching for a fancier model
when a simpler one is underfed.

**The fair rematch:** does SARIMA still lose once it also gets the 8-zone
weather, as exogenous regressors (SARIMAX)? Checked on the same 00:00-only
slice as a fast first pass (`scripts/compare_sarimax.py` — the full
6-hour sweep would take ~60 min, since 8 exogenous regressors makes each
fit ~3.5x slower):

| Model (00:00 only, 89-90 origins) | MAPE |
|---|---|
| SARIMAX(1,0,1)(1,1,1,24) + 8-zone weather | 2.30% |
| XGBoost + Fourier + 8-zone weather | 2.20% |

**Essentially tied**, XGBoost with a thin edge. Giving SARIMA the weather
data closed the rest of the gap — the data mattered far more than the
model family, through two full rounds of giving each model a fairer shot.

| Month | XGBoost + Fourier + 8-zone weather | Naive |
|---|---|---|
| 2026-07 | 1.88% | 3.17% |
| 2026-08 | 2.35% | 2.53% |
| 2026-09 | 2.50% | 3.39% |
| 2026-10 | 3.52% | 14.99% |

![Forecast error by month](results/mape_by_month.png)

The naive baseline is dangerous to trust blindly: in July and August,
when the weather is just "hot," same-hour-last-week is nearly as good as
any model. The gap opens up hard in October once temperatures start
swinging week to week — and even there, XGBoost with real geography
clearly leads.

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

SARIMA beats ARIMA at every hour-of-day tested, but the margin varies —
near-identical at 16:00, widest at 00:00 and 20:00.

![Classical model accuracy vs. forecast horizon](results/classical_horizon_profile.png)

This horizon chart is the clearest explanation of *why* SARIMA (and the
multi-horizon XGBoost model `forecast.py` uses) beat plain ARIMA: ARIMA
has no way to represent the daily load cycle, so its error balloons as
the horizon stretches toward 12-15 hours out. A seasonal term — or a
horizon feature, in XGBoost's case — keeps the daily cycle in view the
whole time, so error climbs far more gently across the full 24 hours.

## Setup (research scripts)

`forecast.py` is self-contained — the Quickstart above is all you need
for a live forecast. The scripts below are how the benchmark numbers
were produced, not required for normal use:

```bash
python3 scripts/fetch_data.py
python3 scripts/train_baseline.py
python3 scripts/compare_classical.py  # ~20-25 min: 535 ARIMA + SARIMA refits
python3 scripts/compare_sarimax.py    # ~10 min: SARIMAX + exogenous weather, 00:00 only
```

## Known limitations

**Live forecast tool:**
- **One real-world check so far (5.41% MAPE, systematically low).** Needs
  tracking over more days before trusting the live tool's accuracy the
  way the 90-day backtests below are trusted.
- **The `past_days` gap-bridging assumes ERCOT's reporting lag stays
  under 3 days.** If it ever lags longer, `forecast.py` will raise a
  clear error rather than silently producing a bad forecast — but the
  fix would be bumping `past_days` in `fetch_data.fetch_weather_forecast`.
- **Retrains from scratch every run** (a few seconds, not a real cost
  today, but worth caching if this ever runs on a schedule).

**Research backtest:**
- **The SARIMAX rematch was only checked at one hour-of-day.** Same risk
  flagged earlier for plain SARIMA — 00:00 happened to be one of SARIMA's
  *better* hours in the 6-hour sweep. A full multi-hour SARIMAX sweep
  (~60 min) is the real remaining check before trusting "XGBoost is
  slightly ahead" very far.
- **Per-zone *load* (not just temperature) is sitting in the data
  unused.** Both ERCOT sources already report load by zone;
  `fetch_data.py` keeps those columns but nothing uses them as features
  yet.
- **Classical models use a short, fixed 21-day trailing window**, refit
  daily. Different window sizes or refit cadences weren't swept.
- **ERCOT archive URLs aren't stable.** The static yearly files are
  hand-coded in `fetch_data.py`; the current year's filename has an
  inconsistent versioning suffix, so the URL list needs occasional manual
  updates.

## Data sources

- ERCOT public load data — [ERCOT MIS](https://www.ercot.com)
  ([hourly load archives](https://www.ercot.com/gridinfo/load/load_hist) +
  live feed via [gridstatus](https://github.com/kmax12/gridstatus))
- Weather — [Open-Meteo](https://open-meteo.com): historical archive for
  training, live forecast API for `forecast.py` (free, no key either way)

## License

MIT
