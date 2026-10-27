"""Give SARIMA the same per-zone weather upgrade XGBoost got, as exogenous
regressors (SARIMAX) — the fair rematch flagged in README limitations
after XGBoost overtook plain SARIMA.

Scoped to one hour-of-day (00:00) across the 90-day test window rather
than the full 6-hour sweep compare_classical.py does, since SARIMAX with
8 exogenous regressors is ~3.5x slower per fit (~6.7s vs ~1.9s) — a full
6-hour sweep would take about an hour. This is a fast, honest first check;
expanding to multiple hours-of-day is a natural follow-up if the result
looks like it matters.
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")

TEST_DAYS = 90
TRAIN_WINDOW_HOURS = 24 * 21
START_HOUR = 0

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "ercot_load_weather.csv"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["Time"]).sort_values("Time").reset_index(drop=True)
    zone_cols = [c for c in df.columns if c.startswith("temp_c_")]
    test_cutoff = df["Time"].max() - pd.Timedelta(days=TEST_DAYS)

    origin_mask = (df["Time"].dt.hour == START_HOUR) & (df["Time"] >= test_cutoff)
    origin_indices = df.index[origin_mask].tolist()

    rows = []
    for origin in tqdm(origin_indices, desc="SARIMAX rolling refit"):
        if origin + 24 > len(df) or origin - TRAIN_WINDOW_HOURS < 0:
            continue
        window = df.iloc[origin - TRAIN_WINDOW_HOURS : origin]
        future = df.iloc[origin : origin + 24]

        model = SARIMAX(
            window["load_mw"], exog=window[zone_cols],
            order=(1, 0, 1), seasonal_order=(1, 1, 1, 24),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = model.forecast(steps=24, exog=future[zone_cols]).to_numpy()

        rows.append({
            "origin": df["Time"].iloc[origin],
            "actual_24h": future["load_mw"].iloc[-1],
            "sarimax_24h_mae": abs(fc[-1] - future["load_mw"].iloc[-1]),
        })

    results = pd.DataFrame(rows)
    mape = (results["sarimax_24h_mae"] / results["actual_24h"]).mean()
    mae = results["sarimax_24h_mae"].mean()
    print(f"\nSARIMAX + 8-zone weather, {START_HOUR:02d}:00 only, {len(results)} origins:")
    print(f"MAE={mae:,.0f} MW  MAPE={mape:.2%}")

    results.to_csv(RESULTS_DIR / "sarimax_rolling_results.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'sarimax_rolling_results.csv'}")


if __name__ == "__main__":
    main()
