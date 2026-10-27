"""Rolling-origin comparison of ARIMA and SARIMA against the naive and
XGBoost baselines already established in train_baseline.py.

v1.1: backtests from multiple hours-of-day (0, 4, 8, 12, 16, 20), not just
midnight. The original version's rolling origins happened to all land on
hour 0 (stepping by exactly 24h from a midnight-aligned test start), which
meant ARIMA/SARIMA were only ever evaluated on one point in the daily cycle
— a real asymmetry against XGBoost's full-24-hours evaluation. This isn't
every hour (that would take ~90 min of fitting), but 6 spread across the
day is a much more representative sample than 1.

Classical ARIMA/SARIMA don't need (and can't efficiently use) 2.5 years of
hourly data in one fit, so this uses a rolling trailing window instead:
at each origin, fit on the trailing TRAIN_WINDOW_HOURS of actual load,
forecast 24 hours ahead, record the forecast, move to the next origin.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

warnings.filterwarnings("ignore")

TEST_DAYS = 90
TRAIN_WINDOW_HOURS = 24 * 21  # 3 weeks trailing
HOURS_OF_DAY = [0, 4, 8, 12, 16, 20]

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "ercot_load_weather.csv"
RESULTS_DIR = ROOT / "results"


def rolling_forecast_for_hour(df: pd.DataFrame, start_hour: int, test_cutoff: pd.Timestamp) -> list[dict]:
    series = df["load_mw"]
    origin_mask = (df["Time"].dt.hour == start_hour) & (df["Time"] >= test_cutoff)
    origin_indices = df.index[origin_mask].tolist()

    rows = []
    for origin in tqdm(origin_indices, desc=f"hour={start_hour:02d}:00", leave=False):
        if origin + 24 > len(series) or origin - TRAIN_WINDOW_HOURS < 0:
            continue
        window = series.iloc[origin - TRAIN_WINDOW_HOURS : origin]
        actual_24h = series.iloc[origin : origin + 24].to_numpy()

        arima_fc = ARIMA(window, order=(2, 1, 2)).fit().forecast(steps=24).to_numpy()
        sarima_fc = (
            SARIMAX(
                window, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False, enforce_invertibility=False,
            )
            .fit(disp=False)
            .forecast(steps=24)
            .to_numpy()
        )

        rows.append({
            "origin": df["Time"].iloc[origin],
            "start_hour": start_hour,
            "actual_24h": actual_24h[-1],
            "arima_24h_mae": abs(arima_fc[-1] - actual_24h[-1]),
            "sarima_24h_mae": abs(sarima_fc[-1] - actual_24h[-1]),
            "arima_path": arima_fc.tolist(),
            "sarima_path": sarima_fc.tolist(),
            "actual_path": actual_24h.tolist(),
        })
    return rows


def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["Time"]).sort_values("Time").reset_index(drop=True)
    test_cutoff = df["Time"].max() - pd.Timedelta(days=TEST_DAYS)

    all_rows = []
    for h in HOURS_OF_DAY:
        all_rows += rolling_forecast_for_hour(df, h, test_cutoff)
        print(f"  hour={h:02d}:00 done ({len([r for r in all_rows if r['start_hour'] == h])} origins)")

    results = pd.DataFrame(all_rows)

    print(f"\nTotal origins across {len(HOURS_OF_DAY)} hours-of-day: {len(results)}")
    mape_arima = (results["arima_24h_mae"] / results["actual_24h"]).mean()
    mape_sarima = (results["sarima_24h_mae"] / results["actual_24h"]).mean()
    print(f"ARIMA(2,1,2):            MAPE={mape_arima:.2%}")
    print(f"SARIMA(1,0,1)(1,1,1,24): MAPE={mape_sarima:.2%}")

    print("\nMAPE by hour-of-day:")
    by_hour = results.groupby("start_hour").apply(
        lambda g: pd.Series({
            "arima": (g["arima_24h_mae"] / g["actual_24h"]).mean(),
            "sarima": (g["sarima_24h_mae"] / g["actual_24h"]).mean(),
        }),
        include_groups=False,
    )
    print(by_hour.to_string(float_format=lambda x: f"{x:.2%}"))

    # Save without the per-step path columns for the main results file
    # (those are large and only needed for the horizon-profile chart below)
    results.drop(columns=["arima_path", "sarima_path", "actual_path"]).to_csv(
        RESULTS_DIR / "classical_rolling_results.csv", index=False
    )
    print(f"\nSaved {RESULTS_DIR / 'classical_rolling_results.csv'}")

    # Horizon-degradation profile, now averaged across all hours-of-day
    arima_paths = np.array(results["arima_path"].tolist())
    sarima_paths = np.array(results["sarima_path"].tolist())
    actual_paths = np.array(results["actual_path"].tolist())
    horizon_mape_arima = [mean_absolute_percentage_error(actual_paths[:, h], arima_paths[:, h]) for h in range(24)]
    horizon_mape_sarima = [mean_absolute_percentage_error(actual_paths[:, h], sarima_paths[:, h]) for h in range(24)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = range(1, 25)
    ax.plot(hours, [v * 100 for v in horizon_mape_arima], marker="o", label="ARIMA(2,1,2)")
    ax.plot(hours, [v * 100 for v in horizon_mape_sarima], marker="o", label="SARIMA(1,0,1)(1,1,1,24)")
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Classical model accuracy vs. forecast horizon\n({len(results)} origins across {len(HOURS_OF_DAY)} hours-of-day, {TEST_DAYS}-day backtest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classical_horizon_profile.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'classical_horizon_profile.png'}")

    # By-hour-of-day accuracy chart — does the model struggle at certain times of day?
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    x = range(len(by_hour))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], by_hour["arima"] * 100, width, label="ARIMA(2,1,2)")
    ax2.bar([i + width / 2 for i in x], by_hour["sarima"] * 100, width, label="SARIMA(1,0,1)(1,1,1,24)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"{h:02d}:00" for h in by_hour.index])
    ax2.set_xlabel("Forecast origin (hour of day)")
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("24h-ahead accuracy by hour-of-day the forecast was made")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "classical_by_hour.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'classical_by_hour.png'}")


if __name__ == "__main__":
    main()
