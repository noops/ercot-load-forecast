"""Rolling-origin comparison of ARIMA and SARIMA against the naive and
XGBoost baselines already established in train_baseline.py.

Classical ARIMA/SARIMA don't need (and can't efficiently use) 2.5 years of
hourly data in one fit, so this uses a rolling trailing window instead:
at the start of each test day, fit on the trailing TRAIN_WINDOW_HOURS of
actual load, forecast 24 hours ahead, record the forecast, advance one day,
repeat. This also produces a full 1h-24h horizon profile as a side effect,
which is a useful diagnostic classical models give you for free.
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

TEST_HOURS = 24 * 90
TRAIN_WINDOW_HOURS = 24 * 21  # 3 weeks trailing

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "ercot_load_weather.csv"
RESULTS_DIR = ROOT / "results"


def rolling_forecast(series: pd.Series, test_start_idx: int, n_days: int) -> dict:
    arima_preds, sarima_preds, actuals, origins = [], [], [], []

    for day in tqdm(range(n_days), desc="Rolling daily refit"):
        origin = test_start_idx + day * 24
        window = series.iloc[origin - TRAIN_WINDOW_HOURS : origin]
        actual_24h = series.iloc[origin : origin + 24].to_numpy()
        if len(actual_24h) < 24:
            break

        arima_fc = ARIMA(window, order=(2, 1, 2)).fit().forecast(steps=24).to_numpy()
        sarima_fc = (
            SARIMAX(
                window,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            .fit(disp=False)
            .forecast(steps=24)
            .to_numpy()
        )

        arima_preds.append(arima_fc)
        sarima_preds.append(sarima_fc)
        actuals.append(actual_24h)
        origins.append(series.index[origin])

    return {
        "arima": np.array(arima_preds),
        "sarima": np.array(sarima_preds),
        "actual": np.array(actuals),
        "origins": origins,
    }


def main() -> None:
    df = pd.read_csv(RAW_PATH, parse_dates=["Time"]).sort_values("Time").reset_index(drop=True)
    series = df["load_mw"]

    test_start_idx = len(series) - TEST_HOURS
    n_days = TEST_HOURS // 24

    result = rolling_forecast(series, test_start_idx, n_days)
    actual_24th_step = result["actual"][:, -1]
    arima_24th_step = result["arima"][:, -1]
    sarima_24th_step = result["sarima"][:, -1]

    print("\n24h-ahead comparison (last step of each daily forecast):")
    for name, preds in [("ARIMA(2,1,2)", arima_24th_step), ("SARIMA(1,0,1)(1,1,1,24)", sarima_24th_step)]:
        mae = mean_absolute_error(actual_24th_step, preds)
        mape = mean_absolute_percentage_error(actual_24th_step, preds)
        print(f"{name:28s} MAE={mae:,.0f} MW  MAPE={mape:.2%}")

    # Horizon-degradation profile: MAPE at each of the 24 forecast steps
    horizon_mape = {
        "arima": [mean_absolute_percentage_error(result["actual"][:, h], result["arima"][:, h]) for h in range(24)],
        "sarima": [mean_absolute_percentage_error(result["actual"][:, h], result["sarima"][:, h]) for h in range(24)],
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = range(1, 25)
    ax.plot(hours, [v * 100 for v in horizon_mape["arima"]], marker="o", label="ARIMA(2,1,2)")
    ax.plot(hours, [v * 100 for v in horizon_mape["sarima"]], marker="o", label="SARIMA(1,0,1)(1,1,1,24)")
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Classical model accuracy vs. forecast horizon (90-day rolling backtest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "classical_horizon_profile.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'classical_horizon_profile.png'}")

    # Save raw results for reuse / monthly breakdown
    out = pd.DataFrame({
        "origin": result["origins"],
        "arima_24h_mae": np.abs(arima_24th_step - actual_24th_step),
        "sarima_24h_mae": np.abs(sarima_24th_step - actual_24th_step),
        "actual_24h": actual_24th_step,
    })
    out.to_csv(RESULTS_DIR / "classical_rolling_results.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'classical_rolling_results.csv'}")


if __name__ == "__main__":
    main()
