"""Train a 24-hour-ahead ERCOT load forecast and compare it to a naive
weekly-seasonal baseline (load exactly 7 days before the target hour).

Simplification for v0: uses the actual recorded temperature at the target
hour as a stand-in for a weather *forecast*, since Open-Meteo's forecast
archive and historical archive aren't the same endpoint. A real forecast
feed is a known v1 upgrade (see README).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

HORIZON_HOURS = 24
TEST_HOURS = 24 * 7  # last 7 days held out

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "ercot_load_weather.csv"
RESULTS_DIR = ROOT / "results"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Time").reset_index(drop=True)
    df["hour"] = df["Time"].dt.hour
    df["dow"] = df["Time"].dt.dayofweek

    out = pd.DataFrame(index=df.index)
    out["target_time"] = df["Time"].shift(-HORIZON_HOURS)
    out["target"] = df["load_mw"].shift(-HORIZON_HOURS)

    out["target_hour"] = df["hour"].shift(-HORIZON_HOURS)
    out["target_dow"] = df["dow"].shift(-HORIZON_HOURS)
    out["target_is_weekend"] = out["target_dow"].isin([5, 6]).astype(int)
    out["target_temp_c"] = df["temp_c"].shift(-HORIZON_HOURS)

    out["load_now"] = df["load_mw"]
    # load at (target_time - 168h) == load at (now - (168 - HORIZON_HOURS))
    out["load_week_before_target"] = df["load_mw"].shift(168 - HORIZON_HOURS)

    return out.dropna().reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_PATH, parse_dates=["Time"])
    data = build_features(raw)

    feature_cols = ["target_hour", "target_dow", "target_is_weekend", "target_temp_c", "load_now", "load_week_before_target"]
    train = data.iloc[: -TEST_HOURS]
    test = data.iloc[-TEST_HOURS:]

    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(train[feature_cols], train["target"])

    pred_model = model.predict(test[feature_cols])
    pred_naive = test["load_week_before_target"].to_numpy()
    actual = test["target"].to_numpy()

    mae_model = mean_absolute_error(actual, pred_model)
    mae_naive = mean_absolute_error(actual, pred_naive)
    mape_model = mean_absolute_percentage_error(actual, pred_model)
    mape_naive = mean_absolute_percentage_error(actual, pred_naive)

    print(f"Rows: train={len(train)}, test={len(test)}")
    print(f"Naive (same hour, 1 week prior):  MAE={mae_naive:,.0f} MW  MAPE={mape_naive:.2%}")
    print(f"XGBoost (24h ahead):               MAE={mae_model:,.0f} MW  MAPE={mape_model:.2%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(test["target_time"], actual, label="Actual", linewidth=1.5)
    ax.plot(test["target_time"], pred_model, label="XGBoost (24h ahead)", linewidth=1.2)
    ax.plot(test["target_time"], pred_naive, label="Naive (1 week prior)", linewidth=1, linestyle="--")
    ax.set_ylabel("System load (MW)")
    ax.set_title("ERCOT system load: 24h-ahead forecast vs. actual (last 7 days)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "forecast_vs_actual.png", dpi=150)
    print(f"Saved plot to {RESULTS_DIR / 'forecast_vs_actual.png'}")


if __name__ == "__main__":
    main()
