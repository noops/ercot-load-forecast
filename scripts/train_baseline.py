"""Train a 24-hour-ahead ERCOT load forecast and compare it to a naive
weekly-seasonal baseline (load exactly 7 days before the target hour).

v1: trains on ~2.5 years of history (2024 through mid-2026) and holds out
the most recent 90 days as test, so the evaluation spans a real seasonal
transition (summer heat into fall) rather than one week of one regime.

Simplification carried over from v0: uses the actual recorded temperature
at the target hour as a stand-in for a weather *forecast*. A real forecast
feed is still a known follow-up (see README).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

HORIZON_HOURS = 24
TEST_DAYS = 90

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
    out["target_month"] = out["target_time"].dt.month
    out["target_is_weekend"] = out["target_dow"].isin([5, 6]).astype(int)
    out["target_temp_c"] = df["temp_c"].shift(-HORIZON_HOURS)

    out["load_now"] = df["load_mw"]
    # load at (target_time - 168h) == load at (now - (168 - HORIZON_HOURS))
    out["load_week_before_target"] = df["load_mw"].shift(168 - HORIZON_HOURS)

    return out.dropna().reset_index(drop=True)


def main() -> None:
    raw = pd.read_csv(RAW_PATH, parse_dates=["Time"])
    data = build_features(raw)

    feature_cols = [
        "target_hour", "target_dow", "target_is_weekend", "target_month",
        "target_temp_c", "load_now", "load_week_before_target",
    ]

    cutoff = data["target_time"].max() - pd.Timedelta(days=TEST_DAYS)
    train = data[data["target_time"] <= cutoff]
    test = data[data["target_time"] > cutoff]

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
    model.fit(train[feature_cols], train["target"])

    test = test.copy()
    test["pred_model"] = model.predict(test[feature_cols])
    test["pred_naive"] = test["load_week_before_target"]

    mae_model = mean_absolute_error(test["target"], test["pred_model"])
    mae_naive = mean_absolute_error(test["target"], test["pred_naive"])
    mape_model = mean_absolute_percentage_error(test["target"], test["pred_model"])
    mape_naive = mean_absolute_percentage_error(test["target"], test["pred_naive"])

    print(f"Train: {train['target_time'].min().date()} to {train['target_time'].max().date()} ({len(train)} rows)")
    print(f"Test:  {test['target_time'].min().date()} to {test['target_time'].max().date()} ({len(test)} rows)")
    print(f"Naive (same hour, 1 week prior):  MAE={mae_naive:,.0f} MW  MAPE={mape_naive:.2%}")
    print(f"XGBoost (24h ahead):               MAE={mae_model:,.0f} MW  MAPE={mape_model:.2%}")

    # Monthly breakdown — does accuracy hold up across the summer->fall transition?
    test["month"] = test["target_time"].dt.to_period("M")
    monthly = test.groupby("month").apply(
        lambda g: pd.Series({
            "mape_model": mean_absolute_percentage_error(g["target"], g["pred_model"]),
            "mape_naive": mean_absolute_percentage_error(g["target"], g["pred_naive"]),
        }),
        include_groups=False,
    )
    print("\nMonthly MAPE:")
    print(monthly.to_string(float_format=lambda x: f"{x:.2%}"))

    RESULTS_DIR.mkdir(exist_ok=True)

    # Monthly MAPE bar chart — the headline v1 result (seasonal robustness)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(monthly))
    width = 0.35
    ax.bar([i - width / 2 for i in x], monthly["mape_naive"] * 100, width, label="Naive (1 week prior)")
    ax.bar([i + width / 2 for i in x], monthly["mape_model"] * 100, width, label="XGBoost (24h ahead)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"24h-ahead forecast error by month (test: last {TEST_DAYS} days)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mape_by_month.png", dpi=150)
    print(f"\nSaved {RESULTS_DIR / 'mape_by_month.png'}")

    # Zoomed-in 14-day snippet for a readable actual-vs-predicted view
    snippet = test.tail(24 * 14)
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.plot(snippet["target_time"], snippet["target"], label="Actual", linewidth=1.5)
    ax2.plot(snippet["target_time"], snippet["pred_model"], label="XGBoost (24h ahead)", linewidth=1.2)
    ax2.plot(snippet["target_time"], snippet["pred_naive"], label="Naive (1 week prior)", linewidth=1, linestyle="--")
    ax2.set_ylabel("System load (MW)")
    ax2.set_title("ERCOT system load: 24h-ahead forecast vs. actual (last 14 days of test period)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "forecast_vs_actual.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'forecast_vs_actual.png'}")


if __name__ == "__main__":
    main()
