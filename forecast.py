"""Produce a real next-24-hour ERCOT load forecast using current data and
an actual weather forecast feed — not a backtest against history that
already happened (see scripts/train_baseline.py and
scripts/compare_classical.py for the research/validation side of this
project).

Usage:
    python3 forecast.py

Trains one XGBoost model across all horizons 1-24h (horizon is itself a
feature) instead of train_baseline.py's fixed-24h-only model, so this can
produce a full next-day hourly profile from "right now," not just a
single point 24 hours out.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import fetch_data  # noqa: E402
from train_baseline import add_fourier_terms, DAILY_HARMONICS, WEEKLY_HARMONICS  # noqa: E402, F401
from zones import ZONE_STATIONS, ZONE_TEMP_COLS, zone_temp_col  # noqa: E402

MAX_HORIZON = 24
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "ercot_load_weather.csv"
RESULTS_DIR = ROOT / "results"


def build_training_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per (origin hour, horizon 1-24) pair, so a single model
    learns the whole next-day shape instead of only a fixed 24h point."""
    df = df.sort_values("Time").reset_index(drop=True)
    epoch = df["Time"].min()

    frames = []
    for h in range(1, MAX_HORIZON + 1):
        row = pd.DataFrame(index=df.index)
        row["target_time"] = df["Time"].shift(-h)
        row["target"] = df["load_mw"].shift(-h)
        row["horizon"] = h
        row["target_hour"] = row["target_time"].dt.hour
        row["target_dow"] = row["target_time"].dt.dayofweek
        row["target_month"] = row["target_time"].dt.month
        row["target_is_weekend"] = row["target_dow"].isin([5, 6]).astype(int)
        for col in ZONE_TEMP_COLS:
            row[f"target_{col}"] = df[col].shift(-h)
        row["load_now"] = df["load_mw"]
        row["load_week_before_target"] = df["load_mw"].shift(168 - h)
        frames.append(row)

    out = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    fourier_cols = add_fourier_terms(out, "target_time", epoch)
    return out, fourier_cols


def feature_columns(fourier_cols: list[str]) -> list[str]:
    return (
        ["horizon", "target_hour", "target_dow", "target_is_weekend", "target_month",
         "load_now", "load_week_before_target"]
        + [f"target_{c}" for c in ZONE_TEMP_COLS] + fourier_cols
    )


def train_model(df: pd.DataFrame) -> tuple[XGBRegressor, list[str], pd.Timestamp]:
    data, fourier_cols = build_training_rows(df)
    cols = feature_columns(fourier_cols)
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(data[cols], data["target"])
    return model, cols, df["Time"].min()


def fetch_zone_forecasts() -> pd.DataFrame:
    """Real forward-looking weather forecast per zone, not reanalysis."""
    frames = {}
    for zone, (lat, lon) in ZONE_STATIONS.items():
        frames[zone_temp_col(zone)] = fetch_data.fetch_weather_forecast(lat, lon, days=2)
    return pd.DataFrame(frames)


def build_forecast_rows(df: pd.DataFrame, zone_forecast: pd.DataFrame, epoch: pd.Timestamp) -> pd.DataFrame:
    origin_time = df["Time"].max()
    load_now = df.loc[df["Time"] == origin_time, "load_mw"].iloc[0]
    load_by_time = df.set_index("Time")["load_mw"]

    rows = []
    for h in range(1, MAX_HORIZON + 1):
        target_time = origin_time + pd.Timedelta(hours=h)
        week_before = target_time - pd.Timedelta(hours=168)
        if target_time not in zone_forecast.index:
            raise ValueError(f"No weather forecast available for {target_time}; try a shorter MAX_HORIZON or refetch.")
        row = {
            "target_time": target_time,
            "horizon": h,
            "target_hour": target_time.hour,
            "target_dow": target_time.dayofweek,
            "target_month": target_time.month,
            "target_is_weekend": int(target_time.dayofweek in (5, 6)),
            "load_now": load_now,
            "load_week_before_target": load_by_time.get(week_before, load_now),
        }
        for col in ZONE_TEMP_COLS:
            row[f"target_{col}"] = zone_forecast.loc[target_time, col]
        rows.append(row)

    out = pd.DataFrame(rows)
    add_fourier_terms(out, "target_time", epoch)
    return out


def main() -> None:
    print("Refreshing load history through yesterday (needed so the forecast origin lines up with the weather forecast window)...")
    fetch_data.main()

    df = pd.read_csv(RAW_PATH, parse_dates=["Time"])
    print(f"Training on {len(df):,} hourly rows ({df['Time'].min().date()} to {df['Time'].max().date()})...")
    model, cols, epoch = train_model(df)

    print("Fetching live weather forecast for all 8 ERCOT zones...")
    zone_forecast = fetch_zone_forecasts()

    forecast_rows = build_forecast_rows(df, zone_forecast, epoch)
    forecast_rows["predicted_load_mw"] = model.predict(forecast_rows[cols])

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "next_24h_forecast.csv"
    forecast_rows[["target_time", "predicted_load_mw"]].to_csv(out_path, index=False)

    peak = forecast_rows.loc[forecast_rows["predicted_load_mw"].idxmax()]
    trough = forecast_rows.loc[forecast_rows["predicted_load_mw"].idxmin()]
    origin_time = df["Time"].max()
    print(f"\nForecast made at {origin_time} (most recent known load: {df['load_mw'].iloc[-1]:,.0f} MW)")
    print(f"Next 24h peak:   {peak['predicted_load_mw']:,.0f} MW at {peak['target_time']}")
    print(f"Next 24h trough: {trough['predicted_load_mw']:,.0f} MW at {trough['target_time']}")
    print(f"Saved hourly forecast to {out_path}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(forecast_rows["target_time"], forecast_rows["predicted_load_mw"], marker="o", markersize=3)
    ax.set_ylabel("Predicted system load (MW)")
    ax.set_title(f"ERCOT 24-hour load forecast, made {origin_time}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "next_24h_forecast.png", dpi=150)
    print(f"Saved plot to {RESULTS_DIR / 'next_24h_forecast.png'}")


if __name__ == "__main__":
    main()
