"""Pull recent ERCOT system load and matching Austin weather, save to data/raw/.

ERCOT's public document feed only retains ~30 days of history for the
"Actual System Load by Weather Zone" report, so this script always pulls
the trailing LOOKBACK_DAYS window. Re-run periodically and append if you
want a longer accumulated history.
"""

import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import gridstatus  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

LOOKBACK_DAYS = 28
AUSTIN_LAT, AUSTIN_LON = 30.27, -97.74
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_ercot_load(start: date, end: date) -> pd.DataFrame:
    iso = gridstatus.Ercot()
    df = iso.get_load_by_weather_zone(start.isoformat(), end=end.isoformat())
    df = df[["Time", "System Total"]].rename(columns={"System Total": "load_mw"})
    df["Time"] = pd.to_datetime(df["Time"]).dt.tz_convert("US/Central").dt.tz_localize(None)
    return df


def fetch_weather(start: date, end: date) -> pd.DataFrame:
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": AUSTIN_LAT,
            "longitude": AUSTIN_LON,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m",
            "timezone": "America/Chicago",
        },
        timeout=30,
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return pd.DataFrame({"Time": pd.to_datetime(hourly["time"]), "temp_c": hourly["temperature_2m"]})


def main() -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS)

    print(f"Fetching ERCOT load {start} to {end}...")
    load_df = fetch_ercot_load(start, end)

    print(f"Fetching Austin weather {start} to {end}...")
    weather_df = fetch_weather(start, end)

    merged = pd.merge(load_df, weather_df, on="Time", how="inner").sort_values("Time")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "ercot_load_weather.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
