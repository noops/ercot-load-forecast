"""Pull ~3 years of ERCOT system load and matching Austin weather.

Combines two ERCOT sources to get continuous history instead of the ~30
days available from the live feed alone:

  1. Static annual "Native Load" archives (ercot.com/gridinfo/load/load_hist)
     - full years, posted with a settlement lag, no auth required.
  2. The live "Actual System Load by Weather Zone" document feed (via
     gridstatus) for the trailing window not yet covered by an archive.

NOTE: the archive URLs below are current as of Oct 2026. ERCOT doesn't
expose a stable index of these files, so the URL for the in-progress
year will need updating periodically (the "-1-" suffix the current-year
file uses is a hard-coded versioning quirk, not a predictable pattern).
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

ARCHIVE_URLS = [
    "https://www.ercot.com/files/docs/2024/02/06/Native_Load_2024.zip",
    "https://www.ercot.com/files/docs/2025/02/11/Native_Load_2025.zip",
    "https://www.ercot.com/files/docs/2026/02/10/Native_Load_2026-1-.zip",
]

AUSTIN_LAT, AUSTIN_LON = 30.27, -97.74
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "_archive_cache"


def fetch_archive_year(url: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = CACHE_DIR / Path(url).name
    if not fname.exists():
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=60)
        resp.raise_for_status()
        fname.write_bytes(resp.content)

    xlsx_path = fname.with_suffix(".xlsx")
    if not xlsx_path.exists():
        import zipfile

        with zipfile.ZipFile(fname) as zf:
            xlsx_name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
            zf.extract(xlsx_name, CACHE_DIR)
            (CACHE_DIR / xlsx_name).rename(xlsx_path)

    df = pd.read_excel(xlsx_path)
    date_part = df["Hour Ending"].str.slice(0, 10)
    hour_part = df["Hour Ending"].str.slice(11, 13).astype(int)
    # Hour Ending HH means the hour that just ended at HH:00; the hour
    # *started* at HH-1:00 on the same calendar date (24:00 -> 23:00 start,
    # still same date under this convention).
    base_date = pd.to_datetime(date_part, format="%m/%d/%Y")
    df["Time"] = base_date + pd.to_timedelta(hour_part - 1, unit="h")
    return df[["Time", "ERCOT"]].rename(columns={"ERCOT": "load_mw"})


def fetch_live_load(start: date, end: date) -> pd.DataFrame:
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
        timeout=60,
    )
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    return pd.DataFrame({"Time": pd.to_datetime(hourly["time"]), "temp_c": hourly["temperature_2m"]})


def main() -> None:
    print("Fetching static archives (2024-2026)...")
    archive_frames = [fetch_archive_year(url) for url in ARCHIVE_URLS]
    archive_df = pd.concat(archive_frames).drop_duplicates("Time").sort_values("Time")
    archive_end = archive_df["Time"].max().date()

    live_start = archive_end + timedelta(days=1)
    live_end = date.today() - timedelta(days=1)
    if live_start <= live_end:
        print(f"Fetching live feed {live_start} to {live_end} (post-archive gap)...")
        live_df = fetch_live_load(live_start, live_end)
        load_df = pd.concat([archive_df, live_df]).drop_duplicates("Time").sort_values("Time")
    else:
        load_df = archive_df

    overall_start = load_df["Time"].min().date()
    overall_end = load_df["Time"].max().date()
    print(f"Combined load history: {overall_start} to {overall_end} ({len(load_df)} hourly rows)")

    print("Fetching Austin weather for the full span...")
    weather_df = fetch_weather(overall_start, overall_end)

    merged = pd.merge(load_df, weather_df, on="Time", how="inner").sort_values("Time")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "ercot_load_weather.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
