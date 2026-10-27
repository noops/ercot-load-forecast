"""ERCOT's 8 official weather zones, with a representative city per zone.
Shared between fetch_data.py (historical) and forecast.py (live forecast)."""

ZONE_STATIONS = {
    "Coast": (29.76, -95.37),          # Houston
    "East": (32.35, -95.30),           # Tyler
    "Far West": (31.997, -102.077),    # Midland
    "North": (33.91, -98.49),          # Wichita Falls
    "North Central": (32.78, -96.80),  # Dallas
    "South Central": (30.27, -97.74),  # Austin
    "Southern": (27.80, -97.40),       # Corpus Christi (ERCOT's "South" zone)
    "West": (31.46, -100.44),          # San Angelo
}


def zone_temp_col(zone: str) -> str:
    return f"temp_c_{zone.lower().replace(' ', '_')}"


ZONE_TEMP_COLS = [zone_temp_col(z) for z in ZONE_STATIONS]
