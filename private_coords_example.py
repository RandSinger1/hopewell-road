"""
private_coords_example.py -- tracked stand-in for private_coords.py.

Only town-centre coordinates (2 decimals, ~1 km) belong here: never exact
earthwork or site locations.  Copy to private_coords.py (gitignored) and fill
in the exact values locally.
"""

CIRCLEVILLE_COORD: tuple = (39.60, -82.95)   # Circleville town centre
NEWARK_COORD: tuple = (40.06, -82.40)        # Newark town centre
CHILLICOTHE_COORD: tuple = (39.33, -82.98)   # Chillicothe town centre
HIGH_BANKS_COORD: tuple = (39.33, -82.98)    # stand-in: Chillicothe town centre

PARABOLA_C: float = CIRCLEVILLE_COORD[0]
PARABOLA_X_OFFSET: float = CIRCLEVILLE_COORD[1]

SCAN_POLYGON: list = [CIRCLEVILLE_COORD, NEWARK_COORD, CHILLICOTHE_COORD]

REPORT_COORDS: list = [
    # {"name": "Site name", "lat": <lat>, "lon": <lon>, "size_m": 200,
    #  "note": "...", "mode": "all", "true_positive_pair_index": 0},
]
