
from __future__ import annotations

# A minimal Overpass `out geom` response: two ways sharing a node (an
# intersection) plus one traffic-signal node, all near the SF Ferry Building.
MOCK_OVERPASS = {
    "elements": [
        {
            "type": "way",
            "id": 1001,
            "tags": {"highway": "primary"},   # -> arterial
            "nodes": [1, 2, 3],
            "geometry": [
                {"lat": 37.7955, "lon": -122.3937},
                {"lat": 37.7960, "lon": -122.3940},
                {"lat": 37.7965, "lon": -122.3943},
            ],
        },
        {
            "type": "way",
            "id": 1002,
            "tags": {"highway": "residential"},  # -> local
            "nodes": [2, 4, 5],                   # shares node 2 with way 1001
            "geometry": [
                {"lat": 37.7960, "lon": -122.3940},
                {"lat": 37.7962, "lon": -122.3935},
                {"lat": 37.7964, "lon": -122.3930},
            ],
        },
        {
            "type": "node",
            "id": 99,
            "lat": 37.7960,
            "lon": -122.3940,
            "tags": {"highway": "traffic_signals"},
        },
    ]
}

# A minimal ACS API response (list-of-lists, header first) for one tract.
MOCK_ACS_ROWS = [
    ["NAME", "B01003_001E", "B08301_001E", "B08301_019E",
     "B25024_001E", "B25024_002E", "B25024_003E",
     "state", "county", "tract"],
    ["Census Tract 1; San Francisco County; California",
     "4200", "2500", "375", "1800", "900", "300",
     "06", "075", "010100"],
]

# A minimal TIGERweb GeoJSON response: one tract polygon covering the ways.
MOCK_TIGER_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"GEOID": "06075010100"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-122.400, 37.790],
                    [-122.385, 37.790],
                    [-122.385, 37.800],
                    [-122.400, 37.800],
                    [-122.400, 37.790],
                ]],
            },
        }
    ],
}

# A minimal FARS accident frame: three crashes, two inside the SF bbox and one
# with a sentinel (bad) coordinate that must be dropped.
MOCK_FARS_RECORDS = [
    {"lat": 37.7960, "lon": -122.3940},   # in bbox
    {"lat": 37.7962, "lon": -122.3935},   # in bbox
    {"lat": 88.8888, "lon": -122.3900},   # sentinel -> dropped
]
