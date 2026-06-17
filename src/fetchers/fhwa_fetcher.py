"""
FHWA Highway Statistics urbanized-area VMT fetcher.

Produces, per city, one real exposure quantity:
    annual_vmt_millions      total annual vehicle-miles traveled (HDV + all
                             on-road traffic) in the city's Census urbanized
                             area, in millions of miles.

Data source: FHWA Highway Statistics Series, table HM-72 ("Functional system
travel -- annual vehicle-miles -- urbanized areas"), published annually as an
Excel workbook. The on-disk layout is:

    https://www.fhwa.dot.gov/policyinformation/statistics/{year}/xls/hm72.xlsx

FHWA reorganizes the static site and reshuffles the header rows from vintage to
vintage; the parser below is deliberately tolerant (it scans for the row that
contains an urbanized-area label and a numeric total rather than assuming fixed
row/column indices). If a future vintage breaks parsing, point FHWA_URL_TEMPLATE
at the new file and adjust _locate_columns.

Network access: required (FHWA static site) on a cold cache. The downloaded
workbook is cached to disk, so reruns are offline. As with the OSM/ACS/FARS
fetchers, this module is written to be run from a machine with outbound network
access; it is not exercised by the unit tests, which use a saved fixture.

NOTE on transport: the file is a binary .xlsx, not JSON, so this module uses a
plain streaming download (not the JSON CachedSession) and caches the workbook
bytes, mirroring the FARS fetcher's approach.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from .config import FetchConfig, URBANIZED_AREA_BY_CITY

FHWA_URL_TEMPLATE = (
    "https://www.fhwa.dot.gov/policyinformation/statistics/"
    "{year}/xls/hm72.xlsx"
)

# FHWA ships many Highway Statistics tables as legacy .xls (OLE2) rather than
# .xlsx. For 2022 the file is hm72.xls; try that first, then .xlsx. The .cfm
# page is an HTML landing page (not a workbook) and is skipped.
FHWA_URL_CANDIDATES = (
    "https://www.fhwa.dot.gov/policyinformation/statistics/{year}/xls/hm72.xls",
    "https://www.fhwa.dot.gov/policyinformation/statistics/{year}/xls/hm72.xlsx",
)

# OLE2 magic bytes (legacy .xls / Compound Document)
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"
# ZIP/OOXML magic bytes (.xlsx is a zip archive)
_XLSX_MAGIC = b"PK\x03\x04"


def _detect_format(content: bytes) -> str:
    """Return 'xls', 'xlsx', or raise if content looks like HTML/text."""
    if content[:4] == _XLS_MAGIC:
        return "xls"
    if content[:4] == _XLSX_MAGIC:
        return "xlsx"
    # Heuristic: HTML response returned with HTTP 200 (redirect, auth wall, etc.)
    preview = content[:120].lower()
    if b"<html" in preview or b"<!doc" in preview or b"host not" in preview:
        raise RuntimeError(
            "FHWA returned HTML instead of a workbook. The URL may have changed "
            "or the server returned a redirect page."
        )
    raise RuntimeError(
        f"Unrecognised workbook format. First 16 bytes: {content[:16].hex()!r}"
    )


def _download_workbook(year: int, cache_dir: Path, user_agent: str) -> bytes:
    """Download and cache the HM-72 workbook bytes for a given vintage."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Accept cache file with either extension (first run writes whichever matched).
    for ext in (".xls", ".xlsx"):
        cached = cache_dir / f"fhwa_hm72_{year}{ext}"
        if cached.exists():
            return cached.read_bytes()

    urls = [u.format(year=year) for u in FHWA_URL_CANDIDATES]
    last_err: Exception | None = None
    for url in urls:
        print(f"[fhwa] downloading {url}")
        try:
            resp = requests.get(
                url, headers={"User-Agent": user_agent}, timeout=300
            )
            resp.raise_for_status()
            fmt = _detect_format(resp.content)        # raises if HTML/unknown
            cached = cache_dir / f"fhwa_hm72_{year}.{fmt}"
            cached.write_bytes(resp.content)
            print(f"[fhwa] saved {cached.name} ({len(resp.content):,} bytes, format={fmt})")
            return resp.content
        except (requests.RequestException, RuntimeError) as exc:
            last_err = exc
            continue

    raise RuntimeError(
        f"Could not download FHWA HM-72 for {year}. Tried: {urls}. "
        f"Last error: {last_err}. If FHWA has reorganised the site, update "
        f"FHWA_URL_CANDIDATES in fhwa_fetcher.py to the current workbook URL."
    )


def _read_all_rows(workbook_bytes: bytes) -> pd.DataFrame:
    """Concatenate data rows from all urbanized-area sheets (A–G).

    HM-72 splits the urbanized-area table across multiple lettered sheets
    (A, B, C … G) sorted by population size, each with the same header layout:
      row 12: "URBANIZED  AREA  (1)"  |  "TOTAL ROADWAY MILES"  |  "TOTAL DVMT (2)"  …
      row 14+: area name               |  numeric                  |  numeric           …

    Sheet 0 is a Crystal Reports artefact ('CRYSTAL_PERSIST'), and the last
    sheet is footnotes; both are skipped. We concatenate rows 14 onward from
    every sheet whose name is a single letter A–G, keeping only the first
    six columns (sufficient to hold name + DVMT + population).
    """
    bio = io.BytesIO(workbook_bytes)
    fmt = _detect_format(workbook_bytes)
    engine = "xlrd" if fmt == "xls" else "openpyxl"

    xl = pd.ExcelFile(bio, engine=engine)
    frames = []
    for sheet in xl.sheet_names:
        if not (len(sheet) == 1 and sheet.isalpha()):
            continue                        # skip CRYSTAL_PERSIST, footnotes
        df = xl.parse(sheet, header=None, dtype=object)
        if len(df) <= 14:
            continue
        data = df.iloc[14:, :6].copy()      # skip banner/header rows 0-13
        data.columns = range(6)
        frames.append(data)

    if not frames:
        raise RuntimeError(
            "HM-72 parse: found no lettered data sheets (A–G). "
            "The workbook structure may have changed."
        )
    return pd.concat(frames, ignore_index=True)


def _locate_columns(raw: pd.DataFrame) -> tuple[int, int]:
    """Find (area_name_column, dvmt_column).

    In HM-72 the layout is fixed across all sheets:
      col 0 – urbanized area name
      col 2 – total DVMT in thousands of daily vehicle-miles

    We verify this by checking that col 0 contains our expected area names and
    that col 2 is predominantly numeric. If the structure has changed we fall
    back to the content-scan approach used in the original version.
    """
    candidates = [s for subs in URBANIZED_AREA_BY_CITY.values() for s in subs]

    def looks_like_area(val: object) -> bool:
        if not isinstance(val, str):
            return False
        low = val.lower()
        return any(sub in low for sub in candidates)

    # Fast path: verify col 0 = names, col 2 = DVMT
    if raw[0].map(looks_like_area).sum() >= 3:
        data_rows = raw[raw[0].map(looks_like_area)].index
        dvmt_numeric = pd.to_numeric(raw.loc[data_rows, 2], errors="coerce").notna().sum()
        if dvmt_numeric >= 3:
            return 0, 2

    # Slow-path fallback: scan all columns as before
    name_hits = {col: int(raw[col].map(looks_like_area).sum()) for col in raw.columns}
    area_col = max(name_hits, key=name_hits.get)
    if name_hits[area_col] == 0:
        raise RuntimeError(
            "FHWA parse: no column contained any expected urbanized-area name. "
            "The workbook layout may have changed; inspect the cached file and "
            "update URBANIZED_AREA_BY_CITY or _locate_columns."
        )
    data_rows = raw[raw[area_col].map(looks_like_area)].index
    numeric_score = {}
    for col in raw.columns:
        if col <= area_col:
            continue
        vals = pd.to_numeric(raw.loc[data_rows, col], errors="coerce")
        numeric_score[col] = int(vals.notna().sum())
    if not numeric_score or max(numeric_score.values()) == 0:
        raise RuntimeError(
            "FHWA parse: found area names but no numeric VMT column to their right."
        )
    threshold = max(1, len(data_rows) // 2)
    eligible = [c for c, n in numeric_score.items() if n >= threshold]
    total_col = max(eligible) if eligible else max(numeric_score, key=numeric_score.get)
    return area_col, total_col


def _dvmt_to_annual_millions(value: float) -> float:
    """Convert HM-72 DVMT column to annual vehicle-miles in millions.

    The HM-72 column header reads 'TOTAL DVMT (2)  (1,000)' meaning the stored
    value is thousands of *daily* vehicle-miles of travel (DVMT).

    Annual VMT (millions) = DVMT_thousands × 1,000 miles/day × 365 days/yr
                            ÷ 1,000,000 to convert to millions
                          = DVMT_thousands × 0.365
    """
    return float(value) * 0.365


def fetch_fhwa_vmt(cfg: FetchConfig) -> pd.DataFrame:
    """Return per-city annual VMT (millions of miles) from FHWA HM-72.

    Output columns: city, urbanized_area, annual_vmt_millions, fhwa_year.
    One row per city in cfg.cities; cities with no matching urbanized area are
    returned with NaN VMT and a warning (the assembler then falls back to the
    synthetic exposure scale for that city).
    """
    cache_dir = Path(cfg.cache_dir)
    workbook = _download_workbook(cfg.fhwa_year, cache_dir, cfg.user_agent)
    raw = _read_all_rows(workbook)
    area_col, total_col = _locate_columns(raw)

    # Build a lookup from each data row's lowercased area string to its VMT.
    rows = []
    for idx in raw.index:
        name = raw.at[idx, area_col]
        if not isinstance(name, str):
            continue
        vmt_raw = pd.to_numeric(pd.Series([raw.at[idx, total_col]]), errors="coerce").iloc[0]
        if pd.isna(vmt_raw):
            continue
        rows.append((name.strip(), name.strip().lower(), float(vmt_raw)))
    area_table = pd.DataFrame(rows, columns=["area_name", "area_low", "vmt_raw"])

    out = []
    for city in cfg.cities:
        subs = URBANIZED_AREA_BY_CITY.get(city.name, ())
        match = None
        for sub in subs:  # most specific first
            hit = area_table[area_table["area_low"].str.contains(sub, regex=False)]
            if len(hit):
                match = hit.iloc[0]
                break
        if match is None:
            print(f"[fhwa] WARNING: no urbanized-area match for {city.name}; "
                  f"VMT will be NaN and the assembler will fall back.")
            out.append((city.name, None, float("nan"), cfg.fhwa_year))
            continue
        vmt_millions = _dvmt_to_annual_millions(match["vmt_raw"])
        # sanity: a US urbanized area annual VMT is ~1,000–300,000 million miles.
        if not (500 <= vmt_millions <= 500_000):
            print(f"[fhwa] WARNING: {city.name} annual VMT {vmt_millions:.0f}M miles is "
                  f"outside the expected range; check HM-72 DVMT units in the cached file.")
        out.append((city.name, match["area_name"], vmt_millions, cfg.fhwa_year))

    df = pd.DataFrame(
        out, columns=["city", "urbanized_area", "annual_vmt_millions", "fhwa_year"]
    )
    matched = int(df["annual_vmt_millions"].notna().sum())
    print(f"[fhwa] matched {matched}/{len(df)} cities to HM-72 urbanized areas "
          f"(vintage {cfg.fhwa_year}).")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch FHWA urbanized-area VMT.")
    ap.add_argument("--year", type=int, default=None,
                    help="FHWA Highway Statistics vintage (default: config 2022).")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default="data/city_vmt.csv")
    args = ap.parse_args()

    config = FetchConfig(
        cache_dir=args.cache_dir,
        fhwa_year=args.year if args.year is not None else 2022,
    )
    table = fetch_fhwa_vmt(config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"[fhwa] wrote {args.out}")
