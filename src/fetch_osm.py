"""Fetch tract-level amenity counts from OpenStreetMap using OSMnx."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely import wkt
from tqdm import tqdm

from config_loader import load_config

INPUT_PATH = Path("data/census_tracts.csv")
OUTPUT_PATH = Path("data/tracts_with_amenities.csv")


def safe_features_from_polygon(polygon, tags: Dict[str, object], attempts: int = 3) -> gpd.GeoDataFrame:
    """Fetch OSM features for a polygon with retries and exponential backoff."""
    for attempt in range(attempts):
        try:
            features = ox.features_from_polygon(polygon, tags=tags)
            return features.reset_index()
        except Exception:
            if attempt == attempts - 1:
                return gpd.GeoDataFrame()
            time.sleep(2**attempt)
    return gpd.GeoDataFrame()


def count_amenities(features: gpd.GeoDataFrame) -> Dict[str, int]:
    """Count target amenity categories from OSM features."""
    if features.empty:
        return {
            "restaurants_cafes": 0,
            "bars_nightclubs": 0,
            "liquor_stores": 0,
            "check_cashing_payday": 0,
            "parks_green_spaces": 0,
            "grocery_stores": 0,
            "pharmacies": 0,
            "schools": 0,
        }

    amenity_col = features.get("amenity", pd.Series(dtype=object)).fillna("")
    shop_col = features.get("shop", pd.Series(dtype=object)).fillna("")
    leisure_col = features.get("leisure", pd.Series(dtype=object)).fillna("")

    return {
        "restaurants_cafes": int(amenity_col.isin(["restaurant", "cafe"]).sum()),
        "bars_nightclubs": int(amenity_col.isin(["bar", "nightclub"]).sum()),
        "liquor_stores": int(shop_col.isin(["alcohol"]).sum()),
        "check_cashing_payday": int(shop_col.isin(["money_lender"]).sum()),
        "parks_green_spaces": int(leisure_col.isin(["park"]).sum()),
        "grocery_stores": int(shop_col.isin(["supermarket", "convenience"]).sum()),
        "pharmacies": int(amenity_col.isin(["pharmacy"]).sum()),
        "schools": int(amenity_col.isin(["school"]).sum()),
    }


def amenity_bucket(row: pd.Series, community_thr: int, underserved_thr: int) -> str:
    """Assign amenity bucket based on a simple positive-minus-negative scoring rule."""
    score = (
        row["parks_green_spaces"]
        + row["restaurants_cafes"]
        + row["schools"]
        + row["grocery_stores"]
        - row["liquor_stores"]
        - row["check_cashing_payday"]
    )
    if score >= community_thr:
        return "community_rich"
    if score <= underserved_thr:
        return "financially_underserved"
    return "commercial_mixed"


def load_or_fetch_counts(tract_fips: str, polygon, cache_dir: Path) -> Dict[str, int]:
    """Load cached amenity counts if present, otherwise query OSM and cache the result."""
    cache_path = cache_dir / f"{tract_fips}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    tags = {
        "amenity": ["restaurant", "cafe", "bar", "nightclub", "pharmacy", "school"],
        "shop": ["alcohol", "money_lender", "supermarket", "convenience"],
        "leisure": ["park"],
    }
    features = safe_features_from_polygon(polygon, tags)
    counts = count_amenities(features)
    cache_path.write_text(json.dumps(counts), encoding="utf-8")
    return counts


def main() -> None:
    """Fetch OSM amenities per tract, merge with census data, and persist merged output."""
    config = load_config()
    cache_dir = Path(config.get("osm_cache_dir", "data/osm_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    thresholds = config.get("amenity_score_threshold", {})
    community_thr = int(thresholds.get("community_rich", 3))
    underserved_thr = int(thresholds.get("financially_underserved", -1))

    census_df = pd.read_csv(INPUT_PATH)
    census_df["geometry"] = census_df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(census_df, geometry="geometry", crs="EPSG:4326")

    amenity_rows = []
    for row in tqdm(gdf.itertuples(index=False), total=len(gdf), desc="Fetching OSM amenities"):
        counts = load_or_fetch_counts(row.tract_fips, row.geometry, cache_dir)
        counts["tract_fips"] = row.tract_fips
        amenity_rows.append(counts)

    amenity_df = pd.DataFrame(amenity_rows)
    merged = census_df.merge(amenity_df, on="tract_fips", how="left")
    merged["amenity_bucket"] = merged.apply(
        amenity_bucket,
        axis=1,
        community_thr=community_thr,
        underserved_thr=underserved_thr,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved tract + amenity dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
