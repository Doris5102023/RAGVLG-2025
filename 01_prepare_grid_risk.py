"""Prepare 1 km industrial-risk grids for Shenzhen and Hong Kong LISA analysis.

The source GeoJSON files contain geometries required for spatial analysis.  Some
bounding boxes were exported multiple times with different cluster labels, while
their continuous risk attributes remain identical.  This script audits that
condition, keeps one row per detected bounding box, and aggregates Priority to
regular grids using intersection-area weights.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


KEY_COLUMNS = ["original_patch", "bbox_index"]
WEIGHTED_COLUMNS = ["Priority", "LST_indus", "LST_excess_local", "NDVI_gap_local"]
CITY_SOURCES = {
    "Hong Kong": {
        "slug": "hong_kong",
        "geojson": "priority_based_risk_clusters_HongKong.geojson",
        "csv": "priority_based_risk_clusters_HongKong.csv",
    },
    "Shenzhen": {
        "slug": "shenzhen",
        "geojson": "priority_based_risk_clusters_Shenzhen.geojson",
        "csv": "priority_based_risk_clusters_ShenZhen.csv",
    },
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "lisa_outputs")
    parser.add_argument("--grid-size", type=float, default=1000.0)
    parser.add_argument("--target-crs", default="EPSG:32650")
    return parser.parse_args()


def key_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return set(map(tuple, frame[KEY_COLUMNS].to_numpy()))


def read_and_audit_city(
    city: str, config: dict[str, str], input_dir: Path, target_crs: str
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    source = gpd.read_file(input_dir / config["geojson"])
    table = pd.read_csv(input_dir / config["csv"])
    if source.crs is None:
        raise ValueError(f"{city}: GeoJSON has no CRS.")

    required = KEY_COLUMNS + WEIGHTED_COLUMNS + ["flag_low_quality", "geometry"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{city}: GeoJSON is missing columns: {missing}")

    grouped = source.groupby(KEY_COLUMNS, dropna=False)
    changing_columns = [
        column
        for column in WEIGHTED_COLUMNS + ["bbox_area", "flag_low_quality"]
        if column in source.columns and (grouped[column].nunique(dropna=False) > 1).any()
    ]
    if changing_columns:
        raise ValueError(
            f"{city}: duplicate bounding-box keys contain differing risk data: "
            f"{changing_columns}"
        )

    unique_source = source.drop_duplicates(KEY_COLUMNS).copy()
    geo_keys = key_set(unique_source)
    csv_keys = key_set(table)
    matched = unique_source.merge(
        table[KEY_COLUMNS + WEIGHTED_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_geo", "_csv"),
    )
    disagreement = 0
    for column in WEIGHTED_COLUMNS:
        disagreement += int(
            (
                (matched[f"{column}_geo"] - matched[f"{column}_csv"]).abs() > 1e-8
            )
            .fillna(False)
            .sum()
        )
    if disagreement:
        raise ValueError(f"{city}: matched CSV and GeoJSON risk attributes disagree.")

    invalid_before = int((~unique_source.geometry.is_valid).sum())
    if invalid_before:
        unique_source["geometry"] = unique_source.geometry.make_valid()
    unique_source = unique_source[~unique_source.geometry.is_empty].copy()
    unique_source["city"] = city
    unique_source["parcel_id"] = [
        f"{config['slug']}_{idx:05d}" for idx in range(len(unique_source))
    ]
    projected = unique_source.to_crs(target_crs)

    audit = {
        "city": city,
        "geojson_rows": len(source),
        "geojson_unique_bbox": len(unique_source),
        "geojson_duplicate_rows_removed": len(source) - len(unique_source),
        "csv_rows": len(table),
        "csv_unique_bbox": len(csv_keys),
        "csv_records_without_geometry": len(csv_keys - geo_keys),
        "spatial_records_without_csv": len(geo_keys - csv_keys),
        "invalid_geometries_repaired": invalid_before,
        "source_crs": str(source.crs),
        "analysis_crs": target_crs,
        "low_quality_bbox": int(projected["flag_low_quality"].astype(bool).sum()),
        "high_quality_bbox": int((~projected["flag_low_quality"].astype(bool)).sum()),
    }
    return projected, audit


def build_grid(
    parcels: gpd.GeoDataFrame,
    city: str,
    slug: str,
    scenario: str,
    grid_size: float,
    target_crs: str,
) -> gpd.GeoDataFrame:
    if parcels.empty:
        raise ValueError(f"{city} / {scenario}: no bounding boxes to aggregate.")

    xmin, ymin, xmax, ymax = parcels.total_bounds
    x0 = math.floor(xmin / grid_size) * grid_size
    y0 = math.floor(ymin / grid_size) * grid_size
    x1 = math.ceil(xmax / grid_size) * grid_size
    y1 = math.ceil(ymax / grid_size) * grid_size

    grid_rows: list[dict[str, object]] = []
    for col, x in enumerate(np.arange(x0, x1, grid_size)):
        for row, y in enumerate(np.arange(y0, y1, grid_size)):
            grid_rows.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "grid_col": col,
                    "grid_row": row,
                    "grid_id": f"{slug}_{scenario}_{col:03d}_{row:03d}",
                    "geometry": box(x, y, x + grid_size, y + grid_size),
                }
            )
    grid = gpd.GeoDataFrame(grid_rows, crs=target_crs)
    piece_columns = ["parcel_id", "flag_low_quality", *WEIGHTED_COLUMNS, "geometry"]
    pieces = gpd.overlay(
        parcels[piece_columns],
        grid,
        how="intersection",
        keep_geom_type=False,
    )
    pieces["intersection_area_m2"] = pieces.geometry.area
    pieces = pieces[pieces["intersection_area_m2"] > 0].copy()
    for column in WEIGHTED_COLUMNS:
        pieces[f"{column}_area_product"] = (
            pieces[column] * pieces["intersection_area_m2"]
        )
    pieces["low_quality_piece"] = pieces["flag_low_quality"].astype(bool).astype(int)

    aggregations: dict[str, tuple[str, str]] = {
        "n_parcels": ("parcel_id", "nunique"),
        "industrial_bbox_area_m2": ("intersection_area_m2", "sum"),
        "n_low_quality_pieces": ("low_quality_piece", "sum"),
    }
    for column in WEIGHTED_COLUMNS:
        aggregations[f"{column}_area_sum"] = (f"{column}_area_product", "sum")
    grouped = (
        pieces.groupby(["city", "scenario", "grid_col", "grid_row", "grid_id"])
        .agg(**aggregations)
        .reset_index()
    )
    occupied = grid.merge(
        grouped,
        on=["city", "scenario", "grid_col", "grid_row", "grid_id"],
        how="inner",
    )
    for column in WEIGHTED_COLUMNS:
        occupied[f"{column}_aw"] = (
            occupied[f"{column}_area_sum"] / occupied["industrial_bbox_area_m2"]
        )
        occupied = occupied.drop(columns=f"{column}_area_sum")
    occupied["risk_aw"] = occupied["Priority_aw"]
    occupied["grid_area_m2"] = grid_size**2
    occupied["industrial_bbox_cover_ratio"] = (
        occupied["industrial_bbox_area_m2"] / occupied["grid_area_m2"]
    )
    return occupied


def summarize_grids(grids: list[gpd.GeoDataFrame]) -> pd.DataFrame:
    rows = []
    for grid in grids:
        city = str(grid["city"].iloc[0])
        scenario = str(grid["scenario"].iloc[0])
        rows.append(
            {
                "city": city,
                "scenario": scenario,
                "occupied_grid_cells": len(grid),
                "risk_aw_min": grid["risk_aw"].min(),
                "risk_aw_mean": grid["risk_aw"].mean(),
                "risk_aw_median": grid["risk_aw"].median(),
                "risk_aw_max": grid["risk_aw"].max(),
                "bbox_intersection_area_km2": grid["industrial_bbox_area_m2"].sum()
                / 1_000_000,
                "parcels_per_cell_mean": grid["n_parcels"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parcels_by_city: dict[str, gpd.GeoDataFrame] = {}
    audits: list[dict[str, object]] = []
    for city, config in CITY_SOURCES.items():
        parcels, audit = read_and_audit_city(
            city, config, input_dir=input_dir, target_crs=args.target_crs
        )
        parcels_by_city[city] = parcels
        audits.append(audit)

    parcels_all = gpd.GeoDataFrame(
        pd.concat(parcels_by_city.values(), ignore_index=True), crs=args.target_crs
    )
    parcels_all.to_file(output_dir / "industrial_parcels_deduplicated.geojson", driver="GeoJSON")

    grids_main: list[gpd.GeoDataFrame] = []
    grids_high_quality: list[gpd.GeoDataFrame] = []
    for city, config in CITY_SOURCES.items():
        parcels = parcels_by_city[city]
        grids_main.append(
            build_grid(
                parcels,
                city,
                config["slug"],
                "main",
                args.grid_size,
                args.target_crs,
            )
        )
        grids_high_quality.append(
            build_grid(
                parcels[~parcels["flag_low_quality"].astype(bool)].copy(),
                city,
                config["slug"],
                "high_quality",
                args.grid_size,
                args.target_crs,
            )
        )

    main_grid = gpd.GeoDataFrame(pd.concat(grids_main, ignore_index=True), crs=args.target_crs)
    hq_grid = gpd.GeoDataFrame(
        pd.concat(grids_high_quality, ignore_index=True), crs=args.target_crs
    )
    main_grid.to_file(output_dir / "grid_risk_1km_main.geojson", driver="GeoJSON")
    hq_grid.to_file(output_dir / "grid_risk_1km_high_quality.geojson", driver="GeoJSON")
    pd.DataFrame(audits).to_csv(output_dir / "data_audit_summary.csv", index=False)
    summarize_grids(grids_main + grids_high_quality).to_csv(
        output_dir / "grid_risk_summary.csv", index=False
    )

    print("Data audit and grid aggregation completed.")
    print(pd.DataFrame(audits).to_string(index=False))
    print(summarize_grids(grids_main + grids_high_quality).to_string(index=False))


if __name__ == "__main__":
    main()
