"""Run global and local Moran analyses for the industrial-risk grids.

The implementation uses row-standardized Queen weights and a fixed-seed,
random-label permutation test.  It intentionally relies only on NumPy, SciPy,
GeoPandas, Rasterio, and Matplotlib because libpysal/esda are not installed in
the available analysis environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.sparse import csr_matrix


PERMUTATIONS = 999
SEED = 20260525
RISK_COLUMN = "risk_aw"
CLUSTER_ORDER = [
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High",
    "Not significant",
    "Isolate",
]
COLORS = {
    "High-High": "#d7191c",
    "Low-Low": "#2c7bb6",
    "High-Low": "#fdae61",
    "Low-High": "#abd9e9",
    "Not significant": "#d9d9d9",
    "Isolate": "#ffffff",
}
IMAGES = {
    "Hong Kong": "Hong Kong_RemoteSensingImagery.tif",
    "Shenzhen": "Shenzhen_RemoteSensingImagery.tif",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=script_dir)
    parser.add_argument("--output-dir", type=Path, default=script_dir / "lisa_outputs")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def queen_weights(gdf: gpd.GeoDataFrame) -> tuple[csr_matrix, np.ndarray]:
    positions = {
        (int(row.grid_col), int(row.grid_row)): index
        for index, row in gdf.iterrows()
    }
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    degrees = np.zeros(len(gdf), dtype=int)
    for (col, row), index in positions.items():
        neighbors = [
            positions[(col + dcol, row + drow)]
            for dcol in (-1, 0, 1)
            for drow in (-1, 0, 1)
            if (dcol != 0 or drow != 0)
            and (col + dcol, row + drow) in positions
        ]
        degrees[index] = len(neighbors)
        if neighbors:
            rows.extend([index] * len(neighbors))
            columns.extend(neighbors)
            values.extend([1.0 / len(neighbors)] * len(neighbors))
    weights = csr_matrix((values, (rows, columns)), shape=(len(gdf), len(gdf)))
    return weights, degrees


def calculate_moran(
    gdf: gpd.GeoDataFrame,
    permutations: int,
    seed: int,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    result = gdf.reset_index(drop=True).copy()
    y = result[RISK_COLUMN].to_numpy(dtype=float)
    if len(y) < 3 or np.std(y) == 0:
        raise ValueError("LISA requires at least three grid cells with non-constant risk.")

    weights, degrees = queen_weights(result)
    z = (y - y.mean()) / y.std(ddof=0)
    lag_z = weights @ z
    scale = (len(z) - 1) / float(z @ z)
    local_i = scale * z * lag_z
    local_i[degrees == 0] = np.nan

    rng = np.random.default_rng(seed)
    randomized = np.stack([rng.permutation(z) for _ in range(permutations)])
    perm_lag = (weights @ randomized.T).T
    perm_local_i = scale * randomized * perm_lag
    p_local = (
        (np.abs(perm_local_i) >= np.abs(np.nan_to_num(local_i))).sum(axis=0) + 1
    ) / (permutations + 1)
    p_local[degrees == 0] = np.nan

    quadrants = np.full(len(z), "Undefined", dtype=object)
    quadrants[(z >= 0) & (lag_z >= 0)] = "High-High"
    quadrants[(z < 0) & (lag_z < 0)] = "Low-Low"
    quadrants[(z >= 0) & (lag_z < 0)] = "High-Low"
    quadrants[(z < 0) & (lag_z >= 0)] = "Low-High"
    significant = (p_local < 0.05) & (degrees > 0)
    clusters = np.full(len(z), "Not significant", dtype=object)
    clusters[significant] = quadrants[significant]
    clusters[degrees == 0] = "Isolate"

    result["neighbor_n"] = degrees
    result["risk_z"] = z
    result["lag_risk_z"] = lag_z
    result["local_I"] = local_i
    result["p_value"] = p_local
    result["lisa_quadrant"] = quadrants
    result["is_significant"] = significant
    result["lisa_cluster"] = clusters

    s0 = float(weights.sum())
    global_i = float(len(z) / s0 * ((z @ lag_z) / (z @ z))) if s0 else np.nan
    perm_global = (
        len(z)
        / s0
        * np.sum(randomized * perm_lag, axis=1)
        / np.sum(randomized * randomized, axis=1)
        if s0
        else np.full(permutations, np.nan)
    )
    global_p = (
        float((np.sum(np.abs(perm_global) >= abs(global_i)) + 1) / (permutations + 1))
        if s0
        else np.nan
    )
    global_summary = {
        "city": result["city"].iloc[0],
        "scenario": result["scenario"].iloc[0],
        "n_cells": len(result),
        "n_isolates": int((degrees == 0).sum()),
        "risk_aw_mean": float(y.mean()),
        "risk_aw_std": float(y.std(ddof=0)),
        "global_moran_I": global_i,
        "global_p_value_two_sided": global_p,
        "permutations": permutations,
        "seed": seed,
    }
    return result, global_summary


def summarize_local(result: gpd.GeoDataFrame) -> pd.DataFrame:
    counts = result["lisa_cluster"].value_counts().reindex(CLUSTER_ORDER, fill_value=0)
    return pd.DataFrame(
        {
            "city": result["city"].iloc[0],
            "scenario": result["scenario"].iloc[0],
            "cluster": CLUSTER_ORDER,
            "count": counts.to_numpy(),
            "percent_of_cells": counts.to_numpy() / len(result) * 100,
        }
    )


def read_background(
    image_path: Path, bounds: np.ndarray, analysis_crs: object
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    if not image_path.exists():
        return None
    with rasterio.open(image_path) as dataset:
        if str(dataset.crs) != str(analysis_crs):
            return None
        requested = from_bounds(*bounds, transform=dataset.transform)
        full = Window(0, 0, dataset.width, dataset.height)
        window = requested.intersection(full).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            return None
        scale = min(1.0, 1400.0 / max(window.width, window.height))
        out_height = max(1, int(window.height * scale))
        out_width = max(1, int(window.width * scale))
        rgb = dataset.read(
            [1, 2, 3],
            window=window,
            out_shape=(3, out_height, out_width),
            masked=True,
        )
        rgb = np.moveaxis(rgb.filled(0).astype(float), 0, -1)
        for band in range(3):
            values = rgb[:, :, band]
            valid = values[values > 0]
            if valid.size:
                low, high = np.percentile(valid, [2, 98])
                if high > low:
                    rgb[:, :, band] = np.clip((values - low) / (high - low), 0, 1)
        left, bottom, right, top = rasterio.windows.bounds(window, dataset.transform)
        extent = (left, right, bottom, top)
        return rgb, extent


def plot_cluster_map(
    result: gpd.GeoDataFrame, input_dir: Path, output_path: Path
) -> None:
    city = str(result["city"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 8))
    background = read_background(
        input_dir / IMAGES[city], result.total_bounds, result.crs
    )
    if background is not None:
        rgb, extent = background
        ax.imshow(rgb, extent=extent, origin="upper")
    for cluster in CLUSTER_ORDER:
        group = result[result["lisa_cluster"] == cluster]
        if not group.empty:
            group.plot(
                ax=ax,
                color=COLORS[cluster],
                alpha=0.78 if cluster not in ("Not significant", "Isolate") else 0.40,
                edgecolor="white",
                linewidth=0.18,
            )
    patches = [
        mpatches.Patch(color=COLORS[label], label=label) for label in CLUSTER_ORDER
    ]
    ax.legend(handles=patches, title="LISA cluster (p < 0.05)", loc="lower left")
    ax.set_title(f"Industrial Risk LISA Cluster Map: {city} (Main Analysis)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_moran_scatter(result: gpd.GeoDataFrame, output_path: Path) -> None:
    city = str(result["city"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cluster in CLUSTER_ORDER:
        group = result[result["lisa_cluster"] == cluster]
        if not group.empty:
            ax.scatter(
                group["risk_z"],
                group["lag_risk_z"],
                color=COLORS[cluster],
                label=cluster,
                s=18,
                alpha=0.75,
                edgecolors="none",
            )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Standardized area-weighted risk")
    ax.set_ylabel("Spatial lag of standardized risk")
    ax.set_title(f"Local Moran Scatterplot: {city} (Main Analysis)")
    ax.legend(title="LISA cluster", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        "main": output_dir / "grid_risk_1km_main.geojson",
        "high_quality": output_dir / "grid_risk_1km_high_quality.geojson",
    }
    result_frames: dict[str, list[gpd.GeoDataFrame]] = {scenario: [] for scenario in scenarios}
    local_summaries: list[pd.DataFrame] = []
    global_summaries: list[dict[str, object]] = []
    for scenario, grid_path in scenarios.items():
        grid = gpd.read_file(grid_path)
        for offset, city in enumerate(IMAGES):
            city_grid = grid[grid["city"] == city].copy().reset_index(drop=True)
            result, global_summary = calculate_moran(
                city_grid,
                permutations=args.permutations,
                seed=args.seed + offset + (10 if scenario == "high_quality" else 0),
            )
            result_frames[scenario].append(result)
            local_summaries.append(summarize_local(result))
            global_summaries.append(global_summary)
            if scenario == "main":
                slug = city.lower().replace(" ", "_")
                plot_cluster_map(
                    result, input_dir, output_dir / f"lisa_cluster_map_{slug}.png"
                )
                plot_moran_scatter(
                    result, output_dir / f"lisa_moran_scatter_{slug}.png"
                )

    for scenario, frames in result_frames.items():
        combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        combined.to_file(output_dir / f"lisa_grid_{scenario}.geojson", driver="GeoJSON")
    local_summary = pd.concat(local_summaries, ignore_index=True)
    global_summary = pd.DataFrame(global_summaries)
    local_summary.to_csv(output_dir / "lisa_cluster_summary.csv", index=False)
    global_summary.to_csv(output_dir / "lisa_global_moran_summary.csv", index=False)
    print("LISA analysis completed.")
    print(global_summary.to_string(index=False))
    print(local_summary.to_string(index=False))


if __name__ == "__main__":
    main()
