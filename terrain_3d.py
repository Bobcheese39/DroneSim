"""
3D Satellite Terrain Visualization

Downloads elevation data (AWS Terrain Tiles) and satellite imagery (ESRI World Imagery)
for a given lat/lon + radius, renders an interactive 3D terrain in Plotly, and optionally
overlays a drone/RC plane GPS track from a CSV file.
"""

import argparse
import io
import math
import os
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Tile math helpers (Web Mercator / slippy-map conventions)
# ---------------------------------------------------------------------------

TILE_SIZE = 256
EARTH_CIRCUMFERENCE_KM = 40075.016686


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon (WGS-84) to slippy-map tile x, y at a given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_bounds(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) for a tile."""
    n = 2 ** zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lat_min, lon_min, lat_max, lon_max


def choose_zoom(radius_km: float, resolution: int) -> int:
    """Pick a zoom level where we get roughly *resolution* pixels across the area."""
    diameter_km = radius_km * 2
    for z in range(1, 16):
        meters_per_pixel = EARTH_CIRCUMFERENCE_KM * 1000 / (TILE_SIZE * 2 ** z)
        pixels_across = diameter_km * 1000 / meters_per_pixel
        if pixels_across >= resolution:
            return z
    return 15


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def bounding_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) for a circle."""
    d_lat = radius_km / 110.540
    d_lon = radius_km / (111.320 * math.cos(math.radians(lat)))
    return lat - d_lat, lon - d_lon, lat + d_lat, lon + d_lon


# ---------------------------------------------------------------------------
# Tile downloading
# ---------------------------------------------------------------------------

SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TERRAIN_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)

_session = requests.Session()
_session.headers.update({"User-Agent": "terrain_3d/1.0"})


def _download_tile(url: str, retries: int = 3) -> Image.Image:
    for attempt in range(retries):
        try:
            resp = _session.get(url, timeout=15)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        except Exception:
            if attempt == retries - 1:
                raise
    raise RuntimeError(f"Failed to download {url}")


def download_tiles(
    url_template: str,
    zoom: int,
    tx_min: int, tx_max: int,
    ty_min: int, ty_max: int,
    label: str = "tiles",
) -> Image.Image:
    """Download a grid of tiles and stitch them into a single PIL image."""
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    total = cols * rows
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))

    tasks: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                url = url_template.format(z=zoom, x=tx, y=ty)
                fut = pool.submit(_download_tile, url)
                tasks[fut] = (tx - tx_min, ty - ty_min)

        done = 0
        for fut in as_completed(tasks):
            col_idx, row_idx = tasks[fut]
            tile_img = fut.result().convert("RGB")
            mosaic.paste(tile_img, (col_idx * TILE_SIZE, row_idx * TILE_SIZE))
            done += 1
            print(f"\r  Downloading {label}: {done}/{total}", end="", flush=True)

    print()
    return mosaic


def download_terrain_tiles_raw(
    zoom: int,
    tx_min: int, tx_max: int,
    ty_min: int, ty_max: int,
) -> np.ndarray:
    """Download terrain tiles and return decoded elevation as a 2-D float array."""
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    total = cols * rows
    raw = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)

    tasks: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                url = TERRAIN_URL.format(z=zoom, x=tx, y=ty)
                fut = pool.submit(_download_tile, url)
                tasks[fut] = (tx - tx_min, ty - ty_min)

        done = 0
        for fut in as_completed(tasks):
            col_idx, row_idx = tasks[fut]
            tile_img = fut.result().convert("RGB")
            arr = np.array(tile_img, dtype=np.float32)
            elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
            y0 = row_idx * TILE_SIZE
            x0 = col_idx * TILE_SIZE
            raw[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = elev
            done += 1
            print(f"\r  Downloading elevation: {done}/{total}", end="", flush=True)

    print()
    return raw


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def to_local_meters(
    lats: np.ndarray, lons: np.ndarray, center_lat: float, center_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to local x/y in metres relative to a centre point."""
    x = (lons - center_lon) * 111320.0 * math.cos(math.radians(center_lat))
    y = (lats - center_lat) * 110540.0
    return x, y


# ---------------------------------------------------------------------------
# Crop & resample helpers
# ---------------------------------------------------------------------------

def crop_and_resample(
    image: Image.Image | np.ndarray,
    mosaic_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    resolution: int,
    is_elevation: bool = False,
) -> np.ndarray:
    """Crop a stitched mosaic (image or elevation array) to *target_bounds* and
    resample to *resolution x resolution*.

    mosaic_bounds / target_bounds: (lat_min, lon_min, lat_max, lon_max)
    """
    lat_min_m, lon_min_m, lat_max_m, lon_max_m = mosaic_bounds
    lat_min_t, lon_min_t, lat_max_t, lon_max_t = target_bounds

    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    px_left = int((lon_min_t - lon_min_m) / (lon_max_m - lon_min_m) * w)
    px_right = int((lon_max_t - lon_min_m) / (lon_max_m - lon_min_m) * w)
    # Latitude axis is inverted in image coordinates (top = north = max lat)
    px_top = int((lat_max_m - lat_max_t) / (lat_max_m - lat_min_m) * h)
    px_bottom = int((lat_max_m - lat_min_t) / (lat_max_m - lat_min_m) * h)

    px_left = max(0, px_left)
    px_right = min(w, px_right)
    px_top = max(0, px_top)
    px_bottom = min(h, px_bottom)

    if is_elevation:
        cropped = image[px_top:px_bottom, px_left:px_right]
        pil_crop = Image.fromarray(cropped, mode="F")
        resampled = pil_crop.resize((resolution, resolution), Image.BILINEAR)
        return np.array(resampled, dtype=np.float32)
    else:
        cropped = image.crop((px_left, px_top, px_right, px_bottom))
        resampled = cropped.resize((resolution, resolution), Image.LANCZOS)
        return resampled


# ---------------------------------------------------------------------------
# Build Plotly figure
# ---------------------------------------------------------------------------

def build_figure(
    sat_img: Image.Image,
    elevation: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    exaggeration: float,
    track_df: pd.DataFrame | None = None,
    locations_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Construct the interactive 3D Plotly figure."""

    res = elevation.shape[0]
    bb = bounding_box(center_lat, center_lon, radius_km)
    lat_min, lon_min, lat_max, lon_max = bb

    lats_1d = np.linspace(lat_max, lat_min, res)   # top-to-bottom = N-to-S
    lons_1d = np.linspace(lon_min, lon_max, res)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)

    x_grid, y_grid = to_local_meters(lat_grid, lon_grid, center_lat, center_lon)
    z_grid = elevation * exaggeration

    # --- quantize satellite image to 256 colours for Plotly colorscale ---
    sat_rgb = sat_img.convert("RGB")
    sat_quantized = sat_rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = sat_quantized.getpalette()[:256 * 3]
    indices = np.array(sat_quantized, dtype=np.float64)

    n_colors = int(indices.max()) + 1
    colorscale = []
    for i in range(n_colors):
        r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
        norm = i / max(n_colors - 1, 1)
        colorscale.append([norm, f"rgb({r},{g},{b})"])

    surface = go.Surface(
        x=x_grid / 1000.0,
        y=y_grid / 1000.0,
        z=z_grid,
        surfacecolor=indices,
        colorscale=colorscale,
        cmin=0,
        cmax=max(n_colors - 1, 1),
        showscale=False,
        lighting=dict(ambient=0.9, diffuse=0.4, specular=0.05, roughness=0.8),
        lightposition=dict(x=0, y=0, z=10000),
        hovertemplate=(
            "East: %{x:.2f} km<br>"
            "North: %{y:.2f} km<br>"
            "Elevation: %{z:.0f} m<extra></extra>"
        ),
    )

    fig = go.Figure(data=[surface])

    # --- optional drone / RC plane track ---
    if track_df is not None and len(track_df) > 0:
        t_x, t_y = to_local_meters(
            track_df["lat"].values, track_df["lon"].values, center_lat, center_lon
        )
        t_z = track_df["alt"].values * exaggeration

        if "timestamp" in track_df.columns:
            color_vals = pd.to_numeric(
                pd.to_datetime(track_df["timestamp"]), errors="coerce"
            )
            color_vals = (color_vals - color_vals.min()) / max(
                (color_vals.max() - color_vals.min()), 1
            )
            color_label = "Time"
            cscale = "Viridis"
        else:
            color_vals = track_df["alt"].values
            color_label = "Altitude (m)"
            cscale = "Hot"

        track_trace = go.Scatter3d(
            x=t_x / 1000.0,
            y=t_y / 1000.0,
            z=t_z,
            mode="lines",
            line=dict(color=color_vals, colorscale=cscale, width=5, showscale=True,
                      colorbar=dict(title=color_label, x=1.02, len=0.5)),
            hovertemplate=(
                "East: %{x:.2f} km<br>"
                "North: %{y:.2f} km<br>"
                "Alt: %{z:.0f} m<extra></extra>"
            ),
            name="Track",
        )
        fig.add_trace(track_trace)

    # --- optional location markers ---
    if locations_df is not None and len(locations_df) > 0:
        for _, row in locations_df.iterrows():
            loc_x, loc_y = to_local_meters(
                np.array([row["lat"]]), np.array([row["lon"]]),
                center_lat, center_lon,
            )
            loc_z = np.array([row["alt"]]) * exaggeration
            fig.add_trace(go.Scatter3d(
                x=loc_x / 1000.0,
                y=loc_y / 1000.0,
                z=loc_z,
                mode="markers",
                marker=dict(
                    size=row["size"],
                    color=row["color"],
                    opacity=row["alpha"],
                    symbol=row["marker"],
                ),
                name=str(row["name"]),
                hovertemplate=(
                    f"{row['name']}<br>"
                    f"Lat: {row['lat']:.6f}, Lon: {row['lon']:.6f}<br>"
                    "East: %{x:.2f} km<br>"
                    "North: %{y:.2f} km<br>"
                    "Alt: %{z:.0f} m<extra></extra>"
                ),
            ))

    # --- layout ---
    elev_range = float(z_grid.max() - z_grid.min()) or 1.0
    x_range = float(x_grid.max() - x_grid.min()) / 1000.0 or 1.0
    y_range = float(y_grid.max() - y_grid.min()) / 1000.0 or 1.0
    horiz_range = max(x_range, y_range)
    z_aspect = elev_range / (horiz_range * 1000.0) * 2.0
    z_aspect = max(z_aspect, 0.05)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="East (km)", showbackground=False),
            yaxis=dict(title="North (km)", showbackground=False),
            zaxis=dict(title="Elevation (m)", showbackground=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=y_range / x_range if x_range else 1, z=z_aspect),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=f"Terrain  ({center_lat:.4f}, {center_lon:.4f})  r={radius_km} km",
            x=0.5,
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3-D satellite terrain viewer"
    )
    parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    parser.add_argument("--radius", type=float, required=True, help="Radius in km")
    parser.add_argument("--track", type=str, default=None,
                        help="CSV file with lat,lon,alt columns (optional timestamp)")
    parser.add_argument("--locations", type=str, default=None,
                        help="CSV with lat,lon,alt,name columns (optional: size,color,alpha,marker)")
    parser.add_argument("--resolution", type=int, default=400,
                        help="Grid resolution NxN (default 400)")
    parser.add_argument("--exaggeration", type=float, default=1.5,
                        help="Vertical exaggeration factor (default 1.5)")
    parser.add_argument("--output", type=str, default="terrain_3d.html",
                        help="Output HTML file (default terrain_3d.html)")
    args = parser.parse_args()

    print(f"Center: ({args.lat}, {args.lon}), radius: {args.radius} km")
    bb = bounding_box(args.lat, args.lon, args.radius)
    lat_min, lon_min, lat_max, lon_max = bb
    print(f"Bounding box: lat [{lat_min:.5f}, {lat_max:.5f}]  "
          f"lon [{lon_min:.5f}, {lon_max:.5f}]")

    zoom = choose_zoom(args.radius, args.resolution)
    print(f"Zoom level: {zoom}")

    tx_min, ty_min_corner = lat_lon_to_tile(lat_max, lon_min, zoom)  # NW corner
    tx_max, ty_max_corner = lat_lon_to_tile(lat_min, lon_max, zoom)  # SE corner
    # tile y increases southward
    ty_min = ty_min_corner
    ty_max = ty_max_corner

    n_tiles = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
    print(f"Tile grid: {tx_max - tx_min + 1} x {ty_max - ty_min + 1}  ({n_tiles} tiles)")

    # Compute the geographic bounds of the full tile mosaic
    tb_nw = tile_bounds(tx_min, ty_min, zoom)
    tb_se = tile_bounds(tx_max, ty_max, zoom)
    mosaic_bounds = (tb_se[0], tb_nw[1], tb_nw[2], tb_se[3])  # (lat_min, lon_min, lat_max, lon_max)

    # Download satellite imagery
    print("Downloading satellite imagery …")
    sat_mosaic = download_tiles(
        SATELLITE_URL, zoom, tx_min, tx_max, ty_min, ty_max, label="satellite"
    )

    # Download & decode elevation
    print("Downloading elevation data …")
    elev_mosaic = download_terrain_tiles_raw(zoom, tx_min, tx_max, ty_min, ty_max)

    # Crop & resample
    print(f"Resampling to {args.resolution}×{args.resolution} …")
    sat_cropped = crop_and_resample(
        sat_mosaic, mosaic_bounds, bb, args.resolution, is_elevation=False
    )
    elev_cropped = crop_and_resample(
        elev_mosaic, mosaic_bounds, bb, args.resolution, is_elevation=True
    )

    # Load optional track
    track_df = None
    if args.track:
        print(f"Loading track: {args.track}")
        track_df = pd.read_csv(args.track)
        required = {"lat", "lon", "alt"}
        if not required.issubset(set(track_df.columns)):
            sys.exit(f"ERROR: CSV must have columns {required}. Found: {list(track_df.columns)}")
        print(f"  {len(track_df)} track points loaded")

    # Load optional locations
    locations_df = None
    if args.locations:
        print(f"Loading locations: {args.locations}")
        locations_df = pd.read_csv(args.locations)
        required_loc = {"lat", "lon", "alt", "name"}
        if not required_loc.issubset(set(locations_df.columns)):
            sys.exit(f"ERROR: Locations CSV must have columns {required_loc}. "
                     f"Found: {list(locations_df.columns)}")
        defaults = {"size": 10, "color": "red", "alpha": 1.0, "marker": "circle"}
        for col, default in defaults.items():
            if col not in locations_df.columns:
                locations_df[col] = default
        print(f"  {len(locations_df)} locations loaded")

    # Build figure
    print("Building 3-D figure …")
    fig = build_figure(
        sat_cropped, elev_cropped,
        args.lat, args.lon, args.radius,
        args.exaggeration, track_df, locations_df,
    )

    # Save & open
    out_path = os.path.abspath(args.output)
    fig.write_html(out_path, include_plotlyjs="cdn")
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"Saved to {out_path}  ({size_mb:.1f} MB)")
    webbrowser.open(f"file:///{out_path}")


if __name__ == "__main__":
    main()
