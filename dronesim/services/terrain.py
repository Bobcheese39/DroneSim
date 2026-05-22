"""Map and terrain service extracted from the terrain_3d prototype."""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image

from dronesim.models import MapSpec, Marker, Waypoint, utc_now, write_json

TILE_SIZE = 256
EARTH_CIRCUMFERENCE_KM = 40075.016686

SATELLITE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

IMAGERY_SOURCES = {
    "esri_world_imagery": SATELLITE_URL,
}
ELEVATION_SOURCES = {
    "aws_terrarium": TERRAIN_URL,
}

ProgressCallback = Callable[[int, int, str], None]

_session = requests.Session()
_session.headers.update({"User-Agent": "DroneSim/0.1 terrain-service"})


@dataclass
class MapAsset:
    """In-memory terrain/map object used by scenario editing and replay."""

    spec: MapSpec
    bounds: tuple[float, float, float, float]
    zoom: int
    satellite: Image.Image
    elevation_m: np.ndarray
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    x_grid_m: np.ndarray
    y_grid_m: np.ndarray
    cache_dir: Path

    @property
    def z_grid_m(self) -> np.ndarray:
        return self.elevation_m * self.spec.vertical_exaggeration

    def elevation_at(self, x_m: float, y_m: float) -> float:
        """Sample terrain elevation at local east/north meters with bilinear interpolation."""
        if self.elevation_m.size == 0:
            raise ValueError("MapAsset has no elevation data")

        x_axis = self.x_grid_m[0, :]
        y_axis = self.y_grid_m[:, 0]
        if x_axis[0] > x_axis[-1]:
            x_axis = x_axis[::-1]
            values = self.elevation_m[:, ::-1]
        else:
            values = self.elevation_m
        if y_axis[0] > y_axis[-1]:
            y_axis = y_axis[::-1]
            values = values[::-1, :]

        x = float(np.clip(x_m, x_axis[0], x_axis[-1]))
        y = float(np.clip(y_m, y_axis[0], y_axis[-1]))
        x_hi = int(np.searchsorted(x_axis, x, side="right"))
        y_hi = int(np.searchsorted(y_axis, y, side="right"))
        x_hi = min(max(x_hi, 1), len(x_axis) - 1)
        y_hi = min(max(y_hi, 1), len(y_axis) - 1)
        x_lo = x_hi - 1
        y_lo = y_hi - 1

        x0, x1 = float(x_axis[x_lo]), float(x_axis[x_hi])
        y0, y1 = float(y_axis[y_lo]), float(y_axis[y_hi])
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

        z00 = float(values[y_lo, x_lo])
        z10 = float(values[y_lo, x_hi])
        z01 = float(values[y_hi, x_lo])
        z11 = float(values[y_hi, x_hi])
        z0 = z00 * (1.0 - tx) + z10 * tx
        z1 = z01 * (1.0 - tx) + z11 * tx
        return z0 * (1.0 - ty) + z1 * ty


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert WGS84 lat/lon to slippy-map tile x/y."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_bounds(tx: int, ty: int, zoom: int) -> tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) for one slippy-map tile."""
    n = 2**zoom
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    return lat_min, lon_min, lat_max, lon_max


def choose_zoom(radius_km: float, resolution: int) -> int:
    """Pick a zoom level with roughly resolution pixels across the requested area."""
    diameter_km = radius_km * 2.0
    for z in range(1, 16):
        meters_per_pixel = EARTH_CIRCUMFERENCE_KM * 1000.0 / (TILE_SIZE * 2**z)
        pixels_across = diameter_km * 1000.0 / meters_per_pixel
        if pixels_across >= resolution:
            return z
    return 15


def bounding_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """Return (lat_min, lon_min, lat_max, lon_max) around a center/radius."""
    d_lat = radius_km / 110.540
    d_lon = radius_km / (111.320 * math.cos(math.radians(lat)))
    return lat - d_lat, lon - d_lon, lat + d_lat, lon + d_lon


def to_local_meters(
    lats: np.ndarray, lons: np.ndarray, center_lat: float, center_lon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to local east/north meters around a map center."""
    x = (lons - center_lon) * 111320.0 * math.cos(math.radians(center_lat))
    y = (lats - center_lat) * 110540.0
    return x, y


def local_to_lat_lon(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    center_lat: float,
    center_lon: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Convert local east/north meters back to WGS84 lat/lon."""
    lat = np.asarray(y_m) / 110540.0 + center_lat
    lon = np.asarray(x_m) / (111320.0 * math.cos(math.radians(center_lat))) + center_lon
    if np.isscalar(x_m) and np.isscalar(y_m):
        return float(lat), float(lon)
    return lat, lon


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


def _resolve_source_url(source: str, sources: dict[str, str], kind: str) -> str:
    try:
        return sources[source]
    except KeyError as exc:
        supported = ", ".join(sorted(sources))
        raise ValueError(f"Unsupported {kind} source '{source}'. Supported sources: {supported}") from exc


def _tile_cache_path(
    cache_root: Path,
    source: str,
    zoom: int,
    tx: int,
    ty: int,
) -> Path:
    return cache_root / "tiles" / source / str(zoom) / str(tx) / f"{ty}.png"


def _save_tile_to_cache(tile: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tile.save(tmp_path, format="PNG")
    tmp_path.replace(path)


def _load_or_download_tile(url: str, cache_path: Path | None = None) -> Image.Image:
    if cache_path is not None and cache_path.exists():
        with Image.open(cache_path) as img:
            return img.convert("RGB")

    tile = _download_tile(url).convert("RGB")
    if cache_path is not None:
        _save_tile_to_cache(tile, cache_path)
    return tile


def _notify(progress: ProgressCallback | None, done: int, total: int, label: str) -> None:
    if progress is not None:
        progress(done, total, label)


def download_tiles(
    url_template: str,
    zoom: int,
    tx_min: int,
    tx_max: int,
    ty_min: int,
    ty_max: int,
    *,
    label: str = "tiles",
    cache_root: str | Path | None = None,
    source_name: str | None = None,
    progress: ProgressCallback | None = None,
) -> Image.Image:
    """Download a grid of image tiles and stitch them into one PIL image."""
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    total = cols * rows
    mosaic = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))

    tasks: dict = {}
    tile_cache_root = Path(cache_root) if cache_root is not None else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                cache_path = (
                    _tile_cache_path(tile_cache_root, source_name, zoom, tx, ty)
                    if tile_cache_root is not None and source_name is not None
                    else None
                )
                fut = pool.submit(
                    _load_or_download_tile,
                    url_template.format(z=zoom, x=tx, y=ty),
                    cache_path,
                )
                tasks[fut] = (tx - tx_min, ty - ty_min)

        done = 0
        for fut in as_completed(tasks):
            col_idx, row_idx = tasks[fut]
            tile_img = fut.result().convert("RGB")
            mosaic.paste(tile_img, (col_idx * TILE_SIZE, row_idx * TILE_SIZE))
            done += 1
            _notify(progress, done, total, label)

    return mosaic


def download_terrain_tiles_raw(
    zoom: int,
    tx_min: int,
    tx_max: int,
    ty_min: int,
    ty_max: int,
    *,
    url_template: str = TERRAIN_URL,
    cache_root: str | Path | None = None,
    source_name: str | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Download Terrarium tiles and return decoded elevation as meters."""
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    total = cols * rows
    raw = np.zeros((rows * TILE_SIZE, cols * TILE_SIZE), dtype=np.float32)

    tasks: dict = {}
    tile_cache_root = Path(cache_root) if cache_root is not None else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                cache_path = (
                    _tile_cache_path(tile_cache_root, source_name, zoom, tx, ty)
                    if tile_cache_root is not None and source_name is not None
                    else None
                )
                fut = pool.submit(
                    _load_or_download_tile,
                    url_template.format(z=zoom, x=tx, y=ty),
                    cache_path,
                )
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
            _notify(progress, done, total, "elevation")

    return raw


def crop_and_resample(
    image: Image.Image | np.ndarray,
    mosaic_bounds: tuple[float, float, float, float],
    target_bounds: tuple[float, float, float, float],
    resolution: int,
    *,
    is_elevation: bool = False,
) -> Image.Image | np.ndarray:
    """Crop a stitched mosaic to target bounds and resample to NxN."""
    lat_min_m, lon_min_m, lat_max_m, lon_max_m = mosaic_bounds
    lat_min_t, lon_min_t, lat_max_t, lon_max_t = target_bounds

    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    px_left = int((lon_min_t - lon_min_m) / (lon_max_m - lon_min_m) * w)
    px_right = int((lon_max_t - lon_min_m) / (lon_max_m - lon_min_m) * w)
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

    cropped = image.crop((px_left, px_top, px_right, px_bottom))
    return cropped.resize((resolution, resolution), Image.LANCZOS)


def _write_map_manifest(
    cache_dir: Path,
    spec: MapSpec,
    bounds: tuple[float, float, float, float],
    zoom: int,
    tile_range: dict[str, int],
    mosaic_bounds: tuple[float, float, float, float],
) -> None:
    write_json(
        cache_dir / "map_manifest.json",
        {
            "created_utc": utc_now(),
            "spec": spec,
            "bounds": {
                "lat_min": bounds[0],
                "lon_min": bounds[1],
                "lat_max": bounds[2],
                "lon_max": bounds[3],
            },
            "zoom": zoom,
            "tile_range": tile_range,
            "mosaic_bounds": {
                "lat_min": mosaic_bounds[0],
                "lon_min": mosaic_bounds[1],
                "lat_max": mosaic_bounds[2],
                "lon_max": mosaic_bounds[3],
            },
            "processed_files": {
                "satellite": "satellite.png",
                "elevation": "elevation.npy",
            },
            "raw_tile_cache": f"tiles/{spec.imagery_source}/... and tiles/{spec.elevation_source}/...",
        },
    )


class TerrainService:
    """Fetch, cache, and build map/terrain assets."""

    def __init__(self, cache_root: str | Path = "maps/cache") -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def build_blank_asset(self, spec: MapSpec) -> MapAsset:
        """Return a no-network placeholder asset with zero elevation."""
        spec.validate()
        bounds = bounding_box(spec.center_lat, spec.center_lon, spec.radius_km)
        res = spec.resolution
        sat = Image.new("RGB", (res, res), (20, 24, 32))
        elevation = np.zeros((res, res), dtype=np.float32)
        lat_min, lon_min, lat_max, lon_max = bounds
        lats_1d = np.linspace(lat_max, lat_min, res)
        lons_1d = np.linspace(lon_min, lon_max, res)
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        x_grid, y_grid = to_local_meters(lat_grid, lon_grid, spec.center_lat, spec.center_lon)
        return MapAsset(
            spec=spec,
            bounds=bounds,
            zoom=choose_zoom(spec.radius_km, spec.resolution),
            satellite=sat,
            elevation_m=elevation,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            x_grid_m=x_grid,
            y_grid_m=y_grid,
            cache_dir=self.cache_root / spec.key(),
        )

    def fetch_map(
        self,
        spec: MapSpec,
        *,
        fetch_remote: bool = True,
        progress: ProgressCallback | None = None,
    ) -> MapAsset:
        """Load a cached map asset or fetch source tiles when requested."""
        spec.validate()
        imagery_url = _resolve_source_url(spec.imagery_source, IMAGERY_SOURCES, "imagery")
        elevation_url = _resolve_source_url(spec.elevation_source, ELEVATION_SOURCES, "elevation")
        cache_dir = self.cache_root / spec.key()
        cache_dir.mkdir(parents=True, exist_ok=True)
        sat_path = cache_dir / "satellite.png"
        elev_path = cache_dir / "elevation.npy"
        manifest_path = cache_dir / "map_manifest.json"

        bounds = bounding_box(spec.center_lat, spec.center_lon, spec.radius_km)
        zoom = choose_zoom(spec.radius_km, spec.resolution)
        lat_min, lon_min, lat_max, lon_max = bounds
        tx_min, ty_min_corner = lat_lon_to_tile(lat_max, lon_min, zoom)
        tx_max, ty_max_corner = lat_lon_to_tile(lat_min, lon_max, zoom)
        ty_min = ty_min_corner
        ty_max = ty_max_corner
        tile_range = {
            "tx_min": tx_min,
            "tx_max": tx_max,
            "ty_min": ty_min,
            "ty_max": ty_max,
        }
        tb_nw = tile_bounds(tx_min, ty_min, zoom)
        tb_se = tile_bounds(tx_max, ty_max, zoom)
        mosaic_bounds = (tb_se[0], tb_nw[1], tb_nw[2], tb_se[3])

        if sat_path.exists() and elev_path.exists():
            sat_cropped = Image.open(sat_path).convert("RGB")
            elev_cropped = np.load(elev_path)
            if not manifest_path.exists():
                _write_map_manifest(cache_dir, spec, bounds, zoom, tile_range, mosaic_bounds)
        elif not fetch_remote:
            return self.build_blank_asset(spec)
        else:
            sat_mosaic = download_tiles(
                imagery_url,
                zoom,
                tx_min,
                tx_max,
                ty_min,
                ty_max,
                label="satellite",
                cache_root=self.cache_root,
                source_name=spec.imagery_source,
                progress=progress,
            )
            elev_mosaic = download_terrain_tiles_raw(
                zoom,
                tx_min,
                tx_max,
                ty_min,
                ty_max,
                url_template=elevation_url,
                cache_root=self.cache_root,
                source_name=spec.elevation_source,
                progress=progress,
            )

            sat_cropped = crop_and_resample(
                sat_mosaic,
                mosaic_bounds,
                bounds,
                spec.resolution,
                is_elevation=False,
            )
            elev_cropped = crop_and_resample(
                elev_mosaic,
                mosaic_bounds,
                bounds,
                spec.resolution,
                is_elevation=True,
            )
            sat_cropped.save(sat_path)
            np.save(elev_path, elev_cropped)
            _write_map_manifest(cache_dir, spec, bounds, zoom, tile_range, mosaic_bounds)

        res = spec.resolution
        lat_min, lon_min, lat_max, lon_max = bounds
        lats_1d = np.linspace(lat_max, lat_min, res)
        lons_1d = np.linspace(lon_min, lon_max, res)
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        x_grid, y_grid = to_local_meters(lat_grid, lon_grid, spec.center_lat, spec.center_lon)

        return MapAsset(
            spec=spec,
            bounds=bounds,
            zoom=zoom,
            satellite=sat_cropped,
            elevation_m=elev_cropped,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            x_grid_m=x_grid,
            y_grid_m=y_grid,
            cache_dir=cache_dir,
        )

    def waypoint_to_local(self, waypoint: Waypoint, spec: MapSpec) -> Waypoint:
        if waypoint.has_local_xy():
            return waypoint
        if waypoint.lat is None or waypoint.lon is None:
            raise ValueError("Waypoint needs lat/lon or local x/y")
        x, y = to_local_meters(
            np.array([waypoint.lat]), np.array([waypoint.lon]), spec.center_lat, spec.center_lon
        )
        waypoint.x_m = float(x[0])
        waypoint.y_m = float(y[0])
        waypoint.z_m = waypoint.alt_m
        return waypoint

    def local_to_waypoint(self, waypoint: Waypoint, spec: MapSpec) -> Waypoint:
        if waypoint.lat is not None and waypoint.lon is not None:
            return waypoint
        if waypoint.x_m is None or waypoint.y_m is None:
            raise ValueError("Waypoint needs local x/y or lat/lon")
        lat, lon = local_to_lat_lon(waypoint.x_m, waypoint.y_m, spec.center_lat, spec.center_lon)
        waypoint.lat = float(lat)
        waypoint.lon = float(lon)
        waypoint.alt_m = float(waypoint.z_m if waypoint.z_m is not None else waypoint.alt_m)
        return waypoint


def build_terrain_figure(
    asset: MapAsset,
    *,
    trajectory_xyz: np.ndarray | None = None,
    waypoints: list[Waypoint] | None = None,
    markers: list[Marker] | None = None,
) -> go.Figure:
    """Build a Plotly 3D terrain/replay figure from a MapAsset."""
    sat_rgb = asset.satellite.convert("RGB")
    sat_quantized = sat_rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = sat_quantized.getpalette()[: 256 * 3]
    indices = np.array(sat_quantized, dtype=np.float64)

    n_colors = int(indices.max()) + 1
    colorscale = []
    for i in range(n_colors):
        r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
        norm = i / max(n_colors - 1, 1)
        colorscale.append([norm, f"rgb({r},{g},{b})"])

    fig = go.Figure(
        data=[
            go.Surface(
                x=asset.x_grid_m / 1000.0,
                y=asset.y_grid_m / 1000.0,
                z=asset.z_grid_m,
                surfacecolor=indices,
                colorscale=colorscale,
                cmin=0,
                cmax=max(n_colors - 1, 1),
                showscale=False,
                lighting=dict(ambient=0.9, diffuse=0.4, specular=0.05, roughness=0.8),
                lightposition=dict(x=0, y=0, z=10000),
                name="Terrain",
            )
        ]
    )

    if trajectory_xyz is not None and len(trajectory_xyz):
        fig.add_trace(
            go.Scatter3d(
                x=trajectory_xyz[:, 0] / 1000.0,
                y=trajectory_xyz[:, 1] / 1000.0,
                z=trajectory_xyz[:, 2],
                mode="lines",
                line=dict(color="#39ff14", width=5),
                name="Trajectory",
            )
        )

    if waypoints:
        local_points = np.array([wp.local_xyz() for wp in waypoints if wp.has_local_xy()])
        if len(local_points):
            fig.add_trace(
                go.Scatter3d(
                    x=local_points[:, 0] / 1000.0,
                    y=local_points[:, 1] / 1000.0,
                    z=local_points[:, 2],
                    mode="markers+text",
                    marker=dict(size=6, color="#58a6ff", symbol="diamond"),
                    text=[wp.label or f"WP{i}" for i, wp in enumerate(waypoints)],
                    textposition="top center",
                    name="Waypoints",
                )
            )

    if markers:
        for marker in markers:
            if not marker.visible or marker.x_m is None or marker.y_m is None:
                continue
            z = marker.z_m if marker.z_m is not None else marker.alt_m
            fig.add_trace(
                go.Scatter3d(
                    x=[marker.x_m / 1000.0],
                    y=[marker.y_m / 1000.0],
                    z=[z],
                    mode="markers+text",
                    marker=dict(size=marker.size, color=marker.color),
                    text=[marker.label],
                    textposition="top center",
                    name=marker.label,
                )
            )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="East (km)", showbackground=False),
            yaxis=dict(title="North (km)", showbackground=False),
            zaxis=dict(title="Elevation / Altitude (m)", showbackground=False),
            aspectmode="data",
        ),
        title=f"{asset.spec.name} ({asset.spec.center_lat:.4f}, {asset.spec.center_lon:.4f})",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
