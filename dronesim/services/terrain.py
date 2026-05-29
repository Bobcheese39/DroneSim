"""Map and terrain service extracted from the terrain_3d prototype."""
from __future__ import annotations

import io
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from PIL import Image

from dronesim.models import MapSpec, Marker, RunResult, Waypoint, utc_now, write_json

logger = logging.getLogger(__name__)

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


_LEGACY_KEY_RE = re.compile(
    r"^(?P<name>.+?)_(?P<lat>m?\d+p\d+)_(?P<lon>m?\d+p\d+)"
    r"_r(?P<radius>\d+p\d+)_n(?P<res>\d+)$"
)
_NEW_KEY_RE = re.compile(
    r"^(?P<lat>m?\d+p\d+)_(?P<lon>m?\d+p\d+)_r(?P<radius>\d+p\d+)_n(?P<res>\d+)$"
)


class MapCacheMiss(RuntimeError):
    """Raised when a requested map asset is not available from any cache.

    Carries enough context for callers (GUI / CLI) to render an actionable
    error message instead of silently substituting a blank placeholder.
    """

    def __init__(
        self,
        spec: MapSpec,
        cache_dir: Path,
        missing_tiles: int,
        available_caches: list[dict],
    ) -> None:
        self.spec = spec
        self.cache_dir = cache_dir
        self.missing_tiles = missing_tiles
        self.available_caches = available_caches
        super().__init__(
            f"No cached map at '{cache_dir.name}' for "
            f"({spec.center_lat:.6f}, {spec.center_lon:.6f}) "
            f"r={spec.radius_km}km n={spec.resolution}. "
            f"{missing_tiles} raw tile(s) also missing. "
            f"{len(available_caches)} other cache(s) available."
        )


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
    origin: str = "unknown"

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


def _load_or_download_tile(
    url: str,
    cache_path: Path | None = None,
    *,
    allow_network: bool = True,
) -> Image.Image:
    if cache_path is not None and cache_path.exists():
        with Image.open(cache_path) as img:
            return img.convert("RGB")

    if not allow_network:
        raise FileNotFoundError(
            f"Raw tile cache miss with allow_network=False: {cache_path}"
        )

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
    allow_network: bool = True,
) -> Image.Image:
    """Download a grid of image tiles and stitch them into one PIL image.

    When ``allow_network`` is False, a missing raw tile cache file raises
    :class:`FileNotFoundError` instead of issuing an HTTP request.
    """
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
                    allow_network=allow_network,
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
    allow_network: bool = True,
) -> np.ndarray:
    """Download Terrarium tiles and return decoded elevation as meters.

    When ``allow_network`` is False, a missing raw tile cache file raises
    :class:`FileNotFoundError` instead of issuing an HTTP request.
    """
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
                    allow_network=allow_network,
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


def encode_cesium_heightmap(
    elevation_m: np.ndarray,
    *,
    vertical_exaggeration: float = 1.0,
    max_dim: int = 1024,
) -> dict[str, float | int | bytes]:
    """Encode an elevation grid for Cesium ``HeightmapTerrainProvider``.

    Returns width, height, height_offset, height_scale, and a little-endian
    uint16 row-major buffer (north row first, west column first).
    """
    z = np.asarray(elevation_m, dtype=np.float64) * vertical_exaggeration
    rows, cols = z.shape

    if rows > max_dim or cols > max_dim:
        scale = max(rows, cols) / max_dim
        out_h = max(2, int(round(rows / scale)))
        out_w = max(2, int(round(cols / scale)))
        pil = Image.fromarray(z.astype(np.float32), mode="F")
        z = np.array(pil.resize((out_w, out_h), Image.BILINEAR), dtype=np.float64)

    min_h = float(z.min())
    max_h = float(z.max())
    if max_h == min_h:
        max_h = min_h + 1.0

    height_scale = (max_h - min_h) / 65535.0
    encoded = np.round((z - min_h) / height_scale).astype(np.uint16)

    return {
        "width": int(encoded.shape[1]),
        "height": int(encoded.shape[0]),
        "height_offset": min_h,
        "height_scale": height_scale,
        "buffer": encoded.tobytes(),
    }


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


def _count_missing_raw_tiles(
    cache_root: Path,
    source_name: str,
    zoom: int,
    tx_min: int,
    tx_max: int,
    ty_min: int,
    ty_max: int,
) -> int:
    """Count how many raw slippy-map tiles are missing from the on-disk cache."""
    missing = 0
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            if not _tile_cache_path(cache_root, source_name, zoom, tx, ty).exists():
                missing += 1
    return missing


class TerrainService:
    """Fetch, cache, and build map/terrain assets."""

    def __init__(self, cache_root: str | Path = "maps/cache") -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_cache_dirs()

    def build_blank_asset(self, spec: MapSpec) -> MapAsset:
        """Return a no-network placeholder asset with zero elevation.

        This is reserved for the initial application placeholder. Cache-miss
        paths now raise :class:`MapCacheMiss` instead of silently returning
        a blank surface.
        """
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
            origin="blank-placeholder",
        )

    def fetch_map(
        self,
        spec: MapSpec,
        *,
        fetch_remote: bool = True,
        progress: ProgressCallback | None = None,
    ) -> MapAsset:
        """Load a cached map asset, rebuild from raw tiles, or download fresh.

        Resolution order:

        1. Load the processed cache (``satellite.png`` + ``elevation.npy``).
        2. If missing, try to rebuild from the raw slippy-tile cache without
           the network. On success, the processed cache is populated.
        3. If raw tiles are also missing and ``fetch_remote`` is True,
           download fresh tiles and write the processed cache.
        4. Otherwise raise :class:`MapCacheMiss` with details that callers can
           surface to the user (looked-up key, missing tile count, list of
           other cached configurations).
        """
        spec.validate()
        imagery_url = _resolve_source_url(spec.imagery_source, IMAGERY_SOURCES, "imagery")
        elevation_url = _resolve_source_url(spec.elevation_source, ELEVATION_SOURCES, "elevation")
        cache_dir = self.cache_root / spec.key()
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

        origin = "unknown"
        sat_cropped: Image.Image | None = None
        elev_cropped: np.ndarray | None = None

        if sat_path.exists() and elev_path.exists():
            logger.info(
                "Map cache hit: loading processed assets from %s", cache_dir
            )
            sat_cropped = Image.open(sat_path).convert("RGB")
            elev_cropped = np.load(elev_path)
            if not manifest_path.exists():
                _write_map_manifest(
                    cache_dir, spec, bounds, zoom, tile_range, mosaic_bounds
                )
            origin = "processed-cache"
        else:
            logger.info(
                "Processed map cache miss at %s; attempting offline rebuild "
                "from raw tile cache",
                cache_dir,
            )
            try:
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
                    allow_network=False,
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
                    allow_network=False,
                )
            except FileNotFoundError as exc:
                logger.info(
                    "Offline rebuild for %s not possible: %s",
                    cache_dir.name,
                    exc,
                )
                if not fetch_remote:
                    missing_imagery = _count_missing_raw_tiles(
                        self.cache_root,
                        spec.imagery_source,
                        zoom,
                        tx_min,
                        tx_max,
                        ty_min,
                        ty_max,
                    )
                    missing_elev = _count_missing_raw_tiles(
                        self.cache_root,
                        spec.elevation_source,
                        zoom,
                        tx_min,
                        tx_max,
                        ty_min,
                        ty_max,
                    )
                    missing_total = missing_imagery + missing_elev
                    available = self._list_available_caches()
                    logger.warning(
                        "MapCacheMiss for '%s': %d raw tile(s) missing across "
                        "imagery+elevation; %d alternate cache(s) available",
                        cache_dir.name,
                        missing_total,
                        len(available),
                    )
                    raise MapCacheMiss(
                        spec=spec,
                        cache_dir=cache_dir,
                        missing_tiles=missing_total,
                        available_caches=available,
                    ) from exc
                logger.info(
                    "Downloading missing tiles for %s (remote fetch enabled)",
                    cache_dir.name,
                )
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
                origin = "network-download"
            else:
                logger.info(
                    "Offline rebuild succeeded for %s; writing processed cache",
                    cache_dir.name,
                )
                origin = "raw-tile-rebuild"

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
            cache_dir.mkdir(parents=True, exist_ok=True)
            sat_cropped.save(sat_path)
            np.save(elev_path, elev_cropped)
            _write_map_manifest(
                cache_dir, spec, bounds, zoom, tile_range, mosaic_bounds
            )

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
            origin=origin,
        )

    def _list_available_caches(self) -> list[dict]:
        """Summarize the cache root so callers can guide the user."""
        records: list[dict] = []
        if not self.cache_root.exists():
            return records
        for entry in sorted(self.cache_root.iterdir()):
            if not entry.is_dir() or entry.name == "tiles":
                continue
            sat_present = (entry / "satellite.png").exists()
            elev_present = (entry / "elevation.npy").exists()
            if not (sat_present and elev_present):
                continue
            record: dict = {
                "key": entry.name,
                "center_lat": None,
                "center_lon": None,
                "radius_km": None,
                "resolution": None,
            }
            manifest_path = entry / "map_manifest.json"
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    spec_dict = data.get("spec") or {}
                    record["center_lat"] = spec_dict.get("center_lat")
                    record["center_lon"] = spec_dict.get("center_lon")
                    record["radius_km"] = spec_dict.get("radius_km")
                    record["resolution"] = spec_dict.get("resolution")
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Could not parse manifest %s: %s", manifest_path, exc
                    )
            records.append(record)
        return records

    def _migrate_legacy_cache_dirs(self) -> None:
        """Rename legacy ``<name>_<lat>_<lon>_r..._n...`` cache dirs to the new
        spatial-only key. Runs once per :class:`TerrainService` instance.
        """
        if not self.cache_root.exists():
            return
        for entry in sorted(self.cache_root.iterdir()):
            if not entry.is_dir() or entry.name == "tiles":
                continue
            new_name = self._compute_new_key_for_dir(entry)
            if new_name is None or new_name == entry.name:
                continue
            target = self.cache_root / new_name
            if target.exists():
                logger.warning(
                    "Skipping migration of '%s' -> '%s' (target already exists)",
                    entry.name,
                    new_name,
                )
                continue
            try:
                entry.rename(target)
            except OSError as exc:
                logger.warning(
                    "Could not migrate cache dir '%s' -> '%s': %s",
                    entry.name,
                    new_name,
                    exc,
                )
                continue
            logger.info(
                "Migrated cache directory '%s' -> '%s'", entry.name, new_name
            )

    @staticmethod
    def _compute_new_key_for_dir(entry: Path) -> str | None:
        """Return the new-format cache key for an existing cache directory.

        Returns ``None`` if the directory name does not look like any known
        cache key format.
        """
        if _NEW_KEY_RE.match(entry.name):
            return entry.name
        manifest_path = entry / "map_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                spec_dict = data.get("spec") or {}
                spec_for_key = MapSpec(
                    center_lat=float(spec_dict["center_lat"]),
                    center_lon=float(spec_dict["center_lon"]),
                    radius_km=float(spec_dict["radius_km"]),
                    resolution=int(spec_dict["resolution"]),
                    name=spec_dict.get("name", "default_map"),
                )
                return spec_for_key.key()
            except (json.JSONDecodeError, KeyError, ValueError, TypeError, OSError) as exc:
                logger.warning(
                    "Cannot derive new cache key from manifest %s: %s",
                    manifest_path,
                    exc,
                )
        match = _LEGACY_KEY_RE.match(entry.name)
        if not match:
            return None
        return (
            f"{match['lat']}_{match['lon']}_r{match['radius']}_n{match['res']}"
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

    def marker_to_local(self, marker: Marker, spec: MapSpec) -> Marker:
        if marker.x_m is not None and marker.y_m is not None:
            return marker
        if marker.lat is None or marker.lon is None:
            raise ValueError("Marker needs lat/lon or local x/y")
        x, y = to_local_meters(
            np.array([marker.lat]), np.array([marker.lon]), spec.center_lat, spec.center_lon
        )
        marker.x_m = float(x[0])
        marker.y_m = float(y[0])
        marker.z_m = marker.alt_m if marker.z_m is None else marker.z_m
        return marker

    def local_to_marker(self, marker: Marker, spec: MapSpec) -> Marker:
        if marker.lat is not None and marker.lon is not None:
            return marker
        if marker.x_m is None or marker.y_m is None:
            raise ValueError("Marker needs local x/y or lat/lon")
        lat, lon = local_to_lat_lon(marker.x_m, marker.y_m, spec.center_lat, spec.center_lon)
        marker.lat = float(lat)
        marker.lon = float(lon)
        marker.alt_m = float(marker.z_m if marker.z_m is not None else marker.alt_m)
        return marker


CLEARANCE_WARN_M = 1.0


def compute_run_clearance_m(run: RunResult, asset: MapAsset) -> list[float]:
    """Return altitude above terrain (m) at each trajectory sample."""
    clearance: list[float] = []
    for pos in run.position_m:
        if len(pos) < 3 or pos[0] is None or pos[1] is None or pos[2] is None:
            clearance.append(float("nan"))
            continue
        try:
            terrain_z = asset.elevation_at(float(pos[0]), float(pos[1]))
            clearance.append(float(pos[2]) - terrain_z)
        except (ValueError, IndexError):
            clearance.append(float("nan"))
    return clearance


def waypoints_from_run_metadata(run: RunResult, default_alt_m: float = 5.0) -> list[Waypoint]:
    """Rebuild waypoint list from frozen run metadata."""
    raw = run.metadata.get("waypoints_local_xy")
    if not raw:
        return []
    waypoints: list[Waypoint] = []
    for i, pt in enumerate(raw):
        if len(pt) < 2:
            continue
        z = float(pt[2]) if len(pt) > 2 else default_alt_m
        waypoints.append(Waypoint.local(float(pt[0]), float(pt[1]), z, label=f"WP{i}"))
    return waypoints


def reference_xyz_from_run(run: RunResult, default_alt_m: float = 5.0) -> np.ndarray | None:
    """Return reference path as Nx3 array from run series or metadata."""
    if run.reference_position_m:
        arr = np.asarray(run.reference_position_m, dtype=float)
        if len(arr):
            return arr
    spline = run.metadata.get("spline_points")
    if spline:
        arr = np.asarray(spline, dtype=float)
        if len(arr) and arr.shape[1] >= 2:
            z = default_alt_m
            cfg = run.metadata.get("cfg_summary") or {}
            if "target_altitude_m" in cfg:
                z = float(cfg["target_altitude_m"])
            return np.column_stack([arr[:, 0], arr[:, 1], np.full(len(arr), z)])
    return None


def build_terrain_figure(
    asset: MapAsset,
    *,
    trajectory_xyz: np.ndarray | None = None,
    reference_xyz: np.ndarray | None = None,
    time_index: int | None = None,
    clearance_m: list[float] | None = None,
    waypoints: list[Waypoint] | None = None,
    markers: list[Marker] | None = None,
    show_clearance_warning: bool = False,
) -> "go.Figure":
    """Build a Plotly 3D terrain/replay figure from a MapAsset.

    ``plotly`` is imported lazily so the core install does not require it; the
    web frontend renders charts client-side and never calls this helper.
    """
    import plotly.graph_objects as go

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
                x=asset.x_grid_m,
                y=asset.y_grid_m,
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

    if reference_xyz is not None and len(reference_xyz):
        fig.add_trace(
            go.Scatter3d(
                x=reference_xyz[:, 0],
                y=reference_xyz[:, 1],
                z=reference_xyz[:, 2],
                mode="lines",
                line=dict(color="#ff7b00", width=4),
                name="Reference",
            )
        )

    if trajectory_xyz is not None and len(trajectory_xyz):
        end_idx = len(trajectory_xyz)
        if time_index is not None:
            end_idx = min(max(int(time_index) + 1, 1), len(trajectory_xyz))
        segment = trajectory_xyz[:end_idx]

        line_color = "#39ff14"
        if show_clearance_warning and clearance_m is not None and len(clearance_m) >= end_idx:
            if any(c < CLEARANCE_WARN_M for c in clearance_m[:end_idx] if c == c):
                line_color = "#ff3860"

        fig.add_trace(
            go.Scatter3d(
                x=segment[:, 0],
                y=segment[:, 1],
                z=segment[:, 2],
                mode="lines",
                line=dict(color=line_color, width=5),
                name="Actual",
            )
        )

        if time_index is not None and 0 <= time_index < len(trajectory_xyz):
            pt = trajectory_xyz[time_index]
            fig.add_trace(
                go.Scatter3d(
                    x=[pt[0]],
                    y=[pt[1]],
                    z=[pt[2]],
                    mode="markers",
                    marker=dict(size=8, color="#58a6ff", symbol="circle"),
                    name="Drone",
                )
            )

    if waypoints:
        local_points = np.array([wp.local_xyz() for wp in waypoints if wp.has_local_xy()])
        if len(local_points):
            fig.add_trace(
                go.Scatter3d(
                    x=local_points[:, 0],
                    y=local_points[:, 1],
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
                    x=[marker.x_m],
                    y=[marker.y_m],
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
            xaxis=dict(title="East (m)", showbackground=False),
            yaxis=dict(title="North (m)", showbackground=False),
            zaxis=dict(title="Elevation / Altitude (m)", showbackground=False),
            aspectmode="data",
        ),
        title=f"{asset.spec.name} ({asset.spec.center_lat:.4f}, {asset.spec.center_lon:.4f})",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
