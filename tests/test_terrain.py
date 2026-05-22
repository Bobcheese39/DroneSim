from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from dronesim.models import MapSpec
from dronesim.services.terrain import (
    TILE_SIZE,
    TerrainService,
    download_tiles,
    local_to_lat_lon,
    to_local_meters,
)


class TerrainServiceTest(unittest.TestCase):
    def test_build_blank_asset_from_map_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = MapSpec(name="unit_map", resolution=32, radius_km=0.25)
            asset = TerrainService(cache_root=tmp).build_blank_asset(spec)

        self.assertEqual(asset.spec, spec)
        self.assertEqual(asset.satellite.size, (32, 32))
        self.assertEqual(asset.elevation_m.shape, (32, 32))
        self.assertEqual(asset.x_grid_m.shape, (32, 32))
        self.assertEqual(asset.y_grid_m.shape, (32, 32))

    def test_local_geographic_round_trip(self) -> None:
        spec = MapSpec(center_lat=37.6188056, center_lon=-122.3754167)
        lat = spec.center_lat + 0.001
        lon = spec.center_lon - 0.0015

        x_m, y_m = to_local_meters(
            np.array([lat]),
            np.array([lon]),
            spec.center_lat,
            spec.center_lon,
        )
        round_trip_lat, round_trip_lon = local_to_lat_lon(
            float(x_m[0]),
            float(y_m[0]),
            spec.center_lat,
            spec.center_lon,
        )

        self.assertAlmostEqual(round_trip_lat, lat, places=9)
        self.assertAlmostEqual(round_trip_lon, lon, places=9)

    def test_elevation_at_interpolates_local_meter_coordinates(self) -> None:
        spec = MapSpec(name="elevation_test", resolution=17, radius_km=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            asset = TerrainService(cache_root=tmp).build_blank_asset(spec)

        asset.elevation_m = (asset.x_grid_m * 0.1 + asset.y_grid_m * 0.2 + 5.0).astype(
            np.float32
        )
        x_m = float((asset.x_grid_m[0, 2] + asset.x_grid_m[0, 3]) / 2.0)
        y_m = float((asset.y_grid_m[2, 0] + asset.y_grid_m[3, 0]) / 2.0)

        self.assertAlmostEqual(asset.elevation_at(x_m, y_m), x_m * 0.1 + y_m * 0.2 + 5.0)

    def test_download_tiles_uses_raw_tile_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            cached_tile = cache_root / "tiles" / "unit_source" / "3" / "4" / "5.png"
            cached_tile.parent.mkdir(parents=True)
            Image.new("RGB", (TILE_SIZE, TILE_SIZE), (12, 34, 56)).save(cached_tile)

            with patch(
                "dronesim.services.terrain._download_tile",
                side_effect=AssertionError("network should not be used for cached tiles"),
            ):
                mosaic = download_tiles(
                    "https://example.invalid/{z}/{x}/{y}.png",
                    3,
                    4,
                    4,
                    5,
                    5,
                    cache_root=cache_root,
                    source_name="unit_source",
                )

        self.assertEqual(mosaic.size, (TILE_SIZE, TILE_SIZE))
        self.assertEqual(mosaic.getpixel((0, 0)), (12, 34, 56))

    def test_fetch_map_writes_manifest_for_processed_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = MapSpec(name="manifest_test", resolution=16, radius_km=0.1)
            cache_dir = Path(tmp) / spec.key()
            cache_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (20, 24, 32)).save(cache_dir / "satellite.png")
            np.save(cache_dir / "elevation.npy", np.zeros((16, 16), dtype=np.float32))

            asset = TerrainService(cache_root=tmp).fetch_map(spec, fetch_remote=False)
            manifest = json.loads((cache_dir / "map_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(asset.elevation_m.shape, (16, 16))
        self.assertEqual(manifest["spec"]["imagery_source"], "esri_world_imagery")
        self.assertEqual(manifest["spec"]["elevation_source"], "aws_terrarium")
        self.assertEqual(manifest["processed_files"]["satellite"], "satellite.png")


if __name__ == "__main__":
    unittest.main()
