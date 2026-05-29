from __future__ import annotations

import importlib.util
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
    MapCacheMiss,
    TerrainService,
    bounding_box,
    build_terrain_figure,
    choose_zoom,
    download_tiles,
    encode_cesium_heightmap,
    lat_lon_to_tile,
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
        spec = MapSpec(center_lat=38.944369, center_lon=-74.891081)
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
            self.assertFalse(spec.key().startswith("manifest_test"))
            cache_dir = Path(tmp) / spec.key()
            cache_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16), (20, 24, 32)).save(cache_dir / "satellite.png")
            np.save(cache_dir / "elevation.npy", np.zeros((16, 16), dtype=np.float32))

            asset = TerrainService(cache_root=tmp).fetch_map(spec, fetch_remote=False)
            manifest = json.loads((cache_dir / "map_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(asset.elevation_m.shape, (16, 16))
        self.assertEqual(asset.origin, "processed-cache")
        self.assertEqual(manifest["spec"]["imagery_source"], "esri_world_imagery")
        self.assertEqual(manifest["spec"]["elevation_source"], "aws_terrarium")
        self.assertEqual(manifest["processed_files"]["satellite"], "satellite.png")

    def test_fetch_map_raises_map_cache_miss_when_no_remote_and_no_cache(self) -> None:
        spec = MapSpec(resolution=16, radius_km=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            service = TerrainService(cache_root=tmp)
            with patch(
                "dronesim.services.terrain._download_tile",
                side_effect=AssertionError("network must not be hit when fetch_remote=False"),
            ):
                with self.assertRaises(MapCacheMiss) as ctx:
                    service.fetch_map(spec, fetch_remote=False)

        self.assertEqual(ctx.exception.spec, spec)
        self.assertGreater(ctx.exception.missing_tiles, 0)
        self.assertEqual(ctx.exception.cache_dir.name, spec.key())
        self.assertIn(spec.key(), str(ctx.exception))

    def test_fetch_map_rebuilds_from_raw_tile_cache(self) -> None:
        spec = MapSpec(resolution=16, radius_km=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            service = TerrainService(cache_root=cache_root)

            bounds = bounding_box(spec.center_lat, spec.center_lon, spec.radius_km)
            zoom = choose_zoom(spec.radius_km, spec.resolution)
            lat_min, lon_min, lat_max, lon_max = bounds
            tx_min, ty_min = lat_lon_to_tile(lat_max, lon_min, zoom)
            tx_max, ty_max = lat_lon_to_tile(lat_min, lon_max, zoom)

            for source in (spec.imagery_source, spec.elevation_source):
                for ty in range(ty_min, ty_max + 1):
                    for tx in range(tx_min, tx_max + 1):
                        tile_path = (
                            cache_root
                            / "tiles"
                            / source
                            / str(zoom)
                            / str(tx)
                            / f"{ty}.png"
                        )
                        tile_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new(
                            "RGB", (TILE_SIZE, TILE_SIZE), (90, 120, 60)
                        ).save(tile_path)

            with patch(
                "dronesim.services.terrain._download_tile",
                side_effect=AssertionError("network must not be hit when raw tiles cached"),
            ):
                asset = service.fetch_map(spec, fetch_remote=False)

            cache_dir = cache_root / spec.key()
            self.assertEqual(asset.origin, "raw-tile-rebuild")
            self.assertEqual(asset.satellite.size, (16, 16))
            self.assertNotEqual(asset.satellite.getpixel((0, 0)), (20, 24, 32))
            self.assertTrue((cache_dir / "satellite.png").exists())
            self.assertTrue((cache_dir / "elevation.npy").exists())
            self.assertTrue((cache_dir / "map_manifest.json").exists())

    @unittest.skipUnless(
        importlib.util.find_spec("plotly") is not None,
        "plotly is optional; build_terrain_figure is legacy and unused by the web frontend",
    )
    def test_build_terrain_figure_emits_meters_on_all_axes(self) -> None:
        spec = MapSpec(name="units_test", resolution=24, radius_km=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            asset = TerrainService(cache_root=tmp).build_blank_asset(spec)
        asset.elevation_m = np.linspace(
            0.0, 250.0, asset.elevation_m.size, dtype=np.float32
        ).reshape(asset.elevation_m.shape)

        fig = build_terrain_figure(asset)
        surface = fig.data[0]

        # x/y should be in meters (not km): plotly extents must match the
        # underlying grid_m extents within float tolerance.
        np.testing.assert_allclose(
            np.asarray(surface.x), asset.x_grid_m, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(surface.y), asset.y_grid_m, rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(surface.z), asset.z_grid_m, rtol=0, atol=1e-6
        )

        # Horizontal extent for radius=0.5 km should be ~1000 m across,
        # which is the regression check against the previous /1000 bug.
        x_extent = float(np.ptp(np.asarray(surface.x)))
        self.assertGreater(x_extent, 800.0)
        self.assertLess(x_extent, 1200.0)

        scene = fig.layout.scene
        self.assertEqual(scene.aspectmode, "data")
        self.assertEqual(scene.xaxis.title.text, "East (m)")
        self.assertEqual(scene.yaxis.title.text, "North (m)")

    def test_migrate_legacy_cache_dir_renames_to_spatial_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            legacy = cache_root / "Old Name_37p618806_m122p375417_r1p000_n400"
            legacy.mkdir(parents=True)
            (legacy / "satellite.png").write_bytes(b"placeholder")

            TerrainService(cache_root=cache_root)

            new = cache_root / "37p618806_m122p375417_r1p000_n400"
            self.assertFalse(legacy.exists())
            self.assertTrue(new.exists())
            self.assertTrue((new / "satellite.png").exists())

    def test_migrate_legacy_cache_dir_uses_manifest_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp)
            legacy = cache_root / "Scenario With Spaces_oldformat"
            legacy.mkdir(parents=True)
            spec = MapSpec(
                name="Scenario With Spaces",
                center_lat=38.944369,
                center_lon=-74.891081,
                radius_km=2.5,
                resolution=200,
            )
            manifest_payload = {
                "spec": {
                    "center_lat": spec.center_lat,
                    "center_lon": spec.center_lon,
                    "radius_km": spec.radius_km,
                    "resolution": spec.resolution,
                    "name": spec.name,
                    "imagery_source": spec.imagery_source,
                    "elevation_source": spec.elevation_source,
                },
            }
            (legacy / "map_manifest.json").write_text(
                json.dumps(manifest_payload), encoding="utf-8"
            )

            TerrainService(cache_root=cache_root)

            new = cache_root / spec.key()
            self.assertFalse(legacy.exists())
            self.assertTrue(new.exists())
            self.assertTrue((new / "map_manifest.json").exists())


class EncodeCesiumHeightmapTest(unittest.TestCase):
    def test_preserves_shape_and_round_trips(self) -> None:
        elev = np.linspace(10.0, 110.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        hm = encode_cesium_heightmap(elev)

        self.assertEqual(hm["width"], 16)
        self.assertEqual(hm["height"], 16)
        self.assertEqual(len(hm["buffer"]), 16 * 16 * 2)

        buf = np.frombuffer(hm["buffer"], dtype=np.uint16).reshape(16, 16)
        decoded = buf.astype(np.float64) * hm["height_scale"] + hm["height_offset"]
        np.testing.assert_allclose(decoded, elev, atol=hm["height_scale"] * 0.6)

    def test_vertical_exaggeration(self) -> None:
        elev = np.full((8, 8), 50.0, dtype=np.float32)
        hm = encode_cesium_heightmap(elev, vertical_exaggeration=2.0)
        buf = np.frombuffer(hm["buffer"], dtype=np.uint16).reshape(8, 8)
        decoded = buf[0, 0] * hm["height_scale"] + hm["height_offset"]
        self.assertAlmostEqual(decoded, 100.0, delta=hm["height_scale"])

    def test_flat_terrain_guard(self) -> None:
        elev = np.zeros((4, 4), dtype=np.float32)
        hm = encode_cesium_heightmap(elev)
        self.assertGreater(hm["height_scale"], 0.0)
        buf = np.frombuffer(hm["buffer"], dtype=np.uint16)
        self.assertTrue(buf.max() <= 65535)

    def test_downsamples_when_exceeding_max_dim(self) -> None:
        elev = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
        hm = encode_cesium_heightmap(elev, max_dim=512)
        self.assertLessEqual(hm["width"], 512)
        self.assertLessEqual(hm["height"], 512)
        self.assertEqual(len(hm["buffer"]), hm["width"] * hm["height"] * 2)


if __name__ == "__main__":
    unittest.main()
