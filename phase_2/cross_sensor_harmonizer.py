"""
Phase-2 cross-sensor harmonisation.

This module handles the co-registration, pyramid level creation, and
(stubbed) normalization across different sensor outputs from Phase 1.

"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import rasterio
from rasterio.warp import transform_bounds

# Set up a logger for this module
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class CrossSensorHarmonizer:
    """
    Handles cross-sensor harmonization and multi-resolution pyramid generation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.common_crs = 'EPSG:32644'  # Fixed CRS for India
        self.wgs84_crs = 'EPSG:4326'   # Standard for GeoJSON
        
        self.pyramid_mapping = {
            'AWiFS':     'P0',
            'LISS3':     'P1',
            'LISS4':     'P2',
            'Landsat8':  'P3',
            'Sentinel2': 'P4',
            'SAR':       'P5',
        }

    def _run_cmd(self, cmd: List[str]):
        """Utility: run subprocess and check result, with improved logging."""
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logging.error(f"Command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
            raise RuntimeError(f"Subprocess failed: {cmd}")
        return result

    def step9_reproject_coregister(self, sensor_tiles_dict: Dict[str, List[str]], output_path: str) -> Dict[str, List[str]]:
        """Reprojects and co-registers all sensor tiles to a common grid."""
        logging.info("Step 9: Reprojecting and co-registering all sensor tiles")
        coregistered_tiles = {}

        for sensor, tile_list in sensor_tiles_dict.items():
            if not tile_list:
                logging.warning(f"No tiles to process for sensor {sensor} in Step 9.")
                continue

            sensor_output_dir = Path(output_path) / sensor
            sensor_output_dir.mkdir(parents=True, exist_ok=True)
            coregistered_tiles[sensor] = []

            for tile_path in tile_list:
                output_tile = sensor_output_dir / f"coregistered_{Path(tile_path).name}"
                
                # FIX: Short-circuit gdalwarp if tile is already in the target CRS
                with rasterio.open(tile_path) as src:
                    if str(src.crs) == self.common_crs:
                        logging.info(f"  - Tile {Path(tile_path).name} is already in target CRS. Copying.")
                        shutil.copy(tile_path, output_tile)
                        coregistered_tiles[sensor].append(str(output_tile))
                        continue

                # If not in the correct CRS, reproject it
                warp_cmd = [
                    'gdalwarp', '-overwrite',
                    '-t_srs', self.common_crs,
                    '-r', 'cubic',
                    '-co', 'TILED=YES', '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2',
                    tile_path, str(output_tile)
                ]
                self._run_cmd(warp_cmd)

                if self._validate_geometric_accuracy(str(output_tile)):
                    coregistered_tiles[sensor].append(str(output_tile))
                else:
                    logging.warning(f"Geometric accuracy validation failed for {output_tile}")

        return coregistered_tiles

    def step10_create_pyramid_levels(self, coregistered_tiles: Dict[str, List[str]], output_path: str) -> Dict[str, List[str]]:
        """Copies tiles into resolution-specific folders based on pyramid mapping."""
        logging.info("Step 10: Creating pyramid levels and assigning sensors")
        pyramid_levels = {}

        for sensor, tile_list in coregistered_tiles.items():
            level = self.pyramid_mapping.get(sensor)
            if level is None:
                logging.warning(f"Sensor '{sensor}' not found in pyramid_mapping. Skipping.")
                continue
            
            level_dir = Path(output_path) / level
            level_dir.mkdir(parents=True, exist_ok=True)
            if level not in pyramid_levels:
                pyramid_levels[level] = []

            for tile_path in tile_list:
                dst_path = level_dir / f"{level}_{Path(tile_path).name}"
                shutil.copy(tile_path, dst_path)
                pyramid_levels[level].append(str(dst_path))

        return pyramid_levels

    def step11_normalize_reflectance(self, pyramid_levels: Dict[str, List[str]], output_path: str) -> Dict[str, List[str]]:
        """Placeholder for cross-sensor radiometric normalization."""
        logging.info("Step 11: Normalizing reflectance across sensors (stubbed)")
        normalized_tiles = {}
        pifs = self._identify_pseudo_invariant_features(pyramid_levels)

        for level, tile_list in pyramid_levels.items():
            level_output_dir = Path(output_path) / level
            level_output_dir.mkdir(parents=True, exist_ok=True)
            normalized_tiles[level] = []

            for tile_path in tile_list:
                normalized_tile = self._apply_radiometric_normalization(tile_path, pifs, str(level_output_dir))
                normalized_tiles[level].append(normalized_tile)

        return normalized_tiles

    def step12_generate_multires_tiles(self, normalized_tiles: Dict[str, List[str]], output_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Generates final tiles with metadata and creates a STAC catalog."""
        logging.info("Step 12: Generating multi-resolution tiles with metadata")
        final_pyramid = {}

        for level, tile_list in normalized_tiles.items():
            level_output_dir = Path(output_path) / level
            level_output_dir.mkdir(parents=True, exist_ok=True)
            final_pyramid[level] = []

            for tile_path in tile_list:
                tile_info = self._create_final_tile_with_metadata(tile_path, level)
                final_pyramid[level].append(tile_info)

        self._create_stac_catalog(final_pyramid, output_path)
        return final_pyramid

    def run_phase2_pipeline(self, phase1_results: Dict[str, List[str]], output_base_path: str) -> Dict[str, Any]:
        """Orchestrates the complete Phase 2 pipeline."""
        logging.info("=== Starting Phase 2 Cross-Sensor Harmonization ===")
        steps_output = {step: Path(output_base_path) / f"step{step}" for step in range(9, 13)}
        for step_dir in steps_output.values():
            step_dir.mkdir(parents=True, exist_ok=True)

        try:
            coreg = self.step9_reproject_coregister(phase1_results, str(steps_output[9]))
            pyramids = self.step10_create_pyramid_levels(coreg, str(steps_output[10]))
            normalized = self.step11_normalize_reflectance(pyramids, str(steps_output[11]))
            final = self.step12_generate_multires_tiles(normalized, str(steps_output[12]))
            logging.info(f"✓ Phase 2 completed successfully with pyramid levels: {list(final.keys())}")
            return final
        except Exception as e:
            logging.error(f"✗ Phase 2 failed: {e}", exc_info=True)
            raise

    # ---------------- Internal Stubs & Helpers ----------------

    def _validate_geometric_accuracy(self, tile_path: str, reference_dem_path: str | None = None) -> bool:
        """
        FIX: High-fidelity stub for geometric accuracy validation.
        A real implementation would check RMSE against a reference DEM.
        """
        if reference_dem_path is None:
            logging.debug(f"No reference DEM provided for {tile_path}. Skipping geometric validation.")
            return True
        
        logging.info(f"Validating {tile_path} against {reference_dem_path} (stubbed).")
        # Placeholder logic:
        # 1. Use gdal.Info(tile_path, format='json') to get corner coordinates.
        # 2. Use gdal.Info(reference_dem_path, format='json') for the same.
        # 3. Calculate the shift and RMSE between the two.
        # 4. Return True if rmse < self.config.get('max_rmse_threshold', 1.0)
        return True

    def _identify_pseudo_invariant_features(self, pyramid_levels: Dict[str, List[str]]) -> Dict[str, Any]:
        """Stub: A real implementation would use SPARCS, Orfeo, or ML models."""
        return {'pifs_model': 'placeholder_v1'}

    def _apply_radiometric_normalization(self, tile_path: str, pifs: Dict, output_dir: str) -> str:
        """Stub: A real implementation would apply histogram matching or MAD transformation."""
        out_path = Path(output_dir) / f"normalized_{Path(tile_path).name}"
        shutil.copy(tile_path, out_path)
        return str(out_path)

    def _create_final_tile_with_metadata(self, tile_path: str, level: str) -> Dict[str, Any]:
        """Creates tile metadata, including a valid GeoJSON geometry and bbox."""
        with rasterio.open(tile_path) as src:
            # Get bounds and reproject to WGS84 for GeoJSON
            bounds_wgs84 = transform_bounds(src.crs, self.wgs84_crs, *src.bounds)
            left, bottom, right, top = bounds_wgs84
            
            # FIX: Create a valid GeoJSON Polygon geometry
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [left, bottom],
                    [right, bottom],
                    [right, top],
                    [left, top],
                    [left, bottom]  # Close the ring
                ]]
            }
            bbox = [left, bottom, right, top]

        return {
            'path': tile_path,
            'level': level,
            'geometry': geometry,
            'bbox': bbox,
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'pyramid_level': level,
                'sensor': self._extract_sensor_from_path(tile_path),
                'tile_name': Path(tile_path).name
            }
        }

    def _create_stac_catalog(self, pyramid: Dict[str, List[Dict]], output_path: str):
        """Creates a more compliant STAC catalog for the output pyramid."""
        logging.info("Creating STAC catalog for pyramid")
        catalog = {
            'type': 'Catalog',
            'id': 'harmonized-india-pyramid',
            'stac_version': '1.0.0',
            'description': 'Multi-resolution STAC pyramid of harmonized Indian EO data',
            'links': [],
            'collections': []
        }

        for level, tiles in pyramid.items():
            collection = {
                'type': 'Collection',
                'id': f'pyramid-{level.lower()}',
                'description': f'Level {level} tiles',
                'license': 'proprietary',
                'extent': {
                    'spatial': {'bbox': [[68.0, 6.0, 98.0, 38.0]]},
                    'temporal': {'interval': [['2023-01-01T00:00:00Z', None]]}
                },
                'links': [],
                'items': []
            }

            for tile in tiles:
                # FIX: Populate geometry and bbox from tile metadata
                item = {
                    'type': 'Feature',
                    'stac_version': '1.0.0',
                    'id': tile['metadata']['tile_name'],
                    'geometry': tile.get('geometry'),
                    'bbox': tile.get('bbox'),
                    'properties': tile['metadata'],
                    'assets': {
                        'cog': {
                            'href': tile['path'],
                            'type': 'image/tiff; application=geotiff; profile=cloud-optimized'
                        }
                    },
                    'collection': collection['id']
                }
                collection['items'].append(item)
            catalog['collections'].append(collection)

        catalog_path = Path(output_path) / 'stac_catalog.json'
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        logging.info(f"STAC catalog saved to {catalog_path}")

    def _extract_sensor_from_path(self, path: str) -> str:
        """Extracts sensor name from a filepath, case-insensitively."""
        path_lower = Path(path).name.lower()
        for sensor_key in self.pyramid_mapping.keys():
            if sensor_key.lower() in path_lower:
                return sensor_key
        return 'Unknown'
