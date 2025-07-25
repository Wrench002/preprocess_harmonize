import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

class CrossSensorHarmonizer:
    """
    Handles cross-sensor harmonization and multi-resolution pyramid generation
    """

    def __init__(self, config):
        self.config = config
        self.common_crs = 'EPSG:32644'  # Fixed CRS for India
        self.tile_origin = (-180.0, 90.0)  # Fixed tile origin
        
        self.pyramid_mapping = {
            'AWiFS':     'P0',
            'LISS3':     'P1',
            'LISS4':     'P2',
            'LANDSAT8':  'P3',
            'SENTINEL2': 'P4',
            'SAR':       'P5',
        }

    def _run_cmd(self, cmd):
        """Utility: run subprocess and check result"""
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logging.error(f"Command failed: {' '.join(cmd)}\n{result.stderr.decode()}")
            raise RuntimeError(f"Subprocess failed: {cmd}")
        return result

    def step9_reproject_coregister(self, sensor_tiles_dict, output_path):
        logging.info("Step 9: Reprojecting and co-registering all sensor tiles")
        coregistered_tiles = {}

        for sensor, tile_list in sensor_tiles_dict.items():
            sensor_output_dir = os.path.join(output_path, sensor)
            os.makedirs(sensor_output_dir, exist_ok=True)
            coregistered_tiles[sensor] = []

            for tile_path in tile_list:
                output_tile = os.path.join(sensor_output_dir, f"coregistered_{os.path.basename(tile_path)}")
                warp_cmd = [
                    'gdalwarp',
                    '-s_srs', 'EPSG:4326',
                    '-t_srs', self.common_crs,
                    '-r', 'cubic',
                    '-co', 'TILED=YES', '-co', 'COMPRESS=DEFLATE', '-co', 'PREDICTOR=2',
                    tile_path, output_tile
                ]
                self._run_cmd(warp_cmd)

                if self._validate_geometric_accuracy(output_tile):
                    coregistered_tiles[sensor].append(output_tile)
                else:
                    logging.warning(f"Geometric accuracy failed for {output_tile}")

        return coregistered_tiles

    def step10_create_pyramid_levels(self, coregistered_tiles, output_path):
        logging.info("Step 10: Creating pyramid levels and assigning sensors")
        pyramid_levels = {}

        for sensor, tile_list in coregistered_tiles.items():
            level = self.pyramid_mapping.get(sensor.upper(), 'Unknown')
            level_dir = os.path.join(output_path, level)
            os.makedirs(level_dir, exist_ok=True)
            pyramid_levels[level] = []

            for tile_path in tile_list:
                level_tile_path = os.path.join(level_dir, f"{level}_{os.path.basename(tile_path)}")
                self._run_cmd(['cp', tile_path, level_tile_path])
                pyramid_levels[level].append(level_tile_path)

        return pyramid_levels

    def step11_normalize_reflectance(self, pyramid_levels, output_path):
        logging.info("Step 11: Normalizing reflectance across sensors")
        normalized_tiles = {}

        pifs = self._identify_pseudo_invariant_features(pyramid_levels)

        for level, tile_list in pyramid_levels.items():
            level_output_dir = os.path.join(output_path, level)
            os.makedirs(level_output_dir, exist_ok=True)
            normalized_tiles[level] = []

            for tile_path in tile_list:
                normalized_tile = self._apply_radiometric_normalization(tile_path, pifs, level_output_dir)
                normalized_tiles[level].append(normalized_tile)

        return normalized_tiles

    def step12_generate_multires_tiles(self, normalized_tiles, output_path):
        logging.info("Step 12: Generating multi-resolution tiles with metadata")
        final_pyramid = {}

        for level, tile_list in normalized_tiles.items():
            level_output_dir = os.path.join(output_path, level)
            os.makedirs(level_output_dir, exist_ok=True)
            final_pyramid[level] = []

            for tile_path in tile_list:
                tile_info = self._create_final_tile_with_metadata(tile_path, level, level_output_dir)
                final_pyramid[level].append(tile_info)

        self._create_stac_catalog(final_pyramid, output_path)
        return final_pyramid

    def run_phase2_pipeline(self, phase1_results, output_base_path):
        logging.info("=== Starting Phase 2 Cross-Sensor Harmonization ===")
        steps_output = {}
        for step in range(9, 13):
            step_dir = os.path.join(output_base_path, f"step{step}")
            os.makedirs(step_dir, exist_ok=True)
            steps_output[step] = step_dir

        try:
            coreg = self.step9_reproject_coregister(phase1_results, steps_output[9])
            pyramids = self.step10_create_pyramid_levels(coreg, steps_output[10])
            normalized = self.step11_normalize_reflectance(pyramids, steps_output[11])
            final = self.step12_generate_multires_tiles(normalized, steps_output[12])
            logging.info(f"✓ Phase 2 completed successfully with pyramid levels: {list(final.keys())}")
            return final
        except Exception as e:
            logging.error(f"✗ Phase 2 failed: {e}")
            raise

    # ---------------- Internal Stubs & Helpers ----------------

    def _validate_geometric_accuracy(self, tile_path):
        # Stub: Replace with RMSE checker
        return True

    def _identify_pseudo_invariant_features(self, pyramid_levels):
        # Stub: Plug-in SPARCS/Orfeo/ML models later
        return {'example': 'pifs_placeholder'}

    def _apply_radiometric_normalization(self, tile_path, pifs, output_dir):
        # Stub: Implement histogram matching / MAD transformation
        out_path = os.path.join(output_dir, f"normalized_{os.path.basename(tile_path)}")
        self._run_cmd(['cp', tile_path, out_path])
        return out_path

    def _create_final_tile_with_metadata(self, tile_path, level, output_dir):
        # Add more accurate STAC metadata later
        return {
            'path': tile_path,
            'level': level,
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'pyramid_level': level,
                'quality_score': 0,
                'sensor': self._extract_sensor_from_path(tile_path),
                'tile_name': os.path.basename(tile_path)
            }
        }

    def _create_stac_catalog(self, pyramid, output_path):
        logging.info("Creating STAC catalog for pyramid")
        catalog = {
            'type': 'Catalog',
            'id': 'harmonized-india-pyramid',
            'description': 'Multi-resolution STAC pyramid of harmonized Indian EO data',
            'links': [],
            'collections': []
        }

        for level, tiles in pyramid.items():
            collection = {
                'type': 'Collection',
                'id': f'pyramid-{level.lower()}',
                'description': f'Level {level} tiles',
                'extent': {
                    'spatial': {'bbox': [[68.0, 6.0, 98.0, 38.0]]},
                    'temporal': {'interval': [['2023-01-01T00:00:00Z', '2025-12-31T23:59:59Z']]}
                },
                'items': []
            }

            for tile in tiles:
                item = {
                    'type': 'Feature',
                    'id': tile['metadata']['tile_name'],
                    'geometry': None,
                    'properties': tile['metadata'],
                    'assets': {
                        'cog': {
                            'href': tile['path'],
                            'type': 'image/tiff; application=geotiff; profile=cloud-optimized'
                        }
                    }
                }
                collection['items'].append(item)

            catalog['collections'].append(collection)

        catalog_path = os.path.join(output_path, 'stac_catalog.json')
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        logging.info(f"STAC catalog saved to {catalog_path}")

    def _extract_sensor_from_path(self, path):
        for sensor in self.pyramid_mapping.keys():
            if sensor.lower() in path.lower():
                return sensor
        return 'Unknown'
