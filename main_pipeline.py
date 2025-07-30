"""
Complete Satellite Data Harmonization Pipeline
Final production-ready version incorporating all audit feedback.
"""
from __future__ import annotations

import os
import json
import argparse
import time
import logging
import warnings
import gc
from pathlib import Path
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from typing import Any

# Import from the phase modules
# Ensure these modules are in the correct path (e.g., in a 'phase_1' and 'phase_2' directory)
from phase_1.sensor_preprocessor import SensorPreprocessor
from phase_2.cross_sensor_harmonizer import CrossSensorHarmonizer

# ------------------------------------------------------------------
# Early GDAL CLI tools check & Environment Setup
# ------------------------------------------------------------------
try:
    from shutil import which
except ImportError:
    from distutils.spawn import find_executable as which

if os.name == "nt" and which("gdalinfo") is None:
    warnings.warn(
        "GDAL command-line tools not found on PATH. "
        "Phase-1 tiling will fall back to the pure-Python implementation.",
        RuntimeWarning,
        stacklevel=2,
    )

# Set GDAL environment variables for consistent CRS handling and performance
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


class SatelliteHarmonizationPipeline:
    """
    Main orchestration class for the complete two-phase harmonization pipeline.
    Enhanced with robust error handling, resource monitoring, and timeout protection.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._validate_config()
        self.sensors = self.config.get('sensors', [])

        # Allow dynamic SAR resolution from config
        sar_res = self.config.get("sar_resolution", 25)

        self.sensor_params = {
            'AWiFS': {'resolution': 56, 'tile_size': 1024, 'type': 'optical'},
            'LISS3': {'resolution': 23.5, 'tile_size': 512, 'type': 'optical'},
            'LISS4': {'resolution': 5.8, 'tile_size': 512, 'type': 'optical'},
            'SAR': {'resolution': sar_res, 'tile_size': 1024, 'type': 'sar'},
            'Landsat8': {'resolution': 30, 'tile_size': 512, 'type': 'optical'},
            'Sentinel2': {'resolution': 10, 'tile_size': 512, 'type': 'optical'}
        }

    def _validate_config(self) -> None:
        """Validate critical configuration parameters."""
        if not self.config.get('sensors'):
            raise ValueError("No 'sensors' listed in the configuration file.")

        try:
            import psutil
            logging.info(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
            logging.info(f"CPU cores: {psutil.cpu_count(logical=False)}")
        except ImportError:
            logging.info("psutil not available for system monitoring.")

    def run_phase1_for_sensor(self, sensor: str, input_base_path: str, output_base_path: str) -> tuple[str, list[str] | None]:
        """
        Enhanced Phase 1 runner with comprehensive error handling.
        """
        try:
            params = self.sensor_params.get(sensor, {})
            preprocessor = SensorPreprocessor(sensor, params)
            sensor_input_path = Path(input_base_path) / sensor

            if not sensor_input_path.exists():
                logging.warning(f"Input directory not found for {sensor}, skipping: {sensor_input_path}")
                return sensor, []

            logging.info(f"==================== Starting Phase 1 for {sensor} ====================")
            tiles = preprocessor.run_phase1_pipeline(str(sensor_input_path), output_base_path)

            tile_count = len(tiles) if tiles else 0
            logging.info(f"✓ Phase 1 for {sensor} produced {tile_count} final tiles.")
            return sensor, tiles

        except subprocess.CalledProcessError as e:
            # FIX: Include the failing command in the log for easier debugging.
            stderr = e.stderr.strip() if e.stderr else "No stderr"
            logging.error(f"✗ {sensor}: External command failed – {e.cmd!s} :: {stderr}")
            return sensor, None
        except MemoryError:
            logging.error(f"✗ {sensor}: Out of memory. Consider reducing workers or processing smaller chunks.")
            return sensor, None
        except Exception as e:
            logging.error(f"✗ Phase 1 pipeline failed for {sensor}: {type(e).__name__}", exc_info=True)
            return sensor, None

    def run_complete_pipeline(self, input_base_path: str, output_base_path: str) -> dict[str, Any]:
        """
        Execute the complete two-phase harmonization pipeline with enhanced monitoring.
        """
        start_time = time.time()
        logging.info("=" * 60)
        logging.info("Starting Complete Satellite Data Harmonization Pipeline")
        logging.info("=" * 60)

        Path(output_base_path).mkdir(parents=True, exist_ok=True)
        phase1_results: dict[str, list[str] | None] = {}

        # --- PHASE 1: Individual sensor preprocessing ---
        if self.config.get('enable_phase_1', True):
            logging.info("--- Running Phase 1: Per-Sensor Preprocessing ---")
            use_parallel = self.config.get('parallel_processing', False)
            max_workers = self.config.get('max_workers', 4)

            if use_parallel and self.sensors:
                logging.info(f"Running in PARALLEL mode with up to {max_workers} workers.")
                phase1_output_path = str(Path(output_base_path) / 'phase1')
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self.run_phase1_for_sensor, sensor, input_base_path, phase1_output_path): sensor
                        for sensor in self.sensors
                    }
                    try:
                        # FIX: Simplified timeout handling with a single global timeout.
                        for future in as_completed(futures, timeout=7200):
                            sensor_name = futures[future]
                            try:
                                # FIX: Removed redundant inner timeout.
                                _, tiles = future.result()
                                phase1_results[sensor_name] = tiles
                            except Exception as e:
                                logging.error(f"✗ {sensor_name}: Task generated an exception: {e}")
                                phase1_results[sensor_name] = None
                    # FIX: Catch the correctly namespaced TimeoutError.
                    except concurrent.futures.TimeoutError:
                        logging.error("✗ Global timeout of 2 hours reached for Phase 1 processing.")

            else:
                logging.info("Running in SEQUENTIAL mode.")
                phase1_output_path = str(Path(output_base_path) / 'phase1')
                for sensor in self.sensors:
                    sensor_name, tiles = self.run_phase1_for_sensor(sensor, input_base_path, phase1_output_path)
                    phase1_results[sensor_name] = tiles

            gc.collect()
            logging.info("✓ Phase 1 finished.")

        # --- PHASE 2: Cross-sensor harmonization ---
        final_pyramid = {}
        # FIX: Guard against running Phase 2 if Phase 1 yielded no valid tiles.
        if self.config.get('enable_phase_2', True):
            if not any(phase1_results.values()):
                logging.warning("Skipping Phase 2: Phase 1 produced zero valid tiles for all sensors.")
            else:
                logging.info("--- Running Phase 2: Cross-Sensor Harmonization ---")
                try:
                    valid_phase1_results = {k: v for k, v in phase1_results.items() if v}
                    harmonizer = CrossSensorHarmonizer(self.config)
                    phase2_output_path = str(Path(output_base_path) / 'phase2')
                    final_pyramid = harmonizer.run_phase2_pipeline(valid_phase1_results, phase2_output_path)
                except Exception as e:
                    logging.error(f"✗ Phase 2 failed: {e}", exc_info=True)
        else:
            logging.info("Phase 2 is disabled in the configuration.")


        total_time = time.time() - start_time
        logging.info("=" * 60)
        logging.info(f"✓ PIPELINE FINISHED in {total_time:.2f} seconds.")
        logging.info(f"Final results available in: {output_base_path}")
        logging.info("=" * 60)
        return final_pyramid

def main(argv: list[str] | None = None) -> int:
    """Main function to configure and run the pipeline from the command line."""
    import sys

    parser = argparse.ArgumentParser(description="Satellite Data Harmonization Pipeline")
    parser.add_argument('--input-path', required=True, help="Base path to raw sensor data folders")
    parser.add_argument('--output-path', required=True, help="Base path for all processed outputs")
    parser.add_argument('--config', required=True, help="Path to the main JSON config file")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing (overrides config)")
    parser.add_argument('--max-workers', type=int, help="Max number of parallel processes")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help="Set logging level (overrides config)")
    args = parser.parse_args(argv)

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse configuration file '{args.config}': {e}", file=sys.stderr)
        return 1

    # FIX: CLI log-level argument explicitly overrides the config file setting.
    log_level_str = args.log_level or config.get('logging_level', 'INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(module)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.parallel:
        config['parallel_processing'] = True
    if args.max_workers:
        config['max_workers'] = args.max_workers

    try:
        pipeline = SatelliteHarmonizationPipeline(config)
        pipeline.run_complete_pipeline(args.input_path, args.output_path)
        return 0
    except KeyboardInterrupt:
        logging.error("Pipeline interrupted by user.")
        raise
    except Exception as e:
        logging.critical(f"✗ PIPELINE FAILED with an unhandled exception: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    try:
        # Pass command-line arguments, excluding the script name itself
        exit_code = main(sys.argv[1:])
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # This handles a Ctrl+C during the setup phase (before main loop)
        sys.exit(130)
