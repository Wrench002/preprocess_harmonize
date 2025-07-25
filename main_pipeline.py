# main_pipeline.py
import os
import json
import argparse
import time
import logging # Improvement: Import logging module
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import the main classes for each phase
from phase_1.sensor_preprocessor import SensorPreprocessor
from phase_2.cross_sensor_harmonizer import CrossSensorHarmonizer

class SatelliteHarmonizationPipeline:
    """
    Main orchestration class for the complete two-phase harmonization pipeline.
    """
    def __init__(self, config):
        self.config = config
        self.sensors = self.config.get('sensors', [])
        if not self.sensors:
            raise ValueError("No sensors listed in the configuration file.")

    def run_phase1_for_sensor(self, sensor, input_base_path, output_base_path):
        """A helper function to run Phase 1 for a single sensor."""
        try:
            preprocessor = SensorPreprocessor(sensor, self.config)
            sensor_input_path = os.path.join(input_base_path, sensor)
            
            if not os.path.exists(sensor_input_path):
                logging.warning(f"Input directory not found for {sensor}, skipping.")
                return sensor, []

            tiles = preprocessor.run_phase1_pipeline(sensor_input_path, output_base_path)
            logging.info(f"✓ {sensor} processing completed successfully.")
            return sensor, tiles
        except Exception as e:
            logging.error(f"✗ Phase 1 pipeline failed for {sensor}.", exc_info=True)
            # exc_info=True will log the full traceback for easier debugging
            return sensor, None # Return None on failure

    def run_complete_pipeline(self, input_base_path, output_base_path):
        """Executes the complete two-phase harmonization pipeline."""
        logging.info("=" * 60)
        logging.info("Starting Complete Satellite Data Harmonization Pipeline")
        logging.info("=" * 60)
        
        # --- PHASE 1: Individual sensor preprocessing ---
        phase1_results = {}
        if self.config.get('enable_phase_1', True):
            logging.info("--- Running Phase 1: Per-Sensor Preprocessing ---")
            
            use_parallel = self.config.get('parallel_processing', False)
            # Improvement: Control max number of workers
            max_workers = self.config.get('max_workers') # None means use default

            if use_parallel:
                logging.info(f"Running in PARALLEL mode with up to {max_workers or 'default'} workers.")
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    phase1_output_path = os.path.join(output_base_path, 'phase1')
                    futures = {executor.submit(self.run_phase1_for_sensor, sensor, input_base_path, phase1_output_path): sensor for sensor in self.sensors}
                    for future in as_completed(futures):
                        sensor_name, tiles = future.result()
                        if tiles is not None:
                            phase1_results[sensor_name] = tiles
            else:
                logging.info("Running in SEQUENTIAL mode.")
                phase1_output_path = os.path.join(output_base_path, 'phase1')
                for sensor in self.sensors:
                    sensor_name, tiles = self.run_phase1_for_sensor(sensor, input_base_path, phase1_output_path)
                    if tiles is not None:
                        phase1_results[sensor_name] = tiles
            
            logging.info("✓ Phase 1 finished.")

        # --- PHASE 2: Cross-sensor harmonization ---
        final_pyramid = {}
        if self.config.get('enable_phase_2', True):
            logging.info("--- Running Phase 2: Cross-Sensor Harmonization ---")
            harmonizer = CrossSensorHarmonizer(self.config)
            phase2_output_path = os.path.join(output_base_path, 'phase2')
            final_pyramid = harmonizer.run_phase2_pipeline(phase1_results, phase2_output_path)
        
        return final_pyramid

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # --- Improvement: Set up logging based on the config file ---
    log_level = config.get('logging_level', 'INFO').upper()
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.parallel:
        config['parallel_processing'] = True
    # Allow overriding max_workers from command line
    if args.max_workers:
        config['max_workers'] = args.max_workers

    pipeline = SatelliteHarmonizationPipeline(config)
    start_time = time.time()

    try:
        final_pyramid = pipeline.run_complete_pipeline(args.input_path, args.output_path)
        total_time = time.time() - start_time
        
        logging.info("=" * 60)
        logging.info(f"✓ PIPELINE FINISHED SUCCESSFULLY in {total_time:.2f} seconds.")
        logging.info(f"Final results available in: {args.output_path}")
        if final_pyramid:
            logging.info("Pyramid Statistics:")
            for level, tiles in sorted(final_pyramid.items()):
                logging.info(f"  - Level {level}: {len(tiles)} tiles")
        logging.info("=" * 60)
        return 0
    except Exception as e:
        total_time = time.time() - start_time
        logging.critical("!" * 60, exc_info=True)
        logging.critical(f"✗ PIPELINE FAILED after {total_time:.2f} seconds.")
        logging.critical(f"An unrecoverable error occurred: {e}")
        logging.critical("!" * 60)
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite Data Harmonization Pipeline")
    parser.add_argument('--input-path', required=True, help="Base path to raw sensor data folders")
    parser.add_argument('--output-path', required=True, help="Base path for all processed outputs")
    parser.add_argument('--config', required=True, help="Path to the main JSON config file")
    parser.add_argument('--parallel', action='store_true', help="Enable parallel processing (overrides config)")
    # Improvement: Add CLI argument for max_workers
    parser.add_argument('--max-workers', type=int, help="Max number of parallel processes (overrides config)")
    
    exit(main(parser.parse_args()))