# eo_qamask/api.py

import os
import rasterio
import yaml
import numpy as np
import logging

# --- Enhancement #5: Add Logging ---
# Use Python's logging module instead of print() for professional-grade output
logger = logging.getLogger(__name__)

# Import logic functions
from .logic.optical import generate_optical_mask
from .logic.remap import remap_landsat8_qa, remap_sentinel2_scl
from .logic.sar import generate_sar_mask
from .bitdefs import QABits # Keep as a fallback

def _infer_sensor_from_filename(image_path: str) -> str | None:
    """
    A simple utility to guess the sensor from common filename prefixes.
    """
    filename = os.path.basename(image_path).lower()
    if 'l8' in filename or 'lc08' in filename:
        return 'landsat8'
    if 's2a' in filename or 's2b' in filename or 's2c' in filename:
        return 'sentinel2'
    if 'awifs' in filename:
        return 'awifs'
    if 'liss3' in filename:
        return 'liss3'
    # Add other sensor patterns here
    return None

def apply(
    image_path: str,
    configs_dir: str = "configs",
    output_path: str = None,
    sensor: str = None,
) -> (np.ndarray, dict):
    """
    Applies QA masking to an image with auto-detection and dynamic configuration.

    Args:
        image_path (str): Path to the input GeoTIFF image.
        configs_dir (str): Path to the directory containing sensor YAML files.
        output_path (str, optional): If provided, saves the QA mask to this location.
        sensor (str, optional): The sensor name (e.g., 'landsat8'). If None,
                                the function will attempt to auto-detect it.

    Returns:
        tuple: A tuple containing the QA mask (np.ndarray) and the original
               image's rasterio metadata profile.
    """
    # --- Enhancement #1: Auto-detect sensor ---
    if not sensor:
        sensor = _infer_sensor_from_filename(image_path)
        if not sensor:
            raise ValueError(
                "Could not auto-detect sensor from filename. "
                "Please specify the 'sensor' argument."
            )
        logger.info(f"Auto-detected sensor: {sensor}")

    # --- Enhancement #2: Auto-load config per sensor ---
    config_path = os.path.join(configs_dir, f"{sensor.lower()}_config.yaml")

    # --- Enhancement #3: Improved Error Messages ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config file: {config_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[eo_qamask] Could not find a config file for '{sensor}' at: {config_path}"
        )
    except yaml.YAMLError as e:
        raise IOError(f"Error parsing YAML config file {config_path}: {e}")

    with rasterio.open(image_path) as src:
        data = src.read()
        profile = src.profile

    # --- Enhancement #4: Bit-Mask Schema from Config (HIGH VALUE) ---
    # Create a bit mapping from the config file if it exists, otherwise use the defaults.
    if 'bit_positions' in config:
        bit_map = {key.upper(): int(value) for key, value in config['bit_positions'].items()}
        logger.info("Using custom bit positions defined in config file.")
    else:
        # Fallback to the hardcoded Enum values
        bit_map = {bit.name: bit.value for bit in QABits}
        logger.info("Using default QABit definitions.")

    # Dictionary mapping 'logic_type' from YAML to the actual function
    logic_map = {
        'optical_rules': generate_optical_mask,
        'remap_qa': remap_landsat8_qa,
        'remap_scl': remap_sentinel2_scl,
        'sar_rules': generate_sar_mask
    }
    
    logic_type = config.get('logic_type')
    logic_func = logic_map.get(logic_type)

    if not logic_func:
        raise ValueError(f"Unknown or missing 'logic_type' in config: {config_path}")

    logger.info(f"Using '{logic_type}' logic for {config.get('sensor', sensor)}")
    
    # Pass the dynamic bit_map to the logic function
    mask = logic_func(data, config, bit_map)
    
    if output_path:
        logger.info(f"Saving QA mask to: {output_path}")
        output_profile = profile.copy()
        output_profile.update(
            dtype=rasterio.uint8, count=1, compress='lzw', nodata=None
        )

        # --- Enhancement #6: Placeholder for other formats ---
        if output_path.lower().endswith(('.nc', '.netcdf')):
            logger.warning("NetCDF output is not yet implemented.")
            # Add NetCDF writing logic here in the future
        else: # Default to GeoTIFF
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(mask, 1)

    return mask, profile