"""
Main API for the eo_qamask library.

This module provides the primary `apply` function for generating quality
assessment masks for various satellite sensors.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import yaml

# --- Import the core logic functions for each mask type ---
# A try/except block makes the API resilient to missing optional dependencies.
try:
    from .bitdefs import QABits
    from .logic.optical import generate_optical_mask
    from .logic.remap import remap_landsat8_qa, remap_sentinel2_scl
    from .logic.sar import generate_sar_mask
except ImportError as e:
    raise ImportError(
        f"Failed to import a core eo_qamask component: {e}. "
        "Please ensure the library is installed correctly."
    ) from e

# --- Set up a logger for professional-grade output ---
logger = logging.getLogger(__name__)


def _infer_sensor_from_filename(image_path: Path) -> str | None:
    """
    A simple utility to guess the sensor from common filename patterns.
    This helps reduce the need for manual user input.
    """
    filename = image_path.name.lower()
    
    # FIX: Use regex for more precise matching.
    sensor_patterns = {
        'landsat8': [r'l[cot]08', r'landsat.*8'],
        'sentinel2': [r's2[abc]_', r'sentinel.*2'],
        'awifs': [r'awifs'],
        'liss3': [r'liss3'],
        'liss4': [r'liss4'],
        'sar': [r'sar', r's1[ab]_'],
    }

    for sensor, patterns in sensor_patterns.items():
        if any(re.search(pattern, filename) for pattern in patterns):
            return sensor
    return None


def _apply_seasonal_adjustments(config: dict) -> dict:
    """
    Adjusts QA thresholds based on the current month to account for
    seasonal variations, as specified in the QA Threshold Blueprint.
    """
    current_month = datetime.now().month
    thresholds = config.get('thresholds', {})
    if not thresholds:
        return config # Do nothing if no thresholds are defined

    # Winter adjustments (November - February)
    if current_month in [11, 12, 1, 2]:
        if 'cloud_brightness' in thresholds:
            thresholds['cloud_brightness'] -= 0.02
        if 'pollution_db_ratio' in thresholds:
            thresholds['pollution_db_ratio'] += 0.02
        logger.info("Applied WINTER seasonal adjustments to thresholds.")

    # Monsoon adjustments (July - September)
    elif current_month in [7, 8, 9]:
        if 'cloud_brightness' in thresholds:
            thresholds['cloud_brightness'] += 0.02
        if 'haze_dark_object' in thresholds:
            thresholds['haze_dark_object'] += 0.005
        logger.info("Applied MONSOON seasonal adjustments to thresholds.")

    config['thresholds'] = thresholds
    return config


def apply(
    image_path: str | Path,
    configs_dir: str | Path,
    output_path: str | Path | None = None,
    sensor: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, dict]:
    """
    The main API function that applies QA masking to a satellite image.

    It features auto-detection of sensors, dynamic loading of configurations,
    seasonal adjustments to thresholds, and the ability to handle runtime
    parameter overrides.

    Args:
        image_path: Path to the input GeoTIFF image.
        configs_dir: Path to the directory containing sensor YAML config files.
        output_path: If provided, saves the generated QA mask to this location.
        sensor: The name of the sensor (e.g., 'landsat8'). If None, the function
                will attempt to auto-detect it from the filename.
        **kwargs: Optional runtime parameter overrides, such as 'cloud_height=2500'.

    Returns:
        A tuple containing:
        - The QA mask as a NumPy ndarray.
        - The original image's rasterio metadata profile.
    """
    image_path = Path(image_path)
    configs_dir = Path(configs_dir)

    # FIX: Add input validation at the beginning.
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not configs_dir.is_dir():
        raise NotADirectoryError(f"Config directory not found or not a directory: {configs_dir}")

    # --- Step 1: Determine the Sensor ---
    if not sensor:
        sensor = _infer_sensor_from_filename(image_path)
        if not sensor:
            raise ValueError(
                "Could not auto-detect sensor from filename. "
                "Please specify the 'sensor' argument."
            )
        logger.info(f"Auto-detected sensor: {sensor}")

    # --- Step 2: Load the Sensor-Specific Configuration ---
    config_path = configs_dir / f"{sensor.lower()}_config.yaml"
    try:
        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config file: {config_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find a config file for '{sensor}' at: {config_path}"
        ) from None
    except yaml.YAMLError as e:
        raise IOError(f"Error parsing YAML config file {config_path}: {e}") from e

    # --- Step 3: Apply Seasonal Adjustments and Runtime Overrides ---
    config = _apply_seasonal_adjustments(config)

    if 'cloud_height' in kwargs:
        config.setdefault('sensor_params', {})['cloud_height'] = kwargs['cloud_height']
        logger.info(f"Runtime override: Set cloud_height to {kwargs['cloud_height']}")

    # --- Step 4: Read Image Data and Metadata ---
    with rasterio.open(image_path) as src:
        # FIX: Add memory efficiency warning for large images.
        if src.width * src.height > 100_000_000:  # ~100MP threshold
            logger.warning("Large image detected. Consider processing in blocks for memory efficiency.")
        data = src.read()
        profile = src.profile

    # --- Step 5: Determine Which Bit Schema to Use ---
    if 'bit_positions' in config:
        bit_map = {key.upper(): int(value) for key, value in config['bit_positions'].items()}
        logger.info("Using custom bit positions defined in the config file.")
    else:
        bit_map = {bit.name: bit.value for bit in QABits}
        logger.info("Using default QABit definitions.")

    # --- Step 6: Select and Execute the Correct Masking Logic ---
    logic_map = {
        'optical_rules': generate_optical_mask,
        'remap_qa': remap_landsat8_qa,
        'remap_scl': remap_sentinel2_scl,
        'sar_rules': generate_sar_mask,
    }

    logic_type = config.get('logic_type')
    if not logic_type or logic_type not in logic_map:
        raise ValueError(f"Unknown or missing 'logic_type' in config: {config_path}")

    logic_func = logic_map[logic_type]
    logger.info(f"Using '{logic_type}' logic for {config.get('sensor', sensor)}")
    mask = logic_func(data, config, bit_map)

    # --- Step 7: Save the Output Mask if a Path is Provided ---
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving QA mask to: {output_path}")
        
        output_profile = profile.copy()
        output_profile.update(
            dtype=rasterio.uint8, count=1, compress='lzw', nodata=None
        )

        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(mask.astype(rasterio.uint8), 1)

    return mask, profile
