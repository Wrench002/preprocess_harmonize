"""
Core logic for generating quality assessment masks for SAR sensors.

This module contains rule-based algorithms for detecting common SAR-specific
features and artifacts, including speckle, radar shadow, and water bodies.
"""
from __future__ import annotations

import logging
import numpy as np
from scipy import ndimage

# --- Set up a module logger ---
logger = logging.getLogger(__name__)

def generate_sar_mask(
    data: np.ndarray,
    config: dict,
    bit_map: dict
) -> np.ndarray:
    """
    Generates a QA bit-mask for SAR (Synthetic Aperture Radar) sensors.

    Applies several rule-based algorithms to detect SAR artifacts and features.
    Assumes the input data is calibrated backscatter, preferably in decibels (dB).

    Args:
        data: SAR data cube (bands, y, x)
        config: Sensor-specific config dictionary
        bit_map: Mapping of QA condition names to bit positions

    Returns:
        An 8-bit integer mask (same x/y shape as image).
    """
    mask = np.zeros(data[0].shape, dtype=np.uint8)
    bands = config.get('bands', {})
    thresholds = config.get('thresholds', {})

    # Step 1: Identify primary SAR band (usually VV or band 0)
    idx = bands.get('vv', 0)
    try:
        primary_band = data[idx].astype(np.float32)
        logger.info(f"Using band index {idx} as the primary SAR band (VV).")
    except Exception:
        logger.error(f"Primary SAR band index {idx} is out of bounds. Skipping mask creation.")
        return mask

    # Step 2: Detection algorithms

    # -- Speckle detection: local coefficient of variation (CV) --
    if 'speckle_cv' in thresholds and 'SPECKLE' in bit_map:
        logger.info("Applying speckle detection...")
        window_size = 5
        mean = ndimage.uniform_filter(primary_band, size=window_size)
        sq_mean = ndimage.uniform_filter(primary_band**2, size=window_size)
        variance = np.maximum(sq_mean - mean**2, 0)
        std_dev = np.sqrt(variance)

        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.zeros_like(primary_band, dtype=np.float32)
            nonzero_mean = mean != 0
            cv[nonzero_mean] = std_dev[nonzero_mean] / mean[nonzero_mean]
        speckle_cond = (cv > thresholds['speckle_cv'])
        mask |= (speckle_cond.astype(np.uint8) << bit_map['SPECKLE'])

    # -- Radar shadow (low backscatter regions) --
    if 'sar_shadow_db' in thresholds and 'SHADOW' in bit_map:
        logger.info("Applying radar shadow detection...")
        shadow_cond = (primary_band < thresholds['sar_shadow_db'])
        mask |= (shadow_cond.astype(np.uint8) << bit_map['SHADOW'])

    # -- Surface water (very low backscatter) --
    if 'sar_water_db' in thresholds and 'WATER' in bit_map:
        logger.info("Applying surface water detection...")
        water_cond = (primary_band < thresholds['sar_water_db'])
        mask |= (water_cond.astype(np.uint8) << bit_map['WATER'])

    # Step 3: No-data bit
    nodata_val = config.get('globals', {}).get('nodata_value')
    if nodata_val is not None and 'NO_DATA' in bit_map:
        # Use np.isclose for float nodata
        nodata_cond = np.isclose(primary_band, nodata_val)
        mask |= (nodata_cond.astype(np.uint8) << bit_map['NO_DATA'])

    # (Optional) Layover/foreshortening detection would go here.

    logger.info("Successfully generated SAR QA mask.")
    return mask
