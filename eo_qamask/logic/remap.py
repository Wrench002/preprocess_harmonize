"""
Core logic for remapping official sensor QA bands to the eo_qamask format.

This module provides functions to translate the specific quality assessment
(QA) bit-bands from sensors like Landsat 8 (QA_PIXEL) and Sentinel-2 (SCL)
into the standardized bitmask used by this library.
"""
from __future__ import annotations

import logging
import numpy as np

# --- Set up a logger for this module ---
logger = logging.getLogger(__name__)

def remap_landsat8_qa(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Translates the official Landsat 8 QA_PIXEL band into the standardized
    eo_qamask format by mapping specific bit flags.
    Reference: https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands
    """
    logger.info("Applying Landsat 8 QA_PIXEL band remapping logic...")

    try:
        qa_band_index = config['bands']['qa_pixel']
        qa_band = data[qa_band_index]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Config for Landsat 8 is missing 'qa_pixel' band index or index is out of bounds: {e}")
        return np.zeros(data[0].shape, dtype=np.uint8)

    mask = np.zeros(qa_band.shape, dtype=np.uint8)

    # Official Landsat 8 QA_PIXEL Bit Definitions (bit positions 0-based)
    L8_DILATED_CLOUD_BIT = 1
    L8_CIRRUS_BIT = 2
    L8_CLOUD_BIT = 3
    L8_CLOUD_SHADOW_BIT = 4
    L8_SNOW_BIT = 5
    L8_WATER_BIT = 7

    # CLOUD: Any of the cloud, cirrus, or dilated_cloud bits set
    if 'CLOUD' in bit_map:
        cloud_condition = (
            ((qa_band >> L8_DILATED_CLOUD_BIT) & 1) |
            ((qa_band >> L8_CIRRUS_BIT) & 1) |
            ((qa_band >> L8_CLOUD_BIT) & 1)
        ).astype(bool)
        mask |= (cloud_condition.astype(np.uint8) << bit_map['CLOUD'])

    # SHADOW: Bit 4
    if 'SHADOW' in bit_map:
        shadow_condition = ((qa_band >> L8_CLOUD_SHADOW_BIT) & 1).astype(bool)
        mask |= (shadow_condition.astype(np.uint8) << bit_map['SHADOW'])

    # SNOW_ICE: Bit 5
    if 'SNOW_ICE' in bit_map:
        snow_condition = ((qa_band >> L8_SNOW_BIT) & 1).astype(bool)
        mask |= (snow_condition.astype(np.uint8) << bit_map['SNOW_ICE'])

    # WATER: Bit 7
    if 'WATER' in bit_map:
        water_condition = ((qa_band >> L8_WATER_BIT) & 1).astype(bool)
        mask |= (water_condition.astype(np.uint8) << bit_map['WATER'])

    logger.info("Successfully remapped Landsat 8 QA_PIXEL band.")
    return mask

def remap_sentinel2_scl(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Translates the official Sentinel-2 Scene Classification Layer (SCL)
    into the standardized eo_qamask format by mapping class values.
    See: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    """
    logger.info("Applying Sentinel-2 SCL remapping logic...")

    try:
        scl_band_index = config['bands']['scl']
        scl_band = data[scl_band_index]
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Config for Sentinel-2 is missing 'scl' band index or index is out of bounds: {e}")
        return np.zeros(data[0].shape, dtype=np.uint8)

    mask = np.zeros(scl_band.shape, dtype=np.uint8)

    # ESA SCL Classification Values
    S2_NO_DATA = 0
    S2_CLOUD_SHADOW = 3
    S2_WATER = 6
    S2_CLOUD_MEDIUM_PROB = 8
    S2_CLOUD_HIGH_PROB = 9
    S2_CIRRUS = 10
    S2_SNOW_ICE = 11

    # CLOUD: 8, 9, 10 (medium/high/cirrus)
    if 'CLOUD' in bit_map:
        cloud_condition = np.isin(scl_band, [S2_CLOUD_MEDIUM_PROB, S2_CLOUD_HIGH_PROB, S2_CIRRUS])
        mask |= (cloud_condition.astype(np.uint8) << bit_map['CLOUD'])

    # SHADOW: 3
    if 'SHADOW' in bit_map:
        shadow_condition = (scl_band == S2_CLOUD_SHADOW)
        mask |= (shadow_condition.astype(np.uint8) << bit_map['SHADOW'])

    # SNOW_ICE: 11
    if 'SNOW_ICE' in bit_map:
        snow_condition = (scl_band == S2_SNOW_ICE)
        mask |= (snow_condition.astype(np.uint8) << bit_map['SNOW_ICE'])

    # WATER: 6
    if 'WATER' in bit_map:
        water_condition = (scl_band == S2_WATER)
        mask |= (water_condition.astype(np.uint8) << bit_map['WATER'])

    # NO_DATA: 0
    if 'NO_DATA' in bit_map:
        nodata_condition = (scl_band == S2_NO_DATA)
        mask |= (nodata_condition.astype(np.uint8) << bit_map['NO_DATA'])

    logger.info("Successfully remapped Sentinel-2 SCL band.")
    return mask
