"""
Core logic for generating quality assessment masks for optical satellite sensors.

This module contains the rule-based algorithms for detecting common features
and artifacts like clouds, shadows, water, snow, and atmospheric haze
using spectral indices and thresholds.
"""
from __future__ import annotations

import logging
import numpy as np

# --- Set up a logger for this module ---
logger = logging.getLogger(__name__)


def _calculate_index(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    """
    Calculates a normalized difference index: (b1 - b2) / (b1 + b2).
    Numerically stable, handling divide-by-zero and NaNs.
    """
    b1 = band1.astype(np.float32)
    b2 = band2.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = b1 + b2
        index = np.zeros_like(b1, dtype=np.float32)
        mask = denominator != 0
        index[mask] = (b1[mask] - b2[mask]) / denominator[mask]
        index[~np.isfinite(index)] = 0
    return index


def _calculate_whiteness(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculates a 'whiteness' index for cloud/snow detection.
    Whiteness = 1 - (variance / mean) of the visible bands.
    """
    vis = np.stack([blue.astype(np.float32), green.astype(np.float32), red.astype(np.float32)], axis=0)
    mean = np.mean(vis, axis=0)
    var = np.var(vis, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_var = np.zeros_like(var)
        mask = mean != 0
        norm_var[mask] = var[mask] / mean[mask]
        whiteness = 1 - norm_var
    return np.clip(whiteness, 0, 1)


def generate_optical_mask(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Generates a QA bit-mask for optical sensors using a dynamic bit_map
    and thresholds defined in the sensor's configuration file.
    """
    mask = np.zeros(data[0].shape, dtype=np.uint8)
    bands = config.get('bands', {})
    thresholds = config.get('thresholds', {})

    # --- Step 1: Extract Core Bands Safely ---
    try:
        red = data[bands['red']]
        green = data[bands['green']]
        blue_idx = bands.get('blue', bands['green'])
        blue = data[blue_idx]
    except (KeyError, IndexError) as e:
        logger.error(f"A required band (red, green, or blue) is missing or index is out of bounds: {e}")
        return mask

    # --- Step 2: Calculate brightness and whiteness ---
    brightness = red.astype(np.float32) + green.astype(np.float32) + blue.astype(np.float32)
    whiteness = _calculate_whiteness(blue, green, red)

    # --- Step 3: Rule-based QA Bit Assignment ---
    # Shadows (only visible bands)
    if 'shadow_brightness' in thresholds and 'SHADOW' in bit_map:
        shadow_cond = (brightness < thresholds['shadow_brightness'])
        mask |= (shadow_cond.astype(np.uint8) << bit_map['SHADOW'])

    # NDVI/Water/Clouds (NIR required)
    if 'nir' in bands:
        try:
            nir = data[bands['nir']]
            ndvi = _calculate_index(nir, red)

            if 'water_ndwi' in thresholds and 'WATER' in bit_map:
                ndwi = _calculate_index(green, nir)
                water_cond = (ndwi > thresholds['water_ndwi'])
                mask |= (water_cond.astype(np.uint8) << bit_map['WATER'])

            if all(k in thresholds for k in ['cloud_brightness', 'cloud_whiteness', 'cloud_ndvi']) and 'CLOUD' in bit_map:
                cloud_cond = (
                    (brightness > thresholds['cloud_brightness']) &
                    (whiteness > thresholds['cloud_whiteness']) &
                    (ndvi < thresholds['cloud_ndvi'])
                )
                mask |= (cloud_cond.astype(np.uint8) << bit_map['CLOUD'])
        except IndexError:
            logger.warning("NIR band index out of bounds. Skipping NIR-based calculations.")
    else:
        # Fallback cloud detection
        if all(k in thresholds for k in ['cloud_brightness', 'cloud_whiteness']) and 'CLOUD' in bit_map:
            logger.warning("NIR band not found. Running cloud detection without NDVI.")
            cloud_cond = (
                (brightness > thresholds['cloud_brightness']) &
                (whiteness > thresholds['cloud_whiteness'])
            )
            mask |= (cloud_cond.astype(np.uint8) << bit_map['CLOUD'])

    # SWIR (Snow, Water, etc.)
    if 'swir' in bands:
        try:
            swir = data[bands['swir']]

            if all(k in thresholds for k in ['snow_ndsi', 'snow_brightness']) and 'SNOW_ICE' in bit_map:
                ndsi = _calculate_index(green, swir)
                snow_cond = (
                    (ndsi > thresholds['snow_ndsi']) &
                    (brightness > thresholds['snow_brightness'])
                )
                mask |= (snow_cond.astype(np.uint8) << bit_map['SNOW_ICE'])

            if 'water_mndwi' in thresholds and 'WATER' in bit_map:
                mndwi = _calculate_index(green, swir)
                water_cond_mndwi = (mndwi > thresholds['water_mndwi'])
                mask |= (water_cond_mndwi.astype(np.uint8) << bit_map['WATER'])
        except IndexError:
            logger.warning("SWIR band index out of bounds. Skipping SWIR-based calculations.")

    # --- NEW: Haze and Pollution Detection ---
    # Haze detection using a simple dark object subtraction approach.
    # This identifies pixels that are bright in the blue band over dark surfaces.
    if 'haze_dark_object' in thresholds and 'HAZE' in bit_map:
        if 'nir' in bands:
            try:
                nir = data[bands['nir']]
                # Condition for dark pixels (e.g., water, dense vegetation)
                dark_pixels = nir < thresholds.get('haze_dark_pixel_max_nir', 0.1)
                # Haze is bright in blue band, over dark pixels, and not already cloud
                is_not_cloud = (mask & (1 << bit_map.get('CLOUD', -1))) == 0
                haze_cond = (blue > thresholds['haze_dark_object']) & dark_pixels & is_not_cloud
                mask |= (haze_cond.astype(np.uint8) << bit_map['HAZE'])
                logger.info("Applied haze detection rule.")
            except (KeyError, IndexError):
                logger.warning("Could not apply haze detection due to missing NIR band.")
        else:
            logger.warning("Skipping haze detection: NIR band is required.")

    # Pollution detection using a simple blue/red ratio.
    # Elevated ratios can indicate aerosols from urban/industrial pollution.
    if 'pollution_br_ratio' in thresholds and 'POLLUTION' in bit_map:
        with np.errstate(divide='ignore', invalid='ignore'):
            # Using a simple ratio is more direct for this purpose
            br_ratio = np.divide(blue, red, where=red != 0)
            br_ratio[~np.isfinite(br_ratio)] = 0
        
        is_not_cloud = (mask & (1 << bit_map.get('CLOUD', -1))) == 0
        pollution_cond = (br_ratio > thresholds['pollution_br_ratio']) & is_not_cloud
        mask |= (pollution_cond.astype(np.uint8) << bit_map['POLLUTION'])
        logger.info("Applied pollution detection rule.")


    # --- Step 4: No-Data Mask ---
    nodata_val = config.get('globals', {}).get('nodata_value')
    if nodata_val is not None and 'NO_DATA' in bit_map:
        nodata_cond = (data[0] == nodata_val)
        mask |= (nodata_cond.astype(np.uint8) << bit_map['NO_DATA'])

    logger.info("Successfully generated optical QA mask.")
    return mask
