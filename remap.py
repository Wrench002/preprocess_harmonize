# preprocess_harmonize/eo_qamask/logic/remap.py
import numpy as np

def remap_landsat8_qa(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Translates the official Landsat 8 QA_PIXEL band into the standardized
    eo_qamask format.
    """
    print("  - Applying Landsat 8 QA band remapping logic...")
    
    try:
        qa_band_index = config['bands']['qa_pixel']
        qa_band = data[qa_band_index]
    except (KeyError, IndexError):
        raise ValueError("Config for Landsat 8 is missing 'qa_pixel' band index.")

    mask = np.zeros(qa_band.shape, dtype=np.uint8)

    L8_DILATED_CLOUD_BIT = 1
    L8_CIRRUS_BIT = 2
    L8_CLOUD_BIT = 3
    L8_CLOUD_SHADOW_BIT = 4
    L8_SNOW_BIT = 5
    L8_WATER_BIT = 7

    cloud_condition = (
        (qa_band & (1 << L8_DILATED_CLOUD_BIT)) |
        (qa_band & (1 << L8_CIRRUS_BIT)) |
        (qa_band & (1 << L8_CLOUD_BIT))
    ).astype(bool)
    mask |= (cloud_condition.astype(np.uint8) << bit_map.get('CLOUD', -1))

    shadow_condition = (qa_band & (1 << L8_CLOUD_SHADOW_BIT)).astype(bool)
    mask |= (shadow_condition.astype(np.uint8) << bit_map.get('SHADOW', -1))

    snow_condition = (qa_band & (1 << L8_SNOW_BIT)).astype(bool)
    mask |= (snow_condition.astype(np.uint8) << bit_map.get('SNOW_ICE', -1))

    water_condition = (qa_band & (1 << L8_WATER_BIT)).astype(bool)
    mask |= (water_condition.astype(np.uint8) << bit_map.get('WATER', -1))

    return mask


def remap_sentinel2_scl(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Translates the official Sentinel-2 Scene Classification Layer (SCL)
    into the standardized eo_qamask format.
    
    This function reads the integer values from the SCL band and maps them
    to our internal bit mask conditions.
    """
    print("  - Applying Sentinel-2 SCL remapping logic...")

    try:
        scl_band_index = config['bands']['scl']
        scl_band = data[scl_band_index]
    except (KeyError, IndexError):
        raise ValueError("Config for Sentinel-2 is missing 'scl' band index.")

    mask = np.zeros(scl_band.shape, dtype=np.uint8)

    # --- Official ESA SCL classification values ---
    # See: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
    S2_CLOUD_SHADOW = 3
    S2_CLOUD_MEDIUM_PROB = 8
    S2_CLOUD_HIGH_PROB = 9
    S2_CIRRUS = 10
    S2_SNOW_ICE = 11

    # --- Map S2 classes to our standardized bits ---

    # Cloud condition (medium, high, or cirrus)
    cloud_condition = (
        (scl_band == S2_CLOUD_MEDIUM_PROB) |
        (scl_band == S2_CLOUD_HIGH_PROB) |
        (scl_band == S2_CIRRUS)
    )
    mask |= (cloud_condition.astype(np.uint8) << bit_map.get('CLOUD', -1))

    # Shadow condition
    shadow_condition = (scl_band == S2_CLOUD_SHADOW)
    mask |= (shadow_condition.astype(np.uint8) << bit_map.get('SHADOW', -1))
    
    # Snow condition
    snow_condition = (scl_band == S2_SNOW_ICE)
    mask |= (snow_condition.astype(np.uint8) << bit_map.get('SNOW_ICE', -1))
    
    return mask