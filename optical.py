# eo_qamask/logic/optical.py

import numpy as np
# The QABits import is no longer needed here

def _calculate_index(band1, band2):
    """Calculates a normalized difference index: (b1 - b2) / (b1 + b2)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        index = np.divide(band1.astype(np.float32) - band2.astype(np.float32),
                          band1.astype(np.float32) + band2.astype(np.float32))
        index[~np.isfinite(index)] = 0
        return index

def generate_optical_mask(data: np.ndarray, config: dict, bit_map: dict) -> np.ndarray:
    """
    Generates a QA bit-mask for optical sensors using a dynamic bit_map.
    """
    mask = np.zeros(data[0].shape, dtype=np.uint8)
    bands = config['bands']
    thresholds = config['thresholds']

    # --- Extract bands ---
    red = data[bands['red']]
    green = data[bands['green']]
    nir = data[bands['nir']]
    blue = data[bands.get('blue', green)]
    brightness = red + green + blue

    # --- Apply rules using the dynamic bit_map ---
    # The .get(BIT, -1) pattern safely skips a rule if the bit is not in the map.
    # Shifting by -1 has no effect.
    
    if 'shadow' in thresholds:
        shadow_cond = (brightness < thresholds['shadow']['brightness_max'])
        mask |= (shadow_cond.astype(np.uint8) << bit_map.get('SHADOW', -1))
    
    if 'water' in thresholds:
        ndwi = _calculate_index(green, nir)
        water_cond = (ndwi > thresholds['water']['ndwi_min'])
        mask |= (water_cond.astype(np.uint8) << bit_map.get('WATER', -1))

    if 'swir' in bands:
        swir = data[bands['swir']]
        ndsi = _calculate_index(green, swir)
        ndvi = _calculate_index(nir, red)
        
        if 'cloud' in thresholds:
            cloud_cond = (
                (brightness > thresholds['cloud']['brightness_min']) &
                (ndsi < thresholds['cloud']['ndsi_max']) &
                (ndvi < thresholds['cloud']['ndvi_max'])
            )
            mask |= (cloud_cond.astype(np.uint8) << bit_map.get('CLOUD', -1))
        
        if 'snow' in thresholds:
            snow_cond = (
                (ndsi > thresholds['snow']['ndsi_min']) &
                (ndvi < thresholds['snow']['ndvi_max'])
            )
            mask |= (snow_cond.astype(np.uint8) << bit_map.get('SNOW_ICE', -1))
    
    if 'globals' in config and 'nodata_value' in config['globals']:
        nodata_cond = (data[0] == config['globals']['nodata_value'])
        mask |= (nodata_cond.astype(np.uint8) << bit_map.get('NO_DATA', -1))
    
    return mask