# eo_qamask/bitdefs.py

from enum import Enum

class QABits(Enum):
    """
    A universal, standardized set of QA bit definitions.
    Each attribute corresponds to a specific bit position in the 8-bit mask.
    """
    CLOUD = 0
    SHADOW = 1
    SNOW_ICE = 2
    WATER = 3
    NO_DATA = 4
    SATURATION = 5
    GEO_DISTORT = 6
    SPECKLE = 7