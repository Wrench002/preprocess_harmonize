"""
Standardized bit definitions for the eo_qamask library.

This module defines the universal, standardized set of QA bit positions
used across all sensor masking logic.
"""
from __future__ import annotations

from enum import Enum

class QABits(Enum):
    """
    Universal 8-bit QA mask for eo_qamask.

    Bit positions (0–7):

      0: CLOUD         (optical)
      1: SHADOW        (optical/SAR)
      2: SNOW_ICE      (optical)
      3: WATER         (optical/SAR)
      4: NO_DATA       (all)
      5: HAZE          (optical, new)
      6: POLLUTION     (optical, new)
      7: SPECKLE       (SAR only)
    """
    CLOUD      = 0
    SHADOW     = 1
    SNOW_ICE   = 2
    WATER      = 3
    NO_DATA    = 4
    HAZE       = 5
    POLLUTION  = 6
    SPECKLE    = 7
