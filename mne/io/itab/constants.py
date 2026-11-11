"""ITAB constants."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from ...utils import BunchConst

ITAB = BunchConst()

# Channel types (``type`` field of each channel record in the .mhd header)
ITAB.ITABV_EEG_CH = 1
ITAB.ITABV_MAG_CH = 2
ITAB.ITABV_REF_EEG_CH = 4
ITAB.ITABV_REF_MAG_CH = 8
ITAB.ITABV_AUX_CH = 16
ITAB.ITABV_PARAM_CH = 32
ITAB.ITABV_DIGIT_CH = 64
ITAB.ITABV_FLAG_CH = 128

# Byte offsets of the blocks of the .mhd header
ITAB.MHD_TIME = 656
ITAB.MHD_NREFCH = 82780
ITAB.MHD_RAWHDTYPE = 85348
ITAB.MHD_CHINFO = 85444
ITAB.MHD_SENSPOS = 295364
ITAB.MHD_NMARKER = 295440
ITAB.MHD_MARKERS = 295700
ITAB.MHD_BESTCHI = 297232

# Size in bytes of one channel record, and number of channel records that fit
# between MHD_CHINFO and MHD_SENSPOS
ITAB.MHD_CH_SIZE = 328
ITAB.MHD_MAX_CH = (ITAB.MHD_SENSPOS - ITAB.MHD_CHINFO) // ITAB.MHD_CH_SIZE

# Number of coil weight / position slots in one channel record
ITAB.MHD_N_COIL_SLOTS = 10

# Value of the ``data_type`` header field for little-endian int32 samples;
# other values are read as big-endian int32
ITAB.DATA_TYPE_LITTLE_ENDIAN = 4
