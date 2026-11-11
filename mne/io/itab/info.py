"""Build measurement info."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from datetime import UTC, datetime

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info
from ..._fiff.tag import _coil_trans_to_loc
from ...annotations import Annotations
from ...surface import _normal_orth
from ...transforms import Transform
from .constants import ITAB


def _convert_time(date_str, time_str):
    """Convert date and time strings to float time."""
    for fmt in ("%d/%m/%Y", "%d-%b-%Y", "%a, %b %d, %Y"):
        try:
            date = datetime.strptime(date_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError(
            f"Illegal date: {date_str}.\nIf the language of the date does not "
            "correspond to your local machine's language try to set the "
            "locale to the language of the date string:\n"
            'locale.setlocale(locale.LC_ALL, "en_US")'
        )

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            time = datetime.strptime(time_str, fmt)
        except ValueError:
            pass
        else:
            break
    else:
        raise RuntimeError(f"Illegal time: {time_str}.")
    # MNE-C uses mktime which uses local time, but here we instead decouple
    # conversion location from the process, and instead assume that the
    # acquisition was in GMT. This will be wrong for most sites, but at least
    # the value we obtain here won't depend on the geographical location
    # that the file was converted.
    res = datetime(
        date.year,
        date.month,
        date.day,
        time.hour,
        time.minute,
        time.second,
        tzinfo=UTC,
    )

    return res


# kind, coil_type and unit of each ITAB channel type; others (auxiliary, parameter
# and flag channels) are misc
_CH_KINDS = {
    ITAB.ITABV_MAG_CH: (
        FIFF.FIFFV_MEG_CH,
        FIFF.FIFFV_COIL_POINT_MAGNETOMETER,
        FIFF.FIFF_UNIT_T,
    ),
    ITAB.ITABV_REF_MAG_CH: (
        FIFF.FIFFV_REF_MEG_CH,
        FIFF.FIFFV_COIL_POINT_MAGNETOMETER,
        FIFF.FIFF_UNIT_T,
    ),
    ITAB.ITABV_EEG_CH: (FIFF.FIFFV_EEG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    ITAB.ITABV_REF_EEG_CH: (FIFF.FIFFV_EEG_CH, FIFF.FIFFV_COIL_EEG, FIFF.FIFF_UNIT_V),
    ITAB.ITABV_DIGIT_CH: (
        FIFF.FIFFV_STIM_CH,
        FIFF.FIFFV_COIL_NONE,
        FIFF.FIFF_UNIT_NONE,
    ),
}
_MISC_KIND = (FIFF.FIFFV_MISC_CH, FIFF.FIFFV_COIL_NONE, FIFF.FIFF_UNIT_V)
_UNIT_SCALES = {"fT": 1e-15, "pT": 1e-12, "uV": 1e-6, "mV": 1e-3}


def _mhdch_2_chs(mhd_ch):
    """Build chs list item from mhd ch list item."""
    kind, coil_type, unit = _CH_KINDS.get(mhd_ch["type"], _MISC_KIND)
    # samples * amvbit / calib are in the header unit, convert them to SI
    cal = mhd_ch["amvbit"] / mhd_ch["calib"] if mhd_ch["calib"] != 0 else 1.0
    cal *= _UNIT_SCALES.get(mhd_ch["unit"], 1.0)
    ch = dict(
        ch_name=mhd_ch["label"],
        kind=kind,
        coil_type=coil_type,
        unit=unit,
        unit_mul=FIFF.FIFF_UNITM_NONE,
        cal=float(cal),
        range=1.0,
        loc=np.full(12, np.nan),
        coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
        logno=int(mhd_ch["number"]),
    )
    if mhd_ch["type"] == ITAB.ITABV_MAG_CH:
        coil = mhd_ch["pos"][0]
        trans = np.eye(4)
        trans[:3, 3] = np.array([coil["posx"], coil["posy"], coil["posz"]]) / 1000
        normal = np.array([coil["orix"], coil["oriy"], coil["oriz"]])
        trans[:3, :3] = _normal_orth(normal / np.linalg.norm(normal)).T
        ch["loc"] = _coil_trans_to_loc(trans)
        ch["coord_frame"] = FIFF.FIFFV_COORD_DEVICE
    return ch


def _mhd2info(mhd):
    """Create meas info and annotations from ITAB mhd data."""
    info = _empty_info(mhd["smpfq"])
    info["meas_date"] = _convert_time(mhd["date"], mhd["time"])
    info["description"] = mhd["notes"]
    info["subject_info"] = dict(
        last_name=mhd["last_name"],
        first_name=mhd["first_name"],
        sex=int(mhd["subinfo"]["sex"][0]),
    )
    info["chs"] = [_mhdch_2_chs(ch) for ch in mhd["ch"]]
    for ci, ch in enumerate(info["chs"], 1):
        ch["scanno"] = ci
    info._update_redundant()
    info["bads"] = [ch["label"] for ch in mhd["ch"] if ch["flag"] == 1]
    if mhd["hw_low_fr"] != 0:
        info["lowpass"] = mhd["hw_low_fr"]
    if mhd["hw_hig_fr"] != 0:
        info["highpass"] = mhd["hw_hig_fr"]

    # The header stores sensor positions and markers relative to the subject's
    # head, so the device and head frames are the same
    info["dev_head_t"] = Transform("meg", "head")
    markers = [[m["posx"], m["posy"], m["posz"]] for m in mhd["marker"]]
    markers = np.array(markers, float).reshape(-1, 3) / 1000
    if len(markers) >= 3:
        # nasion, right and left preauricular points, vertex, then HPI coils
        info["dig"] = _make_dig_points(
            nasion=markers[0],
            rpa=markers[1],
            lpa=markers[2],
            hpi=markers[4:],
            extra_points=markers[3:4],
        )
    info._unlocked = False
    info._check_consistency()

    sfreq = info["sfreq"]
    annotations = Annotations(
        onset=[segment["start"] / sfreq for segment in mhd["segment"]],
        duration=[segment["ntptot"] / sfreq for segment in mhd["segment"]],
        description=[str(segment["type"]) for segment in mhd["segment"]],
    )
    return info, annotations
