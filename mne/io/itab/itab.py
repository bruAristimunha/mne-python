# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np

from ..._fiff._digitization import _make_dig_points
from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _read_segments_file
from ...annotations import Annotations
from ...surface import _normal_orth
from ...transforms import Transform
from ...utils import (
    _check_fname,
    _verbose_control,
    verbose_static,
    warn,
)
from ..base import BaseRaw

# One record of the channel table (328 bytes)
_CH_DTYPE = np.dtype(
    [
        ("type", "<i4"),
        ("number", "<i4"),
        ("label", "S16"),
        ("flag", "<i4"),
        ("amvbit", "<f4"),
        ("calib", "<f4"),
        ("unit", "S8"),
        ("ncoils", "<i4"),
        ("wgt", "<f4", 10),
        ("pos", "<f4", (10, 6)),  # per coil: x, y, z (mm), then the normal
    ]
)
# The used fields of the .raw.mhd header: (name, format, byte offset). The segment
# and marker arrays fill the room up to the next header block.
_MHD_FIELDS = [
    ("first_name", "S32", 22),
    ("last_name", "S32", 54),
    ("notes", "S256", 118),
    ("sex", "u1", 374),
    ("time", "S12", 656),
    ("date", "S16", 668),
    ("nchan", "<i4", 684),
    ("data_type", "<i4", 720),
    ("smpfq", "<f4", 724),
    ("hw_low_fr", "<f4", 728),  # low-pass
    ("hw_hig_fr", "<f4", 732),  # high-pass
    ("ntpdata", "<i4", 748),
    ("no_segments", "<i4", 752),
    ("segment", ("<i4", (4101, 5)), 756),  # start, ntptot, ntppre, type, quality
    ("start_data", "<i4", 85428),
    ("ch", (_CH_DTYPE, 640), 85444),
    ("num_markers", "<i4", 295440),
    ("marker", ("<f4", (63, 6)), 295700),  # x, y, z (mm), then orientation
]
_names, _formats, _offsets = zip(*_MHD_FIELDS)
_MHD_DTYPE = np.dtype(dict(names=_names, formats=_formats, offsets=_offsets))
# ITAB channel type -> MNE channel type, others (aux, parameter, flag) are misc
_CH_TYPES = {1: "eeg", 2: "mag", 4: "eeg", 8: "ref_meg", 64: "stim"}
_UNIT_SCALES = {"fT": 1e-15, "pT": 1e-12, "uV": 1e-6, "mV": 1e-3}


class RawITAB(BaseRaw):
    """Raw object from an ITAB MEG file, see :func:`mne.io.read_raw_itab`."""

    @_verbose_control
    def __init__(self, fname, preload=False, *, verbose=None):
        fname = _check_fname(fname, "read", True, "fname")
        fname_mhd = _check_fname(fname.parent / f"{fname.name}.mhd", "read", True)
        mhd = np.fromfile(fname_mhd, _MHD_DTYPE, count=1)[0]
        chs = mhd["ch"][: mhd["nchan"]]
        ch_names = [label.split(b"\0")[0].decode() for label in chs["label"]]
        units = [unit.split(b"\0")[0].decode() for unit in chs["unit"]]
        ch_types = [_CH_TYPES.get(kind, "misc") for kind in chs["type"]]
        info = create_info(ch_names, float(mhd["smpfq"]), ch_types)
        for ch, mhd_ch, unit in zip(info["chs"], chs, units):
            # samples * amvbit / calib are in the header unit, store SI data
            cal = mhd_ch["amvbit"] / mhd_ch["calib"] if mhd_ch["calib"] else 1.0
            ch["cal"] = float(cal) * _UNIT_SCALES.get(unit, 1.0)
            ch["logno"] = int(mhd_ch["number"])
            if ch["kind"] in (FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH):
                ch["coil_type"] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
            if ch["kind"] == FIFF.FIFFV_MEG_CH:  # first coil, normal as ez
                pos = mhd_ch["pos"][0].astype(float)
                ch["loc"] = np.concatenate(
                    [pos[:3] / 1000, _normal_orth(pos[3:]).ravel()]
                )
        info["bads"] = [name for name, ch in zip(ch_names, chs) if ch["flag"] == 1]

        # date and time are stored as text in one of these formats
        when = " ".join(mhd[key].split(b"\0")[0].decode() for key in ("date", "time"))
        meas_date = None
        for date_fmt, time_fmt in product(
            ("%d/%m/%Y", "%d-%b-%Y", "%a, %b %d, %Y"), ("%H:%M:%S", "%H:%M")
        ):
            try:
                meas_date = datetime.strptime(when, f"{date_fmt} {time_fmt}")
                meas_date = meas_date.replace(tzinfo=UTC)
                break
            except ValueError:
                pass
        else:
            warn(f"Could not parse the measurement date {when!r}, setting it to None.")

        # sensor positions and markers are stored relative to the subject's head,
        # so the device and head frames are the same
        markers = mhd["marker"][: mhd["num_markers"], :3] / 1000
        with info._unlock():
            info["meas_date"] = meas_date
            info["description"] = mhd["notes"].split(b"\0")[0].decode()
            info["subject_info"] = dict(
                first_name=mhd["first_name"].split(b"\0")[0].decode(),
                last_name=mhd["last_name"].split(b"\0")[0].decode(),
                sex=int(mhd["sex"]),
            )
            if mhd["hw_low_fr"]:
                info["lowpass"] = float(mhd["hw_low_fr"])
            if mhd["hw_hig_fr"]:
                info["highpass"] = float(mhd["hw_hig_fr"])
            info["dev_head_t"] = Transform("meg", "head")
            if len(markers) >= 3:  # nasion, RPA, LPA, vertex, then HPI coils
                info["dig"] = _make_dig_points(
                    nasion=markers[0],
                    rpa=markers[1],
                    lpa=markers[2],
                    hpi=markers[4:],
                    extra_points=markers[3:4],
                )

        # multiplexed int32 samples; a Python int offset so that byte offsets past
        # 2 GiB do not overflow
        dtype = "<i4" if mhd["data_type"] == 4 else ">i4"
        raw_extras = dict(dtype=dtype, offset=int(mhd["start_data"]))
        super().__init__(
            info,
            preload,
            last_samps=[int(mhd["ntpdata"]) - 1],
            filenames=[fname],
            raw_extras=[raw_extras],
            orig_units=dict(zip(ch_names, units)),
            orig_format="int",
            verbose=verbose,
        )
        segment = mhd["segment"][: mhd["no_segments"]]
        sfreq = self.info["sfreq"]
        self.set_annotations(
            Annotations(segment[:, 0] / sfreq, segment[:, 1] / sfreq, segment[:, 3])
        )

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        dtype, offset = self._raw_extras[fi]["dtype"], self._raw_extras[fi]["offset"]
        _read_segments_file(
            self, data, idx, fi, start, stop, cals, mult, dtype, offset=offset
        )


@verbose_static("preload")
def read_raw_itab(
    fname: Path | str,
    preload: bool | str = False,
    *,
    verbose: bool | str | int | None = None,
) -> RawITAB:
    """Reader for an ITAB MEG file.

    Parameters
    ----------
    fname : path-like
        Path to the ITAB data file (``.raw``). The header file with the same name
        plus ``.mhd`` (e.g., ``run01.raw.mhd``) must be in the same directory.
    preload : bool | str
        Preload data into memory for data manipulation and faster indexing.
        If True, the data will be preloaded into memory (fast, requires
        large amount of memory). If preload is a string, it is the name of a
        freshly created memory-mapped file used to store the data on the hard
        drive (slower, requires less memory). An existing file is overwritten.
        The caller owns the file and is responsible for removing it after the
        Raw object is no longer in use. For supported Raw readers, the exact string
        ``"auto"`` instead reuses decoded data below the directory configured by
        :func:`mne.set_cache_dir`. Entries persist without a size limit and are mapped
        copy-on-write. Use ``Path("auto")`` for a literal filename.

        .. versionchanged:: 1.13
           Support for the ``"auto"`` decoded-data cache was added.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw : instance of RawITAB
        A Raw object containing ITAB data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawITAB.

    Notes
    -----
    The ``.mhd`` header stores the sensor positions and the head markers relative
    to the subject's head, so ``info["dev_head_t"]`` is the identity. The first three
    markers are read as the nasion, right and left preauricular points, the fourth
    as an extra head point, and any others as HPI coils.

    .. versionadded:: 1.14
    """
    return RawITAB(fname, preload=preload, verbose=verbose)
