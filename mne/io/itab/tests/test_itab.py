# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import struct
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne._fiff.constants import FIFF
from mne.io import read_raw, read_raw_itab
from mne.io.tests.test_raw import _test_raw_reader

# label, ITAB channel type, unit, amvbit, calib, flag, expected cal
_CHANNELS = (
    ("MAG_001", 2, "fT", 2.0, 1.0, 0, 2e-15),
    ("MAG_002", 2, "fT", 2.0, 4.0, 0, 0.5e-15),
    ("REF_001", 8, "pT", 1.0, 1.0, 0, 1e-12),
    ("EEG_001", 1, "uV", 0.5, 1.0, 0, 0.5e-6),
    ("EEG_REF", 4, "uV", 0.5, 1.0, 0, 0.5e-6),
    ("AUX_001", 16, "mV", 1.0, 1.0, 1, 1e-3),
    ("TRIG", 64, "", 1.0, 1.0, 0, 1.0),
)
# nasion, RPA, LPA, vertex, HPI coil (mm)
_MARKERS = ((0, 90, 0), (75, 0, 0), (-75, 0, 0), (0, 0, 100), (10, 60, 50))


@pytest.mark.parametrize("data_type, endian", [(4, "<"), (0, ">")])
def test_itab_synthetic(tmp_path, data_type, endian):
    """Test reading a synthetic ITAB .raw/.raw.mhd pair."""
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, (len(_CHANNELS), 1000))
    data[-1] = rng.integers(0, 4, data.shape[1])
    # header: byte offsets of the ITAB .raw.mhd layout
    mhd = bytearray(297232)
    struct.pack_into("32s32s", mhd, 22, b"Jane", b"Doe")
    struct.pack_into("12s16s", mhd, 656, b"10:20:30", b"15/03/2024")
    struct.pack_into("<i", mhd, 684, len(_CHANNELS))
    # data_type, smpfq, hw_low_fr (low-pass), hw_hig_fr (high-pass)
    struct.pack_into("<i3f", mhd, 720, data_type, 1000.0, 200.0, 0.5)
    # ntpdata, one segment (start, ntptot, ntppre, type, quality)
    struct.pack_into("<ii5i", mhd, 748, data.shape[1], 1, 100, 50, 0, 3, 0)
    struct.pack_into("<i", mhd, 85428, 1024)  # start_data
    for ci, (label, kind, unit, amvbit, calib, flag, _) in enumerate(_CHANNELS):
        offset = 85444 + ci * 328
        struct.pack_into(
            "<ii16sif", mhd, offset, kind, ci + 1, label.encode(), flag, amvbit
        )
        struct.pack_into("<f8si", mhd, offset + 32, calib, unit.encode(), 1)
        # first coil: position (mm), then normal
        struct.pack_into("<6f", mhd, offset + 88, 10.0 * ci, 20.0, 30.0, 0, 0, 2.0)
    struct.pack_into("<i", mhd, 295440, len(_MARKERS))
    for mi, marker in enumerate(_MARKERS):
        struct.pack_into("<3f", mhd, 295700 + 24 * mi, *marker)
    fname = tmp_path / "test_itab.raw"
    (tmp_path / "test_itab.raw.mhd").write_bytes(mhd)
    # multiplexed int32 samples after start_data bytes
    fname.write_bytes(bytes(1024) + data.T.astype(f"{endian}i4").tobytes())

    raw = _test_raw_reader(read_raw_itab, fname=fname)
    cals = np.array([ch[-1] for ch in _CHANNELS])
    assert_allclose(raw.get_data(), data * cals[:, np.newaxis])
    assert raw.get_channel_types() == "mag mag ref_meg eeg eeg misc stim".split()
    assert raw.info["bads"] == ["AUX_001"]
    assert_allclose([raw.info["highpass"], raw.info["lowpass"]], [0.5, 200.0])
    assert raw.info["meas_date"] == datetime(2024, 3, 15, 10, 20, 30, tzinfo=UTC)
    assert raw.info["subject_info"]["first_name"] == "Jane"
    assert raw._orig_units["MAG_001"] == "fT"
    meg = raw.info["chs"][1]
    assert meg["coil_type"] == FIFF.FIFFV_COIL_POINT_MAGNETOMETER
    assert_allclose(meg["loc"][:3], [0.01, 0.02, 0.03])
    assert_allclose(meg["loc"][9:], [0, 0, 1])  # unit coil normal
    kinds = [d["kind"] for d in raw.info["dig"]]
    assert kinds == [FIFF.FIFFV_POINT_CARDINAL] * 3 + [
        FIFF.FIFFV_POINT_HPI,
        FIFF.FIFFV_POINT_EXTRA,
    ]
    nasion = raw.info["dig"][1]
    assert nasion["ident"] == FIFF.FIFFV_POINT_NASION
    assert_allclose(nasion["r"], [0, 0.09, 0])
    assert_allclose(raw.annotations.onset, [0.1])
    assert_allclose(raw.annotations.duration, [0.05])
    assert list(raw.annotations.description) == ["3"]
    assert_allclose(read_raw(fname).get_data(), raw.get_data())
