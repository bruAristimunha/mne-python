# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import struct
from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import pick_types
from mne._fiff.constants import FIFF
from mne.datasets import testing
from mne.io import read_raw, read_raw_fieldtrip, read_raw_itab
from mne.io.itab.constants import ITAB
from mne.io.tests.test_raw import _test_raw_reader

data_path = testing.data_path(download=False)
# Not in mne-testing-data yet: a short ITAB recording (.raw + .raw.mhd) and the
# same data read with FieldTrip and saved as a raw structure, in T and V.
raw_itab_fname = data_path / "itab" / "test_itab.raw"
mat_itab_fname = data_path / "itab" / "test_itab.mat"

# label, type, unit, amvbit, calib, flag
_CHANNELS = (
    ("MAG_001", ITAB.ITABV_MAG_CH, "fT", 2.0, 1.0, 0),
    ("MAG_002", ITAB.ITABV_MAG_CH, "fT", 2.0, 4.0, 0),
    ("REF_001", ITAB.ITABV_REF_MAG_CH, "pT", 1.0, 1.0, 0),
    ("EEG_001", ITAB.ITABV_EEG_CH, "uV", 0.5, 1.0, 0),
    ("EEG_REF", ITAB.ITABV_REF_EEG_CH, "uV", 0.5, 1.0, 0),
    ("AUX_001", ITAB.ITABV_AUX_CH, "mV", 1.0, 1.0, 1),
    ("TRIG", ITAB.ITABV_DIGIT_CH, "", 1.0, 1.0, 0),
)
_SCALES = (2e-15, 0.5e-15, 1e-12, 0.5e-6, 0.5e-6, 1e-3, 1.0)
_MARKERS_MM = ((0, 90, 0), (75, 0, 0), (-75, 0, 0), (0, 0, 100), (10, 60, 50))


def _write_itab(fname, data, sfreq, data_type, start_data=1024):
    """Write a minimal ITAB .raw/.raw.mhd pair with the layout read_raw_itab uses."""
    n_channels, n_times = data.shape
    mhd = bytearray(ITAB.MHD_BESTCHI)
    struct.pack_into("32s32s32s", mhd, 22, b"Jane", b"Doe", b"S01")
    struct.pack_into("12s16s", mhd, ITAB.MHD_TIME, b"10:20:30", b"15/03/2024")
    struct.pack_into("<i", mhd, 684, n_channels)
    struct.pack_into("<i", mhd, 720, data_type)
    # smpfq, hw_low_fr (low-pass), hw_hig_fr (high-pass), then ntpdata
    struct.pack_into("<3f", mhd, 724, sfreq, 200.0, 0.5)
    struct.pack_into("<i", mhd, 748, n_times)
    # one segment (start, ntptot, ntppre, type, quality), no samples
    struct.pack_into("<i5ii", mhd, 752, 1, 100, 50, 0, 3, 0, 0)
    struct.pack_into("<i", mhd, ITAB.MHD_RAWHDTYPE + 80, start_data)
    for ci, (label, kind, unit, amvbit, calib, flag) in enumerate(_CHANNELS):
        offset = ITAB.MHD_CHINFO + ci * ITAB.MHD_CH_SIZE
        struct.pack_into(
            "<ii16sif", mhd, offset, kind, ci + 1, label.encode(), flag, amvbit
        )
        struct.pack_into("<f8si", mhd, offset + 32, calib, unit.encode(), 1)
        # first coil: position in mm, then orientation
        struct.pack_into("<6f", mhd, offset + 88, 10.0 * ci, 20.0, 30.0, 0, 0, 2.0)
    struct.pack_into("<i", mhd, ITAB.MHD_NMARKER, len(_MARKERS_MM))
    for mi, marker in enumerate(_MARKERS_MM):
        struct.pack_into("<3f", mhd, ITAB.MHD_MARKERS + 24 * mi, *marker)
    fname.with_name(fname.name + ".mhd").write_bytes(bytes(mhd))
    endian = "<" if data_type == ITAB.DATA_TYPE_LITTLE_ENDIAN else ">"
    samples = data.T.astype(f"{endian}i4").tobytes()  # multiplexed
    fname.write_bytes(bytes(start_data) + samples)


@pytest.mark.parametrize("data_type", [ITAB.DATA_TYPE_LITTLE_ENDIAN, 0])
def test_itab_synthetic(tmp_path, data_type):
    """Test reading a synthetic ITAB file."""
    rng = np.random.default_rng(0)
    data = rng.integers(-1000, 1000, (len(_CHANNELS), 1000))
    data[-1] = rng.integers(0, 4, data.shape[1])
    fname = tmp_path / "test_itab.raw"
    _write_itab(fname, data, 1000.0, data_type)

    raw = _test_raw_reader(read_raw_itab, fname=fname)
    assert_allclose(raw.get_data(), data * np.array(_SCALES)[:, np.newaxis])
    assert raw.get_channel_types() == [
        "mag",
        "mag",
        "ref_meg",
        "eeg",
        "eeg",
        "misc",
        "stim",
    ]
    assert raw.info["bads"] == ["AUX_001"]
    assert raw.info["sfreq"] == 1000.0
    assert_allclose([raw.info["highpass"], raw.info["lowpass"]], [0.5, 200.0])
    assert raw.info["meas_date"] == datetime(2024, 3, 15, 10, 20, 30, tzinfo=UTC)
    assert raw.info["subject_info"]["first_name"] == "Jane"
    assert raw._orig_units["MAG_001"] == "fT"
    meg = raw.info["chs"][0]
    assert meg["coord_frame"] == FIFF.FIFFV_COORD_DEVICE
    assert_allclose(meg["loc"][:3], [0, 0.02, 0.03])
    assert_allclose(meg["loc"][9:], [0, 0, 1])  # coil normal, unit length
    dig = raw.info["dig"]
    kinds = [d["kind"] for d in dig]
    assert kinds == [FIFF.FIFFV_POINT_CARDINAL] * 3 + [FIFF.FIFFV_POINT_HPI] + [
        FIFF.FIFFV_POINT_EXTRA
    ]
    nasion = [d for d in dig if d["ident"] == FIFF.FIFFV_POINT_NASION][0]
    assert_allclose(nasion["r"], [0, 0.09, 0])
    assert_allclose(raw.annotations.onset, [0.1])
    assert_allclose(raw.annotations.duration, [0.05])
    assert list(raw.annotations.description) == ["3"]
    # read_raw dispatches .raw files to the ITAB reader
    assert_allclose(read_raw(fname).get_data(), raw.get_data())


@testing.requires_testing_data
@pytest.mark.skipif(
    not (raw_itab_fname.is_file() and mat_itab_fname.is_file()),
    reason="ITAB files not yet in mne-testing-data",
)
def test_itab_raw():
    """Test reading ITAB .raw files against FieldTrip."""
    raw = _test_raw_reader(read_raw_itab, fname=raw_itab_fname)
    assert len(pick_types(raw.info, meg=True)) > 0
    ft_raw = read_raw_fieldtrip(mat_itab_fname, info=raw.info, data_name="data")
    assert ft_raw.info["sfreq"] == raw.info["sfreq"]
    assert_allclose(raw.get_data(), ft_raw.get_data(), rtol=1e-6, atol=0)
